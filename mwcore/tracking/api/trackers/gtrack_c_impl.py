import time
from typing import List, Literal, Optional
import numpy as np
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

from mwcore.registry import TRACKERS

from mwcore.tracking.src.algs.gtrack import BatchedData
from ..base import BaseTracker

from mwcore.tracking.src.algs.gtrack_c_impl import GTrackCBuffer 
from mwcore.tracking.src.algs.gtrack_c_impl import GTrackCConfig
from mwcore.tracking.src.algs.gtrack_c.gtrack_interface import GTRACK_STATE_VECTOR_TYPE, GTRACK_VERBOSE_TYPE

import logging
log= logging.getLogger(__name__)

@TRACKERS.register_module()
class GTrackCTracker(BaseTracker):
    def __init__(self, 
                 keep_radial: bool,
                 tracker_params: dict,
                    do_dev2standard: bool = False,
                 radar_cfg: Optional[dict] = None):
        super().__init__(radar_cfg=radar_cfg)
        self.keep_radial = keep_radial
        self.config = make_config_gtrackc(tracker_params)
        self.tracker = GTrackCBuffer(self.config)
        self.do_dev2standard = do_dev2standard
        # Adjust batch size based on what data you're storing
        # Note: The new implementation doesn't use FB_FRAMES_BATCH in the same way
        self.batch = BatchedData(3, np.empty((0, 11 if self.keep_radial else 8)))
        self.last_time = time.time()
        
    def consume(self, 
                det_obj: dict = None, 
                point_array: np.ndarray = None, 
                sort_metric: Optional['Literal["size", "snr", "rel"]'] = None, 
                **kwargs) -> List[np.ndarray]:
        effective_data = self.normalize_data(
            det_obj=det_obj, 
            point_array=point_array, 
            keepRadial=self.keep_radial, 
            transform=self.do_dev2standard
        )
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.tracker.dt = dt
        if effective_data.shape[0] > 0:
            self.tracker.track(effective_data)
        states = [track["position"] for track in self.tracker.effective_tracks]
        # Address different data format (invert x (0) and exchange y (1) and z (2) for (0,2,1))
        states = [np.array([s[1], s[0], s[2]]) for s in states]
        # TODO - Implement sorting based on the sort_metric
        if sort_metric is not None:
            sorted_indices = self.sort_results(metric="confidence", **kwargs)
            states = [states[i] for i in sorted_indices]
        return states

    def sort_results(self, metric: 'Literal["confidence"]' = 'confidence', **kwargs) -> np.ndarray:
        tracks = self.tracker.effective_tracks
        if metric == 'confidence':
            confidence_levels = [track["confidence"] for track in tracks]
            if not confidence_levels:
                return np.array([])
            sorted_indices = np.argsort(confidence_levels)[::-1]
            return sorted_indices

    
def make_config_gtrackc(raw: dict) -> GTrackCConfig:
    """Create a GTrackC configuration from raw parameters with equivalent values to Python tracker."""
    
    # Gating limits [depth, width, height, velocity]
    kf_spread_lim = raw.get("KF_SPREAD_LIM", [1.5, 1.5, 2.0, 1.0])
    gating_limits = [
        kf_spread_lim[0],  # depth (x)
        kf_spread_lim[1],  # width (y)
        kf_spread_lim[2],  # height (z)
        kf_spread_lim[3],  # radial velocity 
    ]
    
    boundary_box = {
        'x1': -5.0,   # Left boundary (negative X)
        'x2': 5.0,    # Right boundary (positive X)  
        'y1': -5.0,   # Back boundary (negative Y)
        'y2': 5.0,    # Front boundary (positive Y)
        'z1': -5.0,   # Bottom boundary (negative Z)
        'z2': 5.0     # Top boundary (positive Z)
    }
    # Scenery parameters with boundary box based on spread limits. Im not using them right now
    scenery_params = {
        'num_boundary_boxes': 1,  # Enable boundary checking with 1 box
        'boundary_boxes': [boundary_box],  # List of boundary boxes
        'num_static_boxes': 0,     # No static boxes for now
        'static_boxes': [],        # Empty list of static boxes
        # 'sensor_orientation': [0, 0],
        # 'sensor_position': [0, 0, 1.0]
    }
    
    # State parameters based on lifetime thresholds. Defaults from the original code
    state_params = {
    'det2actThre': 3,          
    'det2freeThre': 3,         
    'active2freeThre': 10,      
    'static2freeThre': 40,      
    'exit2freeThre': 5,         
    'sleep2freeThre': 1000      
    }
    
    # Allocation parameters based on DBSCAN values
    allocation_params = {
        'pointsThre': raw.get("DB_MIN_SAMPLES_MIN", 5),
        'maxDistanceThre': raw.get("DB_EPS", 2),
        'velocityThre': raw.get("TR_VEL_THRES", 0),
        'snrThre': raw.get("DB_POINTS_THRES", 100),
        'snrThreObscured': raw.get("DB_POINTS_THRES", 100),
        'maxVelThre': 2.0
    }
    
    return GTrackCConfig(
        max_units=raw.get("TR_MAX_TRACKS", 4),
        is_ceiling_mounted=False,
        gating_limits=gating_limits,
        allocation_params=allocation_params,
        scenery_params=scenery_params,
        state_params=state_params,
        frame_rate_ms=1000.0 / raw.get("FRAME_RATE", 10),  # Convert FPS to ms
        stateVectorType=GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DA,
        verbose=GTRACK_VERBOSE_TYPE.GTRACK_VERBOSE_ERROR,
        maxNumPoints=1000,
        maxNumTracks=raw.get("TR_MAX_TRACKS", 4),
        initialRadialVelocity=0,
        maxRadialVelocity=raw.get("TR_VEL_THRES", 0.12) * 5,
        radialVelocityResolution=0.01,
        maxAcceleration=(0.5, 0.5, 0.1),  # x,y,z acceleration limits
        deltaT=1.0 / raw.get("FRAME_RATE", 10)  # Time delta in seconds
    )