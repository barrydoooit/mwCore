import time
from typing import Dict, Any, List, Optional
import numpy as np
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

from mwcore.registry import TRACKERS

from mwcore.tracking.src.algs.gtrack import BatchedData
from ..base import BaseTracker

from mwcore.tracking.src.algs.gtrack_c_impl import GTrackCBuffer 
from mwcore.tracking.src.algs.gtrack_c_impl import GTrackCConfig
from mwcore.tracking.src.algs.gtrack_c.gtrack_interface import GTRACK_STATE_VECTOR_TYPE, GTRACK_VERBOSE_TYPE

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
        
    
        
    def consume(self, det_obj: dict) -> Optional[List[Dict[str, Any]]]:
        """Process detection objects through the tracker"""
        effective_data = self.normalize_data(det_obj=det_obj, keepRadial=self.keep_radial, transform=self.do_dev2standard)

        if effective_data.shape[0] == 0:
            return None
            
        # Update timing
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        # Columns:
            # - x-coordinate
            # - y-coordinate
            # - z-coordinate
            # - Cartesian velocity along the x-axis
            # - Cartesian velocity along the y-axis
            # - Cartesian velocity along the z-axis
            # - doppler
            # - peakval
            # - range (radial distance)
            # - theta (angle in polar coordinates)
            # - radial velocity (ṙ)


        tracks = self.tracker.track(effective_data)
        if tracks:
            x = np.array([t['position'][0] for t in tracks])
            y = np.array([t['position'][1] for t in tracks])
            z = np.array([t['position'][2] if len(t['position']) > 2 else 0.0 for t in tracks])
            centroids = np.column_stack((x, y, z))
            return centroids.tolist()
        return None
    
def make_config_gtrackc(raw: dict) -> GTrackCConfig:
    """Create a GTrackC configuration from raw parameters with equivalent values to Python tracker."""
    
    # Convert spread limits to gating limits [depth, width, height, velocity]
    kf_spread_lim = raw.get("KF_SPREAD_LIM", [0.3, 0.3, 2, 1.2, 1.2, 0.2])
    gating_limits = [
        kf_spread_lim[0],  # depth (x)
        kf_spread_lim[1],  # width (y)
        kf_spread_lim[2],  # height (z)
        kf_spread_lim[3],  # radial velocity 
    ]
    
    # Scenery parameters with boundary box based on spread limits
    scenery_params={
        'boundary_box': [-0.2, 0.2, -0.2, 0.2, 0.0, 2.0],
        'sensor_orientation': [0, 0],
        'sensor_position': [0, 0, 1.0]
    },
    
    # State parameters based on lifetime thresholds
    state_params = {
        'det2actThre': 2,  # Similar to FB_FRAMES_BATCH
        'det2freeThre': raw.get("TR_LIFETIME_DYNAMIC", 3),
        'active2freeThre': raw.get("TR_LIFETIME_DYNAMIC", 3),
        'static2freeThre': raw.get("TR_LIFETIME_STATIC", 7),
        'exit2freeThre': 3,
        'sleep2freeThre': 1000
    }
    
    # Allocation parameters based on DBSCAN values
    allocation_params = {
        'pointsThre': raw.get("DB_MIN_SAMPLES_MIN", 30),
        'maxDistanceThre': raw.get("DB_EPS", 0.7),
        'velocityThre': raw.get("TR_VEL_THRES", 0),
        'snrThre': raw.get("DB_POINTS_THRES", 0) / 10.0,  # Scaled down
        'snrThreObscured': raw.get("DB_POINTS_THRES", 0) / 5.0,
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
        verbose=GTRACK_VERBOSE_TYPE.GTRACK_VERBOSE_DEBUG,
        maxNumPoints=1000,
        maxNumTracks=raw.get("TR_MAX_TRACKS", 4),
        initialRadialVelocity=0,
        maxRadialVelocity=raw.get("TR_VEL_THRES", 0.12) * 5,
        radialVelocityResolution=0.01,
        maxAcceleration=(0.5, 0.5, 0.1),  # x,y,z acceleration limits
        deltaT=1.0 / raw.get("FRAME_RATE", 10)  # Time delta in seconds
    )