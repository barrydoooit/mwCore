import time
from typing import List, Literal, Optional
import numpy as np
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

from mwcore.registry import TRACKERS

from mwcore.tracking.src.algs.gtrack import BatchedData
from mwcore.tracking.src.algs.gtrack_asterios import  ConstAccModel
from ..base import BaseTracker

from mwcore.tracking.src.algs.dawnlh import DawnLhConfig, DawnLHTrackBuffer, KFParameters

@TRACKERS.register_module()
class DawnLHTracker(BaseTracker):
    def __init__(self, 
                 keep_radial: bool,
                 tracker_params: dict,
                 do_dev2standard: bool = False,
                 radar_cfg: Optional[dict] = None):
        super().__init__(radar_cfg=radar_cfg)
        self.keep_radial = keep_radial
        self.config = make_config_dawnLh(tracker_params)
        self.do_dev2standard = do_dev2standard
        self.tracker = DawnLHTrackBuffer(self.config)
        self.batch = BatchedData(self.config.FB_FRAMES_BATCH+1, np.empty((0, 11 if self.keep_radial else 8)))
        self.last_time = time.time()
        
    def consume(self, 
                det_obj: dict = None, 
                point_array: np.ndarray = None, 
                sort_metric: Optional[Literal['size', 'snr', 'rel']] = None, 
                **kwargs) -> List[np.ndarray]:
        effective_data = self.normalize_data(det_obj=det_obj, point_array=point_array, keepRadial=self.keep_radial, transform=self.do_dev2standard)
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.tracker.dt = dt
        if effective_data.shape[0] > 0:
            self.tracker.track(effective_data, self.batch)
        # locations = [track.cluster.centroid[:3] for track in self.tracker.effective_tracks]
        states = [track.kalmanFilter.x.flatten()[:3] for track in self.tracker.effective_tracks]
        # print(f"These are the current states: {states}")
        if sort_metric is not None:
            sorted_indices = self.sort_results(metric=sort_metric, **kwargs)
            # locations = [locations[i] for i in sorted_indices]
            states = [states[i] for i in sorted_indices]
        return states

    def sort_results(self, metric: Literal['size', 'snr', 'rel'] = 'size', **kwargs) -> np.ndarray:
        clusters = [track.cluster for track in self.tracker.effective_tracks]
        if metric == 'size':
            cluster_sizes = [c.point_num for c in clusters]
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            return sorted_indices
        
        if metric == 'snr':
            snr_values = [c.pointcloud[:, 7].mean() for c in clusters]
            sorted_indices = np.argsort(snr_values)[::-1]
            return sorted_indices
                    
        if metric == 'rel':
            anchor = kwargs.get('anchor')
            if anchor is None:
                raise ValueError("The 'anchor' point must be provided for 'rel' sorting.")
            rel_distances = [np.linalg.norm(c.centroid[:3] - anchor) for c in clusters]
            sorted_indices = np.argsort(rel_distances)
            return sorted_indices


def make_config_dawnLh(raw: dict) -> DawnLhConfig:
    # For constant velocity model 
    def KF_F_cv(dt):
        return np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
    
    kf_q_std = raw.get('KF_Q_STD', 25.0)  # Changed default to match MATLAB
    
    def KF_Q_DISCR_cv(dt):
        # Q for constant velocity model
        return block_diag(
            Q_discrete_white_noise(2, dt, var=kf_q_std),
            Q_discrete_white_noise(2, dt, var=kf_q_std),
            Q_discrete_white_noise(2, dt, var=kf_q_std),
        )
    
    def STATE_VEC_cv(init):
        # For 3D position + velocity
        return [init[0], init[1], init[2], 0, 0, 0]
    
    # Define the motion model
    motion_model = ConstAccModel(
        KF_DIM=[6, 3],  # 6D state (pos+vel), 3D measurement (pos only)
        KF_H=np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]]),  # Extract position only
        KF_F=KF_F_cv,
        KF_Q_DISCR=KF_Q_DISCR_cv,
        STATE_VEC=STATE_VEC_cv,
    )
    
    # Create KF Parameters with optional overrides from raw config
    kf_params = KFParameters(
        motion_model=raw.get('motion_model', "ConstantVelocity"),
        measurement_noise=raw.get('measurement_noise', 1.0),
        initial_estimate_error=raw.get('initial_estimate_error', None),
        motion_noise=raw.get('motion_noise', None),
        initial_location=raw.get('initial_location', "Same as first detection")
    )
    
    # Create and return the full configuration
    return DawnLhConfig(
        motion_model=motion_model,
        kf_params=kf_params,
        
        # DBSCAN clustering parameters
        DB_EPS=raw.get('DB_EPS', 0.3),
        DBSCAN_MinPts=raw.get('DBSCAN_MinPts', 20),
        minObjPoints=raw.get('minObjPoints', 10),
        
        # Point cloud filtering parameters
        dpl_thr=raw.get('dpl_thr', 0.1),
        power_thr=raw.get('power_thr', 0.1),
        loc_thr=raw.get('loc_thr', [-50, 50, -50, 50, -50, 50]),
        
        # Track lifecycle management
        TR_LIFETIME_STATIC=raw.get('TR_LIFETIME_STATIC', 10),
        TR_LIFETIME_DYNAMIC=raw.get('TR_LIFETIME_DYNAMIC', 30),
        TR_VEL_THRES=raw.get('TR_VEL_THRES', 0.05),
        invisible_for_too_long=raw.get('invisible_for_too_long', 20),
        age_threshold=raw.get('age_threshold', 5),
        visibility_threshold=raw.get('visibility_threshold', 0.5),
        
        # Tracking parameters
        FB_FRAMES_BATCH=raw.get('FB_FRAMES_BATCH', 5),
        cost_of_non_assignment=raw.get('cost_of_non_assignment', 25.0),
        assignment_dist_weight=raw.get('assignment_dist_weight', 0.6),
        assignment_feature_weight=raw.get('assignment_feature_weight', 0.4),
        feature_cost_multiplier=raw.get('feature_cost_multiplier', 10.0),
    )