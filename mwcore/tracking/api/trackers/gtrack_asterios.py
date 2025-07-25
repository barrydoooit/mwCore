import time
from typing import List, Literal, Optional
import numpy as np
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

from mwcore.registry import TRACKERS

from mwcore.tracking.src.algs.gtrack import BatchedData
from mwcore.tracking.src.algs.gtrack_asterios import AsteriosConfig, ConstAccModel
from ..base import BaseTracker

from mwcore.tracking.src.algs.gtrack_asterios import TrackBuffer as AsteriosTrackBuffer



@TRACKERS.register_module()
class GTrackATracker(BaseTracker):
    def __init__(self, 
                 keep_radial: bool,
                 tracker_params: dict,
                    do_dev2standard: bool = False,
                 radar_cfg: Optional[dict] = None):
        super().__init__(radar_cfg=radar_cfg)
        self.keep_radial = keep_radial
        self.config = make_config_asterios(tracker_params)
        self.do_dev2standard = do_dev2standard
        self.tracker = AsteriosTrackBuffer(self.config)
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
        states = [track.state.x.flatten()[:3] for track in self.tracker.effective_tracks]
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

        
def make_config_asterios(raw: dict) -> AsteriosConfig:
    def KF_F(dt):
        return np.array([
            [1, 0, 0, dt, 0, 0, (0.5 * dt**2), 0, 0],
            [0, 1, 0, 0, dt, 0, 0, (0.5 * dt**2), 0],
            [0, 0, 1, 0, 0, dt, 0, 0, (0.5 * dt**2)],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
    
    def KF_Q_DISCR(dt):
        return block_diag(
            Q_discrete_white_noise(3, dt, var=raw['KF_Q_STD']),
            Q_discrete_white_noise(3, dt, var=raw['KF_Q_STD']),
            Q_discrete_white_noise(3, dt, var=raw['KF_Q_STD']),
        )
    
    def STATE_VEC(init):
        return [*init[:6], 0, 0, 0]
    
    motion_model = ConstAccModel(
        KF_DIM=[9, 6],
        KF_H=np.eye(6, 9),
        KF_F=KF_F,
        KF_Q_DISCR=KF_Q_DISCR,
        STATE_VEC=STATE_VEC,
    )
    
    return AsteriosConfig(
        motion_model=motion_model,
        FB_FRAMES_BATCH=raw["FB_FRAMES_BATCH"],
        FB_FRAMES_BATCH_STATIC=raw.get("FB_FRAMES_BATCH_STATIC"),
        DB_POINTS_THRES=raw.get("DB_POINTS_THRES"),
        DB_SPREAD_THRES=raw.get("DB_SPREAD_THRES"),
        DB_EPS=raw.get("DB_EPS"),
        DB_RANGE_WEIGHT=raw.get("DB_RANGE_WEIGHT"),
        DB_Z_WEIGHT=raw.get("DB_Z_WEIGHT"),
        DB_MIN_SAMPLES_MIN=raw.get("DB_MIN_SAMPLES_MIN"),
        KF_R_STD=raw.get("KF_R_STD"),
        KF_P_INIT=raw.get("KF_P_INIT"),
        KF_GROUP_DISP_EST_INIT=raw.get("KF_GROUP_DISP_EST_INIT"),
        KF_ENABLE_EST=raw.get("KF_ENABLE_EST"),
        KF_A_N=raw.get("KF_A_N"),
        KF_EST_POINTNUM=raw.get("KF_EST_POINTNUM"),
        KF_SPREAD_LIM=raw.get("KF_SPREAD_LIM"),
        KF_A_SPR=raw.get("KF_A_SPR"),
        # MODEL_MIN_INPUT=raw.get("MODEL_MIN_INPUT"),
        # MODEL_DEFAULT_POSTURE=np.zeros(57),
        TR_LIFETIME_DYNAMIC=raw.get("TR_LIFETIME_DYNAMIC"),
        TR_LIFETIME_STATIC=raw.get("TR_LIFETIME_STATIC"),
        TR_GATE=raw.get("TR_GATE"),
        TR_MAX_TRACKS=raw.get("TR_MAX_TRACKS"),
        TR_VEL_THRES=raw.get("TR_VEL_THRES"),
    )
