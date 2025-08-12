# Short for  Real-Time Multiple-Human Tracking and Fall Detection. Honestly did not put much thought into the name.
# https://github.com/DarkSZChao/MMWave_Radar_Human_Tracking_and_Fall_detection
import time
from typing import List, Literal, Optional
import numpy as np
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

from mwcore.registry import TRACKERS

from mwcore.tracking.src.algs.gtrack import BatchedData
from mwcore.tracking.src.algs.gtrack_asterios import  ConstAccModel
from ..base import BaseTracker

from mwcore.tracking.src.algs.rt_mthf import RT_MTFHTrackBuffer, RT_MTFHConfig
@TRACKERS.register_module()
class RT_MTFHTracker(BaseTracker):
    def __init__(self, 
                 keep_radial: bool,
                 tracker_params: dict,
                 do_dev2standard: bool = False,
                 radar_cfg: Optional[dict] = None):
        super().__init__(radar_cfg=radar_cfg)
        self.keep_radial = keep_radial
        self.config = make_config_rt_mthf(tracker_params)
        self.do_dev2standard = do_dev2standard
        self.tracker = RT_MTFHTrackBuffer(self.config)
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
        states = [track.last_position for track in self.tracker.effective_tracks if track.last_position is not None]
        
        # if sort_metric is not None:
            # sorted_indices = self.sort_results(metric=sort_metric, **kwargs)
            # states = [states[i] for i in sorted_indices]
        return states


    def sort_results(self, metric: Literal['size', 'snr', 'rel'] = 'size', **kwargs) -> np.ndarray:
        # Since RT-MTHF doesn't use cluster objects like the original, we need to adapt this. TODO
        tracks = self.tracker.effective_tracks
        
        if metric == 'size':
            track_ages = [track.age for track in tracks]
            sorted_indices = np.argsort(track_ages)[::-1]
            return sorted_indices
        
        if metric == 'snr':
            return np.arange(len(tracks))
                    
        if metric == 'rel':
            anchor = kwargs.get('anchor')
            if anchor is None:
                raise ValueError("The 'anchor' point must be provided for 'rel' sorting.")
            rel_distances = [np.linalg.norm(track.last_position - anchor) for track in tracks if track.last_position is not None]
            sorted_indices = np.argsort(rel_distances)
            return sorted_indices
        
def make_config_rt_mthf(raw: dict) -> RT_MTFHConfig:
    """
    Create RT_MTFHConfig from raw dictionary parameters
    """
    return RT_MTFHConfig(
        FB_FRAMES_BATCH=raw.get('FB_FRAMES_BATCH', 5),
        # Boundary filtering parameters
        global_xlim=tuple(raw.get('global_xlim', (-5.0, 5.0))),
        global_ylim=tuple(raw.get('global_ylim', (0.0, 10.0))),
        global_zlim=tuple(raw.get('global_zlim', (-2.0, 2.0))),
        # Speed filtering
        es_threshold=raw.get('es_threshold', 0.5),
        # Tracking parameters
        obj_bin_number=raw.get('obj_bin_number', 10),
        poss_clus_deque_length=raw.get('poss_clus_deque_length', 3),
        redundant_clus_remove_cp_dis=raw.get('redundant_clus_remove_cp_dis', 0.5),
        # DBSCAN parameters
        dbscan_eps=raw.get('dbscan_eps', 0.5),
        dbscan_min_samples=raw.get('dbscan_min_samples', 3)
    )