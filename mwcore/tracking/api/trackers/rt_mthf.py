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
        tracks = self.tracker.effective_tracks
        states = [track.last_position for track in tracks if track.last_position is not None]

        if sort_metric is not None:
            sorted_indices = self.sort_results(metric=sort_metric, **kwargs)
            states = [states[i] for i in sorted_indices]
        return states


    def sort_results(self, metric: Literal['size', 'snr', 'rel'] = 'size', **kwargs) -> np.ndarray:
        # Since RT-MTHF doesn't use cluster objects like the original, we need to adapt this. TODO
        tracks = self.tracker.effective_tracks
        
        if len(tracks) == 0:
            return np.array([])
        
        if metric == 'size':
            # Use track stability (number of updates) instead of age
            track_stability = [len(track.obj_cp_deque) for track in tracks]
            sorted_indices = np.argsort(track_stability)[::-1]  # Descending order
            return sorted_indices
        
        if metric == 'snr':
            # Use last known size as proxy for track strength/quality
            # Larger clusters typically have higher SNR
            track_sizes = []
            for track in tracks:
                if track.last_size is not None:
                    # Use volume as proxy for SNR (larger clusters = stronger signal)
                    volume = np.prod(track.last_size)
                    track_sizes.append(volume)
                else:
                    track_sizes.append(0.0)
            sorted_indices = np.argsort(track_sizes)[::-1]  # Descending order
            return sorted_indices
                        
        if metric == 'rel':
            anchor = kwargs.get('anchor')
            if anchor is None:
                raise ValueError("The 'anchor' point must be provided for 'rel' sorting.")
            
            # Calculate distances to anchor point
            rel_distances = []
            for track in tracks:
                if track.last_position is not None:
                    distance = np.linalg.norm(track.last_position - anchor)
                    rel_distances.append(distance)
                else:
                    rel_distances.append(float('inf'))  # Put tracks without position at the end
            
            sorted_indices = np.argsort(rel_distances)  # Ascending order (closest first)
            return sorted_indices
        
        # Default fallback
        return np.arange(len(tracks))
    
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
        dbscan_min_samples=raw.get('dbscan_min_samples', 3),
        dbscan_sort=raw.get('dbscan_sort', True),
        dbscan_sort_limit=raw.get('dbscan_sort_limit', 5),
        # Human object parameters
        obj_deque_length=raw.get('obj_deque_length', 10),
        dis_diff_threshold=raw.get('dis_diff_threshold', 0.5),
        dis_diff_threshold_dr=raw.get('dis_diff_threshold_dr', 0.1),
        size_diff_threshold=raw.get('size_diff_threshold', 0.1),
        sub_possibility_proportion=raw.get('sub_possibility_proportion', [0.25, 0.25, 0.25, 0.25]),
        expect_pos=raw.get('expect_pos', {'default': [None, None, None]}),
        expect_shape=raw.get('expect_shape', {'default': [None, None, None]}),
        obj_delete_timeout=raw.get('obj_delete_timeout', 5.0),
        fuzzy_boundary_enter=raw.get('fuzzy_boundary_enter', False),
        fuzzy_boundary_threshold=raw.get('fuzzy_boundary_threshold', 0.5)
    )