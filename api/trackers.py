import time
import numpy as np
from src.tracking.AsteriosTracking import BatchedData
from src.tracking.GTrack import GTrackBuffer
from .base_tracker import TRACKERS, BaseTracker
from src.tracking import config_factory as cfgfactory


@TRACKERS.register_module()
class GTrackTracker(BaseTracker):
    def __init__(self,
                 keep_radial: bool,
                 tracker_config: dict,
                 radar_cfg: dict):
        super().__init__(radar_cfg=radar_cfg)
        self.keep_radial = keep_radial
        self.config = cfgfactory.make_config_gtrack(tracker_config)
        self.tracker = GTrackBuffer(self.config)
        self.batch = BatchedData(self.config.FB_FRAMES_BATCH+1, np.empty((0, 11 if self.keep_radial else 8)))
        self.last_time = time.time()
        
    def consume(self, det_obj: dict):
        effective_data = self.normalize_data(det_obj=det_obj, keepRadial=self.keep_radial, transform=True)
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.tracker.dt = dt
        if effective_data.shape[0] > 0:
            self.tracker.track(effective_data, self.batch)
        locations = [track.cluster.centroid for track in self.tracker.effective_tracks]
        return locations

from src.tracking.AsteriosTracking import TrackBuffer as AsteriosTrackBuffer
@TRACKERS.register_module()
class GTrackATracker(BaseTracker):
    def __init__(self, 
                 keep_radial: bool,
                 tracker_config: dict,
                 radar_cfg: dict):
        super().__init__(radar_cfg=radar_cfg)
        self.keep_radial = keep_radial
        self.config = cfgfactory.make_config_asterios(tracker_config)
        self.tracker = AsteriosTrackBuffer(self.config)
        self.batch = BatchedData(self.config.FB_FRAMES_BATCH+1, np.empty((0, 11 if self.keep_radial else 8)))
        self.last_time = time.time()
        
    def consume(self, det_obj: dict):
        effective_data = self.normalize_data(det_obj=det_obj, keepRadial=self.keep_radial, transform=True)
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.tracker.dt = dt
        if effective_data.shape[0] > 0:
            self.tracker.track(effective_data, self.batch)
        locations = [track.cluster.centroid for track in self.tracker.effective_tracks]
        return locations

from src.tracking.RKFTracking import RKFTrackBuffer
@TRACKERS.register_module()
class RKFTracker(BaseTracker):
    def __init__(self, 
                 keep_radial: bool,
                 tracker_config: dict,
                 radar_cfg: dict):
        super().__init__(radar_cfg=radar_cfg)
        self.keep_radial = keep_radial
        self.config = cfgfactory.make_config_rkf(tracker_config)
        self.tracker = RKFTrackBuffer(self.config)
        self.batch = BatchedData(self.config.FB_FRAMES_BATCH+1, np.empty((0, 11 if self.keep_radial else 8)))
        self.last_time = time.time()
        
    def consume(self, det_obj: dict):
        effective_data = self.normalize_data(det_obj=det_obj, keepRadial=self.keep_radial, transform=True)
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.tracker.dt = dt
        if effective_data.shape[0] > 0:
            self.tracker.track(effective_data, self.batch)
        locations = [track.cluster.centroid for track in self.tracker.effective_tracks]
        return locations