import time
from typing import List, Literal, Optional
import numpy as np
from mwcore.registry import TRACKERS
from mwcore.tracking.src.algs.gtrack import BatchedData
from ..base import BaseTracker
    
from mwcore.tracking.src.algs.recursive_kf import RKFConfig

from mwcore.tracking.src.algs.recursive_kf import RKFTrackBuffer



@TRACKERS.register_module()
class RKFTracker(BaseTracker):
    def __init__(self, 
                 keep_radial: bool,
                 do_dev2standard: bool,
                 tracker_params: dict,
                 radar_cfg: Optional[dict] = None,
                 result_in_polar: bool = False):
        super().__init__(radar_cfg=radar_cfg)
        self.keep_radial = keep_radial
        self.do_dev2standard = do_dev2standard
        self.result_in_polar = result_in_polar
        self.config = make_config_rkf(tracker_params)
        self.tracker = RKFTrackBuffer(self.config)
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
        locations = [track.cluster.centroid for track in self.tracker.effective_tracks]
        if not self.result_in_polar:
            locations = self.polar_to_cartesian(locations)
        if sort_metric is not None:
            sorted_indices = self.sort_results(metric=sort_metric, **kwargs)
            locations = [locations[i] for i in sorted_indices]
        return locations

    def polar_to_cartesian(self, polar_coords: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert a list of polar-coordinate centroids [r, θ, ṙ] back to Cartesian (x, y, z).
        Since we only have range (r) and azimuth (θ), we assume z = 0 in the Cartesian output.

        Parameters
        ----------
        polar_coords : List[np.ndarray]
            List of shape-(3,) arrays, each containing [r, θ, r_dot].

        Returns
        -------
        List[np.ndarray]
            List of shape-(3,) arrays, each containing [x, y, z] in meters.
        """
        cartesian_coords = []
        for p in polar_coords:
            r, theta, r_dot = p  # unpack range, azimuth, radial velocity (we ignore r_dot here)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 0.0
            cartesian_coords.append(np.array([x, y, z]))
        return cartesian_coords
    
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
            is_radial_anchor = kwargs.get('is_radial_anchor', self.result_in_polar)
            if is_radial_anchor:
                anchor = self.polar_to_cartesian([anchor])[0]
            rel_distances = [np.linalg.norm(self.polar_to_cartesian([c.centroid])[0] - anchor) for c in clusters]
            sorted_indices = np.argsort(rel_distances)
            return sorted_indices
        
def make_config_rkf(raw: dict) -> RKFConfig:
    def RKF_F(dt):
        return np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1],
        ])

    def RKF_Q(dt):
        return np.array([
            [0.2, 0, 0, 0],
            [0, 0.2, 0, 0],
            [0, 0, 0.2, 0],
            [0, 0, 0, 0.2]
        ])

    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])

    return RKFConfig(
        RKF_F=RKF_F,
        RKF_Q=RKF_Q,
        H=H,
        KF_GROUP_DISP_EST_INIT=raw.get("KF_GROUP_DISP_EST_INIT"),
        KF_ENABLE_EST=raw.get("KF_ENABLE_EST"),
        KF_A_N=raw.get("KF_A_N"),
        KF_EST_POINTNUM=raw.get("KF_EST_POINTNUM"),
        KF_SPREAD_LIM=raw.get("KF_SPREAD_LIM"),
        KF_A_SPR=raw.get("KF_A_SPR"),
        DB_POINTS_THRES=raw.get("DB_POINTS_THRES"),
        DB_SPREAD_THRES=raw.get("DB_SPREAD_THRES"),
        DB_EPS=raw.get("DB_EPS"),
        DB_RANGE_WEIGHT=raw.get("DB_RANGE_WEIGHT"),
        DB_Z_WEIGHT=raw.get("DB_Z_WEIGHT"),
        DB_MIN_SAMPLES_MIN=raw.get("DB_MIN_SAMPLES_MIN"),
        FB_FRAMES_BATCH=raw.get("FB_FRAMES_BATCH"),
        KF_R_STD=raw.get("KF_R_STD"),
        # MODEL_DEFAULT_POSTURE=np.zeros(57),
        TR_LIFETIME_DYNAMIC=raw.get("TR_LIFETIME_DYNAMIC"),
        TR_LIFETIME_STATIC=raw.get("TR_LIFETIME_STATIC"),
        TR_GATE=raw.get("TR_GATE"),
        TR_MAX_TRACKS=raw.get("TR_MAX_TRACKS"),
        TR_VEL_THRES=raw.get("TR_VEL_THRES"),
        ENABLE_TORSO_TRACKING=raw.get("ENABLE_TORSO_TRACKING", False),  # New flag
        MAX_LIMB_VELOCITY=raw.get("MAX_LIMB_VELOCITY", 2.0),  # m/s
        MIN_TORSO_MOVEMENT=raw.get("MIN_TORSO_MOVEMENT", 0.1),  # meters
        TORSO_DENSITY_RADIUS=raw.get("TORSO_DENSITY_RADIUS", 0.3),  # meters
        MIN_TORSO_POINTS=raw.get("MIN_TORSO_POINTS", 5),  # Minimum points to consider torso tracking
    ) 