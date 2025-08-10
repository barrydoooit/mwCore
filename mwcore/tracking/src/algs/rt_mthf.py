# Short for  Real-Time Multiple-Human Tracking and Fall Detection. Honestly did not put much thought into the name.
# https://github.com/DarkSZChao/MMWave_Radar_Human_Tracking_and_Fall_detection
from functools import partial
import numpy as np
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass, field
import math
import time
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

from .gtrack import BatchedData, PointCluster, Tracker
from .gtrack_asterios import ClusterTrack
from ..utils import (
    RingBuffer,
)
from typing import List, Callable

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


ACTIVE: int = 1
INACTIVE: int = 0
STATIC: bool = True
DYNAMIC: bool = False

@dataclass
class RT_MTFHConfig:
    FB_FRAMES_BATCH: int = 5

class RT_MTFHTrackBuffer(Tracker):
    """
    A class representing a buffer for managing and updating the multiple ClusterTracks of the scene.


    """

    def __init__(self, config: RT_MTFHConfig) -> None:
        """
        Initialize TrackBuffer with empty lists for tracks and effective tracks.
        """
        self.config = config
        self.effective_tracks: List[ClusterTrack] = []
        self.next_track_id: int = 0
        self.dt: float = 0
        self.t: float = time.time()

   

    def track(self, pointcloud: np.array, batch: RingBuffer, clusteringAlgorithm: str = "DBSCAN") -> None:
        pass





class RT_MTHF_KalmanState(KalmanFilter):
    """
    Kalman filter state for the Dawn algorithm, simplified to match MATLAB implementation
    """
    def __init__(self, centroid: np.ndarray, config: RT_MTFHConfig) -> None:
        self.config = config
        self.model = config.motion_model
        # Initialize with 6D state, 3D measurement
        super().__init__(dim_x=self.model.KF_DIM[0], dim_z=self.model.KF_DIM[1])
        self.F = self.model.KF_F(1)
        self.H = self.model.KF_H
        self.Q = self.model.KF_Q_DISCR(1)
        self.R = np.eye(self.model.KF_DIM[1]) * config.kf_params.measurement_noise**2
        self.x = np.array([self.model.STATE_VEC(centroid)]).T
        
        # Use initial_estimate_error from kf_params
        if config.kf_params.initial_estimate_error:
            # Create diagonal P matrix with appropriate dimensions
            self.P = np.diag(config.kf_params.initial_estimate_error * (self.model.KF_DIM[0] // len(config.kf_params.initial_estimate_error)))
        else:
            self.P = np.eye(self.model.KF_DIM[0]) * 1000  # Fallback to default


class RH_MTHFClusterTrack:
    """
    A simplified track class for Dawn algorithm, compatible with DawnLhConfig
    """
    def __init__(self, cluster: "PointCluster", config: RT_MTFHConfig) -> None:
        self.config = config
        self.cluster = cluster
        self.kalmanFilter = RT_MTHF_KalmanState(cluster.centroid, config)
        
        # Track state variables
        self.id = -1  # Will be set when added to tracker
        self.status = ACTIVE
        self.lifetime = 0
        self.state = "normal"  # normal, noise, or lost
        
        # For visualization and history
        self.bbox = None
        self.traj_rec = []  # trajectory recording
        self.bbox_rec = []  # bounding box recording
        self.age = 0
        self.totalVisibleCount = 0
        self.consecutiveInvisibleCount = 0
        self.appear_frame = 0
        self.obj_feature = None
        
    def predict_state(self, dt):
        """Predict the state using the Kalman filter"""
        if dt > 0 and dt < 100:  # Bounds check to prevent overflow, just in case
            self.kalmanFilter.F = self.config.motion_model.KF_F(dt)
            self.kalmanFilter.Q = self.config.motion_model.KF_Q_DISCR(dt)
        self.kalmanFilter.predict()
        self.lifetime +=1  
        return self.kalmanFilter.x

