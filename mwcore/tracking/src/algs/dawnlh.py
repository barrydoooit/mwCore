from functools import partial
import numpy as np
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
import math
import time
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

from .gtrack import BatchedData, PointCluster, Tracker
from .gtrack_asterios import ClusterTrack
from ..utils import (
    RingBuffer,
)
from typing import List, Callable, Any

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class ConstAccModel:
    KF_DIM: List[int]  # e.g. [9, 6]
    KF_H: np.ndarray   # Measurement matrix (6x9? Typically defined in the model)
    KF_F: Callable[[float], np.ndarray]  # Function: dt -> state transition matrix
    KF_Q_DISCR: Callable[[float], np.ndarray]  # Function: dt -> process noise covariance
    STATE_VEC: Callable[[np.ndarray], List[float]]  # Function to build the initial state vector

@dataclass
class DawnLhConfig:
    motion_model: ConstAccModel
    FB_FRAMES_BATCH: int 
    KF_Q_STD: float
    minObjPoints: int = 30  # Minimum points per cluster
    DB_EPS: float = 0.3  # Epsilon for DBSCAN
    DBSCAN_MinPts: int = 30  # Minimum points for DBSCAN
    TR_LIFETIME_STATIC: float = 10.0
    TR_LIFETIME_DYNAMIC: float = 5.0
    TR_VEL_THRES: float = 0.1
    TR_GATE: float = 20.0
    TR_MAX_TRACKS: int = 2
    DB_EPS: float = 0.3  
    DB_MIN_SAMPLES_MIN: int = 30  
    DB_RANGE_WEIGHT: float = 1.0
    DB_Z_WEIGHT: float = 1.0
    KF_GROUP_DISP_EST_INIT: float = 0.1
    KF_Q_STD: float = 1.0 

ACTIVE: int = 1
INACTIVE: int = 0
STATIC: bool = True
DYNAMIC: bool = False


class DawnLHTrackBuffer(Tracker):
    """
    A class representing a buffer for managing and updating the multiple ClusterTracks of the scene.

    Attributes
    ----------
    effective_tracks : List[ClusterTrack]
        List of currently active (non-INACTIVE) tracks in the buffer.
    next_track_id : int
        The id int that will be given to the next active track.
    dt : float
        Time multiplier used for predicting states. Indicates the time passed since the previous
        valid observed frame.
    t : float
        Current time when the TrackBuffer is instantiated / updated.

    Methods
    -------
    _maintain_tracks()
        Update the status of tracks based on their lifetime.

    update_ef_tracks()
        Update the list of effective tracks (excluding INACTIVE tracks).

    has_active_tracks()
        Check if there are active tracks in the buffer.

    _calc_dist_fun(full_set)
        Calculate the Mahalanobis distance matrix for gating.

    _add_tracks(new_clusters)
        Add new tracks to the buffer.

    _predict_all()
        Predict the state of all effective tracks.

    _update_all()
        Update the state of all effective tracks.

    _get_gated_clouds(full_set)
        Gate the pointcloud and return gated and unassigned clouds.

    _associate_points_to_tracks(full_set)
        Associate points to existing tracks and handle inner cluster separation.

    track(pointcloud, batch)
        Perform the tracking process including prediction, association, status update, and clustering.

    estimate_posture(model)
        Estimate the posture of each track in the buffer using a CNN model.

    """

    def __init__(self, config: DawnLhConfig, usePalmar: bool = False) -> None:
        """
        Initialize TrackBuffer with empty lists for tracks and effective tracks.
        """
        self.config = config
        self.effective_tracks: List[ClusterTrack] = []
        self.next_track_id: int = 0
        self.dt: float = 0
        self.t: float = time.time()

    def _maintain_tracks(self) -> None:
        """
        Update the status of tracks based on their mobility and lifetime. Then update the list of effective tracks.
        """
        for track in self.effective_tracks:
            lifetime = (self.config.TR_LIFETIME_DYNAMIC if track.cluster.status == DYNAMIC 
                        else self.config.TR_LIFETIME_STATIC)
            if track.lifetime > lifetime:
                track.status = INACTIVE
        self.effective_tracks[:] = [track for track in self.effective_tracks if track.status != INACTIVE]



    def _predict_all(self) -> None:
        """
        Predict the state of all effective tracks.
        """
        for track in self.effective_tracks:
            track.predict_state(track.lifetime + self.dt)

    def _update_all(self) -> None:
        """
        Update the state of all effective tracks.
        """
        for track in self.effective_tracks:
            track.update_state()

    def update_assigned_tracks(self, assignments, centroids, bboxes, obj_features):
        for track_idx, det_idx in assignments:
            track = self.effective_tracks[track_idx]
            centroid = centroids[det_idx]
            bbox = bboxes[det_idx]
            kf_centroid = track.kalmanFilter.correct(centroid)
            track.bbox = bbox
            track.traj_rec.append(kf_centroid)
            track.bbox_rec.append(bbox)
            track.age += 1
            track.totalVisibleCount += 1
            track.consecutiveInvisibleCount = 0
            track.obj_feature = obj_features[det_idx]

    def update_unassigned_tracks(self, unassignedTracks):
        for track_idx in unassignedTracks:
            track = self.effective_tracks[track_idx]
            track.age += 1
            track.traj_rec.append([np.nan]*3)
            track.consecutiveInvisibleCount += 1

    def update_track_states(self, invisibleForTooLong=20, ageThreshold=5):
        for track in self.effective_tracks:
            if track.age < ageThreshold and (track.totalVisibleCount / track.age) < 0.5:
                track.state = "noise"
            if track.consecutiveInvisibleCount >= invisibleForTooLong:
                track.state = "lost"

    def create_new_tracks(self, unassignedDetections, centroids, bboxes, obj_features, obj_frame, obj_idx):
        nextId = self.next_track_id
        for det_idx in unassignedDetections:
            # Find cluster points for this detection
            cluster_points = obj_frame[obj_idx == det_idx]
            point_cluster = PointCluster(cluster_points, tr_vel_threshold=self.config.TR_VEL_THRES)
            newTrack = DawnClusterTrack(point_cluster, self.config)
            newTrack.id = nextId
            newTrack.bbox = bboxes[det_idx]
            newTrack.traj_rec = [centroids[det_idx]]
            newTrack.bbox_rec = [bboxes[det_idx]]
            newTrack.obj_feature = obj_features[det_idx]
            newTrack.age = 1
            newTrack.appear_frame = self.t
            newTrack.state = "normal"
            newTrack.totalVisibleCount = 1
            newTrack.consecutiveInvisibleCount = 0
            self.effective_tracks.append(newTrack)
            nextId += 1
        self.next_track_id = nextId

    def track(self, pointcloud: np.array, batch: RingBuffer, clusteringAlgorithm: str = "DBSCAN") -> None:
        print(f"Pointcloud shape: {pointcloud.shape}")

        self._predict_all()

        logger.info(f"Pointcloud before denoising: {pointcloud.shape}")
        # Noise filtering
        pointcloud = self.point_cloud_denoise(pointcloud, {
            'dpl_thr': 0.1,  # Doppler threshold
            'power_thr': 0.1,  # Power threshold
            'loc_thr': [-50, 50, -50, 50, -50, 50]  # Location threshold
        })
        logger.info(f"Pointcloud after denoising: {pointcloud.shape}")

        param_det = {
            'minObjPoints': self.config.minObjPoints,  # They suggest a minimum of 30 points per cluster.
            'DBSCAN_epsilon': self.config.DB_EPS,  # Epsilon for DB. They suggest 0.3
            'DBSCAN_MinPts': self.config.DBSCAN_MinPts,  # They suggest a minimum of 30 points per cluster
        }
        centroids, bboxes, obj_frame, obj_idx, obj_features = self.getDetections(pointcloud, param_det)

        assignments, unassignedTracks, unassignedDetections = self.detectionToTrackAssignment(
            self.effective_tracks, centroids, obj_features
        )

        self.update_assigned_tracks(assignments, centroids, bboxes, obj_features)
        self.update_unassigned_tracks(unassignedTracks)
        self.update_track_states()
        self.create_new_tracks(unassignedDetections, centroids, bboxes, obj_features, obj_frame, obj_idx)

        self._maintain_tracks()
        self._update_all()




    def point_cloud_denoise(self, frame: np.ndarray, param: dict) -> np.ndarray:
        dpl_thr = param.get('dpl_thr', 0)
        power_thr = param.get('power_thr', 0)
        loc_thr = param.get('loc_thr', [-50, 50, -50, 50, -50, 50])
        # Doppler threshold (column 7)
        mask = np.abs(frame[:, 7]) > dpl_thr
        # Power threshold (column 8, zero-based index 7)
        mask &= frame[:, 7] > power_thr
        frame_now = frame[mask]
        # Location threshold
        loc_mask = (
            (loc_thr[0] < frame_now[:, 0]) & (frame_now[:, 0] < loc_thr[1]) &
            (loc_thr[2] < frame_now[:, 1]) & (frame_now[:, 1] < loc_thr[3]) &
            (loc_thr[4] < frame_now[:, 2]) & (frame_now[:, 2] < loc_thr[5])
        )
        frame_clean = frame_now[loc_mask]
        return frame_clean

    def calc_centroid(self, points: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        if weights is None or len(weights) != len(points):
            return np.mean(points, axis=0)
        return np.average(points, axis=0, weights=weights)

    def get_detection_feature(self, frame_obj: np.ndarray) -> dict:
        # Use peakval for power and power_value
        return {
            'average_speed': np.mean(frame_obj[:, 6]),  # doppler
            'average_power': np.mean(frame_obj[:, 7]),  # peakval as power
            'max_power': np.max(frame_obj[:, 7])        # peakval as power_value
        }
    
    def getDetections(self, frame: np.ndarray, param_det: dict):
        logger.info(f"Starting detection with frame shape: {frame.shape}")
    
        if frame.shape[0] < param_det['minObjPoints']:
            logger.info(f"Not enough points ({frame.shape[0]}) for detection, minimum required: {param_det['minObjPoints']}")
            return [], [], [], [], []
            
        # DBSCAN clustering on X,Y
        db = DBSCAN(eps=param_det['DBSCAN_epsilon'], min_samples=param_det['DBSCAN_MinPts'])
        logger.info(f"Running DBSCAN with eps={param_det['DBSCAN_epsilon']}, min_samples={param_det['DBSCAN_MinPts']}")
        
        idx = db.fit_predict(frame[:, [0, 1]])
        
        # Count non-noise points
        non_noise_count = np.sum(idx != -1)
        logger.info(f"DBSCAN assigned {non_noise_count}/{len(idx)} points to clusters")
        
        # Remove noise
        obj_frame = frame[idx != -1]
        obj_idx = idx[idx != -1]
        unique_class = np.unique(obj_idx)
        class_num = len(unique_class)
        
        logger.info(f"Found {class_num} unique clusters")
    
        bboxes = np.full((class_num, 6), np.nan)
        centroids = np.full((class_num, 3), np.nan)
        obj_features = []
        for i, cls in enumerate(unique_class):
            frame_obj = obj_frame[obj_idx == cls]
            rect_min = np.min(frame_obj[:, :3], axis=0)
            rect_max = np.max(frame_obj[:, :3], axis=0)
            rect_size = rect_max - rect_min
            rect_center = self.calc_centroid(frame_obj[:, :3], frame_obj[:, 7])
            bboxes[i, :3] = rect_min
            bboxes[i, 3:] = rect_size
            centroids[i] = rect_center
            obj_features.append(self.get_detection_feature(frame_obj))
        logger.info(f"Detection completed with {centroids.shape[0]} centroids")
        logger.info(f"Centroids: {centroids}")
        return centroids, bboxes, obj_frame, obj_idx, obj_features

    def calc_feature_cost(self, objA_feature, obj_features):
        if not obj_features or objA_feature is None:
            return np.full(len(obj_features), np.nan)
        return 10 * np.square(np.array([f['average_speed'] for f in obj_features]) - objA_feature['average_speed'])

    def detectionToTrackAssignment(self, tracks, centroids, obj_features):
        normal_tracks = [t for t in tracks if t.state == "normal"]
        nTracks = len(normal_tracks)
        nDetections = len(centroids)
        
        logger.debug(f"Assignment: {nTracks} normal tracks, {nDetections} detections")
        
        # If there are no tracks or detections, return empty assignments
        if nTracks == 0 or nDetections == 0:
            logger.debug("No tracks or detections for assignment")
            return np.zeros((0, 2), dtype=int), list(range(nTracks)), list(range(nDetections))
        
        cost_dist = np.zeros((nTracks, nDetections))
        cost_feature = np.zeros((nTracks, nDetections))
        
        for i, track in enumerate(normal_tracks):
            try:
                cost_dist[i, :] = track.kalmanFilter.distance(centroids)
                cost_feature[i, :] = self.calc_feature_cost(track.obj_feature, obj_features)
                logger.debug(f"Track {i} costs - distance: {np.mean(cost_dist[i, :]):.2f}, feature: {np.mean(cost_feature[i, :]):.2f}")
            except Exception as e:
                logger.error(f"Error calculating costs for track {i}: {str(e)}")
                raise
        
        cost = 0.6 * cost_dist + 0.4 * cost_feature
        costOfNonAssignment = 25
        
        logger.debug(f"Running linear_sum_assignment with cost matrix shape: {cost.shape}")
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Filter assignments by cost threshold
        valid_assignments = [(r, c) for r, c in zip(row_ind, col_ind) if cost[r, c] < costOfNonAssignment]
        logger.debug(f"Found {len(valid_assignments)}/{len(row_ind)} assignments below cost threshold {costOfNonAssignment}")
        
        # Create a properly shaped 2D array with explicit reshape
        if valid_assignments:
            assignments = np.array(valid_assignments, dtype=int).reshape(-1, 2)
        else:
            assignments = np.zeros((0, 2), dtype=int)
        logger.debug(f"Assignments array shape: {assignments.shape}")
        
        # Get assigned track/detection indices
        if len(assignments) == 0:
            assigned_tracks = set()
            assigned_detections = set()
        else:
            assigned_tracks = set(assignments[:, 0].tolist())
            assigned_detections = set(assignments[:, 1].tolist())
        
        # Get unassigned tracks and detections using the normal_tracks index mapping
        unassignedTracks = [i for i in range(nTracks) if i not in assigned_tracks]
        unassignedDetections = [i for i in range(nDetections) if i not in assigned_detections]
        
        logger.debug(f"Unassigned tracks: {len(unassignedTracks)}, Unassigned detections: {len(unassignedDetections)}")
        
        return assignments, unassignedTracks, unassignedDetections
    

class DawnKalmanState(KalmanFilter):
    """
    Kalman filter state for the Dawn algorithm, simplified to match MATLAB implementation
    """
    def __init__(self, centroid: np.ndarray, config: DawnLhConfig) -> None:
        self.config = config
        self.model = config.motion_model
        # Initialize with 6D state, 3D measurement
        super().__init__(dim_x=self.model.KF_DIM[0], dim_z=self.model.KF_DIM[1])
        self.F = self.model.KF_F(1)
        self.H = self.model.KF_H
        self.Q = self.model.KF_Q_DISCR(1)
        # Use the KF_Q_STD from DawnLhConfig
        self.R = np.eye(self.model.KF_DIM[1]) * config.KF_Q_STD**2
        self.x = np.array([self.model.STATE_VEC(centroid)]).T
        self.P = np.eye(self.model.KF_DIM[0]) * 1000  # Initial error covariance

    def distance(self, centroids):
        """Calculate the Mahalanobis distance between state and measurements"""
        if len(centroids) == 0:
            return np.array([])
        
        # Predict measurement (position only)
        z_pred = self.H @ self.x
        
        # Calculate innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
        
        # Calculate distances
        distances = []
        for i in range(len(centroids)):
            # Innovation (measurement - prediction)
            y = centroids[i] - z_pred.flatten()
            # Mahalanobis distance
            d = np.sqrt(y @ S_inv @ y.T)
            distances.append(d)
            
        return np.array(distances)
    
    def correct(self, z):
        """
        Correct the state using a position measurement.
        """
        # z should already be the right dimension (3D position)
        super().update(z)
        
        # Return the position components of the corrected state
        return self.x[:3].flatten()

class DawnClusterTrack:
    """
    A simplified track class for Dawn algorithm, compatible with DawnLhConfig
    """
    def __init__(self, cluster: "PointCluster", config: DawnLhConfig) -> None:
        self.config = config
        self.cluster = cluster
        self.kalmanFilter = DawnKalmanState(cluster.centroid, config)
        
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
        self.kalmanFilter.predict()
        self.lifetime += 1
        return self.kalmanFilter.x
        
    def update_state(self):
        """Update the Kalman filter state"""
        # The actual correction happens in update_assigned_tracks
        # This is kept for compatibility with the existing code
        pass