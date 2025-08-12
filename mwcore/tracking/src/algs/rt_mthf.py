# Short for  Real-Time Multiple-Human Tracking and Fall Detection. Honestly did not put much thought into the name.
# https://github.com/DarkSZChao/MMWave_Radar_Human_Tracking_and_Fall_detection
import numpy as np
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass, field
from math import hypot
import time
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from collections import deque
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

# Remove the RT_MTHF_KalmanState class entirely
# Remove motion_model and kf_params from RT_MTFHConfig since they're not used

@dataclass
class RT_MTFHConfig:
    FB_FRAMES_BATCH: int = 5
    # Boundary filtering parameters
    global_xlim: tuple = (-5.0, 5.0)
    global_ylim: tuple = (0.0, 10.0) 
    global_zlim: tuple = (-2.0, 2.0)
    # Speed filtering
    es_threshold: float = 0.5
    # Tracking parameters
    obj_bin_number: int = 10
    poss_clus_deque_length: int = 3
    redundant_clus_remove_cp_dis: float = 0.5
    # DBSCAN parameters
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 3


class RT_MTFHTrackBuffer(Tracker):
    """
    A class representing a buffer for managing and updating the multiple ClusterTracks of the scene.
    """

    def __init__(self, config: RT_MTFHConfig) -> None:
        """
        Initialize TrackBuffer with empty lists for tracks and effective tracks.
        """
        self.config = config
        self.effective_tracks: List[RH_MTHFClusterTrack] = []
        self.next_track_id: int = 0
        self.dt: float = 0
        self.t: float = time.time()
        
        # Initialize tracking objects
        self.people_list = []
        for i in range(config.obj_bin_number):
            self.people_list.append(RH_MTHFClusterTrack(None, config))
            
        self.poss_clus_deque = deque([], config.poss_clus_deque_length)

    def boundary_filter(self, data_points: np.ndarray) -> np.ndarray:
        """
        Filter points outside the defined boundaries
        """
        if len(data_points) == 0:
            return data_points
            
        # Filter by x, y, z limits
        mask = (
            (data_points[:, 0] >= self.config.global_xlim[0]) & 
            (data_points[:, 0] < self.config.global_xlim[1]) &
            (data_points[:, 1] >= self.config.global_ylim[0]) & 
            (data_points[:, 1] < self.config.global_ylim[1]) &
            (data_points[:, 2] >= self.config.global_zlim[0]) & 
            (data_points[:, 2] < self.config.global_zlim[1])
        )
        return data_points[mask]

    def es_speed_filter(self, data_points: np.ndarray) -> tuple:
        """
        Filter points based on speed threshold
        Assumes data format: [x, y, z, vx, vy, vz, doppler, peakval]
        """
        if len(data_points) == 0 or data_points.shape[1] < 8:
            return data_points, np.array([])
        
        # Calculate total speed from velocity components (vx, vy, vz)
        vx, vy, vz = data_points[:, 3], data_points[:, 4], data_points[:, 5]
        total_speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        speed_mask = total_speed > self.config.es_threshold
        filtered_points = data_points[speed_mask]
        noise_points = data_points[~speed_mask]
        
        return filtered_points, noise_points

    def cluster_points(self, data_points: np.ndarray, eps: float = None, min_samples: int = None) -> List[np.ndarray]:
        """
        Cluster points using DBSCAN
        Uses x, y, z coordinates for clustering (columns 0, 1, 2)
        """
        if eps is None:
            eps = self.config.dbscan_eps
        if min_samples is None:
            min_samples = self.config.dbscan_min_samples
            
        if len(data_points) < min_samples:
            return []
            
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(data_points[:, :3])  # Use x,y,z for clustering
        
        clusters = []
        for label in set(cluster_labels):
            if label != -1:  # Ignore noise points
                cluster_points = data_points[cluster_labels == label]
                clusters.append(cluster_points)
                
        return clusters

    def calculate_cluster_properties(self, clusters: List[np.ndarray]) -> tuple:
        """
        Calculate central points and sizes for clusters
        Returns centroids and sizes based on spatial coordinates (x, y, z)
        """
        obj_cp_total = []
        obj_size_total = []
        
        for cluster in clusters:
            if len(cluster) == 0:
                continue
                
            # Calculate boundaries using spatial coordinates (x, y, z)
            x_min, x_max = np.min(cluster[:, 0]), np.max(cluster[:, 0])
            y_min, y_max = np.min(cluster[:, 1]), np.max(cluster[:, 1]) 
            z_min, z_max = np.min(cluster[:, 2]), np.max(cluster[:, 2])
            
            # Central point
            cp = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
            obj_cp_total.append(cp)
            
            # Size
            size = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
            obj_size_total.append(size)
            
        return np.array(obj_cp_total), np.array(obj_size_total)

    def update_possibility_matrix(self, clusters: List[np.ndarray]):
        """
        Update tracking using possibility matrix approach from the original code
        """
        if not clusters:
            return
            
        # Add clusters to deque
        self.poss_clus_deque.append(clusters)
        
        # Flatten all clusters from the deque
        all_clusters = []
        for cluster_list in self.poss_clus_deque:
            all_clusters.extend(cluster_list)
            
        if not all_clusters:
            return
            
        # Calculate cluster properties
        obj_cp_total, obj_size_total = self.calculate_cluster_properties(all_clusters)
        
        if len(obj_cp_total) == 0:
            return
            
        # Calculate possibility matrix using the track's own method
        poss_matrix = np.zeros([len(all_clusters), len(self.people_list)], dtype=np.float32)
        
        for c in range(len(all_clusters)):
            for p in range(len(self.people_list)):
                poss_matrix[c, p] = self.people_list[p].check_clus_possibility(obj_cp_total[c], obj_size_total[c])
        
        # Assignment using possibility matrix (same as original)
        while poss_matrix.size > 0 and np.max(poss_matrix) > 0:
            max_index = divmod(np.argmax(poss_matrix), poss_matrix.shape[1])
            c, p = max_index[0], max_index[1]
            
            # Update the corresponding track using original method name
            self.people_list[p].update_info(all_clusters[c], obj_cp_total[c], obj_size_total[c])
            
            # Remove redundant clusters
            obj_cp_used = obj_cp_total[c]
            for i in range(len(obj_cp_total)):
                diff = obj_cp_total[i] - obj_cp_used
                dis_diff = hypot(diff[0], diff[1])
                if dis_diff < self.config.redundant_clus_remove_cp_dis:
                    poss_matrix[i, :] = 0
            poss_matrix[:, p] = 0

    def track(self, pointcloud: np.array, batch: RingBuffer, clusteringAlgorithm: str = "DBSCAN") -> None:
        """
        Main tracking method integrating the ported functionality
        """
        current_time = time.time()
        self.dt = current_time - self.t
        self.t = current_time
        
        if len(pointcloud) == 0:
            return
            
        # Apply filtering pipeline
        filtered_points = self.boundary_filter(pointcloud)
        filtered_points, noise_points = self.es_speed_filter(filtered_points)
        
        # Cluster the filtered points
        clusters = self.cluster_points(filtered_points)
        
        # Update tracking using possibility matrix
        self.update_possibility_matrix(clusters)
        
        # Update effective tracks from active people
        self.effective_tracks = [person for person in self.people_list if person.is_active()]


class RH_MTHFClusterTrack:
    """
    Simplified track class matching the original HumanObject implementation
    """
    def __init__(self, cluster: "PointCluster", config: RT_MTFHConfig) -> None:
        self.config = config
        self.cluster = cluster
        
        # Track state variables
        self.id = -1  # Will be set when added to tracker
        self.status = ACTIVE
        self.lifetime = 0
        self.state = "normal"  # normal, noise, or lost
        
        # Simple tracking properties (no Kalman filter)
        self.last_position = None
        self.last_size = None
        self.last_update_time = 0
        self.max_invisible_count = 5
        self.consecutiveInvisibleCount = 0
        self.totalVisibleCount = 0
        self.age = 0
        
        # Trajectory recording
        self.traj_rec = []  # trajectory recording
        self.appear_frame = 0

    def check_clus_possibility(self, obj_cp: np.ndarray, obj_size: np.ndarray) -> float:
        """
        Calculate possibility score for cluster assignment (from original HumanObject)
        """
        if self.last_position is None:
            return 0.1  # Small possibility for new tracks
            
        # Simple distance-based possibility
        distance = np.linalg.norm(obj_cp - self.last_position)
        possibility = max(0, 1.0 - distance / 2.0)  # Simple distance-based scoring
        
        return possibility

    def update_with_cluster(self, cluster_points: np.ndarray, centroid: np.ndarray, size: np.ndarray):
        """
        Update track with new cluster information (simplified, no Kalman filter)
        """
        self.last_position = centroid.copy()
        self.last_size = size.copy()
        self.last_update_time = time.time()
        self.consecutiveInvisibleCount = 0
        self.totalVisibleCount += 1
        self.age += 1
        
        # Update trajectory
        self.traj_rec.append(centroid.copy())
        if len(self.traj_rec) > 100:  # Keep last 100 positions
            self.traj_rec.pop(0)

    def is_active(self) -> bool:
        """
        Check if track is still active
        """
        return (self.consecutiveInvisibleCount < self.max_invisible_count and 
                self.totalVisibleCount > 0)

    def predict_state(self, dt):
        """
        Predict the next state of the track. Note that this code does not use
        a Kalman filter or any advanced prediction model.
        """
        return self.last_position if self.last_position is not None else np.zeros(3)

    def update_info(self, cluster_points: np.ndarray, obj_cp: np.ndarray, obj_size: np.ndarray):
        """
        Method name matching the original HumanObject.update_info
        """
        self.update_with_cluster(cluster_points, obj_cp, obj_size)