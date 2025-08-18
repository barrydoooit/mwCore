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
    dbscan_sort: bool = True
    dbscan_sort_limit: int = 5  # Maximum number of clusters to keep
    # Dynamic ES levels
    dynamic_es_levels: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0])
    # Human object parameters
    obj_deque_length: int = 10
    dis_diff_threshold: float = 0.5
    dis_diff_threshold_dr: float = 0.1
    size_diff_threshold: float = 0.1
    sub_possibility_proportion: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    expect_pos: dict = field(default_factory=lambda: {'default': [None, None, None]})
    expect_shape: dict = field(default_factory=lambda: {'default': [None, None, None]})

    obj_delete_timeout: float = 5.0
    fuzzy_boundary_enter: bool = False
    fuzzy_boundary_threshold: float = 0.5

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
        Enhanced DBSCAN clustering closer to the original implementation
        """
        if eps is None:
            eps = self.config.dbscan_eps
        if min_samples is None:
            min_samples = self.config.dbscan_min_samples
            
        if len(data_points) < min_samples:
            return []
            
        # Run DBSCAN on spatial coordinates only (matching original: data_points[:, 0:3])
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(data_points[:, :3])
        
        # Filter DBSCAN noise (matching original logic)
        noise_mask = cluster_labels == -1
        valid_points = data_points[~noise_mask]
        valid_labels = cluster_labels[~noise_mask]
        
        if len(valid_points) == 0:
            return []
        
        # Get info for each cluster including central point position, size and label
        # (matching original: cluster_info_total with 7 columns)
        cluster_info_total = []
        valid_labels_unique = np.unique(valid_labels)
        
        for label in valid_labels_unique:
            cluster_points = valid_points[valid_labels == label]
            
            # Calculate boundaries using same method as original
            x_vals = cluster_points[:, 0]
            y_vals = cluster_points[:, 1] 
            z_vals = cluster_points[:, 2]
            
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            y_min, y_max = np.min(y_vals), np.max(y_vals)
            z_min, z_max = np.min(z_vals), np.max(z_vals)
            
            # Central point (matching original calculation)
            cp_pos = np.array([
                (x_min + x_max) / 2, 
                (y_min + y_max) / 2, 
                (z_min + z_max) / 2
            ], dtype=np.float16)
            
            # Size calculation (matching original)
            size = np.array([
                x_max - x_min,
                y_max - y_min, 
                z_max - z_min
            ], dtype=np.float16)
            
            # Store cluster info: [cp_pos_x, cp_pos_y, cp_pos_z, size_x, size_y, size_z, label]
            cluster_info = np.concatenate([cp_pos, size, np.array([label], dtype=np.float16)])
            cluster_info_total.append({
                'info': cluster_info,
                'points': cluster_points,
                'cp_pos': cp_pos,
                'size': size,
                'label': label
            })
        
        # Apply filters (matching original filter sequence)
        filtered_cluster_info = []
        for cluster_data in cluster_info_total:
            cp_pos = cluster_data['cp_pos']
            size = cluster_data['size']
            
            # Apply position filters (matching original boundary checks)
            if not (self.config.global_xlim[0] <= cp_pos[0] < self.config.global_xlim[1]):
                continue
            if not (self.config.global_ylim[0] <= cp_pos[1] < self.config.global_ylim[1]):
                continue
            if not (self.config.global_zlim[0] <= cp_pos[2] < self.config.global_zlim[1]):
                continue
                
            filtered_cluster_info.append(cluster_data)
        
        # DBSCAN sort process
        if hasattr(self.config, 'dbscan_sort') and self.config.dbscan_sort:
            # Sort clusters by point count (high to low)
            filtered_cluster_info.sort(key=lambda x: len(x['points']), reverse=True)
            
            # Only return the biggest several clusters if sort limit is set
            sort_limit = getattr(self.config, 'dbscan_sort_limit', len(filtered_cluster_info))
            filtered_cluster_info = filtered_cluster_info[:sort_limit]
        
        # Return the actual point arrays
        return [cluster_data['points'] for cluster_data in filtered_cluster_info]



    def update_possibility_matrix(self, clusters: List[np.ndarray]):
        """Update tracking using possibility matrix approach from the original code"""
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
            
        # Calculate cluster properties inline
        obj_cp_total = []
        obj_size_total = []
        
        for cluster in all_clusters:
            if len(cluster) == 0:
                continue
                
            # Calculate boundaries using spatial coordinates (x, y, z)
            x_min, x_max = np.min(cluster[:, 0]), np.max(cluster[:, 0])
            y_min, y_max = np.min(cluster[:, 1]), np.max(cluster[:, 1]) 
            z_min, z_max = np.min(cluster[:, 2]), np.max(cluster[:, 2])
            
            # Central point and size
            cp = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
            size = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
            obj_cp_total.append(cp)
            obj_size_total.append(size)
        
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
            
            # Update the corresponding track
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
            # Still update effective tracks even with no points
            self.effective_tracks = [person for person in self.people_list if person.is_active()]
            return
            
        # Apply filtering pipeline
        filtered_points = self.boundary_filter(pointcloud)
        filtered_points, noise_points = self.es_speed_filter(filtered_points)
        
        # Cluster the filtered points
        clusters = self.cluster_points(filtered_points)
        
        # Update tracking using possibility matrix
        self.update_possibility_matrix(clusters)
        
        # Update effective tracks from active people (this will handle timeouts)
        self.effective_tracks = [person for person in self.people_list if person.is_active()]


class RH_MTHFClusterTrack:
    """
    Simplified track class matching the original HumanObject implementation
    """
    def __init__(self, cluster: "PointCluster", config: RT_MTFHConfig) -> None:
        self.config = config
        self.cluster = cluster

        deque_length = getattr(config, 'obj_deque_length', 10)
        self.obj_cp_deque = deque([], maxlen=deque_length)
        self.obj_size_deque = deque([], maxlen=deque_length)
        self.obj_status_deque = deque([], maxlen=deque_length)
        self.obj_speed_deque = deque([], maxlen=deque_length)
        self.obj_timestamp_deque = deque([], maxlen=deque_length)
        
        # Use config thresholds
        self.dis_diff_threshold = config.dis_diff_threshold
        self.dis_diff_threshold_dr = config.dis_diff_threshold_dr
        self.size_diff_threshold = config.size_diff_threshold
        self.sub_possibility_proportion = config.sub_possibility_proportion
        self.expect_pos = config.expect_pos
        self.expect_shape = config.expect_shape
        
        # Tracking properties 
        self.last_position = None
        self.last_size = None
        self.last_update_time = 0
        self.max_invisible_count = 5
        self.consecutiveInvisibleCount = 0
        self.totalVisibleCount = 0
        self.age = 0
        
        # Trajectory recording
        self.traj_rec = []  # trajectory recording

    def check_clus_possibility(self, obj_cp: np.ndarray, obj_size: np.ndarray) -> float:
        """
        Calculate possibility score for cluster assignment (from original HumanObject)
        """
        # If no previous points saved
        if len(self.obj_cp_deque) == 0:
            dis_possibility = 0
            size_possibility = 0
            # Enable central point starts around scene boundaries (if configured)
            if hasattr(self.config, 'fuzzy_boundary_enter') and self.config.fuzzy_boundary_enter:
                if not self._boundary_fuzzy_area(obj_cp):
                    return 0
        else:
            # Distance-based possibility
            diff = obj_cp - self.obj_cp_deque[-1]
            dis_diff = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
            
            # Dynamic distance threshold
            dyn_dis_threshold = self.dis_diff_threshold + abs(self.obj_speed_deque[-1]) * self.dis_diff_threshold_dr
            
            if dis_diff < dyn_dis_threshold:
                dis_possibility = (dyn_dis_threshold - dis_diff) / dyn_dis_threshold
            else:
                return 0
                
            # Size-based possibility
            size_diff = abs(np.prod(obj_size) - np.prod(self.obj_size_deque[-1]))
            if size_diff < self.size_diff_threshold:
                size_possibility = (self.size_diff_threshold - size_diff) / self.size_diff_threshold
            else:
                return 0
        
        # Self possibility based on expected position and shape
        pos_possibility, shape_possibility = self._get_self_possibility(obj_cp, obj_size, 
                                                                    self.expect_pos['default'], 
                                                                    self.expect_shape['default'])
        
        # Weighted combination
        point_taken_possibility = sum(np.array([dis_possibility, size_possibility, pos_possibility, shape_possibility]) * 
                                    np.array(self.sub_possibility_proportion))
        return point_taken_possibility
    

    def is_active(self) -> bool:
        """
        Check if track is still active based on timeout logic 
        """
        if len(self.obj_timestamp_deque) == 0:
            return False
            
        # Check for deletion timeout
        if (time.time() - self.obj_timestamp_deque[-1]) >= self.config.obj_delete_timeout:
            # Clear all deques when timeout
            self.obj_cp_deque.clear()
            self.obj_size_deque.clear()
            self.obj_status_deque.clear()
            self.obj_speed_deque.clear()
            self.obj_timestamp_deque.clear()
            return False
            
        return len(self.obj_cp_deque) > 0

    def _get_self_possibility(self, obj_cp, obj_size, expect_pos, expect_shape):
        """Calculate possibility based on expected position and shape"""
        # Position possibility
        pos_diff_list = []
        for i in range(len(expect_pos)):
            if expect_pos[i] is not None:
                pos_diff_axis = abs(obj_cp[i] - expect_pos[i])
            else:
                pos_diff_axis = 0
            pos_diff_list.append(pos_diff_axis)
        pos_diff = np.sqrt(sum([x**2 for x in pos_diff_list]))
        pos_possibility = np.exp(-pos_diff)
        
        # Shape possibility
        shape_diff_list = []
        for i in range(len(expect_shape)):
            if expect_shape[i] is not None:
                shape_diff_axis = abs(obj_size[i] - expect_shape[i])
            else:
                shape_diff_axis = 0
            shape_diff_list.append(shape_diff_axis)
        shape_diff = np.array(shape_diff_list, dtype=np.float16)
        shape_possibility = sum(np.exp(-shape_diff) * [0.2, 0.2, 0.6])
        
        return pos_possibility, shape_possibility

    def _get_speed(self, data_points):
        """Calculate cluster average speed matching original"""
        if data_points.shape[1] >= 6:  # Has velocity components vx, vy, vz
            # Calculate speed from velocity components (columns 3, 4, 5). Our points are like: {x, y, z, vx, vy, vz, doppler, snr}
            vx, vy, vz = data_points[:, 3], data_points[:, 4], data_points[:, 5]
            speeds = np.sqrt(vx**2 + vy**2 + vz**2)
            speeds = speeds[speeds != 0]  # Filter out zero speeds
            return float(np.mean(speeds)) if len(speeds) > 0 else 0
        elif data_points.shape[1] > 3:
            # Fallback to single speed column. This is from the original code
            speed_np = data_points[:, 3]
            speed_np = speed_np[speed_np != 0]
            return float(np.mean(speed_np)) if speed_np.size > 0 else 0
        return 0

    def _boundary_fuzzy_area(self, data_point):
        """Check if point is in boundary fuzzy area"""
        if not hasattr(self.config, 'fuzzy_boundary_threshold'):
            return True
            
        threshold = self.config.fuzzy_boundary_threshold
        
        # Define fuzzy areas around boundaries
        x_fuzzy1 = (self.config.global_xlim[0] - threshold, self.config.global_xlim[0] + threshold)
        x_fuzzy2 = (self.config.global_xlim[1] - threshold, self.config.global_xlim[1] + threshold)
        y_fuzzy1 = (self.config.global_ylim[0] - threshold, self.config.global_ylim[0] + threshold)
        y_fuzzy2 = (self.config.global_ylim[1] - threshold, self.config.global_ylim[1] + threshold)
        z_fuzzy1 = (self.config.global_zlim[0] - threshold, self.config.global_zlim[0] + threshold)
        z_fuzzy2 = (self.config.global_zlim[1] - threshold, self.config.global_zlim[1] + threshold)

        x, y, z = data_point[0], data_point[1], data_point[2]
        
        # Check if point is in any fuzzy boundary area
        in_fuzzy = ((x_fuzzy1[0] <= x < x_fuzzy1[1]) or (x_fuzzy2[0] <= x < x_fuzzy2[1]) or
                    (y_fuzzy1[0] <= y < y_fuzzy1[1]) or (y_fuzzy2[0] <= y < y_fuzzy2[1]) or
                    (z_fuzzy1[0] <= z < z_fuzzy1[1]) or (z_fuzzy2[0] <= z < z_fuzzy2[1]))
        
        return in_fuzzy
    
    def predict_state(self, dt):
        """
        Predict the next state of the track. Note that this code does not use
        a Kalman filter.
        """
        return self.last_position if self.last_position is not None else np.zeros(3)

    def update_info(self, cluster_points: np.ndarray, obj_cp: np.ndarray, obj_size: np.ndarray):
        """Update track with cluster information matching original"""
        # Update deques
        self.obj_cp_deque.append(obj_cp)
        self.obj_size_deque.append(obj_size)
        self.obj_speed_deque.append(self._get_speed(cluster_points))
        self.obj_timestamp_deque.append(time.time())
        
        # Update other properties
        self.last_position = obj_cp.copy()
        self.last_size = obj_size.copy()
        self.last_update_time = time.time()
        self.consecutiveInvisibleCount = 0
        self.totalVisibleCount += 1
        self.age += 1
        
        # Update trajectory
        self.traj_rec.append(obj_cp.copy())
        if len(self.traj_rec) > 100:
            self.traj_rec.pop(0)