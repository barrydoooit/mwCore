from functools import partial
import numpy as np
from .gtrack_asterios import Tracker,  BatchedData, PointCluster
from .gtrack_asterios import ACTIVE, INACTIVE, STATIC, DYNAMIC
import time
from ..utils import (
    altered_EuclideanDist,
    apply_clustering,
)
from typing import Any, List, Tuple
from ..palmar import AdaptiveOrderHMM, CPDA
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class RKFConfig:
    # Functions to compute state transition and process noise covariance given dt.
    RKF_F: Callable[[float], np.ndarray]  # e.g. lambda dt: np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    RKF_Q: Callable[[float], np.ndarray]  # e.g. lambda dt: np.array([[0.2,0,0,0],[0,0.2,0,0],[0,0,0.2,0],[0,0,0,0.2]])
    
    # Fixed measurement matrix for polar observations: observes [r, θ]
    H: np.ndarray  # shape (2,4), e.g. np.array([[1,0,0,0],[0,0,1,0]])
    
    KF_GROUP_DISP_EST_INIT: float  # e.g. 0.1
    KF_ENABLE_EST: bool  # e.g. False
    KF_A_N: float  # e.g. 0.9
    KF_EST_POINTNUM: int  # e.g. 10
    KF_SPREAD_LIM: List[float]  # e.g. [0.2, 0.2, 2, 1.2]
    KF_A_SPR: float  # e.g. 0.9
    
    DB_POINTS_THRES: int  # e.g. 40
    DB_SPREAD_THRES: float  # e.g. 0.7
    DB_EPS: float
    DB_RANGE_WEIGHT: float
    DB_Z_WEIGHT: float
    DB_MIN_SAMPLES_MIN: int
    FB_FRAMES_BATCH: int  # e.g. 2
    
    KF_R_STD: float  # e.g. 0.1
    
    # MODEL_DEFAULT_POSTURE: np.ndarray  # e.g. a posture array
    
    TR_LIFETIME_DYNAMIC: float  # e.g. 3.0 (seconds)
    TR_LIFETIME_STATIC: float  # e.g. 7.0 (seconds)
    TR_GATE: float  # e.g. 4.5
    TR_MAX_TRACKS: int  # e.g. 4
    TR_VEL_THRES: float

    ENABLE_TORSO_TRACKING: bool  # New flag
    MAX_LIMB_VELOCITY: float         # m/s
    MIN_TORSO_MOVEMENT: float       # meters
    TORSO_DENSITY_RADIUS: float     # meters
    MIN_TORSO_POINTS: int 

def voxelize(pointcloud, voxel_size):
    """
    Perform voxelization on the point cloud data.

    Parameters
    ----------
    pointcloud : np.array
        Pointcloud data.
    voxel_size : float
        Size of the voxel.

    Returns
    -------
    np.array
        Voxelized point cloud data.
    """
    # Calculate voxel grid coordinates
    voxel_grid = np.floor(pointcloud[:, :3] / voxel_size).astype(np.int32)

    # Use a dictionary to store voxelized points
    voxel_dict = {}
    for i, voxel in enumerate(voxel_grid):
        voxel_key = tuple(voxel)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(pointcloud[i])

    # Calculate the average point for each voxel
    voxelized_pointcloud = []
    for voxel_key, points in voxel_dict.items():
        voxelized_pointcloud.append(np.mean(points, axis=0))

    return np.array(voxelized_pointcloud)

class RecursiveKalmanFilter:
    def __init__(self, dt=0.05, Q=None, R=None):
        self.dt = dt  # Sampling time (50ms)
        
        # System State Transition Matrix (F) in polar coordinates
        self.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ])
        
        # Measurement Matrix (H) - Observes range and angle
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        # Process noise covariance (Q)
        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.array([
                [0.2, 0, 0, 0],
                [0, 0.2, 0, 0],
                [0, 0, 0.2, 0],
                [0, 0, 0, 0.2]
            ])
        
        # Measurement noise covariance (R)
        self.R = R if R is not None else np.eye(2) * 1e-2
        
        # Initial state estimate (x) and covariance matrix (P)
        self.x = np.zeros((4, 1))  # [r, r_dot, theta, theta_dot]
        self.P = np.eye(4)

    def predict(self):
        # Predict state
        self.x = self.F @ self.x
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R=None):
        R = R if R is not None else self.R


        # Measurement residual (innovation)
        z = z.reshape(-1, 1)
        y = z - (self.H @ self.x)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R
        
        # Recursively calculate Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Recursively update error covariance matrix
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T
        
    
class RKFClusterTrack:
    """
    A class representing a tracked cluster with a Recursive Kalman Filter.

    Parameters
    ----------
    cluster : PointCluster
        The initial point cluster associated with the track.

    Attributes
    ----------
    N_est : int
        Estimated number of points in the cluster.
    spread_est : numpy.ndarray
        Estimated spread of measurements in each dimension.
    group_disp_est : numpy.ndarray
        Estimated group dispersion matrix.
    cluster : PointCluster
        PointCluster associated with the track.
    batch : BatchedData
        The collection of overlaying previous frames
    state : RecursiveKalmanFilter
        The state of the Kalman filter.
    status : int (INACTIVE or ACTIVE)
        Current status of the track.
    lifetime : int
        Number of frames the track has been active.
    keypoints : list of floats
        Keypoint x, y, z coordinates of the tracked 19 joints
    color : numpy.ndarray
        Random color assigned to the track for visualization (for visualization purposes).
    predict_x : numpy.ndarray
        Predicted state vector (for visualization purposes).

    Methods
    -------
    predict_state(dt)
        Predict the state of the Kalman filter based on the time multiplier.

    _estimate_point_num()
        Estimate the number of points in the cluster.

    _estimate_measurement_spread()
        Estimate the spread of measurements in each dimension.

    _estimate_group_disp_matrix()
        Estimate the group dispersion matrix.

    _get_D()
        Calculate and get the dispersion matrix for the track.

    associate_pointcloud(pointcloud)
        Associate a new pointcloud with the track.

    get_Rm()
        Get the measurement covariance matrix.

    _get_Rc()
        Get the combined covariance matrix.

    update_state()
        Update the state of the Kalman filter based on the associated pointcloud.

    update_lifetime(reset=False)
        Update the track lifetime.

    seek_inner_clusters()
        Seek inner clusters within the current track.

    """
    def __init__(self, cluster: PointCluster, config: RKFConfig, usePalmar: bool = False) -> None:
        self.config = config
        self.N_est = 0  # For RKF the state vector has 4 elements: [r, r_dot, θ, θ_dot]
        self.spread_est = np.zeros(4)
        self.group_disp_est = np.eye(4) * config.KF_GROUP_DISP_EST_INIT
        self.cluster = cluster
        self.batch = BatchedData(self.config.FB_FRAMES_BATCH + 1, cluster.pointcloud)
        self.state = RecursiveKalmanFilter(R=np.eye(4) * config.KF_R_STD**2)
        self.status = ACTIVE  # ACTIVE status
        self.lifetime = 0
        self.predict_x = self.state.x
        self.color = np.random.rand(3)
        self.usePalmar = usePalmar
        self.useTorsoTracking = config.ENABLE_TORSO_TRACKING
        r, theta, r_dot = cluster.centroid[-3:]  # Assume last 3 are [r, θ, r_dot]
        self.state.x = np.array([
            [r], 
            [r_dot],  # Use measured Doppler
            [theta], 
            [0.0]     # θ_dot unknown
        ])
        self.hits = 1  
        self.misses = 0  
        self.torso_center = None  # Center of torso for tracking
        self.last_valid_position = None  # Last valid position for torso tracking
        self.body_state = deepcopy(self.state)  # Separate state for body tracking
        self.limb_points = []  # Stores outlier points

        if usePalmar:
            self.ao_hmm = AdaptiveOrderHMM()

    def _estimate_point_num(self) -> None:
        if self.config.KF_ENABLE_EST:
            if self.cluster.point_num > self.N_est:
                self.N_est = self.cluster.point_num
            else:
                self.N_est = (1 - self.config.KF_A_N) * self.N_est + self.config.KF_A_N * self.cluster.point_num
        else:
            self.N_est = max(self.config.KF_EST_POINTNUM, self.cluster.point_num)

    def _estimate_measurement_spread(self) -> None:
        for m in range(len(self.cluster.min_vals)):
            spread = self.cluster.max_vals[m] - self.cluster.min_vals[m]
            if self.cluster.point_num != 1:
                spread = spread * (self.cluster.point_num + 1) / (self.cluster.point_num - 1)
            spread = min(2 * self.config.KF_SPREAD_LIM[m], spread)
            spread = max(self.config.KF_SPREAD_LIM[m], spread)
            if spread > self.spread_est[m]:
                self.spread_est[m] = spread
            else:
                self.spread_est[m] = (1.0 - self.config.KF_A_SPR) * self.spread_est[m] + self.config.KF_A_SPR * spread

    def _get_D(self) -> np.ndarray:
        dimension = 4  # [r, r_dot, θ, θ_dot]
        # Use only the last 3 polar dimensions from the cluster: [r, θ, r_dot]
        pointcloud = self.cluster.pointcloud[:, -3:]
        centroid = self.cluster.centroid
        # Transform measurements to state vector coordinates: assume θ_dot = 0
        transformed_pointcloud = np.zeros((pointcloud.shape[0], dimension))
        for i, point in enumerate(pointcloud):
            r, theta, r_dot = point
            transformed_pointcloud[i] = [r, r_dot, theta, 0.0]
        transformed_centroid = np.array([centroid[0], centroid[2], centroid[1], 0.0])
        disp = np.zeros((dimension, dimension), dtype=float)
        for i in range(dimension):
            for j in range(dimension):
                disp[i, j] = np.mean((transformed_pointcloud[:, i] - transformed_centroid[i]) *
                                    (transformed_pointcloud[:, j] - transformed_centroid[j]))
        return disp

    def _estimate_group_disp_matrix(self) -> None:
        a = self.cluster.point_num / self.N_est
        self.group_disp_est = (1 - a) * self.group_disp_est + a * self._get_D()

    def _get_Rc(self) -> np.ndarray:
        N = self.cluster.point_num
        N_est = self.N_est
        R_m = self.get_Rm()  # 2x2 measurement covariance
        # Project group dispersion to measurement space using H from config.
        D_projected = self.config.H @ self._get_D() @ self.config.H.T
        return (R_m / N) + (((N_est - N) / ((N_est - 1) * N)) * D_projected)

    def associate_pointcloud(self, pointcloud: np.array, useBatch: bool = False):
        # First classify points
        self.body_points, self.limb_points = self._classify_points(pointcloud)
        
        # print(f"Classified {len(self.body_points)} body points and {len(self.limb_points)} limb points")  # Debug line
        # Update body state if possible
        if len(self.body_points) >= self.config.MIN_TORSO_POINTS:
            print(f"Updating body state with {len(self.body_points)} points")  # Debug line
        
        # Original processing 
        if not useBatch:
            self.cluster = PointCluster(pointcloud, polar=True)
            self.batch.add_frame(self.cluster.pointcloud)
        else:
            self.batch.add_frame(pointcloud)
            fused, weights = self.batch._compute_effective_data()
            self.cluster = PointCluster(fused, tr_vel_threshold=self.config.TR_VEL_THRES,
                                    weights=weights, isFrame=True, polar=True)
        
        self._estimate_point_num()
        self._estimate_measurement_spread()
        self._estimate_group_disp_matrix()

    def predict_state(self, dt: float):
        # Predict both states
        self.state.F = self.config.RKF_F(dt)
        self.state.predict()
        
        self.predict_x = self.state.x

    def get_Rm(self) -> np.ndarray:
        diagonal_elements = (self.spread_est[:2] / 2) ** 2
        return np.diag(diagonal_elements)


    def update_state(self) -> None:
        z = self.cluster.centroid[:2]
        
        # Apply the update
        self.state.update(z, R=self.state.R)
        

    def update_lifetime(self, dt: float, reset: bool = False) -> None:
        if reset:
            self.lifetime = 0
        else:
            self.lifetime += dt

    def update_status(self):
        if self.hits >= 2: 
            self.status = ACTIVE  # Confirmed track
        elif self.misses > 5:
            self.status = 0  # Mark inactive

            
    def observation_to_index(self, observation: np.ndarray) -> int:
        r, theta = observation
        r_bins = [0.000, 2.141, 3.257, 4.275]
        theta_bins = [0.000, 1.413, 1.602, 1.941]
        r_bin = np.digitize(r, r_bins) - 1
        theta_bin = np.digitize(theta, theta_bins) - 1
        obs_index = r_bin * 3 + theta_bin
        return obs_index

    
    def _classify_points(self, pointcloud: np.array) -> Tuple[np.array, np.array]:
        """Separate body points from limb points"""
        if len(pointcloud) == 0:
            return np.empty((0,3)), np.empty((0,3))
        
        body_points = []
        limb_points = []
        
        for point in pointcloud:
            r, theta, r_dot = point[:3]
            if abs(r_dot - self.body_state.x[1,0]) < self.config.MAX_LIMB_VELOCITY:
                body_points.append(point)
            else:
                limb_points.append(point)
                
        return np.array(body_points), np.array(limb_points)


class RKFTrackBuffer(Tracker):
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

    def __init__(self, config: RKFConfig, usePalmar: bool = False) -> None:
        self.config = config
        self.effective_tracks = []
        self.next_track_id = 0
        self.dt = 0
        self.t = time.time()
        self.usePalmar = usePalmar

    def _maintain_tracks(self) -> None:
        for track in self.effective_tracks:
            lifetime = self.config.TR_LIFETIME_DYNAMIC if track.cluster.status != 0 else self.config.TR_LIFETIME_STATIC
            lifetime_expired = track.lifetime > lifetime
            
            too_many_misses = track.misses > 4
            
            if lifetime_expired or too_many_misses:
                track.status = 0  # INACTIVE
                
        self.effective_tracks = [track for track in self.effective_tracks if track.status != 0]


    def _calc_dist_fun(self, full_set: np.array) -> np.ndarray:
        num_points = full_set.shape[0]
        num_tracks = len(self.effective_tracks)
        dist_matrix = np.full((num_points, num_tracks), np.inf)  # Initialize with infinity
        associated_track_for = np.full(num_points, None, dtype=object)
        H = self.config.H


        
        for j, track in enumerate(self.effective_tracks):
            # Use predicted state for current time step
            H_i = np.dot(H, track.state.x).flatten()
            
            # Calculate the measurement covariance matrix
            C_g_j = track.state.P + track._get_Rm() + track.group_disp_est
            C_g_j_obs = H @ C_g_j @ H.T 
            for i, point in enumerate(full_set):
                r, theta, _ = point[-3:]  # Polar coordinates
                z = np.array([r, theta])
                y_ij = z - H_i
                
                # Mahalanobis distance
                try:
                    dist = (np.log(np.abs(np.linalg.det(C_g_j_obs))) +
                            y_ij.T @ np.linalg.inv(C_g_j_obs) @ y_ij)
                except np.linalg.LinAlgError:
                    dist = np.inf
                # Store distance for potential assignment
                dist_matrix[i, j] = dist

                # Vertically gate the point to the track


                if dist < self.config.TR_GATE:
                    if associated_track_for[i] is None:
                        associated_track_for[i] = j
                    elif dist < dist_matrix[i, associated_track_for[i]]:
                        associated_track_for[i] = j
                        
        return associated_track_for

    def _add_tracks(self, new_clusters: List[np.array]) -> None:
        for new_cluster in new_clusters:
            new_track = RKFClusterTrack(PointCluster(np.array(new_cluster), tr_vel_threshold=self.config.TR_VEL_THRES, polar=True), self.config, usePalmar=self.usePalmar)
            self.next_track_id += 1
            self.effective_tracks.append(new_track)

    def _predict_all(self) -> None:
        for i, track in enumerate(self.effective_tracks):
            # print(f"Track {i} pre-predict state: {track.state.x.flatten()}")  # Debug
            track.predict_state(self.dt)
            # print(f"Track {i} post-predict state: {track.state.x.flatten()}")  # Debug

    def _update_all(self) -> None:
        for i, track in enumerate(self.effective_tracks):
            if track.hits > 0:
                # print(f"Updating track {i} with centroid {track.cluster.centroid[:2]}")  # Debug
                track.update_state()



    def _get_gated_clouds(self, full_set: np.array) -> Tuple[np.ndarray, List[List[np.array]]]:
        unassigned = np.empty((0, full_set.shape[1]), dtype=float)
        clusters = [[] for _ in range(len(self.effective_tracks))]
        associated_track_for = self._calc_dist_fun(full_set)
        
        for i, point in enumerate(full_set):
            if associated_track_for[i] is None:
                unassigned = np.append(unassigned, [point], axis=0)
            else:
                track_idx = associated_track_for[i]
                clusters[track_idx].append(point)
                
        return unassigned, clusters

    def _associate_points_to_tracks(self, full_set: np.array) -> np.ndarray:
        unassigned, clouds = self._get_gated_clouds(full_set)
        
        for j, track in enumerate(self.effective_tracks):
            if len(clouds[j]) < 1:
                # Update miss counter and lifetime
                track.misses += 1
                track.update_lifetime(dt=self.dt)
            else:
                # Reset miss counter and lifetime
                track.misses = 0
                track.update_lifetime(dt=self.dt, reset=True)
                track.associate_pointcloud(np.array(clouds[j]), useBatch=True)
                
        return unassigned
    


    def track(self, pointcloud: np.array, batch: BatchedData, clusteringAlgorithm: str = "DBSCAN") -> None:
        if clusteringAlgorithm not in ["DBSCAN", "BIRCH", "both"]:
            raise ValueError("Invalid clustering algorithm. Please use 'DBSCAN', 'BIRCH', or 'both'.")
        pointcloud = self.remove_outliers(pointcloud)
        self._predict_all()
        unassigned = self._associate_points_to_tracks(pointcloud)

        # print(f"Unassigned points after association: {len(unassigned)}")
        # print(f"Active tracks: {len([t for t in self.effective_tracks if t.status != 0])}")
    
        self._maintain_tracks()
        self._update_all()
        new_clusters = []
        batch.add_frame(unassigned)
        if batch.effective_data.size > 0 and len(self.effective_tracks) < self.config.TR_MAX_TRACKS:
            new_clusters = apply_clustering(batch.effective_data, 
                                            clusteringAlgorithm, 
                                            metric=partial(altered_EuclideanDist,
                                                db_range_weight=self.config.DB_RANGE_WEIGHT,
                                                db_z_weight=self.config.DB_Z_WEIGHT),
                                            eps=self.config.DB_EPS,
                                            min_samples=self.config.DB_MIN_SAMPLES_MIN
                                            )
            if new_clusters:
                batch.clear()
            # print(f"Creating {len(new_clusters)} new tracks")  # Debug line
            self._add_tracks(new_clusters)
    
    def remove_outliers(self, pointcloud: np.array, 
                   min_range: float = 0.3, 
                   max_range: float = 50.0,
                   z_threshold: float = 2.0,
                   use_statistical: bool = False) -> np.array:
        """
        Remove outliers from point cloud using multi-stage filtering
        
        Parameters
        ----------
        pointcloud : np.array
            Input point cloud with structure [x, y, z, ..., r, azimuth, ...]
        min_range : float, optional
            Minimum valid radial distance (meters). Default 0.3m
        max_range : float, optional
            Maximum valid radial distance (meters). Default 50.0m
        z_threshold : float, optional
            Maximum valid z-height (meters). Default 2.0m
        use_statistical : bool, optional
            Enable statistical outlier removal. Default True
        
        Returns
        -------
        np.array
            Filtered point cloud
        """
        if pointcloud.size == 0:
            return pointcloud
        
        # Remember that each point in the pointcloud is structured as:
        # [x, y, z, x_vel, y_vel, z_vel, doppler, peakVal, r, theta, r_dot]

        # 1. Range-based filtering using radial distance
        radial_dists = pointcloud[:, -3]  # Radial distance is stored at index -3
        range_mask = (radial_dists >= min_range) & (radial_dists <= max_range)
        
        # 2. Height filtering (remove flying objects/unrealistic heights)
        z_vals = pointcloud[:, 2]  # Z-coordinate at index 2
        height_mask = np.abs(z_vals) <= z_threshold
        
        # 3. Statistical Outlier Removal (optional)
        if use_statistical and np.sum(range_mask & height_mask) > 1:
            filtered = pointcloud[range_mask & height_mask]
            
            # Calculate mean distance to K-nearest neighbors
            k = min(10, len(filtered) - 1)  # Ensure k < num_points
            nn_distances = np.sort(np.linalg.norm(filtered[:, None, :3] - filtered[None, :, :3], axis=-1))[:, 1:k+1]  # Exclude self-distance
            
            mean_dists = np.mean(nn_distances, axis=1)
            dist_threshold = np.percentile(mean_dists, 90)  
            statistical_mask = mean_dists <= dist_threshold
            
            final = filtered[statistical_mask]
        else:
            final = pointcloud[range_mask & height_mask]
        
        # 4. Velocity filtering 
        if pointcloud.shape[1] > 7:  # Check if velocity data exists
            velocities = final[:, -1]
            velocity_mask = np.abs(velocities) <= 25.0  # 25 m/s = 90 km/h
            final = final[velocity_mask]
        
        # removed = len(pointcloud) - len(final)
        # if removed > 0:
        #     print(f"Removed {removed} outliers "
        #         f"(range: [{min_range},{max_range}]m, "
        #         f"height: ±{z_threshold}m)")
        
        return final
def transform_measurement(measurement):
    """
    Transform a measurement from [r, θ, ṙ] to [r, ṙ, θ, θ̇].

    Parameters
    ----------
    measurement : np.array
        Measurement vector [r, θ, ṙ].

    Returns
    -------
    np.array
        Transformed measurement vector [r, ṙ, θ, θ̇].
    """
    r, θ, ṙ = measurement
    return np.array([r, ṙ, θ, 0.0])  # Assume θ̇ = 0 (or estimate it if available)