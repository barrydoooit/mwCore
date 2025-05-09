from functools import partial
import numpy as np
from dataclasses import dataclass
import math
import time
from filterpy.kalman import KalmanFilter
from .utils import (
    altered_EuclideanDist,
    apply_clustering,
    RingBuffer,
)
from typing import List, Callable, Any
from abc import ABC, abstractmethod


@dataclass
class ConstAccModel:
    KF_DIM: List[int]  # e.g. [9, 6]
    KF_H: np.ndarray   # Measurement matrix (6x9? Typically defined in the model)
    KF_F: Callable[[float], np.ndarray]  # Function: dt -> state transition matrix
    KF_Q_DISCR: Callable[[float], np.ndarray]  # Function: dt -> process noise covariance
    STATE_VEC: Callable[[np.ndarray], List[float]]  # Function to build the initial state vector

@dataclass
class AsteriosConfig:
    motion_model: ConstAccModel
    FB_FRAMES_BATCH: int
    FB_FRAMES_BATCH_STATIC: int
    DB_POINTS_THRES: int
    DB_SPREAD_THRES: float
    DB_EPS: float
    DB_RANGE_WEIGHT: float
    DB_Z_WEIGHT: float
    DB_MIN_SAMPLES_MIN: int
    KF_R_STD: float
    KF_P_INIT: float
    KF_GROUP_DISP_EST_INIT: float
    KF_ENABLE_EST: bool
    KF_A_N: float
    KF_EST_POINTNUM: int
    KF_SPREAD_LIM: List[float]
    KF_A_SPR: float
    # MODEL_MIN_INPUT: int
    # MODEL_DEFAULT_POSTURE: np.ndarray
    TR_LIFETIME_DYNAMIC: float
    TR_LIFETIME_STATIC: float
    TR_GATE: float
    TR_MAX_TRACKS: int
    TR_VEL_THRES: float


ACTIVE: int = 1
INACTIVE: int = 0
STATIC: bool = True
DYNAMIC: bool = False


class BatchedData(RingBuffer):
    """
    A class to manage and combine frames into a batch.

    Attributes:
    ----------
    - effective_data (numpy.ndarray): An array to store effective data frames.

    Methods:
    -------
    - empty(): Reset the buffer and create an empty effective_data array.
    - add_frame(new_data: numpy.ndarray): Add a new frame of data to the buffer.
    - clear(): Clear the buffer and reset effective_data.
    - change_buffer_size(new_size): Change the size of the buffer.
    - pop_frame(): Remove the oldest frame from the buffer.
    """

    def __init__(self, batch_size, init_data=np.empty((0, 8))):
        super().__init__(batch_size, init_val=init_data) # size = const.FB_FRAMES_BATCH + 1
        self.weights = self._compute_weights()  # Compute weights for the frames
        self.effective_data = np.concatenate(self.buffer, axis=0)  # Combine all frames into a single array

    def _compute_weights(self):
        """
        Compute weights for the frames in the buffer.

        Returns:
        -------
        list: A list of weights for each frame in the buffer.
        """
        # Assign higher weights to more recent frames
        return [1.0 - (i * 0.2) for i in range(len(self.buffer))]  # Example: [1.0, 0.8, 0.6, ...]
    
    def _compute_effective_data(self):
        """
        Compute the effective data by combining frames with temporal weighting.

        Returns:
        -------
        numpy.ndarray: The combined data from all frames in the buffer, weighted by their age.
        """
        if not self.buffer:
            return (np.empty((0, 8)), np.empty(0))

        # return frames and weights
        return (self.buffer, self.weights)

    def add_frame(self, new_data: np.array):
        """
        Add a new frame of data to the buffer.

        Parameters:
        ----------
        - new_data (numpy.ndarray): New frame of data to add to the buffer.
        """
        if len(self.buffer) >= self.size:
            self.pop_frame()  # Remove the oldest frame if the buffer is full

        super().append(new_data)
        self.weights = self._compute_weights()
        self.effective_data = np.concatenate(self.buffer, axis=0)
    def clear(self):
        """
        Clear the buffer and reset effective_data.
        """
        self.buffer.clear()
        self.weights = self._compute_weights()
        self.effective_data = np.array([])

    def change_buffer_size(self, new_size):
        """
        Change the size of the buffer.
        """
        while len(self.buffer) > new_size:
            self.pop_frame()  # Remove excess frames if the new size is smaller
        self.size = new_size
        self.weights = self._compute_weights()  # Recompute weights
        self.effective_data = np.concatenate(self.buffer, axis=0)
    
    def pop_frame(self):
        """
        Remove the oldest frame from the buffer.
        """
        if len(self.buffer) > 0:
            self.buffer.popleft()
            self.weights = self._compute_weights()
            self.effective_data = np.concatenate(self.buffer, axis=0)


class KalmanState(KalmanFilter):
    """
    A class representing the state of a Kalman filter for motion tracking.

    Attributes:
    ----------
    - centroid: The centroid of the track used for initializing this Kalman filter instance.

    Methods:
    -------
    - __init__(centroid: np.ndarray): Initialize the Kalman filter with default parameters based on the centroid.
    """
    def __init__(self, centroid: np.ndarray, config: AsteriosConfig) -> None:
        self.config = config
        model = config.motion_model
        super().__init__(dim_x=model.KF_DIM[0], dim_z=model.KF_DIM[1])
        self.F = model.KF_F(1)
        self.H = model.KF_H
        self.Q = model.KF_Q_DISCR(1)
        self.R = np.eye(model.KF_DIM[1]) * config.KF_R_STD**2
        self.x = np.array([model.STATE_VEC(centroid)]).T
        self.P = np.eye(model.KF_DIM[0]) * config.KF_P_INIT

class PointCluster:
    """
    A class representing a cluster of 3D points and its attributes.

    Attributes:
    ----------
    - pointcloud (numpy.ndarray): An array of 3D points in the form (x, y, z, x', y', z', r', s).
    - point_num (int): The number of points in the cluster.
    - centroid (numpy.ndarray): The centroid of the cluster.
    - min_vals (numpy.ndarray): The minimum values in each dimension of the pointcloud.
    - max_vals (numpy.ndarray): The maximum values in each dimension of the pointcloud.
    - status (bool): The cluster movement status (STATIC: True, DYNAMIC: False)

    Methods:
    -------
    - __init__(pointcloud: numpy.ndarray):
        Initialize PointCluster with a given pointcloud.

    """

    def __init__(self, pointcloud: np.array, tr_vel_threshold, polar=False, weights=None, isFrame=False):
        """
        Initialize PointCluster with a given pointcloud.
        - pointcloud: Can be either:
            - A flat 2D numpy array of points (shape: [N, D]), or
            - A list/array of frames, where each frame is a 2D numpy array of points.
        - polar: A boolean flag to indicate if the pointcloud is in polar coordinates.
        - weights: An array of weights for each frame in the pointcloud.
        - isFrame: If True, treats `pointcloud` as an array of frames 
        """
        if isFrame:
            # Input is an array of frames and frame-level weights
            self.pointcloud = np.concatenate(pointcloud, axis=0)  # Combine frames into a single point cloud
            self.point_num = pointcloud[0].shape[0]  # Number of points in the first frame
            if weights is not None:
                # Expand frame-level weights into point-level weights
                self.weights = np.concatenate([
                    np.full(frame.shape[0], weight) for frame, weight in zip(pointcloud, weights)
                ])
            else:
                self.weights = np.ones(self.pointcloud.shape[0])  # Default weights (all 1.0)
        else:
            # Input is a flat array of points and point-level weights
            self.pointcloud = pointcloud
            self.weights = weights if weights is not None else np.ones(pointcloud.shape[0])  # Default weights (all 1.0)
            self.point_num = self.pointcloud.shape[0]


        # NOTE: the input is now a list of 8 entries
        self.centroid = np.average(self.pointcloud[:, :6], axis=0, weights=self.weights)
        self.min_vals = np.min(self.pointcloud[:, :6], axis=0)
        self.max_vals = np.max(self.pointcloud[:, :6], axis=0)

        velocity = math.sqrt(np.sum((self.centroid[3:6] ** 2)))

        # Last 3 values are radial measurements (r, θ, ṙ)
        if polar:
            # Compute centroid in polar coordinates (r, θ, ṙ)
            self.centroid = np.average(self.pointcloud[:, -3:], axis=0, weights=self.weights) #Last 3 values are radial measurements (r, θ, ṙ)
            self.min_vals = np.min(self.pointcloud[:, -3:], axis=0)
            self.max_vals = np.max(self.pointcloud[:, -3:], axis=0)


        if velocity < tr_vel_threshold:
            # if pointcloud[6] < const.TR_VEL_THRES:
            self.status = STATIC
        else:
            self.status = DYNAMIC


class ClusterTrack:
    """
    A class representing a tracked cluster with a Kalman filter for motion estimation.

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
    state : KalmanState
        KalmanState instance for motion estimation.
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

    def __init__(self, cluster: "PointCluster", config: AsteriosConfig) -> None:
        self.config = config
        model = config.motion_model
        self.N_est: int = 0
        self.spread_est: np.ndarray = np.zeros(model.KF_DIM[1])
        self.group_disp_est: np.ndarray = np.eye(model.KF_DIM[1]) * config.KF_GROUP_DISP_EST_INIT
        self.cluster = cluster

        # Pass the batch size from configuration
        self.batch = BatchedData(self.config.FB_FRAMES_BATCH+1, cluster.pointcloud)
        self.state = KalmanState(cluster.centroid, config)
        self.status: int = ACTIVE
        self.lifetime: int = 0
        # self.keypoints: np.ndarray = config.MODEL_DEFAULT_POSTURE
        self.predict_x: np.ndarray = self.state.x
        self.color: np.ndarray = np.random.rand(3)
        
    def _estimate_point_num(self) -> None:
        """
        Estimate the expected number of points in the cluster.
        """
        if self.config.KF_ENABLE_EST:
            if self.cluster.point_num > self.N_est:
                self.N_est = self.cluster.point_num
            else:
                self.N_est = ((1 - self.config.KF_A_N) * self.N_est +
                            self.config.KF_A_N * self.cluster.point_num)
        else:
            self.N_est = max(self.config.KF_EST_POINTNUM, self.cluster.point_num)

    def _estimate_measurement_spread(self) -> None:
        """
        Estimate the spread of measurements in each dimension.
        """
        for m in range(len(self.cluster.min_vals)):
            spread = self.cluster.max_vals[m] - self.cluster.min_vals[m]
            if self.cluster.point_num != 1:
                spread = spread * (self.cluster.point_num + 1) / (self.cluster.point_num - 1)
            spread = min(2 * self.config.KF_SPREAD_LIM[m], spread)
            spread = max(self.config.KF_SPREAD_LIM[m], spread)
            if spread > self.spread_est[m]:
                self.spread_est[m] = spread
            else:
                self.spread_est[m] = ((1.0 - self.config.KF_A_SPR) * self.spread_est[m] +
                                    self.config.KF_A_SPR * spread)

    def _get_D(self) -> np.ndarray:
        """
        Calculate and get the dispersion matrix for the track.

        Returns
        -------
        numpy.ndarray
            Dispersion matrix for the cluster.
        """
        dimension = self.config.motion_model.KF_DIM[1]
        pointcloud = self.cluster.pointcloud
        centroid = self.cluster.centroid
        disp = np.zeros((dimension, dimension), dtype=float)
        for i in range(dimension):
            for j in range(dimension):
                disp[i, j] = np.mean((pointcloud[:, i] - centroid[i]) *
                                    (pointcloud[:, j] - centroid[j]))
        return disp
    
    def _estimate_group_disp_matrix(self) -> None:
        """
        Estimate the group dispersion matrix.
        """
        a = self.cluster.point_num / self.N_est
        self.group_disp_est = (1 - a) * self.group_disp_est + a * self._get_D()

    def _get_Rc(self):
        """
        Get the combined covariance matrix.

        Returns
        -------
        numpy.ndarray
            Combined covariance matrix for the cluster.
        """
        N = self.cluster.point_num
        N_est = self.N_est
        return (self.get_Rm() / N) + (((N_est - N) / ((N_est - 1) * N)) * self.group_disp_est)

    def associate_pointcloud(self, pointcloud: np.array) -> None:
        """
        Associate a point cloud with the track.

        Parameters
        ----------
        pointcloud : np.array
            2D NumPy array representing the point cloud.

        Notes
        -----
        This method performs the following steps:
        1. Initializes a PointCluster with the given point cloud.
        2. Adds the point-cluster to the track's frames batch.
        3. Estimates the number of points in the cluster.
        4. Estimates the spread of measurements in the cluster.
        5. Estimates the dispersion matrix of the point groups in the cluster.

        """
        # This is the original code. No temporal fusion is done here.
        self.cluster = PointCluster(pointcloud, tr_vel_threshold=self.config.TR_VEL_THRES)
        # Add the new frame to the batch
        self.batch.add_frame(self.cluster.pointcloud)
        self._estimate_point_num()
        self._estimate_measurement_spread()
        self._estimate_group_disp_matrix()

        # Save the current height and width of the pointcloud projection to the screen in the ringbuffers.
        # TODO: This approach needs to change
        # self.height_buffer.append(
        #     calc_projection_points(
        #         value=self.cluster.max_vals[2] - 0.01,
        #         y=self.cluster.min_vals[1],
        #         vertical_axis=True,
        #     )
        # )
        # self.width_buffer.append(
        #     calc_projection_points(
        #         value=self.cluster.max_vals[0], y=self.cluster.min_vals[1]
        #     )
        #     - calc_projection_points(
        #         value=self.cluster.min_vals[0], y=self.cluster.min_vals[1]
        #     )
        # )

    def get_Rm(self) -> np.ndarray:
        """
        Get the measurement covariance matrix

        Returns
        -------
        numpy.ndarray
            Measurement covariance matrix for the cluster.
        """
        return np.diag(((self.spread_est / 2) ** 2))

    def predict_state(self, dt: float) -> None:
        """
        Predict the state of the Kalman filter based on the time multiplier.

        Parameters
        ----------
        dt : float
            Time multiplier for the prediction.
        """
        self.state.predict(
            F=self.config.motion_model.KF_F(dt),
            Q=self.config.motion_model.KF_Q_DISCR(dt),
        )
        self.predict_x = self.state.x

    def update_state(self) -> None:
        """
        Update the state of the Kalman filter based on the associated measurement (pointcloud centroid).
        """
        z = np.array(self.cluster.centroid)
        self.state.update(z, R=self._get_Rc())
        variance = z[:1] - self.state.x[:1, 0]
        if abs(variance.any()) > 0.6 and self.lifetime == 0:
            self.state.x[:1, 0] += variance * 0.4

    def update_lifetime(self, dt, reset=False) -> None:
        """
        Update the track lifetime.
        """
        if reset:
            self.lifetime = 0
        else:
            self.lifetime += dt

    def seek_inner_clusters(self) -> List:
        """
        Seek inner clusters within the current cluster.

        This method uses DBSCAN to identify inner clusters within the current cluster's pointcloud.
        It helps in separating inner clusters and filtering noise when a single cluster is detected.

        Returns
        -------
        list
            List of new inner track clusters (PointCluster instances).

        """

        new_track_clusters = []
        spread = self.cluster.max_vals[:1] - self.cluster.min_vals[:1]
        if (self.cluster.point_num > self.config.DB_POINTS_THRES and
            spread.any() > self.config.DB_SPREAD_THRES):
            if self.cluster.status == STATIC:
                # Change batch size for static clusters
                while len(self.batch.buffer) > self.config.FB_FRAMES_BATCH_STATIC:
                    self.batch.buffer.popleft()
            else:
                while len(self.batch.buffer) > self.config.FB_FRAMES_BATCH:
                    self.batch.buffer.popleft()
            self.batch.add_frame(self.cluster.pointcloud)
            pointcloud = np.concatenate(list(self.batch.buffer), axis=0)
            # Use DBSCAN with the configured inner eps
            track_clusters = apply_clustering(pointcloud, method="DBSCAN")
            if len(track_clusters) > 1:
                new_track_clusters = [track_clusters[1]]
        return new_track_clusters


class Tracker(ABC):
    """
    An abstract class representing a tracker.
    """

    @abstractmethod
    def track(self, pointcloud, batch, clusteringAlgorithm="DBSCAN"):
        pass

    # @abstractmethod
    # def estimate_posture(self, model):
    #     pass

    # @abstractmethod
    # def update_real_posture(self, real_data):
    #     pass

class TrackBuffer(Tracker):
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
       
    def __init__(self, config: AsteriosConfig, usePalmar: bool = False) -> None:
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

    def _calc_dist_fun(self, full_set: np.array) -> np.ndarray:
        """
        Calculate the Mahalanobis distance matrix for gating.

        Parameters
        ----------
        full_set : np.ndarray
            Full set of points.

        Returns
        -------
        np.ndarray
            An array representing the associated track (entry) for each point (index).
            If no track is associated with a point, the entry is set to None.
        """
        dist_matrix = np.empty((full_set.shape[0], len(self.effective_tracks)))
        associated_track_for = np.full(full_set.shape[0], None, dtype=object)
        for j, track in enumerate(self.effective_tracks):
            H_i = np.dot(self.config.motion_model.KF_H, track.state.x).flatten()
            C_g_j = track.state.P[:6, :6] + track.get_Rm() + track.group_disp_est
            for i, point in enumerate(full_set):
                y_ij = np.array(point[:6]) - H_i
                dist_matrix[i][j] = (np.log(np.abs(np.linalg.det(C_g_j))) +
                                    np.dot(np.dot(y_ij.T, np.linalg.inv(C_g_j)), y_ij))
                if dist_matrix[i][j] < self.config.TR_GATE:
                    if associated_track_for[i] is None:
                        associated_track_for[i] = j
                    else:
                        if dist_matrix[i][j] < dist_matrix[i][int(associated_track_for[i])]:
                            associated_track_for[i] = j
        return associated_track_for

    def _add_tracks(self, new_clusters: List[np.array]) -> None:
        """
        Add new tracks to the buffer.

        Parameters
        ----------
        new_clusters : list
            List of new clusters to be added as tracks.
        """
        for new_cluster in new_clusters:
            new_track = ClusterTrack(PointCluster(np.array(new_cluster), tr_vel_threshold=self.config.TR_VEL_THRES), self.config)
            self.next_track_id += 1
            self.effective_tracks.append(new_track)

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

    def _get_gated_clouds(self, full_set: np.array) -> tuple[np.ndarray, List[List[np.array]]]:
        """
        Split the pointcloud according to the formed gates and return gated and unassigned clouds.

        Parameters
        ----------
        full_set : np.array
            Full set of points.

        Returns
        -------
        tuple
            Tuple containing unassigned points and clustered clouds.
        """
        unassigned = np.empty((0, 8), dtype=float)
        clusters = [[] for _ in range(len(self.effective_tracks))]
        associated_track_for = self._calc_dist_fun(full_set)
        for i, point in enumerate(full_set):
            if associated_track_for[i] is None:
                unassigned = np.append(unassigned, [point], axis=0)
            else:
                clusters[associated_track_for[i]].append(point)
        return unassigned, clusters

    def _associate_points_to_tracks(self, full_set: np.array) -> np.ndarray:
        """
        Associate points to existing tracks and handle inner cluster separation.

        Parameters
        ----------
        full_set : np.array
            Full set of sensed points.

        Returns
        -------
        np.ndarray
            Unassigned points.
        """
        unassigned, clouds = self._get_gated_clouds(full_set)
        new_inner_clusters = []
        for j, track in enumerate(self.effective_tracks):
            if len(clouds[j]) == 0:
                track.update_lifetime(dt=self.dt)
            else:
                track.update_lifetime(dt=self.dt, reset=True)
                track.associate_pointcloud(np.array(clouds[j]))
        for inner_cluster in new_inner_clusters:
            self._add_tracks(inner_cluster)
        return unassigned

    def track(self, pointcloud: np.array, batch: RingBuffer, clusteringAlgorithm: str = "DBSCAN") -> None:
        """
        Perform the tracking process including prediction, association, maintenance, update, and clustering.

        Parameters
        ----------
        pointcloud : np.array
            Pointcloud data.
        batch : BatchedData
            BatchedData instance for managing frames.
        clusteringAlgorithm : str
            Clustering algorithm to be used. Default is DBSCAN. Accepted values are "DBSCAN", "BIRCH", or "both".
        Returns
        -------
        None
        """
        if clusteringAlgorithm not in ["DBSCAN", "BIRCH", "both"]:
            raise ValueError("Invalid clustering algorithm. Please use 'DBSCAN', 'BIRCH', or 'both'.")
        self._predict_all()
        unassigned = self._associate_points_to_tracks(pointcloud)
        self._maintain_tracks()
        self._update_all()
        new_clusters = []
        batch.add_frame(unassigned)
        effective_data = np.concatenate(list(batch.buffer), axis=0)
        if (effective_data.size > 0 and len(self.effective_tracks) < self.config.TR_MAX_TRACKS):
            new_clusters = apply_clustering(batch.effective_data, 
                                            clusteringAlgorithm, 
                                            metric=partial(altered_EuclideanDist,
                                                db_range_weight=self.config.DB_RANGE_WEIGHT,
                                                db_z_weight=self.config.DB_Z_WEIGHT),
                                            eps=self.config.DB_EPS,
                                            min_samples=self.config.DB_MIN_SAMPLES_MIN,
                                            )
            if new_clusters:
                batch.buffer.clear()
            self._add_tracks(new_clusters)

    # def estimate_posture(self, model: Any) -> None:
    #     """
    #     Format the pointcloud, estimate and save the posture of the target of each track using a CNN model.

    #     Parameters
    #     ----------
    #     model : Model
    #         The CNN model used for posture estimation.

    #     Returns
    #     -------
    #     None
    #     """
    #     frame_matrices = []
    #     indexes = []
    #     for index, track in enumerate(self.effective_tracks):
    #         if len(np.concatenate(list(track.batch.buffer), axis=0)) > self.config.MODEL_MIN_INPUT:
    #             rel_track_points = relative_coordinates(list(track.batch.buffer), track.cluster.centroid[:2])
    #             frame_matrices.append(format_single_frame(rel_track_points))
    #             indexes.append(index)
    #     frame_matrices_array = np.array(frame_matrices)
    #     if frame_matrices_array.size > 0:
    #         frame_keypoints = model.predict(frame_matrices_array)
    #         for i, idx in enumerate(indexes):
    #             self.effective_tracks[idx].keypoints = frame_keypoints[i]

    # def update_real_posture(self, real_data: np.array) -> List[tuple]:
    #     """
    #     Update the real posture of the target of each track using the real data.

    #     Parameters
    #     ----------
    #     real_data : np.array
    #         Real data for posture estimation.

    #     Returns
    #     -------
    #     An array of tuples containing the centroid and joint 0 of each track. They will be reformatted to (x, y, z) as the coordinate system is like that.
    #     """
    #     centralValues = []
    #     for index, track in enumerate(self.effective_tracks):
    #         try:
    #             kinect_coords = real_data[index]
    #         except Exception:
    #             print("Warning. No more than one skeleton detected; using the same skeleton for all tracks.", time.time())
    #             kinect_coords = real_data[0]
    #         track.ground_truth = np.array(kinect_coords)
    #         reshaped_keypoints = track.ground_truth.copy().reshape(3, -1)
    #         reshaped_keypoints[0] *= -1
    #         reshaped_keypoints = reshaped_keypoints[[0, 2, 1]]
    #         centroid = np.mean(reshaped_keypoints, axis=1)
    #         centralValues.append((centroid, reshaped_keypoints[:, 0]))
    #     return centralValues
