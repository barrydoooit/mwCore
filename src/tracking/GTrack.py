from dataclasses import dataclass
from functools import partial
from .AsteriosTracking import Tracker,  BatchedData, PointCluster
import numpy as np
import time
from filterpy.kalman import KalmanFilter
from .utils import (
    altered_EuclideanDist,
    apply_clustering,
    apply_DBscan,
)
from typing import Callable, List

ACTIVE = 1
INACTIVE = 0

STATIC = True
DYNAMIC = False


@dataclass 
class ConstVelModel: 
    KF_DIM: List[int] # e.g. [6, 6] 
    KF_H: np.ndarray # Measurement matrix (6x6 identity) 
    KF_F: Callable[[float], np.ndarray] # Function: dt -> state transition matrix 
    KF_Q_DISCR: Callable[[float], np.ndarray] # Function: dt -> process noise covariance 

@dataclass
class GTrackConfig:
    const_vel_model: ConstVelModel
    KF_GROUP_DISP_EST_INIT: float  # e.g. 0.1
    KF_ENABLE_EST: bool  # e.g. False
    KF_A_N: float  # e.g. 0.9
    KF_EST_POINTNUM: int  # e.g. 10
    KF_SPREAD_LIM: List[float]  # e.g. [0.2, 0.2, 2, 1.2, 1.2, 0.2]
    KF_A_SPR: float  # e.g. 0.9
    DB_POINTS_THRES: int  # e.g. 40
    DB_SPREAD_THRES: float  # e.g. 0.7
    FB_FRAMES_BATCH_STATIC: int  # e.g. 2
    FB_FRAMES_BATCH: int  # e.g. 2
    DB_INNER_EPS: float  # e.g. 0.1
    DB_EPS: float
    DB_RANGE_WEIGHT: float
    DB_Z_WEIGHT: float
    DB_MIN_SAMPLES_MIN: int
    KF_R_STD: float  # e.g. 0.1
    KF_P_INIT: float  # e.g. 0.1
    # MODEL_DEFAULT_POSTURE: np.ndarray  # e.g. a NumPy array of shape (64,) or the defined shape
    TR_LIFETIME_DYNAMIC: float  # e.g. 3 (seconds)
    TR_LIFETIME_STATIC: float  # e.g. 7 (seconds)
    TR_GATE: float  # e.g. 4.5
    TR_MAX_TRACKS: int  # e.g. 4
    TR_VEL_THRES: float
    
class ClusterGTrack:
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

    def __init__(self, cluster: 'PointCluster', config: GTrackConfig) -> None:
        self.config = config
        model = config.const_vel_model
        self.N_est: int = 0
        self.spread_est: np.ndarray = np.zeros(model.KF_DIM[1])
        self.group_disp_est: np.ndarray = np.eye(model.KF_DIM[1]) * config.KF_GROUP_DISP_EST_INIT
        self.cluster = cluster
        self.batch = BatchedData(cluster.pointcloud)
        self.state = ExtendedKalmanState(cluster.centroid, config)
        self.status: int = ACTIVE
        self.lifetime: int = 0
        # self.keypoints: np.ndarray = config.MODEL_DEFAULT_POSTURE
        self.predict_x: np.ndarray = self.state.x
        self.color: np.ndarray = np.random.rand(3)

    def _estimate_point_num(self):
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

    def _estimate_measurement_spread(self):
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
        dimension = self.config.const_vel_model.KF_DIM[1]
        pointcloud = self.cluster.pointcloud
        centroid = self.cluster.centroid
        disp = np.zeros((dimension, dimension), dtype=float)
        for i in range(dimension):
            for j in range(dimension):
                disp[i, j] = np.mean((pointcloud[:, i] - centroid[i]) *
                                    (pointcloud[:, j] - centroid[j]))
        return disp

    def _estimate_group_disp_matrix(self):
        """
        Estimate the group dispersion matrix.
        """
        a = self.cluster.point_num / self.N_est
        self.group_disp_est = (1 - a) * self.group_disp_est + a * self._get_D()

    def _get_Rc(self) -> np.ndarray:
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

    def associate_pointcloud(self, pointcloud: np.array):
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

        Parameters
        ----------
        pointcloud : np.array
            2D NumPy array representing the point cloud.
        """
        self.cluster = PointCluster(pointcloud, tr_vel_threshold=self.config.TR_VEL_THRES)
        self.batch.add_frame(self.cluster.pointcloud)
        self._estimate_point_num()
        self._estimate_measurement_spread()
        self._estimate_group_disp_matrix()


    def get_Rm(self) -> np.ndarray:
        """
        Get the measurement covariance matrix

        Returns
        -------
        numpy.ndarray
            Measurement covariance matrix for the cluster.
        """
        return np.diag(((self.spread_est / 2) ** 2))

    def predict_state(self, dt: float):
        """
        Predict the state of the Kalman filter based on the time multiplier.

        Parameters
        ----------
        dt : float
            Time multiplier for the prediction.
        """
        self.state.predict(
            F=self.config.const_vel_model.KF_F(dt),
            Q=self.config.const_vel_model.KF_Q_DISCR(dt),
        )
        self.predict_x = self.state.x

    def update_state(self):
        """
        Update the state of the Kalman filter based on the associated measurement (pointcloud centroid).
        """
        z = np.array(self.cluster.centroid)
        self.state.update(z, R=self._get_Rc())
        # If the variance between the predicted and measured position
        variance = z[:1] - self.state.x[:1, 0]
        if abs(variance.any()) > 0.6 and self.lifetime == 0:
            self.state.x[:1, 0] += variance * 0.4

    def update_lifetime(self, dt, reset=False):
        """
        Update the track lifetime.
        """
        if reset:
            self.lifetime = 0
        else:
            self.lifetime += dt

    def seek_inner_clusters(self) -> List[np.array]:
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
        if (
            self.cluster.point_num > self.config.DB_POINTS_THRES and
            spread.any() > self.config.DB_SPREAD_THRES
        ):
            # Allow frame fusion for better resolution
            # NOTE: State machine is better here
            if self.cluster.status == STATIC:
                self.batch.change_buffer_size(self.config.FB_FRAMES_BATCH_STATIC)
            else:
                self.batch.change_buffer_size(self.config.FB_FRAMES_BATCH)

            self.batch.add_frame(self.cluster.pointcloud)
            pointcloud = self.batch.effective_data

            # Apply clustering to identify inner clusters
            track_clusters = apply_DBscan(pointcloud=pointcloud, 
                                          eps=self.config.DB_INNER_EPS,
                                          min_samples=self.config.DB_MIN_SAMPLES_MIN,
                                          metric=partial(altered_EuclideanDist, 
                                                         db_range_weight=self.config.DB_RANGE_WEIGHT, 
                                                         db_z_weight=self.config.DB_Z_WEIGHT),
                                          )
            if len(track_clusters) > 1:
                new_track_clusters = [track_clusters[1]]
        return new_track_clusters
    
class ExtendedKalmanState(KalmanFilter):
    """
    A class representing the state of a Kalman filter for motion tracking.

    Attributes:
    ----------
    - centroid: The centroid of the track used for initializing this Kalman filter instance.

    Methods:
    -------
    - __init__(centroid: np.ndarray): Initialize the Kalman filter with default parameters based on the centroid.
    """

    def __init__(self, centroid: np.ndarray, config: GTrackConfig) -> None:
        self.config = config
        model = config.const_vel_model
        super().__init__(dim_x=model.KF_DIM[0], dim_z=model.KF_DIM[1])
        self.F = model.KF_F(1)
        self.H = model.KF_H
        self.Q = model.KF_Q_DISCR(1)
        self.R = np.eye(model.KF_DIM[1]) * config.KF_R_STD**2
        self.x = np.array([model.STATE_VEC(centroid)]).T
        self.P = np.eye(model.KF_DIM[0]) * config.KF_P_INIT


class GTrackBuffer(Tracker):
    def __init__(self, config: GTrackConfig) -> None:
        """
        Initialize TrackBuffer with empty lists for tracks and effective tracks.
        """
        self.config = config
        self.effective_tracks: List[ClusterGTrack] = []
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
            An array mapping each point to an associated track index, or None if no track was associated.
        """
        dist_matrix = np.empty((full_set.shape[0], len(self.effective_tracks)))
        associated_track_for = np.full(full_set.shape[0], None, dtype=object)
        for j, track in enumerate(self.effective_tracks):
            H_i = np.dot(self.config.const_vel_model.KF_H, track.state.x).flatten()
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
        new_clusters : List[np.array]
            List of new clusters to be added as tracks.
        """
        for new_cluster in new_clusters:
            new_track = ClusterGTrack(PointCluster(np.array(new_cluster), tr_vel_threshold=self.config.TR_VEL_THRES), self.config)
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
        full_set : np.ndarray
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

    def track(self, pointcloud: np.array, batch: BatchedData, clusteringAlgorithm: str = "DBSCAN") -> None:
        """
        Perform the tracking process including prediction, association, maintenance, update, and clustering.

        Parameters
        ----------
        pointcloud : np.array
            Pointcloud data.
        batch : BatchedData
            BatchedData instance for managing frames.
        clusteringAlgorithm : str, optional
            Clustering algorithm to be used, by default "DBSCAN".

        Raises
        ------
        ValueError
            If an invalid clustering algorithm is provided.
        """
        if clusteringAlgorithm not in ["DBSCAN", "BIRCH", "both"]:
            raise ValueError("Invalid clustering algorithm. Please use 'DBSCAN', 'BIRCH', or 'both'.")
        self._predict_all()
        unassigned = self._associate_points_to_tracks(pointcloud)
        self._maintain_tracks()
        self._update_all()
        new_clusters = []
        batch.add_frame(unassigned)
        if len(batch.effective_data) > 0 and len(self.effective_tracks) < self.config.TR_MAX_TRACKS:
            new_clusters = apply_clustering(batch.effective_data, 
                                            clusteringAlgorithm, 
                                            metric=partial(altered_EuclideanDist,
                                                db_range_weight=self.config.DB_RANGE_WEIGHT,
                                                db_z_weight=self.config.DB_Z_WEIGHT),
                                            eps=self.config.DB_EPS,
                                            min_samples=self.config.DB_MIN_SAMPLES_MIN,
                                            )
            if len(new_clusters) > 0:
                batch.clear()
            self._add_tracks(new_clusters)

    # def estimate_posture(self, model: any) -> None:
    #     """
    #     Format the pointcloud, estimate and save the posture of the target of each track using a CNN model.

    #     Parameters
    #     ----------
    #     model : any
    #         The CNN model used for posture estimation.
    #     """
    #     frame_matrices = []
    #     indexes = []
    #     for index, track in enumerate(self.effective_tracks):
    #         if len(track.batch.effective_data) > self.config.MODEL_MIN_INPUT:
    #             rel_track_points = relative_coordinates(list(track.batch.buffer), track.cluster.centroid[:2])
    #             frame_matrices.append(format_single_frame(rel_track_points))
    #             indexes.append(index)
    #     frame_matrices_array = np.array(frame_matrices)
    #     if len(frame_matrices_array) > 0:
    #         frame_keypoints = model.predict(frame_matrices_array)
    #         for i, index in enumerate(indexes):
    #             self.effective_tracks[index].keypoints = frame_keypoints[i]

    # def update_real_posture(self, real_data: np.array) -> List[tuple]:
    #     """
    #     Update the real posture of the target of each track using the real data.

    #     Parameters
    #     ----------
    #     real_data : np.array
    #         Real data for posture estimation.

    #     Returns
    #     -------
    #     List[tuple]
    #         An array of tuples containing the centroid and joint 0 of each track.
    #     """
    #     centralValues = []
    #     for index, track in enumerate(self.effective_tracks):
    #         try:
    #             kinect_coords = real_data[index]
    #         except Exception:
    #             print("Warning. No more than one skeleton detected, showing the same skeleton for all tracks. ", time.time())
    #             kinect_coords = real_data[0]
    #         track.ground_truth = np.array(kinect_coords)
    #         reshaped_keypoints = track.ground_truth.copy().reshape(3, -1)
    #         reshaped_keypoints[0] *= -1
    #         reshaped_keypoints = reshaped_keypoints[[0, 2, 1]]
    #         centroid = np.mean(reshaped_keypoints, axis=1)
    #         centralValues.append((centroid, reshaped_keypoints[:, 0]))
    #     return centralValues

