from functools import partial
import numpy as np
from dataclasses import dataclass
import math
import time
from filterpy.kalman import KalmanFilter

from .gtrack import BatchedData, PointCluster, Tracker
from ..utils import (
    altered_EuclideanDist,
    apply_clustering,
    RingBuffer,
)
from typing import List, Callable, Any
from abc import ABC, abstractmethod

ACTIVE = 1
INACTIVE = 0

STATIC = True
DYNAMIC = False

@dataclass
class ConstAccModel:
    KF_DIM: List[int]
    KF_H: np.ndarray
    KF_F: Callable[[float], np.ndarray]
    KF_Q_DISCR: Callable[[float], np.ndarray]
    STATE_VEC: Callable[[np.ndarray], List[float]]


@dataclass
class KaloyanConfig:
    FB_FRAMES_BATCH: int
    FB_FRAMES_BATCH_STATIC: int
    DB_POINTS_THRES: int
    DB_SPREAD_THRES: float
    DB_EPS: float
    DB_RANGE_WEIGHT: float
    DB_Z_WEIGHT: float
    DB_MIN_SAMPLES_MIN: int
    KF_R_STD: float
    KF_Q_STD: float
    KF_P_INIT: float
    KF_GROUP_DISP_EST_INIT: float
    KF_ENABLE_EST: bool
    KF_A_N: float
    KF_EST_POINTNUM: int
    KF_SPREAD_LIM: List[float]
    KF_A_SPR: float
    TR_LIFETIME_DYNAMIC: float
    TR_LIFETIME_STATIC: float
    TR_GATE: float
    TR_MAX_TRACKS: int
    TR_VEL_THRES: float
    motion_model: ConstAccModel
    NUM_DYNAMIC_POINTS_THRESHOLD: int
    DOPPLER_THRESHOLD: float
    MIN_VELOCITY_STOP_NO_POINTS: float
    MIN_VELOCITY_STOP_NO_DYNAMIC_POINTS: float
    MIN_VELOCITY_SLOW_DOWN: float



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
    num_points_associated_last : int
        Number of points associated with the track in the last frame.
    num_dynamic_points_associated_last : int
        Number of dynamic points associated with the track in the last frame.
    track_status : Status
        The status of the track (STATIC or DYNAMIC).
    color : numpy.ndarray
        Random color assigned to the track for visualization (for visualization purposes).

    Methods
    -------
    compute_cartesian_velocity()
        Compute the cartesian velocity of the track using the a priori state, with respect to the x and y dimensions.
    
    __get_num_dynamic_points_associated(pointcloud)
        Get the number of dynamic points associated with the track.

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

    seek_inner_clusters()
        Seek inner clusters within the current track.

    """

    def __init__(self, cluster: PointCluster, config: KaloyanConfig):
        self.config = config
        self.N_est = 0
        self.spread_est = np.zeros(self.config.motion_model.KF_DIM[1])
        self.group_disp_est = (
            np.eye(self.config.motion_model.KF_DIM[1]) * self.config.KF_GROUP_DISP_EST_INIT
        )
        self.cluster = cluster
        self.batch = BatchedData(self.config.FB_FRAMES_BATCH + 1)
        self.state = KalmanState(cluster.centroid, self.config)
        self.lifetime = 0
        self.status = ACTIVE
        self.num_points_associated_last = cluster.point_num
        self.num_dynamic_points_associated_last = self._get_num_dynamic_points_associated(cluster.pointcloud)
        self.track_status = DYNAMIC if self.num_dynamic_points_associated_last > self.config.NUM_DYNAMIC_POINTS_THRESHOLD else STATIC
        self.color = np.random.rand(3)
        
    def compute_cartesian_velocity(self):
        """
        Compute the cartesian velocity of the track using the a priory state, with respect to the x and y dimensions.
        """
        return math.sqrt(np.sum((self.state.x_prior[3:5] ** 2)))
    
    def _get_num_dynamic_points_associated(self, pointcloud: np.array):
        """
        Get the number of dynamic points associated with the track.

        A point is considered dynamic if its Doppler value is greater than the DOPPLER_THRESHOLD.
        """
        return 0 if not len(pointcloud) else np.sum(pointcloud[:, 6] > self.config.DOPPLER_THRESHOLD)

    def _estimate_point_num(self):
        """
        Estimate the expected number of points in the cluster.
        """
        if self.config.KF_ENABLE_EST:
            # TODO: Instead of self.cluster.point_num, use my_good_points
            if self.cluster.point_num > self.N_est:
                self.N_est = self.cluster.point_num
            else:
                # Weighted average between the current number of points and the estimated number of points
                self.N_est = (
                    1 - self.config.KF_A_N
                ) * self.N_est + self.config.KF_A_N * self.cluster.point_num
        else:
            self.N_est = max(self.config.KF_EST_POINTNUM, self.cluster.point_num)

    def _estimate_measurement_spread(self):
        """
        Estimate the spread of measurements in each dimension.
        """
        if self.cluster.point_num > 1:
            for m in range(len(self.cluster.min_vals)):
                # Difference between max and min values in the cluster (in one dimension)
                spread = self.cluster.max_vals[m] - self.cluster.min_vals[m]

                # Unbiased spread estimation - the more points we have, the tighter the spread we create is
                spread = (
                    # TODO: Use my_good_points instead of self.cluster.point_num
                    spread * (self.cluster.point_num + 1) / (self.cluster.point_num - 1)
                )

                # Map the spread to a range between 1 and 2 times between the configured spread limits
                spread = min(2 * self.config.KF_SPREAD_LIM[m], spread)
                spread = max(self.config.KF_SPREAD_LIM[m], spread)

                if spread > self.spread_est[m]:
                    # This would most likely be the case when we have few samples
                    self.spread_est[m] = spread
                else:
                    # Weighed average between calculated spread and the previous spread estimation
                    self.spread_est[m] = (1.0 - self.config.KF_A_SPR) * self.spread_est[
                        m
                    ] + self.config.KF_A_SPR * spread

    def _get_D(self):
        """
        Calculate and get the dispersion matrix for the current cluster.

        Returns
        -------
        numpy.ndarray
            Dispersion matrix for the cluster.
        """
        dimension = self.config.motion_model.KF_DIM[1]
        pointcloud = self.cluster.pointcloud
        centroid = self.cluster.centroid
        disp = np.zeros((dimension, dimension), dtype="float")

        for i in range(dimension):
            for j in range(dimension):
                disp[i, j] = np.mean(
                    (pointcloud[:, i] - centroid[i]) * (pointcloud[:, j] - centroid[j])
                )

        return disp

    def _estimate_group_disp_matrix(self):
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
        return (self.get_Rm() / N) + (
            (N_est - N) / ((N_est - 1) * N)
        ) * self.group_disp_est

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
        1. Updates the number of points associated with the track.
        2. Updates the number of dynamic points associated with the track.
        3. In case there are point associated with this track, perform the other steps.
        4. Initializes a PointCluster with the given point cloud.
        5. Adds the point-cluster to the track's frames batch.
        """



        # Update the number of points and dynamic associated with the track.
        self.num_points_associated_last = len(pointcloud)
        self.num_dynamic_points_associated_last = self._get_num_dynamic_points_associated(pointcloud)

        # print('points associated with the track -- ', len(pointcloud))
        # print('dynamic points associated with the track -- ', self.num_dynamic_points_associated_last)
        # print('pointcloud -- ', pointcloud)
        # If there are points associated with this track, update the track.
        if len(pointcloud):
            self.cluster = PointCluster(pointcloud, tr_vel_threshold=self.config.TR_VEL_THRES)
            self.batch.add_frame(self.cluster.pointcloud)

    def get_Rm(self):
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
        if self.track_status is DYNAMIC:
            self.state.predict(
                F=self.config.motion_model.KF_F(dt),
                Q=self.config.motion_model.KF_Q_DISCR(dt),
            )
        
    def update_state(self):
        """
        Update the track.
        """
        # TODO: Calculate my_good_points - dynamic (Doppler more than 0) and unique (association with only one track)
        vel = self.compute_cartesian_velocity()
        if not self.num_points_associated_last:
            if self.track_status is DYNAMIC:
                if vel < self.config.MIN_VELOCITY_STOP_NO_POINTS:
                    # If the track is dynamic and no points are associated, force zero velocity.
                    self.state.x[3:6] = 0
                    # If the track is dynamic and no points are associated, transition to STATIC.
                    self.track_status = STATIC
                else: 
                    self._move_target()
            else:
                # If the track is static and no points are associated, do not update the state.
                return
        elif self.num_dynamic_points_associated_last < self.config.NUM_DYNAMIC_POINTS_THRESHOLD + 1:
            if self.track_status is STATIC:
                # TODO: Update confidence.
                return
            else:
                if vel < self.config.MIN_VELOCITY_STOP_NO_DYNAMIC_POINTS:
                    # If the track is dynamic and no dynamic points are associated, force zero velocity.
                    self.state.x[3:6] = 0
                    # If the track is dynamic and no dynamic points are associated, transition to STATIC.
                    self.track_status = STATIC 
                    # TODO: If there are many STATIC points, increase confidence.
                elif vel < self.config.MIN_VELOCITY_SLOW_DOWN:
                    # If the track is dynamic and no dynamic points are associated, decrease the velocity.
                    self.state.x[3:6] *= 0.5
                    self._move_target()
                else:
                    # TODO: Increase confidence.
                    self._move_target() 
        elif self.num_dynamic_points_associated_last > self.config.NUM_DYNAMIC_POINTS_THRESHOLD:
            self._estimate_point_num()
            self._estimate_measurement_spread()
            self._estimate_group_disp_matrix()

            self._move_target()

    def _move_target(self):
        """
        Move the target.
        """
        self.track_status = DYNAMIC
        z = np.array(self.cluster.centroid)
        self.state.update(z, R=self._get_Rc())

        variance = z[:1] - self.state.x[:1, 0]
        self.state.x[:1, 0] += variance * 0.4

class TrackBuffer:
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

    def __init__(self, config: KaloyanConfig):
        self.config = config
        self.effective_tracks: List[ClusterTrack] = []
        self.next_track_id = 0
        self.dt = 0
        self.t = time.time()

    def _find_closest_track(self, full_set: np.array):
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
        bidding_score = np.empty((full_set.shape[0], len(self.effective_tracks)))
        associated_track_for = np.full(full_set.shape[0], None, dtype=object)
        for j, track in enumerate(self.effective_tracks):
            H_i = np.dot(self.config.motion_model.KF_H, track.state.x_prior).flatten()
            # Group residual covariance matrix
            C_g_j = track.state.P_prior[:6, :6] + track.get_Rm() + track.group_disp_est

            for i, point in enumerate(full_set):
                # Innovation for each measurement
                y_ij = np.array(point[:6]) - H_i

                # Mahalanobis Distance (squared)
                d_squared = np.dot(np.dot(y_ij.T, np.linalg.inv(C_g_j)), y_ij)

                # bidding score (squared)
                bidding_score[i][j] = np.log(np.abs(np.linalg.det(C_g_j))) + d_squared
                # Perform Gate threshold check
                if bidding_score[i][j] < self.config.TR_GATE:
                    # Just choose the closest mahalanobis distance
                    if associated_track_for[i] is None:
                        associated_track_for[i] = j
                    else:
                        if (
                            bidding_score[i][j]
                            < bidding_score[i][int(associated_track_for[i])]
                        ):
                            associated_track_for[i] = j
                
        return associated_track_for

    def _add_tracks(self, new_clusters):
        """
        Add new tracks to the buffer.

        Parameters
        ----------
        new_clusters : list
            List of new clusters to be added as tracks.
        """
        for new_cluster in new_clusters:
            new_track = ClusterTrack(PointCluster(np.array(new_cluster), tr_vel_threshold=self.config.TR_VEL_THRES), self.config)
            # new_track.id = self.next_track_id
            self.next_track_id += 1
            self.effective_tracks.append(new_track)

    def _predict_all(self):
        """
        Predict the state of all effective tracks.
        """
        for track in self.effective_tracks:
            # TODO: Maybe, accumulate dt for this track in case it is not updated.
            track.predict_state(self.dt)

    def _update_all(self):
        """
        Update the state of all effective tracks.
        """
        for track in self.effective_tracks:
            # TODO: Update only when the track is active (static or dynamic). Delete FREE tracks.
            track.update_state()

    def _get_gated_clouds(self, full_set: np.array):
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
        unassigned = np.empty((0, 8), dtype="float")
        clusters = [[] for _ in range(len(self.effective_tracks))]
        # Simple matrix has len = len(full_set) and has the index of the chosen track.
        point_to_closest_track_assignment = self._find_closest_track(full_set)

        for i, point in enumerate(full_set):
            # print(f"Point {i} is assigned to track {point_to_closest_track_assignment[i]}")
            if point_to_closest_track_assignment[i] is None:
                unassigned = np.append(unassigned, [point], axis=0)
            else:
                clusters[point_to_closest_track_assignment[i]].append(point)
        return unassigned, clusters

    def _assign_points_to_tracks_and_get_unassigned(self, full_set: np.array):
        """
        Associate points to existing tracks.

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

        for j, track in enumerate(self.effective_tracks):
            track.associate_pointcloud(np.array(clouds[j]))

        return unassigned

    def _maintain_tracks(self):
        """
        Update the status of tracks based on their mobility and lifetime. Then update the list of effective tracks.
        """
        for track in self.effective_tracks:
            if track.cluster.status == DYNAMIC:
                lifetime = self.config.TR_LIFETIME_DYNAMIC
            else:
                lifetime = self.config.TR_LIFETIME_STATIC

            if track.lifetime > lifetime:
                track.status = INACTIVE

        self.effective_tracks[:] = [
            track for track in self.effective_tracks if track.status != INACTIVE
        ]

    def track(self, pointcloud, batch: BatchedData, isBetweenFrame: bool = False):
        """
        Perform the tracking process including prediction, association, maintenance, update, and clustering.

        Parameters
        ----------
        pointcloud : np.array
            Pointcloud data.
        batch : BatchedData
            BatchedData instance for managing frames.

        Returns
        -------
        None
        """
        # Prediction Step
        self._predict_all()
        # Association Step
        unassigned = self._assign_points_to_tracks_and_get_unassigned(pointcloud)
        # Update Step
        self._update_all()

        # TODO: Move Allocation step before maintenance.

        # TODO: Maintenance Step
        # self._maintain_tracks()
        # Clustering of the remainder points Step
        new_clusters = []
        batch.add_frame(unassigned)

        if (
            len(batch.effective_data) > 0
            and len(self.effective_tracks) < self.config.TR_MAX_TRACKS
        ):
            new_clusters = apply_DBscan(batch.effective_data)

            if len(new_clusters) > 0:
                batch.clear()

            # Create new track for every new cluster
            self._add_tracks(new_clusters)

    def update_real_posture(self, real_data):
        """
        Update the real posture of the target of each track using the real data.

        Parameters
        ----------
        real_data : np.array
            Real data for posture estimation.

        Returns
        -------
        An array of tuples containing the centroid and joint 0 of each track. They will be reformatted to (x, y, z) as the coordinate system is like that.
        """
        centralValues = []
        for index, track in enumerate(self.effective_tracks):
            try:
                kinect_coords = real_data[index]
              
            except:
                print("Warning. No more than one skeleton detected, showing the same skeleton for all tracks. ", time.time())
                kinect_coords = real_data[0]
            track.ground_truth = np.array(kinect_coords)
            reshaped_keypoints = track.ground_truth.copy().reshape(3, -1)

            reshaped_keypoints[0] *= -1
            # reshaped_keypoints[0] += track.state.x[0]
            # reshaped_keypoints[2] += track.state.x[1]
            # print(f"Track {index}: {reshaped_keypoints}")
            # Swap y and z coordinates to get x, y, z format
            reshaped_keypoints = reshaped_keypoints[[0, 2, 1]]
            # Calculate the centroid
            centroid = np.mean(reshaped_keypoints, axis=1)
            centralValues.append((centroid, reshaped_keypoints[:, 0]))
            # print(f"Centroid: {centroid}, Joint 0: {reshaped_keypoints[0]}")
        return centralValues