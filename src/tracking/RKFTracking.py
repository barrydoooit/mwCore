import numpy as np
from tracking.AsteriosTracking import Tracker,  BatchedData, PointCluster
from tracking.AsteriosTracking import ACTIVE, INACTIVE, STATIC, DYNAMIC
import time
import constants as const
from Utils import (
    apply_clustering,
    polar_to_cartesian,
    RingBuffer,
    relative_coordinates,
    format_single_frame,
)
from typing import List

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
        
    def step(self, measurement=None):
        self.predict()
        if measurement is not None:
            self.update(measurement)
        return self.x.flatten()
    
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

    def __init__(self, cluster: PointCluster):
        self.N_est = 0
        self.spread_est = np.zeros(4)  # Initialize with 4 elements for [r, _r, θ, _θ]
        self.group_disp_est = (
            np.eye(4) * const.KF_GROUP_DISP_EST_INIT  
        )
        self.cluster = cluster
        self.batch = BatchedData(cluster.pointcloud)
        self.state = RecursiveKalmanFilter()
        self.status = ACTIVE
        self.lifetime = 0
        self.keypoints = const.MODEL_DEFAULT_POSTURE
        # self.height_buffer = RingBuffer(
        #     const.FB_HEIGHT_FRAME_PERIOD, init_val=self.cluster.max_vals[2] - 0.01
        # )
        # self.width_buffer = RingBuffer(const.FB_WIDTH_FRAME_PERIOD)
        # NOTE: For visualizing purposes only
        self.predict_x = self.state.x
        self.color = np.random.rand(
            3,
        )

    def _estimate_point_num(self):
        """
        Estimate the expected number of points in the cluster.
        """
        if const.KF_ENABLE_EST:
            if self.cluster.point_num > self.N_est:
                self.N_est = self.cluster.point_num
            else:
                self.N_est = (
                    1 - const.KF_A_N
                ) * self.N_est + const.KF_A_N * self.cluster.point_num
        else:
            self.N_est = max(const.KF_EST_POINTNUM, self.cluster.point_num)

    def _estimate_measurement_spread(self):
        """
        Estimate the spread of measurements in each dimension.
        """
        for m in range(len(self.cluster.min_vals)):
            spread = self.cluster.max_vals[m] - self.cluster.min_vals[m]

            # Unbiased spread estimation
            if self.cluster.point_num != 1:
                spread = (
                    spread * (self.cluster.point_num + 1) / (self.cluster.point_num - 1)
                )

            # Ensure the computed spread estimation is between 1x and 2x of configured limits
            spread = min(2 * const.KF_SPREAD_LIM[m], spread)
            spread = max(const.KF_SPREAD_LIM[m], spread)

            if spread > self.spread_est[m]:
                self.spread_est[m] = spread
            else:
                self.spread_est[m] = (1.0 - const.KF_A_SPR) * self.spread_est[
                    m
                ] + const.KF_A_SPR * spread

    def _get_D(self):
        """
        Calculate and get the dispersion matrix for the track in state vector coordinates [r, ṙ, θ, θ̇].

        Returns
        -------
        numpy.ndarray
            Dispersion matrix for the cluster (4x4 for [r, ṙ, θ, θ̇]).
        """
        dimension = 4  # State vector dimensions: [r, ṙ, θ, θ̇]
        pointcloud = self.cluster.pointcloud[:, -3:]  # Use only the last 3 polar dimensions [r, θ, ṙ]
        centroid = self.cluster.centroid  # Centroid is [r, θ, ṙ]

        # Transform measurements to state vector coordinates [r, ṙ, θ, θ̇]
        transformed_pointcloud = np.zeros((pointcloud.shape[0], dimension))
        for i, point in enumerate(pointcloud):
            r, θ, ṙ = point
            transformed_pointcloud[i] = [r, ṙ, θ, 0.0]  # Assume θ̇ = 0 (or estimate it if available)

        # Compute the centroid in state vector coordinates
        transformed_centroid = np.array([centroid[0], centroid[2], centroid[1], 0.0])

        # Compute the covariance of the transformed point cloud around the transformed centroid
        disp = np.zeros((dimension, dimension), dtype="float")
        for i in range(dimension):
            for j in range(dimension):
                disp[i, j] = np.mean(
                    (transformed_pointcloud[:, i] - transformed_centroid[i]) *
                    (transformed_pointcloud[:, j] - transformed_centroid[j])
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
            Combined covariance matrix for the cluster (2x2 for [r, θ]).
        """
        N = self.cluster.point_num
        N_est = self.N_est

        # Handle edge cases
        if N == 0:
            raise ValueError("Cluster has no points (N = 0). Cannot compute Rc.")
        if N_est == 1:
            raise ValueError("Estimated number of points is 1 (N_est = 1). Cannot compute Rc.")

        # Get measurement noise covariance matrix (R_m)
        R_m = self.get_Rm()  # 2x2 for [r, θ]

        # Get group dispersion matrix (D)
        D = self._get_D()  # 4x4 for [r, ṙ, θ, θ̇]

        # Project D into measurement space
        H = np.array([
            [1, 0, 0, 0],  # Observe range (r)
            [0, 0, 1, 0]   # Observe angle (θ)
        ])
        D_projected = H @ D @ H.T  # 2x2

        # Compute the combined covariance matrix
        Rc = (R_m / N) + ((N_est - N) / ((N_est - 1) * N)) * D_projected

        return Rc

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

        """
        self.cluster = PointCluster(pointcloud, polar=True)
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

    def get_Rm(self):
        """
        Get the measurement covariance matrix for the observed dimensions [r, θ].
        """
        # Use only the first 2 elements of spread_est (r and θ)
        diagonal_elements = (self.spread_est[:2] / 2) ** 2
        R_m = np.diag(diagonal_elements)
        return R_m

    def predict_state(self, dt: float):
        """
        Predict the state of the Kalman filter based on the time multiplier.

        Parameters
        ----------
        dt : float
            Time multiplier for the prediction.
        """
        # Update the transition matrix (F) with the new time step (dt)
        self.state.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ])
        
        # Predict the state using the updated transition matrix and process noise covariance
        self.state.predict()
        self.predict_x = self.state.x

    def update_state(self):
        """
        Update the state of the Kalman filter based on the associated measurement (pointcloud centroid).
        """
        z = np.array(self.cluster.centroid)[:2]
        x_prev = self.state.x[:2, 0]
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

    def __init__(self):
        """
        Initialize TrackBuffer with empty lists for tracks and effective tracks.
        """
        self.effective_tracks: List[RKFClusterTrack] = []
        self.next_track_id = 0
        self.dt = 0
        self.t = time.time()

    def _maintain_tracks(self):
        """
        Update the status of tracks based on their mobility and lifetime. Then update the list of effective tracks.
        """
        for track in self.effective_tracks:
            if track.cluster.status == DYNAMIC:
                lifetime = const.TR_LIFETIME_DYNAMIC
            else:
                lifetime = const.TR_LIFETIME_STATIC

            if track.lifetime > lifetime:
                track.status = INACTIVE

        self.effective_tracks[:] = [
            track for track in self.effective_tracks if track.status != INACTIVE
        ]

    def _calc_dist_fun(self, full_set: np.array):
        """
        Calculate the Mahalanobis distance matrix for gating using polar measurements.

        Parameters
        ----------
        full_set : np.ndarray
            Full set of points, where the last 3 elements of each point are radial
            measurements (r, θ, ṙ).

        Returns
        -------
        np.ndarray
            An array representing the associated track (entry) for each point (index).
            If no track is associated with a point, the entry is set to None.
        """
        dist_matrix = np.empty((full_set.shape[0], len(self.effective_tracks)))
        associated_track_for = np.full(full_set.shape[0], None, dtype=object)

        # Measurement matrix (observes range and angle)
        H = np.array([
            [1, 0, 0, 0],  # Observe range (r)
            [0, 0, 1, 0]   # Observe angle (θ)
        ])

        for j, track in enumerate(self.effective_tracks):
            # Predicted measurement in polar coordinates
            H_i = np.dot(H, track.state.x).flatten()

            # Group residual covariance matrix
            C_g_j = H @ (track.state.P + track.group_disp_est) @ H.T + track.get_Rm()

            for i, point in enumerate(full_set):
                # Extract polar measurements (r, θ, ṙ)
                r, θ, ṙ = point[-3:]

                # Construct measurement vector [r, θ]
                z = np.array([r, θ])

                # Innovation for each measurement
                y_ij = z - H_i

                # Distance function (d^2)
                dist_matrix[i][j] = np.log(np.abs(np.linalg.det(C_g_j))) + np.dot(
                    np.dot(y_ij.T, np.linalg.inv(C_g_j)), y_ij
                )

                # Perform Gate threshold check
                if dist_matrix[i][j] < const.TR_GATE:
                    # Just choose the closest Mahalanobis distance
                    if associated_track_for[i] is None:
                        associated_track_for[i] = j
                    else:
                        if (
                            dist_matrix[i][j]
                            < dist_matrix[i][int(associated_track_for[i])]
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
            new_track = RKFClusterTrack(PointCluster(np.array(new_cluster), polar=True))
            # new_track.id = self.next_track_id
            self.next_track_id += 1
            self.effective_tracks.append(new_track)

    def _predict_all(self):
        """
        Predict the state of all effective tracks.
        """
        for track in self.effective_tracks:
            track.predict_state(track.lifetime + self.dt)

    def _update_all(self):
        """
        Update the state of all effective tracks.
        """
        for track in self.effective_tracks:
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
        unassigned = np.empty((0, 11), dtype="float")
        clusters = [[] for _ in range(len(self.effective_tracks))]
        # Simple matrix has len = len(full_set) and has the index of the chosen track.
        associated_track_for = self._calc_dist_fun(full_set)

        for i, point in enumerate(full_set):
            if associated_track_for[i] is None:
                unassigned = np.append(unassigned, [point], axis=0)
            else:
                clusters[associated_track_for[i]].append(point)
        return unassigned, clusters

    def _associate_points_to_tracks(self, full_set: np.array):
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
                # If no points are associated with the track, just update the lifetime
                track.update_lifetime(dt=self.dt)
            else:
                # If points are associated with the track, update the lifetime and associate the pointcloud
                track.update_lifetime(dt=self.dt, reset=True)
                track.associate_pointcloud(np.array(clouds[j]))

                # inner cluster separation
                # new_inner_clusters.append(track.seek_inner_clusters())

        # In case inner clusters are found, create new tracks for them
        for inner_cluster in new_inner_clusters:
            self._add_tracks(inner_cluster)

        return unassigned

    def track(self, pointcloud, batch: BatchedData, clusteringAlgorithm="DBSCAN"):
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
        if (clusteringAlgorithm not in ["DBSCAN", "BIRCH", "both"]):
            raise ValueError("Invalid clustering algorithm. Please use 'DBSCAN', 'BIRCH', or 'both'.")

        # Prediction Step. This modifies only the kalman state of the tracks
        self._predict_all()

        # Association Step
        unassigned = self._associate_points_to_tracks(pointcloud)
        self._maintain_tracks()

        # Update Step
        self._update_all()

        # Clustering of the remainder points Step
        new_clusters = []
        batch.add_frame(unassigned)

        if (len(batch.effective_data) > 0 and len(self.effective_tracks) < const.TR_MAX_TRACKS):
            new_clusters = apply_clustering(
                batch.effective_data, clusteringAlgorithm
            )
            if len(new_clusters) > 0:
                batch.clear()

            # Create new track for every new cluster
            self._add_tracks(new_clusters)

    def estimate_posture(self, model):
        """
        Format the pointcloud, estimate and save the posture of the target of each track using a CNN model.

        Parameters
        ----------
        model : Model
            The CNN model used for posture estimation.

        Returns
        -------
        None
        """
        frame_matrices = []
        indexes = []
        for index, track in enumerate(self.effective_tracks):
            if len(track.batch.effective_data) > const.MODEL_MIN_INPUT:
                rel_track_points = relative_coordinates(
                    list(track.batch.buffer),
                    track.cluster.centroid[:2],
                )
                # The inputs are in the form of [x, y, z, x', y', z', r', s]
                frame_matrices.append(format_single_frame(rel_track_points))
                indexes.append(index)

        frame_matrices_array = np.array(frame_matrices)
        if len(frame_matrices_array) > 0:
            frame_keypoints = model.predict(frame_matrices_array)
            for i, index in enumerate(indexes):
                self.effective_tracks[index].keypoints = frame_keypoints[i]

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

            # Convert state to cartesian coordinates
            state_cartesian = polar_to_cartesian(track.state.x.flatten())

            reshaped_keypoints[0] *= -1
            # reshaped_keypoints[0] += state_cartesian[0]
            # reshaped_keypoints[2] += state_cartesian[1]
            # print(f"Track {index}: {reshaped_keypoints}")
            # Swap y and z coordinates to get x, y, z format
            reshaped_keypoints = reshaped_keypoints[[0, 2, 1]]
            # Calculate the centroid
            centroid = np.mean(reshaped_keypoints, axis=1)
            centralValues.append((centroid, reshaped_keypoints[:, 0]))
            # print(f"Centroid: {centroid}, Joint 0: {reshaped_keypoints[0]}")
        return centralValues
    
    
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