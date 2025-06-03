from sklearn.cluster import DBSCAN, Birch
from collections import deque
import math
import csv
import numpy as np
import os


class RingBuffer:
    """
    Circular buffer with a fixed size that automatically discards the oldest elements
    when new elements are added.

    Attributes
    ----------
    size : int
        Maximum size of the buffer.

    buffer : collections.deque
        Deque representing the circular buffer.

    Methods
    -------
    append(item)
        Add a new element to the buffer. If the buffer is full, the oldest element is removed.

    get_max()
        Get the maximum value in the buffer.

    get_mean()
        Get the mean value of the elements in the buffer.
    """

    def __init__(self, size, init_val=None):
        self.size = size
        self.buffer = deque(maxlen=size)
        if init_val is None:
            self.append(0)
        else:
            self.append(init_val)

    def append(self, item):
        self.buffer.append(item)

    def get_max(self):
        return np.max(self.buffer)

    def get_mean(self):
        return np.mean(self.buffer)

def altered_EuclideanDist(p1, p2, db_range_weight=0.03, db_z_weight=0.4):
    """
    Calculate an altered Euclidean distance between two points in 3D space.

    This distance metric incorporates modifications to better suit the characteristics of cylinder-shaped point clouds,
    especially those representing the human silhouette. It achieves this by applying the following adjustments:

    1. **Vertical Weighting**: Reduces the impact of the vertical distance by using a constant `const.DB_Z_WEIGHT`.
    This is beneficial for improved clustering of cylinder-shaped point clouds.

    2. **Inverse Proportional Weighting**: Introduces a weight to the result inversely proportional to the points' y-axis values.
    This ensures that the distance outputs are lower when the point cloud is further away from the sensor and thus, more sparse.

    Returns
    -------
    float
        The adjusted Euclidean distance between the two points.
    """
    # NOTE: The z-axis has less weight in the distance metric since the sillouette of a person is tall and thin.
    # Also, the further away from the sensor the more sparse the points, so we need a weighing factor
    weight = 1 - ((p1[1] + p2[1]) / 2) * db_range_weight
    return weight * (
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + db_z_weight * ((p1[2] - p2[2]) ** 2)
    )


def apply_DBscan(pointcloud, eps=0.3, min_samples=35, metric=altered_EuclideanDist):
    """
    Apply DBSCAN clustering to a 3D point cloud using an altered Euclidean distance metric.

    Parameters
    ----------
    pointcloud : array-like
        The 3D point cloud represented as a list or NumPy array.

    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        Default is const.DB_EPS.

    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        Default is const.DB_MIN_SAMPLES.

    Returns
    -------
    list
        A list of clustered point clouds, where each cluster is represented as a list of points.
    """
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )

    labels = dbscan.fit_predict(pointcloud)

    # label of -1 means noice so we exclude it
    filtered_labels = set(labels) - {-1}

    # Assign points to clusters
    clustered_points = {label: [] for label in filtered_labels}
    for i, label in enumerate(labels):
        if label != -1:
            clustered_points[label].append(pointcloud[i])

    # Return a list of clustered pointclouds
    clusters = list(clustered_points.values())
    return clusters

def apply_Birch(pointcloud, threshold=4, branching_factor=10):
    """
    Apply BIRCH clustering to a 3D point cloud.

    Parameters
    ----------
    pointcloud : array-like
        The 3D point cloud represented as a list or NumPy array.

    threshold : float, optional
        The radius of the sub-cluster obtained by merging a new sample and the closest sub-cluster.
        Default is 

    branching_factor : int, optional
        Maximum number of CF sub-clusters in each node.
        Default is

    Returns
    -------
    list
        A list of clustered point clouds, where each cluster is represented as a list of points.
    """
    if len(pointcloud) < 35:
        # Not enough points to form clusters
        return []
    
    birch = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=1)
    labels = birch.fit_predict(pointcloud)

    # label of -1 means noise so we exclude it
    filtered_labels = set(labels) - {-1}

    # Assign points to clusters
    clustered_points = {label: [] for label in filtered_labels}
    for i, label in enumerate(labels):
        if label != -1:
            clustered_points[label].append(pointcloud[i])

    # Return a list of clustered pointclouds
    clusters = list(clustered_points.values())
    return clusters

def apply_clustering(pointcloud, method="DBSCAN", **kwargs):
    """
    Apply a clustering algorithm to a 3D point cloud.

    Parameters
    ----------
    pointcloud : array-like
        The 3D point cloud represented as a list or NumPy array.

    method : str, optional
        The clustering algorithm to use. Supported methods are "DBSCAN" and "BIRCH".
        Default is "DBSCAN".

    Returns
    -------
    list
        A list of clustered point clouds, where each cluster is represented as a list of points.
    """
    if method == "DBSCAN":
        return apply_DBscan(pointcloud, **kwargs)
    elif method == "BIRCH":
        return apply_Birch(pointcloud)
    else:
        raise ValueError(f"Clustering method '{method}' is not supported.")

def polar_to_cartesian(state_polar):
        """
        Convert a state vector from polar coordinates [r, _r, θ, _θ] to Cartesian coordinates [x, y, vx, vy].

        Parameters
        ----------
        state_polar : np.array
            State vector in polar coordinates [r, _r, θ, _θ].

        Returns
        -------
        np.array
            State vector in Cartesian coordinates [x, y, vx, vy].
        """
        r, r_dot, theta, theta_dot = state_polar[:4]

        # Position
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Velocity
        vx = r_dot * np.cos(theta) - r * theta_dot * np.sin(theta)
        vy = r_dot * np.sin(theta) + r * theta_dot * np.cos(theta)

        return np.array([x, y, vx, vy])