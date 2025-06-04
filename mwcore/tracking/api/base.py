from copy import deepcopy
import math
from typing import Optional, Protocol
import numpy as np

from mwcore.utils.tranforms import dev2standard



class TrackingFunctionality(Protocol):
    """Protocol for tracking functionality."""
    def consume(self, pcd_data_dict: dict): ...
    

class BaseTracker(TrackingFunctionality):
    def __init__(self, radar_cfg: dict):
        self.radar_cfg = deepcopy(radar_cfg)
    
    def normalize_data(self, 
                       det_obj: Optional[dict] = None, 
                       point_array: Optional[np.ndarray] = None, 
                       keepRadial: bool = False, 
                       transform: bool = True):
        """
        Preprocesses the point cloud data from the sensor.

        This function filters the input point cloud, converts radial to Cartesian velocity,
        and transforms the coordinates to the standard vertical-horizontal plane axis system.

        Parameters
        ----------
        det_obj : dict
            Dictionary containing the raw detection data with keys:
            - "x": x-coordinate
            - "y": y-coordinate
            - "z": z-coordinate
            - "doppler": Doppler velocity
            - "peakVal": Signal Intensity
        point_array : np.ndarray, optional
            If provided, this array should contain the point cloud data in the format:
            [[x1, y1, z1, doppler1, peakVal1],
             [x2, y2, z2, doppler2, peakVal2], ...]
        transform : bool, optional
            If True, applies a transformation to the points to align them with the standard vertical-horizontal plane axis system.
        keepRadial : bool, optional
        If True, retains the original radial measurements (r, θ, ṙ) alongside Cartesian-transformed values.

        Returns
        -------
        np.ndarray
            Preprocessed data in the standard vertical-horizontal plane axis system.
            Columns:
            - x-coordinate
            - y-coordinate
            - z-coordinate
            - Cartesian velocity along the x-axis
            - Cartesian velocity along the y-axis
            - Cartesian velocity along the z-axis
            - doppler
            - peakval
        """
        if det_obj is not None:
            input_data = np.vstack(
                (det_obj["x"], det_obj["y"], det_obj["z"], det_obj["doppler"], det_obj["peakVal"])
            ).T
        if point_array is not None:
            input_data = point_array
        
        ef_data = np.empty((0, 8), dtype="float") if not keepRadial else np.empty((0, 11), dtype="float")

        for index in range(len(input_data)):
            x, y, z, doppler, peakVal = input_data[index]

            # Compute polar coordinates
            r = math.sqrt(x**2 + y**2 + z**2)
            theta = math.atan2(y, x)  # Angle in radians
            r_dot = doppler  # Radial velocity remains unchanged

            # Transform the radial velocity into Cartesian
            r =  math.sqrt(x**2 + y**2 + z**2)
            if r == 0:
                vx = 0
                vy = doppler
                vz = 0
            else:
                if (
                    input_data[index, 0] is None
                    or input_data[index, 1] is None
                    or input_data[index, 2] is None
                    or input_data[index, 3] is None
                ):
                    print(f"Error: {input_data[index, :]}")

                vx = doppler * x / r
                vy = doppler * y / r
                vz = doppler * z / r

            # Translate points to new coordinate system
            transformed_point = self.point_transform_to_standard_axis(
                np.array([x, y, z, vx, vy, vz])
            ) if transform else np.array([x, y, z, vx, vy, vz])

            transformed_point = np.append(
                transformed_point, (input_data[index, 3], input_data[index, 4])
            )

            # To be used with the Recursive Kalmann Filter
            if keepRadial:
                transformed_point = np.append(transformed_point, [r, theta, r_dot])  
            # Perform scene constraints filtering
            if (
                transformed_point[2] <= 2.5
                and transformed_point[2] > 0
                and transformed_point[1] > 0
            ):
                ef_data = np.append(
                    ef_data,
                    [transformed_point],
                    axis=0,
                )

        return ef_data
    
    @property
    def cfg_sensor_height(self):
        return self.radar_cfg.get('sensor_height')
    
    @property
    def cfg_sensor_tilt(self):
        return self.radar_cfg.get('sensor_tilt')
    
    def point_transform_to_standard_axis(self, input: np.ndarray) -> np.ndarray:
        """
        Transform 3D point coordinates and velocities to a standard axis.

        The transformation includes translation and rotation to bring the input point into a standard coordinate system.

        Parameters
        ----------
        input : array-like
            Input point represented as a 6-element array or list, where the first three elements are coordinates (x, y, z),
            and the last three elements are velocities along the corresponding axes.

        Returns
        -------
        np.array
            Transformed point with coordinates and velocities in the standard axis system.
        """
        # Translation Matrix (T)
        M = dev2standard(self.cfg_sensor_height, self.cfg_sensor_tilt)

        coords_h = np.concatenate((input[:3], [1.0]))
        transformed_coords_h = M @ coords_h

        if len(input) < 6:
            return np.array(
                [
                    transformed_coords_h[0],
                    transformed_coords_h[1],
                    transformed_coords_h[2],
                ]
            )
        vel_h = np.concatenate((input[3:], [0.0]))
        transformed_velocities_h = M @ vel_h
        return np.array(
            [
                transformed_coords_h[0],
                transformed_coords_h[1],
                transformed_coords_h[2],
                transformed_velocities_h[0],
                transformed_velocities_h[1],
                transformed_velocities_h[2],
            ]
        ) 
