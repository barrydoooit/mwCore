from enum import Enum
import numpy as np
import logging
import csv
import os
from typing import Tuple, Dict, List, Optional


log = logging.getLogger(__name__)
from mwcore.registry import READERS
from .base import OfflineReader


@READERS.register_module()
class AsteriosOfflineReader(OfflineReader):
    """Reader for Asterios offline data"""
    def __init__(self, data: str = None, source_file: str = None, source_dir: str = None, **kwargs):
        """
        Initialize the offline reader
        
        Parameters
        ----------
        data : str
            Path to the directory with data
        source_file : str
            Path to a specific file to read (alternative to data)
        source_dir : str
            Path to a directory with data files (alternative to data)
        """
        super().__init__(data, source_file, source_dir, **kwargs)
        # Initialize the offline manager
        self.manager = OfflineManager(self.path)
        
    def register_config(self, config):
        """Store config data if needed"""
        self._config = config

    @property
    def config(self):
        return self._config
        
    def read(self) -> Tuple[int, int, Dict, Optional[np.ndarray]]:
        """
        Read a frame from the offline data
        
        Returns
        -------
        Tuple[int, int, Dict]
            A tuple containing:
            - data_ok: 1 if data was read successfully, 0 otherwise
            - frame_number: The frame number
            - det_obj: Dictionary with detected object data
        """
        try:
            # Read the next frame from the offline manager
            exists, frame_number, data, groundTruth = self.manager.get_data2()
            
            if not exists or data is None:
                log.warning(f"No data for frame {frame_number}")
                if frame_number == -1:
                    log.info("Reached the end of the data stream")
                    self.current_frame = -1
                    return 0, -1, {}, []
                return 0, frame_number, {}, []
                
            # Prepare the data in the expected format
            det_obj = {
                "numObj": len(data["x"]) if "x" in data else 0,
                "doppler": np.array(data.get("doppler", [])),
                "peakVal": np.array(data.get("peakVal", [])),
                "x": np.array(data.get("x", [])),
                "y": np.array(data.get("y", [])),
                "z": np.array(data.get("z", []))
            }
            # print(f"Offline frame - Num Points: {det_obj['numObj']}")
            self.current_frame = frame_number
            return 1, frame_number, det_obj, groundTruth
            
        except Exception as e:
            log.error(f"Error reading offline data: {e}")
            return 0, self.current_frame, {}, []
        

class OfflineManager:
    """
    A class for managing the reading of frames from an offline experiment file.

    Attributes
    ----------
    experiment_path : str
        The path to the directory containing the experiment files.
    frame_count : int
        The count of frames read so far.
    pointer : List[int]
        A list containing two integers: the index of the last read frame and the index of the current file being read.
    pointclouds : Dict[int, Dict[str, List[float]]]
        A dictionary containing point cloud data for each frame.
    last_frame : int or None
        The index of the last frame read from the experiment file, or None if the experiment has finished.

    Methods
    -------
    read_next_frames()
        Read the next batch of frames from the experiment file.
    get_data()
        Get the data for the current frame.
    """

    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        
        self.frame_count = 0
        self.pointer = [0, 1]
        # self.read_next_frames()
        self.last_frame = None

        self.read_next_frames2()

    def read_next_frames(self):
        """
        Read the next batch of frames from the given experiment file starting from the specified frame number.
        """
        self.pointclouds = {}
        self.last_frame = None

        while len(self.pointclouds) < 40:
            file_path = os.path.join(self.experiment_path, f"{self.pointer[1]}.csv")
            try:
                with open(file_path, "r") as file:
                    csv_reader = csv.reader(file)
                    for index, row in enumerate(csv_reader):
                        # Pass previously parsed frames
                        if index < self.pointer[0]:
                            continue

                        framenum = int(row[0])
                        coords = [
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                            int(row[6]),
                        ]

                        # Read only the frames in the specified range
                        if framenum in self.pointclouds:
                            # Append coordinates to the existing lists
                            for key, value in zip(
                                ["x", "y", "z", "doppler", "peakVal", "posix"], coords
                            ):
                                self.pointclouds[framenum][key].append(value)
                        else:
                            # If not, create a new dictionary for the framenum
                            self.pointclouds[framenum] = {
                                "x": [coords[0]],
                                "y": [coords[1]],
                                "z": [coords[2]],
                                "doppler": [coords[3]],
                                "peakVal": [coords[4]],
                                "posix": [coords[5]],
                            }

                        self.last_frame = framenum

                        if len(self.pointclouds) >= 40:
                            # Break the loop once const.FB_READ_BUFFER_SIZE frames are read
                            self.pointer[0] = index + 1
                            break
                    else:
                        self.pointer[0] = 0
                        self.pointer[1] += 1

            except FileNotFoundError:
                break

    def read_next_frames2(self):
        """
        Read the next batch of frames from the given experiment files starting from the specified frame number. Real data included. Work in progress.
        """
        self.pointclouds = {}
        self.kinect_joints = {}
        frame_count = 0
        mmwave_path = self.experiment_path.replace("preprocessed/?", "log/mmWave")
        kinect_path = self.experiment_path.replace("preprocessed/?", "log/kinect") + ".csv"

        mmwave_file_path = os.path.join(mmwave_path, f"{self.pointer[1]}.csv")
        try:
            with open(mmwave_file_path, "r") as mmwave_file:
                mmwave_csv_reader = csv.reader(mmwave_file)
                kinect_data = []

                # Read Kinect data once
                with open(kinect_path, "r") as kinect_file:
                    kinect_csv_reader = csv.reader(kinect_file)
                    kinect_data = list(kinect_csv_reader)

                for index, mmwave_row in enumerate(mmwave_csv_reader):
                    # Pass previously parsed frames
                    if index < self.pointer[0]:
                        continue

                    framenum = int(mmwave_row[0])
                    pointcloud_coords = [
                        float(mmwave_row[1]),
                        float(mmwave_row[2]),
                        float(mmwave_row[3]),
                        float(mmwave_row[4]),
                        float(mmwave_row[5]),
                        int(mmwave_row[6]),
                    ]

                    frame_index = self.pointer[0] + index

                    # Read only the frames in the specified range
                    if framenum in self.pointclouds:
                        # Append coordinates to the existing lists
                        for key, value in zip(
                            ["x", "y", "z", "doppler", "peakVal", "posix"], pointcloud_coords
                        ):
                            self.pointclouds[framenum][key].append(value)
                    else:
                        # If not, create a new dictionary for the framenum
                        self.pointclouds[framenum] = {
                            "x": [pointcloud_coords[0]],
                            "y": [pointcloud_coords[1]],
                            "z": [pointcloud_coords[2]],
                            "doppler": [pointcloud_coords[3]],
                            "peakVal": [pointcloud_coords[4]],
                            "posix": [pointcloud_coords[5]],
                        }

                        # Read corresponding Kinect data:
                        # kinect_row = kinect_data[frame_count]
                        closest_row = min(kinect_data, key=lambda row: abs(float(row[0]) - pointcloud_coords[5]))
                        # Format (x1, y1, z1, x2, y2, z2, ...)
                        kinect_coords = [float(closest_row[i]) for i in range(2, len(closest_row)-1)]
                        kinect_coords=np.array(kinect_coords).reshape(-1, 3).T.flatten()


                        self.kinect_joints[framenum] = [kinect_coords] 

                        frame_count += 1

                    

                    self.last_frame = framenum

                    if len(self.pointclouds) >= 40:
                        # Break the loop once const.FB_READ_BUFFER_SIZE frames are read
                        self.pointer[0] = index + 1
                        break
                else:
                    self.pointer[0] = 0
                    self.pointer[1] += 1

        except FileNotFoundError:
            print(f"File not found: {mmwave_file_path} or {kinect_path}")
            raise FileNotFoundError(f"File not found: {mmwave_file_path} or {kinect_path}")


    def get_data(self):
        """
        Get the data for the current frame.

        Returns
        -------
        exists : bool
            True if data for the current frame exists, False otherwise.
        frame_count : int
            The count of frames read so far.
        data : dict or None
            The point cloud data for the current frame, or None if data for the frame is not available.
        """

        self.frame_count += 1
        # If the read buffer is parsed, read more frames from the experiment file
        if self.frame_count > self.last_frame:
            self.read_next_frames()

        if self.frame_count in self.pointclouds:
            return True, self.frame_count, self.pointclouds[self.frame_count]
        else:
            return False, self.frame_count, None
        
    def get_data2(self):
        """
        Get the data for the current frame. Real data included. Work in progress.

        Returns
        -------
        exists : bool
            True if data for the current frame exists, False otherwise.
        frame_count : int
            The count of frames read so far.
        data : dict or None
            The point cloud data for the current frame, or None if data for the frame is not available.
        kinect_data : list or None
            The Kinect joint data for the current frame, or None if data for the frame is not available.
        """

        self.frame_count += 1
        # If the read buffer is parsed, read more frames from the experiment file
        try:
            if self.frame_count > self.last_frame:
                try:
                    self.read_next_frames2()
                except FileNotFoundError:
                    print(f"File not found. End of experiment reached at frame {self.frame_count}.")
                    return False, -1, None, None
        except Exception as e:
            print(f"Error reading next frames: {e}")
            return False, -1, None, None

        if self.frame_count in self.pointclouds:
            return True, self.frame_count, self.pointclouds[self.frame_count], self.kinect_joints[self.frame_count]
        else:
            print(f"Frame {self.frame_count} not found.")
            return False, self.frame_count, None, None

    def get_real_data(self):
        """
        Get the real data for the current frame.
        """
        current_frame = self.kinect_joints[self.frame_count]
        return current_frame
