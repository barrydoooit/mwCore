from enum import Enum
import numpy as np
import logging
import os
import pickle
import csv

log = logging.getLogger(__name__)
from mwcore.registry import READERS
from .base import OfflineReader
from typing import Dict, Tuple, List, Union, Optional



@READERS.register_module()
class MRIOfflineReader(OfflineReader):
    """Reader for MRI offline data"""
    
    def __init__(self, data: str = None, source_file: str = None, source_dir: str = None, **kwargs):
        """
        Initialize the MIR offline reader
        
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
        self.read_buffer_size = 100

        self.mmwave_path = os.path.join("data", "mri", "dataset_release", "aligned_data", "radar","singleframe", f"subject{self.path}.csv")
        self.gt_path = os.path.join("data", "mri", "dataset_release", "aligned_data", "pose_labels", f"subject{self.path}_all_labels.cpl")
        
        self.frame_count = 0
        self.last_read = None
        self.last_frame = None
        self.pointer = [0, 1]
        self.read_next_frames()

       

        
    def read(self) -> Tuple[int, int, Dict, Optional[np.ndarray]]:
        """
        Read a frame from the MIR offline data
        
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
            exists, frame_number, data, groundTruth = self.get_data()

            groundTruth = np.array([
                groundTruth["x"] if "x" in groundTruth else [],
                groundTruth["y"] if "y" in groundTruth else [],
                groundTruth["z"] if "z" in groundTruth else []
            ])
            
            if not exists or data is None:
                log.warning(f"No data for frame {frame_number}")
                if frame_number == -1:
                    log.info("Reached the end of the data stream")
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
            return 1, frame_number, det_obj, groundTruth
            
        except Exception as e:
            log.error(f"Error reading MIR offline data: {e}")
            self.current_frame = -1
            return 0, self.last_read, {}, []

    def read_next_frames(self):
        """
        Read the next batch of frames from the given experiment files starting from the specified frame number. Real data included
        """
        self.pointclouds = {}
        self.gt_joints = {}
        ground_truth_is_between = []
        self.last_frame = self.last_read if self.last_read is not None else None

        try:
            with open(self.mmwave_path, "r") as mmwave_file:
                mmwave_csv_reader = csv.reader(mmwave_file)
                # Read gt data once
                with open(self.gt_path, "rb") as gt_file:
                    gt_data = pickle.load(gt_file)
                    ground_truth_is_between = gt_data['video_label']['walk']  # [beginning_frame, end_frame]
                    begin_frame, end_frame = ground_truth_is_between
                    self.last_read = begin_frame if self.last_read is None else self.last_read
                    gt_data = gt_data['refined_gt_kps']


                for index, mmwave_row in enumerate(mmwave_csv_reader):
                    if index == 0:
                        # Skip header row
                        continue

                    gt_frame = int(mmwave_row[8])
                    pointcloud_coords = [
                        float(mmwave_row[2]),
                        float(mmwave_row[3]),
                        float(mmwave_row[4]),
                        float(mmwave_row[5]),
                        float(mmwave_row[6]),
                        float(mmwave_row[7]),
                    ]

                    # Only process frames within the specified ground truth range
                    if  ((not (begin_frame <= gt_frame <= end_frame))
                         or (gt_frame < self.last_read)):
                        continue

                    # If this is a new gt_frame, create a new entry
                    if gt_frame not in self.pointclouds:
                        self.pointclouds[gt_frame] = {
                            "x": [pointcloud_coords[0]],
                            "y": [pointcloud_coords[1]],
                            "z": [pointcloud_coords[2]],
                            "doppler": [pointcloud_coords[3]],
                            "peakVal": [pointcloud_coords[4]],
                            "posix": [pointcloud_coords[5]],
                        }
                        # Add ground truth for this frame
                        if gt_frame < len(gt_data):
                            self.gt_joints[gt_frame] = {
                                "x": gt_data[gt_frame, 0, :].tolist(),
                                "y": gt_data[gt_frame, 1, :].tolist(),
                                "z": gt_data[gt_frame, 2, :].tolist(),
                            }
                        self.last_frame = gt_frame

                        if len(self.pointclouds) >= self.read_buffer_size:
                            print(f"Read from frame number {self.last_frame-self.read_buffer_size} to {self.last_frame}")
                            break
                    else:
                        # Same gt_frame: append points
                        for key, value in zip(
                            ["x", "y", "z", "doppler", "peakVal", "posix"], pointcloud_coords
                        ):
                            self.pointclouds[gt_frame][key].append(value)


        except FileNotFoundError:
            print(f"File not found: {self.mmwave_path} or {self.gt_path}. Please check the paths.")

    def get_data(self):
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
        gt_data : list or None
            The Kinect joint data for the current frame, or None if data for the frame is not available.
        """

        # If the read buffer is parsed, read more frames from the experiment file
        if self.last_read >= self.last_frame:
            try:
                self.read_next_frames()
            except FileNotFoundError:
                print(f"File not found. End of experiment reached at frame {self.frame_count}.")
                return False, -1, None, None
        try:
            gt_data = self.gt_joints.get(self.last_read, None)
            pointcloud_data = self.pointclouds.get(self.last_read, None)
            if pointcloud_data is None:
                print(f"No point cloud data for frame {self.last_read}")
                return False, self.frame_count, None, None
            data_tuple= (
                True,
                self.last_read,
                pointcloud_data,
                gt_data
            )
            self.last_read = self.last_read + 1
            return data_tuple
        except Exception as e:
            print(f"Error: {e}. No data for frame {self.last_read}")
            return False, self.last_read, None, None
        