from enum import Enum
import numpy as np
import logging
import os
import pickle
from typing import Dict, Tuple, List, Union, Optional
import random


log = logging.getLogger(__name__)
from mwcore.registry import READERS
from .base import OfflineReader

@READERS.register_module()
class MiliPointReader(OfflineReader):
    """Reader for MMR keypoint data (mmWave points + ground truth)"""
    
    def __init__(self, data: str = None, source_file: str = None, source_dir: str = None, **kwargs):
        super().__init__(data, source_file, source_dir, **kwargs)
        self.read_buffer_size = 100
        self.frame_count = 0
        self.last_read = None
        self.last_frame = None
        
        # Configuration (matches MMRKeypointData defaults)
        self.max_points = kwargs.get('max_points', 300)
        self.num_keypoints = 18
        
        # Path configuration
        self.raw_data_path = os.path.join('data', 'MiliPoint_data', 'raw')
        self.processed_data = os.path.join('data', 'MiliPoint_data', 'processed', f'data.pkl')

        self.kp18_names = ['NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 
                  'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW', 
                  'LEFT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE', 
                  'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE', 
                  'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE', 
                  'RIGHT_EAR', 'LEFT_EAR']
        
        # Initialize data structures
        self.pointclouds = {}  # mmWave data storage
        self.gt_joints = {}    # ground truth keypoints storage
        self._load_data()

    def _load_data(self):
        """Load and process the data from files"""
        # Check if processed data exists
        if not os.path.exists(self.processed_data):
            self._process_raw_data()
        
        # Load processed data
        with open(self.processed_data, 'rb') as f:
            full_data = pickle.load(f)
            print(f"Loaded processed data from {self.processed_data}, total frames: {len(full_data)}")
            # print(f"First frame data: {full_data[0] if full_data else 'No data'}")
            self.dataset = full_data
        
        # Initialize reading pointers
        self.last_read = random.randint(0, len(self.dataset) - 1)
        self.last_frame = len(self.dataset) - 1

    def _process_raw_data(self):
        """Process raw data files (similar to MMRKeypointData._process)"""
        data_list = []
        
        # Load all raw files (0.pkl to 18.pkl)
        for i in range(19):
            fn = f'{self.raw_data_path}/{i}.pkl'
            with open(fn, 'rb') as f:
                print(f"Loading raw data from {fn}")
                data_list.extend(pickle.load(f))
        
        # Process keypoints (convert 18 to 9 if needed)
        if self.num_keypoints == 9:
            data_list = self._transform_keypoints(data_list)
        
        # Process point clouds (padding if needed)
        # data_list = self._process_pointclouds(data_list)
        
        
        # Save processed data
        processed_data =data_list
        os.makedirs(os.path.dirname(self.processed_data), exist_ok=True)
        with open(self.processed_data, 'wb') as f:
            pickle.dump(processed_data, f)

    def _transform_keypoints(self, data_list):
        """Convert 18 keypoints to 9 keypoints format"""
        kp9_names = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'LEFT_ELBOW', 
                    'RIGHT_HIP', 'RIGHT_KNEE', 'LEFT_HIP', 'LEFT_KNEE', 'HEAD']
        head_names = ['NOSE', 'RIGHT_EYE', 'LEFT_EYE', 'RIGHT_EAR', 'LEFT_EAR']
        
        kp9_idx = [self.kp18_names.index(n) for n in kp9_names[:-1]]
        head_idx = [self.kp18_names.index(n) for n in head_names]
        
        for data in data_list:
            kpts = data['y']
            # Select main keypoints
            kpts_new = kpts[kp9_idx]
            # Calculate head position as average of head components
            head = np.mean(kpts[head_idx], axis=0)
            kpts_new = np.concatenate((kpts_new, head[None]))
            data['y'] = kpts_new
        return data_list

    def _process_pointclouds(self, data_list):
        """Apply padding to point clouds if needed"""
        for data in data_list:
            points = data['x']
            if len(points) < self.max_points:
                # Pad with zeros
                pad_size = self.max_points - len(points)
                points = np.pad(points, ((0, pad_size), (0, 0)), 'constant')
            elif len(points) > self.max_points:
                # Randomly sample down
                points = points[np.random.choice(len(points), self.max_points, replace=False)]
            data['new_x'] = points
        return data_list

    def read(self) -> Tuple[int, int, Dict, np.ndarray]:
        """
        Read a frame of mmWave data and corresponding ground truth
        
        Returns:
            tuple: (success_flag, frame_number, pointcloud_data, keypoints)
        """
        try:
            exists, frame_number, pc_data, gt_data = self.get_data()
            
            if not exists:
                return (0, frame_number, {}, np.array([]))
            
            print(f"Reading frame {frame_number} with {pc_data['x'].shape[0]} points")
            # Format point cloud data
            pointcloud = {
                'numObj': pc_data['x'].shape[0],
                'x': pc_data['x'][:, 0],
                'y': pc_data['x'][:, 1],
                'z': pc_data['x'][:, 2],
                'doppler': np.zeros(pc_data['x'].shape[0]),  # placeholder
                'peakVal': np.zeros(pc_data['x'].shape[0])   # placeholder
            }
            
            # Format keypoints (ground truth)
            keypoints = np.array([
                gt_data[:, 0],  # x coordinates
                gt_data[:, 2],  # y coordinates
                gt_data[:, 1]   # z coordinates
            ])
            
            return (1, frame_number, pointcloud, keypoints)
            
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            return (0, self.last_read or -1, {}, np.array([]))

    def get_data(self):
        """Get next frame of data"""
        if self.last_read >= self.last_frame:
            return (False, -1, None, None)
        
        data = self.dataset[self.last_read]
        result = (
            True,
            self.last_read,
            {'x': data['x']},  # point cloud
            data['y']  # keypoints
        )
        self.last_read += 1
        return result