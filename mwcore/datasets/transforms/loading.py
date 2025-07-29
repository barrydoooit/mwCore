from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from .base import BaseTransform
from mwcore.registry import TRANSFORMS



@TRANSFORMS.register_module()
class LoadMultiFrame3DPoseEstimDatasetFromH5(BaseTransform):
    POINTCLOUD_MODALITY_KEY = 'pcd'
    SKELETON_MODALITY_KEY = 'skel'
    INDEX = 'index'
    DATA = 'data'
    LOCAL_IDX = 'local_idx'
    DATA_FILE = 'data_file'
    FRAME_IDX = 'frame_idx'
    PCD_FRAMES = 'pcd_frames'
    SKEL_FRAMES = 'skel_frames'
    TARGET_NUM_FRAMES = 'target_num_frames'
    
    def __init__(self,
                 load_pcd_dim: int,
                 num_frames: int,
                 backup_frames: int = 0,
                 empty_frame_op: Literal['zero', 'prev', 'error'] = 'error',
                 with_skeleton: bool = True
                 ):
        super().__init__()
        self.backup_frames = backup_frames
        self.load_pcd_dim = load_pcd_dim
        self.num_frames = num_frames
        self.empty_frame_op = empty_frame_op
        self.with_skeleton = with_skeleton

        self._empty_frame_count = 0
    
    def transform(self, input):
        local_idx: int = input[self.LOCAL_IDX]
        data_files: dict = input[self.DATA_FILE]
        pcd_file = data_files[self.POINTCLOUD_MODALITY_KEY]
        
        total = self.backup_frames + self.num_frames
        start_idx = local_idx - total + 1
        if start_idx == 0:
            input['starting_flag'] = True
        else:
            input['starting_flag'] = False
        input[self.TARGET_NUM_FRAMES] = self.num_frames
        
        with h5py.File(pcd_file, 'r') as pf:
            grp = pf[self.POINTCLOUD_MODALITY_KEY]
            index = grp[self.INDEX][:]
            
            if local_idx < 0 or local_idx >= index[-1]:
                raise IndexError(f'local_idx {local_idx} is out of bounds [0, {index[-1]})')
            
            pcd_frames = []
            last_valid = None
            
            for i in range(total):
                cur = start_idx + i
                if cur < 0 or cur >= index[-1]:
                    pcd_frame = self.handle_empty_pcd(last_valid)
                else:
                    start = index[cur]
                    end = index[cur + 1] if cur + 1 < len(index) else None
                    pcd_data = grp[self.DATA][start:end][:, :self.load_pcd_dim]
                    if pcd_data.size == 0:
                        pcd_frame = self.handle_empty_pcd(last_valid)
                    else:
                        pcd_frame = pcd_data
                        last_valid = pcd_frame
                pcd_frames.append(pcd_frame)
            input[self.PCD_FRAMES] = tuple(pcd_frames)
        

        if not self.with_skeleton:
            return input
    
        skel_file = data_files[self.SKELETON_MODALITY_KEY]
        with h5py.File(pcd_file, 'r') as pf, h5py.File(skel_file, 'r') as sf:
            frame_idx_map = pf[self.POINTCLOUD_MODALITY_KEY].get(self.FRAME_IDX, None)
            skel_seq = sf[self.SKELETON_MODALITY_KEY]
            skel_frames = []
            
            for i in range(total):
                cur = start_idx + i
                skel_idx = int(frame_idx_map[cur] - 1) if frame_idx_map is not None else cur
                skel_frames.append(skel_seq[skel_idx])
        
            input[self.SKEL_FRAMES] = tuple(skel_frames)
            
        input[self.TARGET_NUM_FRAMES] = self.num_frames
        return input

    def handle_empty_pcd(self, prev_frame):
        self._empty_frame_count += 1
        if self._empty_frame_count % 100 == 0:
            print(f"Total empty frames: {self._empty_frame_count}")
        if self.empty_frame_op == 'zero':
            return np.zeros((0, self.load_pcd_dim), dtype=np.float32)
        elif self.empty_frame_op == 'prev':
            if prev_frame is not None:
                return prev_frame
            else:
                return np.zeros((1, self.load_pcd_dim), dtype=np.float32)
        elif self.empty_frame_op == 'error':
            raise ValueError("Empty frame found in the dataset. Set empty_frame_op to 'zero' or 'prev' to handle empty frames.")