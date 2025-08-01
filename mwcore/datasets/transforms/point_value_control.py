from typing import Tuple
import numpy as np

from .base import BaseTransform, OnlineEnabled
from mwcore.registry import TRANSFORMS



@OnlineEnabled
@TRANSFORMS.register_module()
class AddRangeDimension(BaseTransform):
    def __init__(self,
                 insert_idx: int = 4,
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.insert_idx = insert_idx
    
    @staticmethod
    def _calc_range(pcd_frame: np.ndarray):
        range = np.linalg.norm(pcd_frame[:, :3], axis=1)
        return range
    
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        added_frames = []
        for pcd_frame in pcd_frames:
            range = self._calc_range(pcd_frame)
            pcd_frame = np.insert(pcd_frame, self.insert_idx, range, axis=1)
            added_frames.append(pcd_frame)
        input['pcd_frames'] = tuple(added_frames)
        return input

@OnlineEnabled
@TRANSFORMS.register_module()
class NormalizePointAttr(BaseTransform):
    def __init__(self,
                 attr_indices: Tuple[int],
                 means: Tuple[float],
                 stds: Tuple[float],
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.attr_indices = attr_indices
        self.means = means
        self.stds = stds

    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        normed_frames = []
        for pcd_frame in pcd_frames:
            pcd_frame[:, self.attr_indices] = (pcd_frame[:, self.attr_indices] - self.means) / self.stds
            normed_frames.append(pcd_frame)
        input['pcd_frames'] = tuple(normed_frames)
        return input