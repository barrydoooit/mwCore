from copy import copy, deepcopy
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
from mmengine.dataset import Compose
from mmengine.fileio import join_path
import numpy as np

from mwcore.datasets.transforms.base import BaseTransform

from mwcore.registry import DATASETS



@DATASETS.register_module()
class PoseEstim3DDataset:
    def __init__(self,
                 info_path: str,
                 data_root: str,
                 data_prefix: dict, # e.g. {'pcd': 'mmwave_filtered', 'skel': 'skeleton'}
                 pipeline: list,
                 sequence_length: int = 1,
                 allow_pad_sequence: bool = False
                 ):
        self.info_path = Path(info_path)
        self.data_root = Path(data_root)
        self.data_prefix = copy(data_prefix)
        assert self.info_path.exists(), f'{self.info_path} does not exist.'
        assert self.data_root.exists(), f'{self.data_root} does not exist.'
        
        self.sequence_length = sequence_length
        self.allow_pad_sequence = allow_pad_sequence
        
        with open(self.info_path, 'rb') as f:
            self.info: list = pickle.load(f)

        self.cum_frames, self.file_names = self._build_global_idx_to_file_table()
        self.total_frames = self.cum_frames[-1] if self.cum_frames.size > 0 else 0
        
        self.pipeline: list['BaseTransform'] = Compose(pipeline) # NOTE: This will only work when default range is moved to mwpose3d from mmengine (like in the init of runner class)
        print("MotionDataset initialized with total_frames:", self.total_frames)
    
    def _build_global_idx_to_file_table(self):
        cum_list = []
        file_lists = {modality: [] for modality in list(self.data_prefix.keys())} # {'pcd': [], 'skel': []}
        total_frames = 0
        for info_single in self.info:
            if isinstance(info_single['frame_count'], dict):
                info_single['frame_count'] = info_single['frame_count'][self.data_prefix['pcd']]
            frame_count = info_single['frame_count']
            if self.sequence_length > 1 and not self.allow_pad_sequence:
                valid_frames = max(frame_count - (self.sequence_length - 1), 0)
            else:
                valid_frames = frame_count
            total_frames += valid_frames
            cum_list.append(total_frames)
            for modality in list(self.data_prefix.keys()):
                file_lists[modality].append(join_path(self.data_root, self.data_prefix[modality], info_single[f'{self.data_prefix[modality]}_path']))
        cum_frames = np.array(cum_list, dtype=np.int64)
        file_names = {modality: np.array(file_lists[modality]) for modality in list(self.data_prefix.keys())}
        return cum_frames, file_names
    
    def get_info_by_global_idx(self, global_idx: int):
        if global_idx < 0 or global_idx >= self.total_frames:
            raise IndexError(f"global_idx {global_idx} is out of bounds [0, {self.total_frames})")
        
        file_idx = np.searchsorted(self.cum_frames, global_idx, side='right')
        info: dict = deepcopy(self.info[file_idx])

        if file_idx == 0:
            valid_local_idx = global_idx
        else:
            valid_local_idx = global_idx - self.cum_frames[file_idx - 1]
        
        if self.sequence_length > 1 and not self.allow_pad_sequence:
            local_idx = valid_local_idx + (self.sequence_length - 1)
        else:
            local_idx = valid_local_idx
        
        if "snippets" in info:
            assert len(info["snippets"]) == 1, "Only one snippet is supported for now."
            snippet = info["snippets"][0]
            local_idx += snippet[0]
        

        info.update({
            "global_idx": global_idx,
            "local_idx": local_idx,
            "data_file": {f'{modality}': self.file_names[modality][file_idx] for modality in list(self.data_prefix.keys())}
        })

        return info
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx) -> dict:
        sample = self.pipeline(self.get_info_by_global_idx(idx))
        return sample