from typing import Dict, Optional, Tuple, Union

import numpy as np
from mwcore.datasets.utils import pseudo_collate
from .base import OfflineReader
from mwcore.radario.readers.offlineReaders.framedata import FrameData
from mwcore.registry import DATASETS, READERS
import torch
from torch.utils.data import DataLoader
import logging

log = logging.getLogger(__name__)



@READERS.register_module()
class PoseEstim3DDatasetReader(OfflineReader):
    """Reader for MRI offline data"""
    
    def __init__(self, dataloader: Union[DataLoader, dict]):
        super().__init__()
        if isinstance(dataloader, DataLoader):
            self.dataloader = dataloader
        elif isinstance(dataloader, dict):
            self.dataloader = self._build_dataloader(dataloader)
    
    @property
    def dataset_iterator(self):
        if not hasattr(self, '_dataset_iterator'):
            self._dataset_iterator = iter(self.dataloader)
        return self._dataset_iterator
    
    @staticmethod
    def _build_dataloader(dataloader_cfg: dict) -> DataLoader:
        dataset = DATASETS.build(dataloader_cfg.pop("dataset"))
        print(type(dataset))
        dataloader = DataLoader(
            dataset,
            collate_fn=pseudo_collate,
            **dataloader_cfg
        )
        return dataloader
    
    @property
    def total_num_frames(self) -> int:
        return len(self.dataloader)
    
    def read(self) -> Tuple[int, int, Dict, Optional[np.ndarray], FrameData]:
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
            batch = next(self.dataset_iterator)
        except StopIteration:
            self._finished = True
            return 0, -1, {}, None
        except Exception as e:
            log.info(f"Error reading from dataset: {e}")
            return 0, -1, {}, None
        
        frm_idx = self._current_frame
        self.frame_inc()
        pcd = batch["pcd_frames"][0] # 0 for single batch
        if isinstance(pcd, (tuple, list)):
            pcd = pcd[0]
        num_pts = pcd.shape[0]
        det_obj = dict(
            numObj=num_pts,
            x=pcd[:, 0],
            y=pcd[:, 1],
            z=pcd[:, 2],
            doppler=pcd[:, 3],
            peakVal=pcd[:, 4],
        )

        skel = batch.get("skel_frames", None)
        if skel is not None:
            skel = skel[0]
            if isinstance(skel, (tuple, list)):
                skel = skel[0]
        
        if skel is not None and skel.size:
            gt = skel.reshape(-1, 3).T
        else:
            gt = None

        return 1, frm_idx, det_obj, gt, FrameData(frm_idx, det_obj, gt, None)