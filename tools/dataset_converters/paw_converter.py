import os
import re
import pickle
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import h5py



logging.basicConfig(level=logging.WARNING)

class AsteriosPawDatasetConverter:
    def __init__(self,
                 input_root: Union[str, Path],
                 output_root: Union[str, Path],
                 seed: int = 42,
                 split_ratios = {'train': 0.6, 'val': 0.2, 'test': 0.2},):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.out_mmwave = self.output_root / 'mmwave'
        self.out_skeleton = self.output_root / 'skeleton'
        self.out_mmwave.mkdir(parents=True, exist_ok=True)
        self.out_skeleton.mkdir(parents=True, exist_ok=True)
        self.split_ratios = split_ratios
        total = sum(split_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'Split ratios must sum to 1, got {total}')
        self.seed = seed
        random.seed(seed)
        self.records: Dict[str, List[dict]]  = {split: [] for split in split_ratios}

    @staticmethod
    def load_radar(csv_path: str,
                   valid_frames: Optional[List[int]] = None,
                   ANGLE_DEG: float = 6.5
                   ) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
        radar_df = pd.read_csv(csv_path, header=None)
        if radar_df.empty:
            return (np.empty((0, 5), np.float32),
                    np.array([0], np.int64),
                    [],
                    ['x', 'y', 'z', 'vel', 'snr'])
        radar_df.columns = ['seq', 'x', 'y', 'z', 'vel', 'snr', 'ts']
        grouped = radar_df.groupby('seq', sort=True)
        frame_ids = sorted(radar_df['seq'].unique())
        if valid_frames is not None:
            frame_ids = [fid for fid in frame_ids if fid in valid_frames]

        pts_list: List[np.ndarray] = []
        idx: List[int] = [0]
        theta = np.radians(-ANGLE_DEG)
        R = np.array([
            [1, 0, 0],
            [0,  np.cos(theta), -np.sin(theta)],
            [0,  np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)

        for fid in frame_ids:
            arr = grouped.get_group(fid)[['x','y','z','vel','snr']].to_numpy(np.float32)
            arr[:, :3] = arr[:, :3] @ R.T
            xyz = arr[:, :3]
            bad_mask = ~np.isfinite(xyz).all(axis=1)
            if bad_mask.any():
                print(f"Found {bad_mask.sum()} bad rows in frame {fid}:")
                print(xyz[bad_mask])
                print(csv_path)
            pts_list.append(arr)
            idx.append(idx[-1] + arr.shape[0])

        pcd_data = (np.concatenate(pts_list, axis=0)
                    if pts_list else np.empty((0,5), np.float32))
        pcd_idx = np.array(idx, dtype=np.int64)
        pcd_cols = ['x','y','z','vel','snr']
        return pcd_data, pcd_idx, frame_ids, pcd_cols

    @staticmethod
    def translate_skel(skel_frame: np.ndarray,
                       X: float = 0.22,
                       Z: float = 0.8) -> np.ndarray:
        joints = skel_frame.reshape(-1, 3).copy()
        joints += np.array([X, 0.0, Z], dtype=joints.dtype)
        joints[:, 0] *= -1
        joints = joints[:, [0, 2, 1]]

        return joints.ravel()
    @staticmethod
    def load_skel(csv_path: str,
                  valid_frames: Optional[List[int]]) -> Tuple[np.ndarray, List[str]]:
        df = pd.read_csv(csv_path, header=None)
        skel_idx = list(df.index)
        if valid_frames is not None:
            df = df.iloc[[i for i in valid_frames if i in skel_idx]]
        raw = df.values[:,2:59].astype(np.float32)
        num_joints = raw.shape[1]//3
        data_list = []
        for frame in raw:
            tr = AsteriosPawDatasetConverter.translate_skel(frame)
            data_list.append(tr)
        skel_data = np.stack(data_list,axis=0) if data_list else np.empty((0,3*num_joints),np.float32)
        skel_cols = [f"joint{j}_{axis}" for j in range(num_joints) for axis in ('x','y','z')]
        return skel_data, skel_cols
    
    def write_single_clip(self,
                          pcd_data: np.ndarray,
                          pcd_idx: np.ndarray,
                          pcd_cols: List[str],
                          skel_data: np.ndarray,
                          skel_cols: List[str],
                          subject: str,
                          split: str) -> None:
        fname = f"{subject}.h5"
        skel_path = self.out_skeleton / fname
        with h5py.File(skel_path, 'w') as h5f:
            ds = h5f.create_dataset('skel', data=skel_data)
            ds.attrs['columns'] = np.array(skel_cols, dtype='S')
        
        mmw_path = self.out_mmwave / fname
        with h5py.File(mmw_path, 'w') as h5f:
            grp = h5f.create_group('pcd')
            grp.create_dataset('data', data=pcd_data)
            grp.create_dataset('index', data=pcd_idx)
            grp.attrs['columns'] = np.array(pcd_cols, dtype='S')
        
        self.records.setdefault(split, []).append({
            'subject': subject,
            'frame_count': skel_data.shape[0],
            'mmwave_path': mmw_path.name,
            'skeleton_path': skel_path.name,
        })
    
    def process_single(self,
                       subject: str,
                       kf: Path,
                       mf: Path,
                       split: str) -> None:
        df_r = pd.read_csv(mf, header=None)
        df_r.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_r.dropna(subset=[1,2,3], inplace=True)
        df_k = pd.read_csv(kf, header=None)

        pairs: List[Tuple[int, int]] = []
        unique_frames = df_r.drop_duplicates(subset=0)
        for _, row in unique_frames.iterrows():

            radar_id = int(row[0])
            t1 = float(row[6])
            diffs = (df_k[0].astype(float) - t1).abs()
            skel_idx = int(diffs.idxmin())
            t2 = float(df_k.iloc[skel_idx, 0])
            if abs(t1 - t2) < 20:
                pairs.append((radar_id, skel_idx))
        
        if not pairs:
            logging.warning(f"No valid pairs found for subject {subject}. Skipping.")
            return
        
        radar_ids = [p[0] for p in pairs]
        skel_idxs = [p[1] for p in pairs]

        pcd_data, pcd_idx, _, pcd_cols = self.load_radar(str(mf), valid_frames=radar_ids)
        skel_data, skel_cols = self.load_skel(str(kf), valid_frames=skel_idxs)

        self.write_single_clip(
            pcd_data=pcd_data, pcd_idx=pcd_idx, pcd_cols=pcd_cols,
            skel_data=skel_data, skel_cols=skel_cols,
            subject=subject, split=split
        )
    
    def process_all(self) -> None:
        kinect_dir = self.input_root / 'kinect'
        mmwave_dir = self.input_root / 'mmWave'
        pattern = re.compile(r'([AB])(\d)(\d)')
        all_seqs: List[Tuple[str, Path, Path]] = []

        for kf in kinect_dir.glob('*.csv'):
            m = pattern.match(kf.stem)
            if not m: continue
            base = f"{m.group(1)}{m.group(2)}{m.group(3)}"
            mw_folder = mmwave_dir / base
            if not mw_folder.exists():
                logging.warning(f"Matching mmwave folder not found for {kf.name}. Skipping."); continue
            for mf in mw_folder.glob('*.csv'):
                if mf.stat().st_size == 0:
                    logging.warning(f"Empty mmwave file {mf.name}. Skipping."); continue
                seq_id = f"{base}-{mf.stem}"
                all_seqs.append((seq_id, kf, mf))
        
        random.shuffle(all_seqs)
        n = len(all_seqs)
        splits = dict()
        _counter = 0
        for i, (split, ratio) in enumerate(self.split_ratios.items()):
            count = int(n * ratio)
            if i == len(self.split_ratios) - 1:
                splits[split] = all_seqs[_counter:]
            else:
                splits[split] = all_seqs[_counter:_counter + count]
            _counter += count
        
        for split, seqs in splits.items():
            for subject, kf, mf in seqs:
                self.process_single(subject, kf, mf, split)
        
        for split, records in self.records.items():
            pkl = self.output_root / f'info_{split}.pkl'
            with open(pkl, 'wb') as f:
                pickle.dump(records, f)