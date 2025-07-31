#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import h5py


class MRIDatasetConverter:
    def __init__(self, 
                 input_root: Path,
                 output_root: Path,
                 split: Literal[1, 2] = 2,
                 protocol: Literal[1, 2] = 2,
                 ratio: float = 0.8,
                 seed: int = 42):
        self.input_root = input_root
        self.output_root = output_root

        # Directories in the mRI release
        self.pose_dir = self.input_root / 'pose_labels'
        self.radar_dir = self.input_root / 'radar' / 'singleframe'

        # Output subfolders
        self.out_skeleton = self.output_root / 'skeleton'
        self.out_mmwave = self.output_root / 'mmwave'
        self.out_skeleton.mkdir(parents=True, exist_ok=True)
        self.out_mmwave.mkdir(parents=True, exist_ok=True)

        assert split in [1, 2], "Split must be 1 or 2"
        assert protocol in [1, 2, 3], "Protocol must be 1, 2, or 3"
        self.split = split
        self.protocol = protocol
        self.ratio = ratio
        # List to accumulate metadata records
        if self.protocol == 1:
            self.selected_labels = [f'pose_{i}' for i in range(1, 11)] + ['free_form', 'walk']
        elif self.protocol == 2:
            self.selected_labels = [f'pose_{i}' for i in range(1, 11)]
        elif self.protocol == 3:
            self.selected_labels = ['walk']
        
        self.records = dict(
            train=[],
            val=[],
            test=[]
        )

        self.seed = seed
        np.random.seed(self.seed)

    @staticmethod
    def load_cpl(cpl_path: Path):
        """
        Load a .cpl pickle file. Extract:
          - refined_gt_kps: (N, 3, 17) numpy array of keypoints
          - gt_avail_frames: [start, end] inclusive camera-frame indices for skeleton
          - radar_avail_frames: [start, end] inclusive camera-frame indices for radar
          - video_label: dict of label: [start, end] exclusive camera-frame indices
        """
        with open(cpl_path, 'rb') as f:
            data = pickle.load(f)
        gt_avail = data['gt_avail_frames']
        assert len(gt_avail) == 2
        radar_avail = data['radar_avail_frames']
        assert len(radar_avail) == 2
        refined_gt_kps = data['refined_gt_kps']
        video_label = data['video_label']
        return refined_gt_kps, gt_avail, radar_avail, video_label

    @staticmethod
    def load_radar(csv_path: Path, start: int, end: int):
        """
        Read the single-frame radar CSV, group points by 'Camera Frame' between start and end (inclusive),
        and return:
          - pcd_data: concatenated point arrays (M, 5) for x, y, z, vel, snr
          - pcd_idx: offsets into pcd_data for each frame (length = num_frames + 1)
          - pcd_cols: ['x', 'y', 'z', 'vel', 'snr']
        """
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            'Camera Frame': 'seq',
            'X': 'x', 'Y': 'y', 'Z': 'z',
            'Doppler': 'vel', 'Intensity': 'snr'
        })[['seq', 'x', 'y', 'z', 'vel', 'snr']]
        grouped = df.groupby('seq', sort=True)
        pts_list = []
        idx = [0]
        for fid in range(start, end + 1):
            if fid in grouped.groups:
                arr = grouped.get_group(fid)[['x','y','z','vel','snr']].to_numpy(np.float32)
            else:
                raise ValueError(f"Frame {fid} not found in radar data")
            pts_list.append(arr)
            idx.append(idx[-1] + arr.shape[0])
        pcd_data = np.concatenate(pts_list, axis=0) if pts_list else np.empty((0,5), dtype=np.float32)
        pcd_idx = np.array(idx, dtype=np.int64)
        pcd_cols = ['x','y','z','vel','snr']
        return pcd_data, pcd_idx, pcd_cols

    def write_clip(
        self,
        subject: str,
        label: str,
        skel_data: np.ndarray,
        pcd_data: np.ndarray,
        pcd_idx: np.ndarray,
        skel_cols: list,
        pcd_cols: list,
    ):
        """
        Write one pose clip to H5 under skeleton and mmwave dirs.
        """
        fname = f"{subject}_{label}.h5"
        # skeleton
        skel_path = self.out_skeleton / fname
        with h5py.File(skel_path, 'w') as h5f:
            ds = h5f.create_dataset('skel', data=skel_data)
            ds.attrs['columns'] = np.array(skel_cols, dtype='S')
        # mmwave
        mmw_path = self.out_mmwave / fname
        with h5py.File(mmw_path, 'w') as h5f:
            grp = h5f.create_group('pcd')
            dsd = grp.create_dataset('data', data=pcd_data)
            dsd.attrs['columns'] = np.array(pcd_cols, dtype='S')
            grp.create_dataset('index', data=pcd_idx)
        return fname

    def process_subject(self, cpl_file: Path, subjects: list, all_entries: list):
        subject = cpl_file.stem.replace('_all_labels','')
        subjects.append(subject)
        skel_full, gt_avail, radar_avail, video_label = self.load_cpl(cpl_file)
        # per label
        for lbl, (s0, e0) in video_label.items():
            lbl_key = lbl.replace(' ', '_')
            if lbl_key not in self.selected_labels:
                continue
            # intersect with availability
            start = max(s0, gt_avail[0], radar_avail[0] + 1)
            end = min(e0-1, gt_avail[1], radar_avail[1])
            if end < start:
                continue
            n_frames = end - start + 1
            # skeleton trim & flatten
            sk = skel_full[start:end+1]  # (n,3,J)
            J = sk.shape[2]
            sk = sk.transpose(0,2,1)
            sk[:, :, [1, 2]] = sk[:, :, [2, 1]]  # swap y and z so as to TI coordinate
            sk[:, :, 0] = -sk[:, :, 0]  # flip x to TI coordinate
            sk = sk.reshape(n_frames, -1)  # (n,3*J)
            skel_cols = [f'joint{j}_{ax}' for j in range(J) for ax in ('x','y','z')]
            # radar
            csv_file = self.radar_dir / f'{subject}.csv'
            pcd, idx, pcd_cols = self.load_radar(csv_file, start, end)
            # write files
            fname = self.write_clip(subject, lbl_key, sk, pcd, idx, skel_cols, pcd_cols)
            # record entry
            entry = {
                'subject': subject,
                'label': lbl_key,
                'frame_count': n_frames,
                'skeleton_path': fname,
                'mmwave_path': fname,
            }
            all_entries.append(entry)

    def process_all(self):
        all_entries = []
        subjects = []
        # iterate subjects
        for cpl_file in sorted(self.pose_dir.glob('*_all_labels.cpl')):
            # process each subject
            self.process_subject(cpl_file, subjects, all_entries)

        # perform split
        np.random.shuffle(all_entries)
        if self.split == 1:
            n_train = int(self.ratio * len(all_entries))
            train = all_entries[:n_train]
            test = all_entries[n_train:]
            self.records['train'] = train
            self.records['test'] = test
            # val = test for random split
            self.records['val'] = list(test)
        else:
            # subject-level
            subjects = sorted(set(subjects))
            np.random.shuffle(subjects)
            n_train_subj = int(self.ratio * len(subjects))
            train_subj = set(subjects[:n_train_subj])
            train, test = [], []
            for ent in all_entries:
                if ent['subject'] in train_subj:
                    train.append(ent)
                else:
                    test.append(ent)
            self.records['train'] = train
            self.records['test'] = test
            self.records['val'] = list(test)

        # write info files
        for sp in ['train','test','val']:
            with open(self.output_root / f'info_{sp}.pkl', 'wb') as f:
                pickle.dump(self.records[sp], f)
        with open(self.output_root / 'info_all.pkl', 'wb') as f:
            pickle.dump(all_entries, f)
        
    MRI_KEYPOINT_TYPE = dict(
        NOSE=0,
        EYE_LEFT=1,
        EYE_RIGHT=2,
        EAR_LEFT=3,
        EAR_RIGHT=4,
        SHOULDER_LEFT=5,
        SHOULDER_RIGHT=6,
        ELBOW_LEFT=7,
        ELBOW_RIGHT=8,
        WRIST_LEFT=9,
        WRIST_RIGHT=10,
        HIP_LEFT=11,
        HIP_RIGHT=12,
        KNEE_LEFT=13,
        KNEE_RIGHT=14,
        ANKLE_LEFT=15,
        ANKLE_RIGHT=16,
    )