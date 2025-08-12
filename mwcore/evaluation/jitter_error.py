import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from mwcore.evaluation.base import Evaluator
from mwcore.registry import EVALUATORS

log = logging.getLogger(__name__)

@EVALUATORS.register_module()
class SkelJitterEvaluator(Evaluator):
    def __init__(self, 
                 keypoints_involved: list[int],
                 model_name: str = "Unnamed",
                 dataset_name: str = "Unnamed",
                 coords_involved: list[int] = [0, 1],
                 out_path: Optional[Union[str, Path]] = None
                 ):
        self.keypoints_involved = keypoints_involved
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.coords_involved = coords_involved
        self.out_path = Path(out_path) if out_path else None
        self.reset()

    def process_sample(self, gt: np.ndarray, pred: list[np.ndarray]) -> None:
        try:
            # Extract GT position
            gt_pos = self._extract_pos_from_gt(gt)
            # Extract predicted position
            pred_pos = pred[0][self.coords_involved]
            self._gt_positions.append(gt_pos)
            self._pred_positions.append(pred_pos)
        except Exception as e:
            log.error(f"Error processing sample. Skipped.")

    def evaluate(self, *args, **kwargs):
        log.info(f"{self.__class__.__name__}: Evaluating jitter (jerk)...")
        gt_positions = np.array(self._gt_positions)
        pred_positions = np.array(self._pred_positions)
        if len(gt_positions) < 4 or len(pred_positions) < 4:
            log.warning("Not enough samples to compute jerk.")
            return None
        # Compute jerk for GT and prediction
        gt_jerk = np.diff(np.diff(np.diff(gt_positions, axis=0), axis=0), axis=0)
        pred_jerk = np.diff(np.diff(np.diff(pred_positions, axis=0), axis=0), axis=0)
        gt_jerk_norm = np.linalg.norm(gt_jerk, axis=1)
        pred_jerk_norm = np.linalg.norm(pred_jerk, axis=1)
        # Compare jerk norms
        jerk_error = pred_jerk_norm - gt_jerk_norm
        mean = np.mean(pred_jerk_norm)
        median = np.median(pred_jerk_norm)
        std = np.std(pred_jerk_norm)
        print(f"Jitter (jerk). Mean: {mean:.4f}, Median: {median:.4f}, Std: {std:.4f}")
        print(f"Expected Jitter (jerk) from GT: {np.mean(gt_jerk_norm):.4f}, Median: {np.median(gt_jerk_norm):.4f}, Std: {np.std(gt_jerk_norm):.4f}")
        metrics = {'mean': mean, 'median': median, 'std': std}
        if self.out_path is not None:
            if self.out_path.is_dir() or self.out_path.suffix != '.npy':
                self.out_path = self.out_path / f"{self.model_name}_{self.dataset_name}_jitter.npy"
            if not self.out_path.parent.exists():
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.out_path, jerk_error)
            log.info(f"Saved jitter metrics to {self.out_path}")
        return metrics

    def reset(self) -> None:
        self._gt_positions: list[np.ndarray] = []
        self._pred_positions: list[np.ndarray] = []

    def _extract_pos_from_gt(self, gt: np.ndarray) -> np.ndarray:
        reshaped_data = gt.reshape((3, -1))
        gt_to_consider = reshaped_data[:, self.keypoints_involved]
        gt_reduced = np.mean(gt_to_consider, axis=1)
        return gt_reduced[self.coords_involved]