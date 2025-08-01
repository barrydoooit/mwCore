import logging

from mwcore.registry import EVALUATORS
log = logging.getLogger(__name__)
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from mwcore.evaluation.base import Evaluator



@EVALUATORS.register_module()
class SkelPositionalErrorEvaluator(Evaluator):
    def __init__(self, 
                 keypoints_involved: list[int],
                 model_name: str = "Unnamed",
                 dataset_name: str = "Unnamed",
                 coords_involved: list[int] = [0, 1], # x, y (ground plane)
                 out_path: Optional[Union[str, Path]] = None
                 ):
        self.keypoints_involved = keypoints_involved
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.coords_involved = coords_involved
        self.out_path = Path(out_path) if out_path else None
        self.reset()

    def process_sample(self, gt: np.ndarray, pred: list[np.ndarray]) -> Optional[np.ndarray]:
        try:
            gt = self._extract_pos_from_gt(gt)
            # print(len(pred), pred[0], gt)
            pred = pred[0][self.coords_involved] # NOTE: While tracker returns multiple objects, we only consider the first one for evaluation
            error_per_axis = gt - pred # array([dx, dy])
            self._errors.append(error_per_axis)
            return error_per_axis
        except Exception as e:
            log.error(f"Error processing sample. Skipped.")
            return None

    def evaluate(self, *args, **kwargs) -> float:
        log.info(f"{self.__class__.__name__}: Evaluating...")
        errors = np.abs(np.vstack(self._errors).transpose(1, 0))
        norm_errors = np.linalg.norm(errors, axis=0)
        errors = np.concatenate((errors, np.expand_dims(norm_errors, axis=0)), axis=0)
        axis_labels = {0: 'x', 1: 'y', 2: 'z', 3: 'all'}
        header = f"{'Axis':>4} | {'Mean':>10} | {'Median':>10} | {'Std':>10}"
        print(header)
        metrics: dict[int, dict[str, float]] = {}
        for i, axis_idx in enumerate(self.coords_involved + [3]):
            axis_errors = errors[i, :]
            mean = np.mean(axis_errors)
            median = np.median(axis_errors)
            std = np.std(axis_errors)
            label = axis_labels.get(axis_idx, f"Axis {axis_idx}")
            print(f"{label:>4} | {mean:>10.2f} | {median:>10.2f} | {std:>10.2f}")
            metrics[axis_idx] = {'mean': mean, 'median': median, 'std': std}
        
        if self.out_path is not None:
            if self.out_path.is_dir() or self.out_path.suffix != '.npy':
                self.out_path = self.out_path / f"{self.model_name}_{self.dataset_name}_poserror.npy"
            if not self.out_path.parent.exists():
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.out_path, errors)
            log.info(f"Saved positional error metrics to {self.out_path}")
        
        return metrics


    def reset(self) -> None:
        self._errors: list[float] = []

    def _extract_pos_from_gt(self, gt: np.ndarray) -> np.ndarray:
        reshaped_data = gt.reshape((3, -1))
        gt_to_consider = reshaped_data[:, self.keypoints_involved]
        gt_reduced = np.mean(gt_to_consider, axis=1)
        return gt_reduced[self.coords_involved]