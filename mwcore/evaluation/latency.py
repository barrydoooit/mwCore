import logging

from mwcore.registry import EVALUATORS
log = logging.getLogger(__name__)
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from mwcore.evaluation.base import Evaluator


@EVALUATORS.register_module()
class LatencyEvaluator(Evaluator):
    def __init__(self, 
                 model_name: str = "Unnamed",
                 dataset_name: str = "Unnamed",
                 out_path: Optional[Union[str, Path]] = None
                 ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.out_path = Path(out_path) if out_path else None
        self.reset()

    def process_sample(self, latency: float) -> None:
        self._latencies.append(latency)

    def evaluate(self, *args, **kwargs) -> dict:
        log.info(f"{self.__class__.__name__}: Evaluating latency...")
        if not self._latencies:
            log.info("No latency data to evaluate.")
            return {}
        latency_mean = np.mean(self._latencies)
        latency_median = np.median(self._latencies)
        latency_std = np.std(self._latencies)
        log.info(f"Latency (ms): mean={latency_mean * 1000:.2f}, median={latency_median * 1000:.2f}, std={latency_std * 1000:.2f}")
        metrics = {
            'latency': {
                'mean': latency_mean,
                'median': latency_median,
                'std': latency_std
            }
        }
        if self.out_path is not None:
            if self.out_path.is_dir() or self.out_path.suffix != '.npy':
                self.out_path = self.out_path / f"{self.model_name}_{self.dataset_name}_latency.npy"
            if not self.out_path.parent.exists():
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.out_path, np.array(self._latencies))
            log.info(f"Saved latency metrics to {self.out_path}")
        return metrics

    def reset(self) -> None:
        self._latencies: list[float] = []