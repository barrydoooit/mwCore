from typing import Generic, List, Tuple, TypeVar, Union
from PySide6.QtCore import QObject, Signal, Slot, QThread
import numpy as np

from mwcore.evaluation.base import Evaluator
from mwcore.registry import EVALUATORS
from mwcore.radario.readers.offlineReaders.framedata import FrameData

import logging
log = logging.getLogger(__name__)


class EvaluationWorker(QObject):
    metrics_updated = Signal(dict)
    signal_evaluation_complete = Signal()

    def __init__(self, evaluators: List[Union[Evaluator, dict]]):
        super().__init__()
        if not isinstance(evaluators, list):
            evaluators = [evaluators]

        self.evaluators: List[Evaluator] = []
        for ev in evaluators:
            if isinstance(ev, dict):
                ev = EVALUATORS.build(ev)
            self.evaluators.append(ev)

        self._last_frame_number: int = -1
        self.final_frame_number = None

    @Slot(object)
    def process_framedata(self, framedata: FrameData):
        gt = framedata.ground_truth
        pred = framedata.prediction
        latency = getattr(framedata, 'latency', None)
        if gt is None or pred is None:
            return

        for evaluator in self.evaluators:
            if evaluator.__class__.__name__ == "LatencyEvaluator" and latency is not None:
                evaluator.process_sample(latency)
            else:
                evaluator.process_sample(gt, pred)
        
        self._last_frame_number = framedata.frame_id
        if self.final_frame_number is not None and self._last_frame_number == self.final_frame_number:
            self.upon_evaluation_complete()
    
    @Slot(int)
    def update_final_frame_number(self, frame_number: int):
        self.final_frame_number = frame_number
        log.info(f"Final frame index updated: {frame_number} expected")
        if self.final_frame_number == self._last_frame_number:
            self.upon_evaluation_complete()

    def upon_evaluation_complete(self):
        metrics = {}
        for evaluator in self.evaluators:
            name = evaluator.name if hasattr(evaluator, 'name') else evaluator.__class__.__name__
            log.info(f"Finalizing evaluator: {name}")
            metrics[name] = evaluator.evaluate()
        self.metrics_updated.emit(metrics)
        self.signal_evaluation_complete.emit()
    
    @classmethod
    def build_with_thread(cls,
                          evaluators: List[Union[Evaluator, dict]]) -> Tuple['EvaluationWorker', QThread]:
        worker = cls(evaluators)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(lambda: None)
        return worker, thread



