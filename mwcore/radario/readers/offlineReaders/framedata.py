from dataclasses import dataclass
from typing import Any



@dataclass
class FrameData:
    frame_id: int
    input_data: Any
    ground_truth: Any = None
    prediction: Any = None
    latency: float = None  # Optional latency field to store processing time
    