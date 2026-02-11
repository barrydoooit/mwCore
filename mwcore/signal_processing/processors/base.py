from abc import ABC
from mwcore.signal_processing.config import ProcessingMode
from ..frame import RadarFrame
from typing import Protocol, runtime_checkable




@runtime_checkable
class Supports2D(Protocol):
    def process_2d(self, frame: RadarFrame) -> None:
        """Execute 2D specific logic (Azimuth only)."""
        ...

@runtime_checkable
class Supports3D(Protocol):
    def process_3d(self, frame: RadarFrame) -> None:
        """Execute 3D specific logic (Azimuth + Elevation)."""
        ...
    
class BaseSignalProcess(ABC):
    def __init__(self, name: str = "BaseProcess"):
        self.name = name

    def execute(self, frame: RadarFrame) -> None:
        """
        Smart Execute:
        Dispatches to process_2d or process_3d based on configuration
        and available implementation.
        """
        mode = frame.config.mode
        
        # 1. Try Specific Implementations
        if mode == ProcessingMode.MODE_2D and isinstance(self, Supports2D):
            self.process_2d(frame)
            return
            
        if mode == ProcessingMode.MODE_3D and isinstance(self, Supports3D):
            self.process_3d(frame)
            return

        # 2. Fallback to generic implementation (if class doesn't specialize)
        # Many modules (like RangeFFT) are identical for 2D/3D.
        self._process_generic(frame)

    def _process_generic(self, frame: RadarFrame) -> None:
        """Default implementation for mode-agnostic processors."""
        pass