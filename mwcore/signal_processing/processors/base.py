from abc import ABC
from mwcore.signal_processing.config import ProcessingMode
from ..frame import RadarFrame
from typing import Any, Protocol, Set, runtime_checkable




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
    """
    Base class for all radar signal processors. 
    Subclasses must declare what attributes they read and write using sets.
    """
    requires: Set[str] = set()
    provides: Set[str] = set()
    
    STRICT_MODE: bool = True 

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    def _validate_contract(self, frame: RadarFrame, keys: Set[str], is_input: bool) -> None:
        if not self.STRICT_MODE or not keys: return
        for key in keys:
            if not frame.has(key):
                context = "required input before execution" if is_input else "promised output"
                raise RuntimeError(f"[{self.name}] Missing {context}: '{key}'.")

    def _check_permission(self, key: str, allowed_keys: Set[str], action: str) -> None:
        if self.STRICT_MODE and key not in allowed_keys:
            target_set = 'requires' if action == 'read' else 'provides'
            raise PermissionError(
                f"[{self.name}] Access Denied: Tried to {action} '{key}', "
                f"but it was not declared in the '{target_set}' set."
            )

    def read(self, frame: RadarFrame, key: str) -> Any:
        self._check_permission(key, self.requires, "read")
        return getattr(frame, key)

    def write(self, frame: RadarFrame, key: str, value: Any) -> None:
        self._check_permission(key, self.provides, "write")
        setattr(frame, key, value)

    def execute(self, frame: RadarFrame) -> None:
        self._validate_contract(frame, self.requires, is_input=True)
        mode = frame.config.mode
        if mode == ProcessingMode.MODE_2D and isinstance(self, Supports2D):
            self.process_2d(frame)
        elif mode == ProcessingMode.MODE_3D and isinstance(self, Supports3D):
            self.process_3d(frame)
        else:
            self._process_generic(frame)
        self._validate_contract(frame, self.provides, is_input=False)

    def _process_generic(self, frame: RadarFrame) -> None:
        pass