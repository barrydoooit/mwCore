from typing import Optional, List
import numpy as np
from .config import RadarConfig

class RadarFrame:
    """
    Central Data Structure acting as a dynamic blackboard.
    Processors attach data here dynamically during the pipeline execution.
    """
    def __init__(self, raw_bytes: bytes, config: Optional[RadarConfig] = None):
        self._config = config
        self.raw_bytes = raw_bytes
    
    @property
    def config(self) -> RadarConfig:
        if self._config is None:
            raise ValueError("RadarConfig not set yet.")
        return self._config

    def keys(self) -> List[str]:
        """Return a list of currently attached processing attributes."""
        # Filters out private attributes and methods
        return [k for k in self.__dict__.keys() if not k.startswith('_')]

    def has(self, key: str) -> bool:
        """Helper to safely check if a specific attribute has been populated."""
        return hasattr(self, key) and getattr(self, key) is not None

    def __repr__(self) -> str:
        """Neat string representation showing what the frame currently holds."""
        attrs = ", ".join(self.keys())
        return f"<RadarFrame: [{attrs}]>"