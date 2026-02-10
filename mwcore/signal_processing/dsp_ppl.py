from typing import List, Dict, Union, Any
from mwcore.registry import ADCPROCESSORS
from .frame import RadarFrame, RadarConfig

class Compose:
    """
    Composes a sequence of signal processing operations.
    Similar to torchvision.transforms.Compose or mmengine.Compose.
    """
    def __init__(self, transforms: List[Union[dict, Any]]):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                # Build from config dict using registry
                self.transforms.append(ADCPROCESSORS.build(transform))
            elif hasattr(transform, 'execute'):
                # Already an instantiated processor
                self.transforms.append(transform)
            else:
                raise TypeError(f"Transform must be a dict or have 'execute' method. Got {type(transform)}")

    def __call__(self, frame: RadarFrame) -> RadarFrame:
        for t in self.transforms:
            t.execute(frame)
        return frame

class DspPipeline:
    """
    A generic Radar DSP Pipeline configurable via dictionary.
    """
    def __init__(self, radar_config: RadarConfig, pipeline_cfg: List[dict]):
        self.config = radar_config
        self.runner = Compose(pipeline_cfg)

    @classmethod
    def from_cfg(cls, cfg: dict):
        """
        Initialize pipeline from a full configuration dictionary.
        
        Args:
            cfg (dict): Must contain:
                - 'radar_cfg': dict of arguments for RadarConfig
                - 'pipeline': list of dicts defining the processing steps
        """
        radar_args = cfg.get('radar_cfg', {})
        radar_config = RadarConfig(**radar_args)
        pipeline_steps = cfg.get('pipeline', [])
        return cls(radar_config, pipeline_steps)

    def run(self, raw_bytes: bytes) -> RadarFrame:
        """Process a single frame of raw data."""

        frame = RadarFrame(raw_bytes, self.config)
        self.runner(frame)
        
        return frame