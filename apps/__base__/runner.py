from mmengine.registry import Registry

APPS = Registry("apps")



class MWAppRunner:
    @classmethod
    def from_cfg(cls, cfg: dict):
        return NotImplementedError
    
    def start(self):
        return NotImplementedError
    