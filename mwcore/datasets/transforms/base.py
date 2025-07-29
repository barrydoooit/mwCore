from abc import ABC, abstractmethod
from mwcore.registry import TRANSFORMS

@TRANSFORMS.register_module()
class BaseTransform:
    def __init__(self, online_mode=False):
        self.online_mode = online_mode
        if self.online_mode:
            self.exec = self.transform_online
        else:
            self.exec = self.transform
    
    def transform(self, input):
        raise NotImplementedError

    def transform_online(self, input):
        return self.transform(input)
    
    def __call__(self, *args, **kwargs):
        return self.exec(*args, **kwargs)

WITH_ONLINE_FUNCTIONALITY = []

def OnlineEnabled(cls):
    WITH_ONLINE_FUNCTIONALITY.append(cls.__name__)
    return cls

def is_online_enabled(cls_or_name):
    if isinstance(cls_or_name, str):
        return cls_or_name in WITH_ONLINE_FUNCTIONALITY
    return cls_or_name.__name__ in WITH_ONLINE_FUNCTIONALITY