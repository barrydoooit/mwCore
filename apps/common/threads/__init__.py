from mmengine.registry import Registry

THREADS = Registry('threads')

from .online_reader import OnlineReaderThread
from .online_tracking import OnlineTrackingThread