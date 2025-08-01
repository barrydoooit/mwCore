from mmengine.registry import Registry

from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS



DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['mwcore.datasets'])
TRANSFORMS = Registry(
    'transform', parent=MMENGINE_TRANSFORMS, locations=['mwcore.datasets'])

APPS = Registry(name="mmwave_apps", locations=["mwcore.apps"])
READERS = Registry(name="mmwave_readers", locations=["mwcore.radario.readers"])
TRACKERS = Registry(name="mmwave_trackers", locations=["mwcore.tracking.api"])
THREADS = Registry(name="mmwave_threads", locations=["mwcore.threads"])
VISUALIZERS = Registry(name="mmwave_visualizers", locations=["mwcore.visualization.visualizers"])
EVALUATORS = Registry(name="mmwave_evaluators", locations=["mwcore.evaluation"])
