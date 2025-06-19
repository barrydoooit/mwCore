from mmengine.registry import Registry



APPS = Registry(name="mmwave_apps", locations=["mwcore.apps"])
READERS = Registry(name="mmwave_readers", locations=["mwcore.radario.readers"])
TRACKERS = Registry(name="mmwave_trackers", locations=["mwcore.tracking.api"])
THREADS = Registry(name="mmwave_threads", locations=["mwcore.threads"])
VISUALIZERS = Registry(name="mmwave_visualizers", locations=["mwcore.visualization.visualizers"])