from mmengine.registry import Registry



READERS = Registry(name="mmwave_readers", locations=["mwcore.radario.readers"])
TRACKERS = Registry(name="mmwave_trackers", locations=["mwcore.tracking.api"])
THREADS = Registry(name="mmwave_threads", locations=["mwcore.threads"])