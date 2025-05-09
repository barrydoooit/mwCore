import sys
import serial
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from api.base_tracker import TRACKERS, BaseTracker
from apps.common.threads import THREADS
from radario.base import READERS, BaseBufferedReader
from radario.chirpConfig.chirpConfigIWR6843 import ChirpConfigIWR6843
from vistools.visualizer import MainVisualizer
from PySide2.QtCore import QThread, Signal
from PySide2.QtWidgets import QApplication



class VisAndTrackRunner:
    def __init__(self,
                 reader_cfg: dict,
                 tracker_cfg: dict,
                 vis_cfg: dict = None):
        self.sensor_started = reader_cfg.pop('sensor_started', False)
        self.reader = self._make_reader(reader_cfg)
        self.reader_thread = THREADS.build(dict(
            type="OnlineReaderThread",
            reader=self.reader
        ))
        self.tracker =  self._make_tracker(tracker_cfg)
        self.tracker_thread = THREADS.build(dict(
            type="OnlineTrackingThread",
            tracker=self.tracker
        ))
        self.reader_thread.raw_data.connect(self.tracker_thread.process_frame)
    
    def _make_reader(self, reader_cfg: dict) -> BaseBufferedReader:
        cli_port = reader_cfg.get("CLI_port")
        cli_port = serial.Serial(cli_port, BaseBufferedReader.CLI_BAUDRATE)
        assert reader_cfg.get("type") == "BufferedPcdReaderIWR6843"
        chirp_cfg = ChirpConfigIWR6843(reader_cfg.pop("config_file_path"), CLI_port=cli_port)
        
        if not self.sensor_started:
            print("Sending config to sensor...")
            chirp_cfg.send_config(False)
            self.sensor_started = True
            
        reader_cfg.update(dict(
            CLI_port=cli_port,
            Data_port=serial.Serial(reader_cfg.get("Data_port"), BaseBufferedReader.DATA_BAUDRATE, timeout=1.0)
        ))
        reader = READERS.build(reader_cfg)
        reader.register_config(chirp_cfg)
        return reader

    def _make_tracker(self, tracker_cfg: dict) -> 'BaseTracker':
        tracker = TRACKERS.build(tracker_cfg)
        return tracker

    def start(self):
        self.app = QApplication(sys.argv)
        def _on_close(event):
            self.reader_thread.requestInterruption()
            self.reader_thread.wait()

        self.main_window = MainVisualizer(
            new_data=self.reader_thread.array_data,
            on_close=_on_close
        )
        self.tracker_thread.tracking_data.connect(self.main_window.update_tracking)
        self.reader_thread.start()
        self.tracker_thread.start()
        self.main_window.show()
        sys.exit(self.app.exec_())
        
    @classmethod
    def from_cfg(cls, cfg: dict):
        tracker_cfg = dict(
                           type=cfg.get("tracker_type"),
                           tracker_config=cfg.get("tracker_config"),
                           keep_radial=cfg.get("keep_radial", False),
                           radar_cfg=cfg.get("radar_cfg", dict(
                               sensor_tilt=0,
                               sensor_height=0
                           )))
        vis_cfg = cfg.get("vis_cfg", dict())
        return cls(
            reader_cfg=cfg.get("reader_cfg"),
            tracker_cfg=tracker_cfg,
            vis_cfg=vis_cfg
        )
        
if __name__ == "__main__":
    import argparse
    import debugpy
    from mmengine.config import Config, DictAction


    from apps.vis_and_track.runner import VisAndTrackRunner

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="mmwave breakout application")
        parser.add_argument("config", type=str, help="Path to the configuration file")
        parser.add_argument('--cfg-options', nargs='+', action=DictAction)
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        args = parser.parse_args()
        if args.debug:
            debugpy.listen(("0.0.0.0", 5678))
            print("Waiting for debugger attach...")
            debugpy.wait_for_client()
            print("Debugger attached.")

        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        
        app_type = cfg.pop("type")
        if app_type is None:
            raise ValueError("Application type is not specified in the configuration file")
        
        if app_type == "vis_and_track":
            app = VisAndTrackRunner.from_cfg(cfg)
            
        app.start()