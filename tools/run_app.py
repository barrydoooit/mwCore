import argparse
import sys
import os
from typing import Type
import debugpy
from mmengine.config import Config, DictAction

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mwcore.apps.base import BaseMWApp
from mwcore.registry import APPS

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
    
    app_type = cfg.get("type")
    if app_type is None:
        raise ValueError("Application type is not specified in the configuration file")
    app_class: Type['BaseMWApp']= APPS.get(app_type)
    app = app_class.from_cfg(cfg)
    
    app.start()