import sys
import time
import os
import cProfile
import pstats
from PyQt5.QtWidgets import QApplication
from wakepy import keep
import constants as const

path_to_kaloyanBarry = os.path.join(os.path.dirname(__file__), '..', '..', 'kaloyanBarry')
sys.path.append(os.path.abspath(path_to_kaloyanBarry))

from radario.readDataIWR6843 import BufferedPcdReaderIWR6843
from radario.chirpConfig.chirpConfigIWR6843 import ChirpConfigIWR6843
from apps.lateral_tracking.constants import P_CONFIG_PATH, P_CLI_PORT, P_DATA_PORT
from Visualizer import VisualManager
from keras.models import load_model
from Utils import (
    normalize_data,
)
from tracking.AsteriosTracking import (
    TrackBuffer,
    BatchedData,
)
import serial

def main():
    # IWR1443 = ReadIWR14xx(
    #     const.P_CONFIG_PATH, CLIport=const.P_CLI_PORT, Dataport=const.P_DATA_PORT
    # )
    config = ChirpConfigIWR6843(P_CONFIG_PATH, P_CLI_PORT)
    config.send_config(close_port=False)
    reader = BufferedPcdReaderIWR6843(P_CLI_PORT, serial.Serial(P_DATA_PORT, BufferedPcdReaderIWR6843.DATA_BAUDRATE, timeout=1.0))
    SLEEPTIME = 0.001 * IWR1443.framePeriodicity  # Sleeping period (sec)

    app = QApplication(sys.argv)

    trackbuffer = TrackBuffer()
    batch = BatchedData()
    visual = VisualManager()
    model = load_model(const.P_MODEL_PATH)

    # Disable screen sleep/screensaver
    with keep.presenting():
        # Control loop
        while True:
            try:
                t0 = time.time()

                # Online mode
                dataOk, _, detObj = reader.read()

                if dataOk:
                    now = time.time()
                    trackbuffer.dt = now - trackbuffer.t
                    trackbuffer.t = now
                    # Apply scene constraints, translation
                    effective_data = normalize_data(detObj)
                    ef_shape = effective_data.shape[0]

                    if ef_shape != 0:
                        # Tracking Module
                        trackbuffer.track(effective_data, batch)

                    visual.update(trackbuffer, detObj)

                    if ef_shape != 0:
                        # Posture Estimation module
                        trackbuffer.estimate_posture(model)

                    t_code = time.time() - t0
                    t_sleep = max(0, SLEEPTIME - t_code)

                    # NOTE: The exact SLEEPTIME might cause the system to occasionally lose
                    # frames. In that case try a shorted sleep duration.
                    time.sleep(t_sleep)

            except KeyboardInterrupt:
                del IWR1443
                # app.exit()
                break


if __name__ == "__main__":
    if const.PROFILING:
        if not os.path.exists(const.P_PROFILING_PATH):
            os.makedirs(const.P_PROFILING_PATH)

        cProfile.run("main()", f"{const.P_PROFILING_PATH}perf_stats")

        with open(f"{const.P_PROFILING_PATH}profiling_results", "w") as f:
            p = pstats.Stats(f"{const.P_PROFILING_PATH}perf_stats", stream=f)
            p.sort_stats("cumulative").print_stats()
    else:
        main()
