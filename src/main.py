import sys
import time
import os
import cProfile
import pstats
from PyQt5.QtWidgets import QApplication
from wakepy import keep
import constants as const

path_to_kaloyanBarry = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.append(os.path.abspath(path_to_kaloyanBarry))
path_to_kaloyanBarry = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'kaloyanBarry')
sys.path.append(os.path.abspath(os.path.join(path_to_kaloyanBarry, 'radario')))

from src.kaloyanBarry.radario.readDataIWR6843 import BufferedPcdReaderIWR6843
from src.kaloyanBarry.radario.chirpConfig.chirpConfigIWR6843 import ChirpConfigIWR6843
from src.kaloyanBarry.apps.lateral_tracking.constants import P_CONFIG_PATH, P_CLI_PORT, P_DATA_PORT
from Visualizer import VisualManager
from keras.models import load_model
from Utils import (
    normalize_data,
)
from tracking.AsteriosTracking import (
    TrackBuffer,
    BatchedData,
)
from tracking.RKFTracking import RKFTrackBuffer
from PyQt5.QtCore import QTimer
import numpy as np

def main():
    polarExperiment = True
    # Initialize configuration and reader
    config = ChirpConfigIWR6843(P_CONFIG_PATH, P_CLI_PORT)
    config.send_config(close_port=True)
    reader = BufferedPcdReaderIWR6843(P_CLI_PORT, P_DATA_PORT)
    SLEEPTIME = 0.001 * 10

    # Initialize Qt application
    app = QApplication(sys.argv)

    # Initialize modules
    trackbuffer = RKFTrackBuffer()
    batch = BatchedData(np.empty((0, 11 if polarExperiment else 8)))
    visual = VisualManager(raw_cloud=True, b_boxes=True, posture=True, polar=True)
    model = load_model(const.P_MODEL_PATH)

    # Disable screen sleep/screensaver
    with keep.presenting():
        # Control loop
        def control_loop():
            try:
                t0 = time.time()

                # Online mode
                dataOk, _, detObj = reader.read()

                if dataOk:
                    now = time.time()
                    trackbuffer.dt = now - trackbuffer.t
                    trackbuffer.t = now

                    # Apply scene constraints, translation
                    effective_data = normalize_data(detObj, keepRadial=polarExperiment, transform=False)
                    ef_shape = effective_data.shape[0]
                    print(f"Detected points: {len(effective_data)}")

                    if ef_shape != 0:
                        # Tracking Module
                        trackbuffer.track(effective_data, batch)

                    visual.update(trackbuffer, detObj)

                    if ef_shape != 0:
                        # Posture Estimation module
                        trackbuffer.estimate_posture(model)

                    t_code = time.time() - t0
                    t_sleep = max(0, SLEEPTIME - t_code)

                    # Schedule the next iteration of the control loop
                    QTimer.singleShot(int(t_sleep * 1000), control_loop)
                else:
                    # If no data is available, schedule the next iteration immediately
                    QTimer.singleShot(0, control_loop)

            except KeyboardInterrupt:
                cleanup()

        # Cleanup function
        def cleanup():
            print("Cleaning up resources...")
            nonlocal reader
            if reader:
                reader.close()  # Ensure the reader is properly closed
                del reader  # Ensure the reader is properly closed
            visual.visual.clear()  # Clear the visualizer
            print("Resources cleaned up.")
            app.quit()  # Quit the Qt application
            sys.exit(0)  # Exit the program

        # Connect cleanup to application exit
        app.aboutToQuit.connect(cleanup)

        # Start the control loop
        QTimer.singleShot(0, control_loop)

        # Start the application event loop
        sys.exit(app.exec_())


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