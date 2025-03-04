import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from wakepy import keep
import constants as const
from Visualizer import VisualManager
from Utils import OfflineManager, normalize_data
# from keras.models import load_model
import numpy as np
import time 
from Tracking import (
    TrackBuffer,
    BatchedData,
)

########### Set the experiment path here ############

EXPERIMENT_PATH = "src/asterios/dataset/preprocessed/?/B25"

#####################################################


def offline_main():
    if not os.path.exists(str.replace(EXPERIMENT_PATH, "?", "mmWave")):
        raise ValueError(f"No experiment file found in the path: {EXPERIMENT_PATH}")

    sensor_data = OfflineManager(EXPERIMENT_PATH)
    SLEEPTIME = 0.1  # from radar config "frameCfg"

    app = QApplication(sys.argv)

    visual = VisualManager()
    trackbuffer = TrackBuffer()
    # model = load_model(const.P_MODEL_PATH)
    batch = BatchedData()
    first_iter = [True]  # Use a list to make it mutable
    accumulated_errors = {
        "joint0": [],
        "centroid": [],
    }
    
    def cleanup():
        visual.visual.clear()
        # Compute and display the average error
        if accumulated_errors:
            average_error = {
                key: np.median(errors) for key, errors in accumulated_errors.items()
            }
            print(f"Median errors: {average_error}")
            # print("Visualizer error: ", np.mean(visual.visual.errors))
        else:
            print("No errors accumulated.")
        app.quit()
        sys.exit(0)

    app.aboutToQuit.connect(cleanup)

    def control_loop():
        errors = 0

        if not sensor_data.is_finished():
            try:
                dataOk, _, detObj, kinectJoints = sensor_data.get_data2()

                if dataOk:
                    if first_iter[0]:  # Access the mutable value
                        trackbuffer.dt = SLEEPTIME
                        first_iter[0] = False  # Update the mutable value
                    else:
                        trackbuffer.dt = detObj["posix"][0] / 1000 - trackbuffer.t

                    trackbuffer.t = detObj["posix"][0] / 1000
                    # Apply scene constraints, point translation and axis normalization
                    effective_data = normalize_data(detObj)

                    if effective_data.shape[0] != 0:
                        # Tracking module
                        # time_start = time.time()
                        trackbuffer.track(effective_data, batch)
                        # time_end = time.time()
                        # print("Tracking time: ", (time_end - time_start) * 1000, "ms")
                        # Posture Estimation module
                        
        
                        # trackbuffer.estimate_posture(model)
                        # if len(trackbuffer.effective_tracks) > 0:
                        #     print("Estimated posture: ", trackbuffer.effective_tracks[0].keypoints)
                        centralValues=trackbuffer.update_real_posture(kinectJoints)
                        
                        for centroid, joint0 in centralValues:
                            for track in trackbuffer.effective_tracks:
                                error = np.linalg.norm(centroid - track.state.x[:3].flatten())
                                accumulated_errors["centroid"].append(error)
                                error = np.linalg.norm(joint0 - track.state.x[:3].flatten())
                                accumulated_errors["joint0"].append(error)

                    visual.update(trackbuffer, detObj)

                # Schedule the next call to control_loop
                QTimer.singleShot(0, control_loop)
            except KeyboardInterrupt:
                cleanup()
        else:
            cleanup()
            

    # Start the control loop
    QTimer.singleShot(0, control_loop)

    # Start the application event loop
    sys.exit(app.exec_())

offline_main()