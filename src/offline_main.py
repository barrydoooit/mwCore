import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from wakepy import keep
import constants as const
from Visualizer import VisualManager
from Utils import OfflineManager, normalize_data, polar_to_cartesian
# from keras.models import load_model
import numpy as np
import time 
from tracking.AsteriosTracking import (
    TrackBuffer,
    BatchedData,
)
from tracking.GTrack import GTrackBuffer
from tracking.RKFTracking import RKFTrackBuffer
from tracking.KaloyanTracking import import_kaloyan_tracking
import pandas as pd

########### Set the experiment path here ############

EXPERIMENT_PATH = "src/asterios_mars_reproduce/dataset/preprocessed/?/A65"

#####################################################


def offline_main():
    kaloyanModule = import_kaloyan_tracking()
    polarExperiment = False

    if not os.path.exists(str.replace(EXPERIMENT_PATH, "?", "mmWave")):
        raise ValueError(f"No experiment file found in the path: {EXPERIMENT_PATH}")

    sensor_data = OfflineManager(EXPERIMENT_PATH)
    SLEEPTIME = 0.1  # from radar config "frameCfg"

    app = QApplication(sys.argv)

    visual = VisualManager(polar=polarExperiment, b_boxes=True, raw_cloud=True, ground_truth=True)
    trackbuffer =TrackBuffer(usePalmar=polarExperiment)
    # model = load_model(const.P_MODEL_PATH)
    batch = BatchedData(np.empty((0, 11 if polarExperiment else 8)))
    # batch = BatchedData() #Change to 11 for polar
    first_iter = [True]  
    accumulated_errors = {
        "joint0": [],
        "centroid": [],
    }
    timeToTrack = []
    frames = 0

    
    
    def cleanup():
        visual.visual.clear()
        # Compute and display the average error
        if accumulated_errors:
            stats = {key: (np.median(errors), np.max(errors), np.min(errors)) for key, errors in accumulated_errors.items()}
            print(f"Max errors: { {key: max for key, (_, max, _) in stats.items()} }")
            print(f"Min errors: { {key: min for key, (_, _, min) in stats.items()} }")
            print(f"Median errors: { {key: median for key, (median, _, _) in stats.items()} }")
            print(f"Median tracking time: {np.median(timeToTrack)} ms")
            # Save it to a file
            df = pd.DataFrame(accumulated_errors)
            df_time = pd.DataFrame({'timeToTrack': timeToTrack})
            df_combined = pd.concat([df, df_time], axis=1)
            df_combined.to_csv("src/asterios_mars_reproduce/errors/asteriosUsingBatch.csv", index=False)
                
            # print("Visualizer error: ", np.mean(visual.visual.errors))
        else:
            print("No errors accumulated.")
        app.quit()
        sys.exit(0)

    app.aboutToQuit.connect(cleanup)

    def control_loop():
        nonlocal frames 

        if not sensor_data.is_finished():
            try:
                dataOk, _, detObj, kinectJoints = sensor_data.get_data2()
                frames += 1
                time_start = 0
                time_end = 0
                if dataOk:
                    if first_iter[0]:  # Access the mutable value
                        trackbuffer.dt = SLEEPTIME
                        first_iter[0] = False  # Update the mutable value
                    else:
                        trackbuffer.dt = detObj["posix"][0] / 1000 - trackbuffer.t

                    trackbuffer.t = detObj["posix"][0] / 1000
                    # Apply scene constraints, point translation and axis normalization
                    effective_data = normalize_data(detObj, keepRadial=polarExperiment)

                    if effective_data.shape[0] != 0:
                        # Tracking module
                        time_start = time.time()
                        trackbuffer.track(effective_data, batch)
                        time_end = time.time()
                        timeToTrack.append((time_end - time_start) * 1000)
                        # Posture Estimation module
                        
        
                        # trackbuffer.estimate_posture(model)
                        # if len(trackbuffer.effective_tracks) > 0:
                        #     print("Estimated posture: ", trackbuffer.effective_tracks[0].keypoints)
                        centralValues=trackbuffer.update_real_posture(kinectJoints)
                        
                        for centroid, joint0 in centralValues:
                            for track in trackbuffer.effective_tracks:
                                if polarExperiment:
                                    state_cartesian = polar_to_cartesian(track.state.x.flatten())
                                    state_position = state_cartesian[:2]  # [x, y]
                                    error_centroid = np.linalg.norm(centroid[:2] - state_position)
                                    accumulated_errors["centroid"].append(error_centroid)
                                    error_joint0 = np.linalg.norm(joint0[:2] - state_position)
                                    accumulated_errors["joint0"].append(error_joint0)
                                else:
                                    error = np.linalg.norm(centroid - track.state.x[:3].flatten())
                                    accumulated_errors["centroid"].append(error)
                                    error = np.linalg.norm(joint0 - track.state.x[:3].flatten())
                                    accumulated_errors["joint0"].append(error)
                                

                                # error = np.linalg.norm(centroid[:2] - track.state.x[:2].flatten())
                                # accumulated_errors["centroid"].append(error)
                                # error = np.linalg.norm(joint0[:2] - track.state.x[:2].flatten())
                                # accumulated_errors["joint0"].append(error)

                    visual.update(trackbuffer, detObj)
                if frames % 10 == 0:
                    print(f"Frame: {frames}")
                    print("Tracking time: ", (time_end - time_start) * 1000, "ms")

                if frames >= 1000:
                    cleanup()
                else:
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