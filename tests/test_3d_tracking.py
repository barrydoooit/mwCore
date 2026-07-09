import numpy as np
import pytest
from mwcore.tracking.api.trackers.gtrack_asterios import GTrackATracker

def generate_synthetic_torso_frame(center, num_points=20, spread=0.3):
    """
    Generates a noisy cluster of points representing a human torso.
    Format required by tracking base: [x, y, z, doppler, peakVal]
    """
    points = np.random.normal(loc=0.0, scale=spread, size=(num_points, 5))
    # Apply center offset to x, y, z
    points[:, 0] += center[0]
    points[:, 1] += center[1]
    points[:, 2] += center[2]
    # Synthetic doppler and peakVal
    points[:, 3] = np.random.uniform(-1.0, 1.0, size=num_points)
    points[:, 4] = np.random.uniform(50, 100, size=num_points)
    
    # Add a few extreme random noise points (outliers) to test the SOR filter
    outliers = np.random.uniform(-5.0, 5.0, size=(5, 5))
    outliers[:, 3] = 0.0
    outliers[:, 4] = 10.0
    
    return np.vstack((points, outliers))

def test_asterios_torso_tracking():
    # Tracker parameters mimicking a real TI sensor configuration
    tracker_params = {
        "FB_FRAMES_BATCH": 1,
        "ENABLE_TORSO_TRACKING": True,
        "KF_Q_STD": 0.5,
        "TR_GATE": 1.0,
        "TR_MAX_TRACKS": 5,
        "DB_EPS": 0.5,
        "DB_MIN_SAMPLES_MIN": 2,
        "DB_POINTS_THRES": 10,
        "DB_RANGE_WEIGHT": 0.03,
        "DB_Z_WEIGHT": 0.4,
        "TR_VEL_THRES": 0.5,
        "KF_GROUP_DISP_EST_INIT": 5.0,
        "KF_P_INIT": 5.0,
        "KF_R_STD": 0.1,
        "TR_LIFETIME_DYNAMIC": 5,
        "TR_LIFETIME_STATIC": 5
    }
    
    # Sensor height configuration required by BaseTracker if transforming
    radar_cfg = {
        "sensor_height": 1.5,
        "sensor_tilt": 0.0
    }
    
    # Instantiate tracker with SOR filter ENABLED
    tracker = GTrackATracker(keep_radial=False, tracker_params=tracker_params, radar_cfg=radar_cfg)
    
    # Simulate a person walking in a straight line for 5 frames
    trajectory = [
        [1.0, 2.0, 0.0],
        [1.0, 2.2, 0.0],
        [1.0, 2.4, 0.0],
        [1.0, 2.6, 0.0],
        [1.0, 2.8, 0.0]
    ]
    
    tracked_states = []
    
    for i, center in enumerate(trajectory):
        # Generate synthetic frame with noise
        frame_points = generate_synthetic_torso_frame(center)
        
        # We manually monkey-patch filter_outliers directly through the consume dictionary if kwargs allow,
        # but since consume doesn't explicitly pass kwargs to normalize_data yet, we'll patch the instance method
        # for testing purposes or assert behavior directly.
        # Let's call normalize_data separately to verify SOR logic.
        
        # 1. Test SOR Filter mathematically
        normalized_with_filter = tracker.normalize_data(point_array=frame_points, filter_outliers=True, sor_neighbors=5, sor_std_ratio=1.0)
        normalized_without_filter = tracker.normalize_data(point_array=frame_points, filter_outliers=False)
        
        # Outliers should be dropped
        assert len(normalized_with_filter) < len(normalized_without_filter), "SOR filter failed to drop outliers."
        
        # 2. Consume data in Tracker
        states = tracker.consume(point_array=normalized_with_filter)
        
        if states:
            # Get primary tracked state (Centroid)
            tracked_states.append(states[0])
            
    # Allow a few frames for the Kalman Filter to initialize and establish a Track
    assert len(tracked_states) >= 2, "Tracker failed to initialize and maintain track on the synthetic torso."
    
    # Check if the final tracked state is near the final trajectory point
    final_state = tracked_states[-1]
    final_true_pos = trajectory[-1]
    
    # Calculate Euclidean distance between tracker state [x, y, z] and true position
    error = np.linalg.norm(final_state[:3] - final_true_pos)
    assert error < 1.5, f"Tracker lost the target. Error {error} is too high."
