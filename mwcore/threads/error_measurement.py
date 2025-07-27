import numpy as np
import time
import logging
from PySide6.QtCore import QThread, Signal
from pathlib import Path
import json
import csv
import os

from mwcore.registry import THREADS

log = logging.getLogger(__name__)

@THREADS.register_module()
class ErrorMeasurementThread(QThread):
    """Thread to calculate error metrics between ground truth and tracking data"""
    
    error_data = Signal(object)  # Will emit error metrics

    def __init__(self, stats_dir='error_stats', tracker_name=None, dataset_name=None, error_cfg=None):
        super().__init__()
        self.ground_truth = None
        self.tracking_data = None
        self.metrics_history = {
            'position_error': [],
            'timestamps': [],
            'frame_numbers': [],
        }
        self.frame_count= 0
        self.last_calculation_time = 0
        self.calculation_interval = 0.07  # Calculate metrics every 70ms
        
        self.polar = error_cfg.get("polar", False) if error_cfg else False
        self.save_stats = error_cfg.get("save_stats", True) if error_cfg else False
        self.full_metrics = error_cfg.get("full_metrics", False) if error_cfg else False
        # self.stats_dir = Path(stats_dir)
        self.stats_dir =  Path(error_cfg.get("stats_dir", stats_dir)) if error_cfg else Path(stats_dir)
        self.experiment_name = error_cfg.get("experiment_name", "default_experiment") if error_cfg else "default_experiment"
        self.tracker_name = tracker_name
        self.dataset_name = dataset_name
        if self.save_stats and not self.stats_dir.exists():
            self.stats_dir.mkdir(parents=True, exist_ok=True)
            
        log.info("Initialized ErrorMeasurementThread")
    
    
    def update_ground_truth(self, ground_truth):
        """Update the latest ground truth data"""
        self.ground_truth = ground_truth
        self._calculate_metrics()
        self.frame_count += 1

    def update_tracking(self, tracking_data):
        """Update the latest tracking data"""
        self.tracking_data = tracking_data
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate error metrics if both ground truth and tracking data are available"""
        current_time = time.time()
        if current_time - self.last_calculation_time < self.calculation_interval:
            return  # Don't calculate too frequently
            
        if self.ground_truth is None or self.tracking_data is None:
            print("Waiting for both ground truth and tracking data to be available")    
            return  # Need both data sources
        
        # Calculate position error (Euclidean distance)
        try:
            # Extract positions from ground truth and tracking data


            gt_position = self._extract_position_from_ground_truth(self.ground_truth, type='spine')
            tracking_position = self._extract_position_from_tracking(self.tracking_data)

            # print(f"Ground Truth Position: {gt_position}")
            # print(f"Tracking Data Position: {tracking_position}")
            
            if gt_position is not None and tracking_position is not None:
                # Calculate Euclidean distance
                if self.polar:
                    position_error = np.linalg.norm(gt_position[:2] - tracking_position[:2])
                else:
                    position_error = np.linalg.norm(gt_position - tracking_position)
                print(gt_position, tracking_position)
                
                # Store error metrics
                self.metrics_history['position_error'].append(position_error)
                self.metrics_history['timestamps'].append(current_time)
                self.metrics_history['frame_numbers'].append(self.frame_count)
                
                # Calculate current metrics
                metrics = {
                    'current_error': position_error,
                    'avg_error': np.mean(self.metrics_history['position_error']),
                    'max_error': np.max(self.metrics_history['position_error']),
                    'min_error': np.min(self.metrics_history['position_error']),
                    'frame': self.frame_count
                }
                
                # Emit metrics
                self.error_data.emit(metrics)
                self.last_calculation_time = current_time
        
        except Exception as e:
            log.error(f"Error calculating metrics: {str(e)}")
    
    def _extract_position_from_ground_truth(self, ground_truth, type='spine'):
        """Extract position from ground truth data"""
        if type not in ['spine', 'centroid']:
            log.error(f"Invalid type '{type}' for ground truth extraction")
            return None
        try:
            dataset = self.dataset_name.lower() if self.dataset_name else 'unknown'
            tracker = self.tracker_name.lower() if self.tracker_name else 'unknown'

            if len(ground_truth) > 0:
                if isinstance(ground_truth, np.ndarray):
                    # Assuming ground_truth is a 2D array with shape (N, 3) for N points
                    raise NotImplementedError("Ground truth as ndarray not implemented yet")
                elif isinstance(ground_truth, list):
                    if type == 'spine':
                        if dataset.find('asterios') != -1:
                            first_gt = ground_truth[0]
                            reshaped_data = np.array(first_gt).reshape((3, -1))
                            # mirror on 0 axis
                            reshaped_data[0, :] = -reshaped_data[0, :]
                            return np.array([reshaped_data[0, 0], reshaped_data[2, 0], reshaped_data[1, 0]])
                        elif dataset.find('mri') != -1:
                            reshaped_data = np.array(ground_truth).reshape((3, -1))  
                            to_consider = [11, 12, 5, 6]  # HipLeft, HipRight, Left shoulder, Right shoulder
                            extra_point = reshaped_data[:, to_consider]                             
                            mean_spine = np.mean([extra_point[0, :], extra_point[2, :], extra_point[1, :]], axis=1)
                            mean_spine[0] = -mean_spine[0]  # Mirror x coordinate
                            return mean_spine.flatten()
                        elif dataset.find('mili') != -1:
                            reshaped_data = np.array(ground_truth).reshape((3, -1))  
                            to_consider = [1, 8, 11]
                            extra_point = reshaped_data[:, to_consider]
                            mean_spine = np.mean([extra_point[0, :], extra_point[2, :], extra_point[1, :]], axis=1)
                            return mean_spine.flatten()
                    elif type == 'centroid':
                        return np.mean(np.array(reshaped_data), axis=0).flatten()
            return None
        except Exception as e:
            log.error(f"Error extracting ground truth position: {str(e)}")
            return None
    
    def _extract_position_from_tracking(self, tracking_data):
        """Extract position from tracking data"""
        try:
            if isinstance(tracking_data, np.ndarray) and tracking_data.size >= 3:
                if self.polar:
                    # Assuming polar coordinates are in the first three columns
                    r, theta, r_speed = tracking_data[:3]
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    return np.array([x, y, r_speed])
                return tracking_data[:3].flatten()
            elif isinstance(tracking_data, list) and len(tracking_data) > 0:
                if self.polar:
                    # Assuming polar coordinates are in the first three elements
                    r, theta, r_speed = tracking_data[0][:3]
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    return np.array([x, y, r_speed])
                return np.array(tracking_data[0][:3])
            return None
        except Exception as e:
            log.error(f"Error extracting tracking position: {str(e)}")
            return None
        
    def generate_statistics_report(self):
        """Generate a comprehensive statistics report"""
        if len(self.metrics_history['position_error']) == 0:
            return "No error data collected"
            
        errors = np.array(self.metrics_history['position_error'])
        
        stats = {
            "total_frames": self.frame_count,
            "error_samples": len(errors),
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_dev": float(np.std(errors)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "error_percentiles": {
                "25%": float(np.percentile(errors, 25)),
                "50%": float(np.percentile(errors, 50)),
                "75%": float(np.percentile(errors, 75)),
                "90%": float(np.percentile(errors, 90)),
                "95%": float(np.percentile(errors, 95)),
                "99%": float(np.percentile(errors, 99))
            }
        }
        
        # Save statistics to file
        if self.save_stats:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            dataset_name = self.dataset_name or "unknown_dataset"
            tracker_name = self.tracker_name or "unknown_tracker"
            stats_file = self.stats_dir / f"{tracker_name}{dataset_name}_stats_{timestamp}.json"
            error_data_file = self.stats_dir / f"{tracker_name}{dataset_name}_error_data_{timestamp}.csv"
            master_csv_file = self.stats_dir / "master_stats.csv"

            if self.full_metrics:
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)

            # Prepare flat row for master CSV
            row = {
                "timestamp": timestamp,
                "tracker_name": tracker_name,
                "dataset_name": dataset_name,
                "experiment_name": self.experiment_name,
                "total_frames": stats["total_frames"],
                "error_samples": stats["error_samples"],
                "mean_error": stats["mean_error"],
                "median_error": stats["median_error"],
                "std_dev": stats["std_dev"],
                "min_error": stats["min_error"],
                "max_error": stats["max_error"],
                "percentile_25%": stats["error_percentiles"]["25%"],
                "percentile_50%": stats["error_percentiles"]["50%"],
                "percentile_75%": stats["error_percentiles"]["75%"],
                "percentile_90%": stats["error_percentiles"]["90%"],
                "percentile_95%": stats["error_percentiles"]["95%"],
                "percentile_99%": stats["error_percentiles"]["99%"],
            }

            # Write header only if file does not exist
            write_header = not os.path.exists(master_csv_file) or os.path.getsize(master_csv_file) == 0
            with open(master_csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

            # Save raw error data for plotting
            if self.full_metrics:
                with open(error_data_file, 'w') as f:
                    f.write("frame,timestamp,error\n")
                    for frame, ts, err in zip(
                        self.metrics_history['frame_numbers'],
                        self.metrics_history['timestamps'],
                        self.metrics_history['position_error']
                    ):
                        f.write(f"{frame},{ts},{err}\n")

            if self.full_metrics:
                log.info(f"Saved error statistics to {stats_file}")
                log.info(f"Saved raw error data to {error_data_file}")
            log.info(f"Appended summary to {master_csv_file}")
        
        # Format for printing
        report = "\n" + "="*50 + "\n"
        report += "             TRACKING ERROR STATISTICS             \n"
        report += "="*50 + "\n"
        report += f"Total frames processed: {stats['total_frames']}\n"
        report += f"Error samples collected: {stats['error_samples']}\n\n"
        report += f"Mean error:      {stats['mean_error']:.4f} meters\n"
        report += f"Median error:    {stats['median_error']:.4f} meters\n"
        report += f"Standard dev:    {stats['std_dev']:.4f} meters\n"
        report += f"Min error:       {stats['min_error']:.4f} meters\n"
        report += f"Max error:       {stats['max_error']:.4f} meters\n\n"
        report += "Error Percentiles:\n"
        report += f"  25%:           {stats['error_percentiles']['25%']:.4f} meters\n"
        report += f"  50%:           {stats['error_percentiles']['50%']:.4f} meters\n"
        report += f"  75%:           {stats['error_percentiles']['75%']:.4f} meters\n"
        report += f"  90%:           {stats['error_percentiles']['90%']:.4f} meters\n"
        report += f"  95%:           {stats['error_percentiles']['95%']:.4f} meters\n"
        report += f"  99%:           {stats['error_percentiles']['99%']:.4f} meters\n"
        report += "="*50 + "\n"
        
        if self.save_stats:
            report += f"Statistics saved to {stats_file}\n"
            report += f"Raw data saved to {error_data_file}\n"
        
        return report
        
    def run(self):
        log.info("Starting ErrorMeasurementThread")
        while not self.isInterruptionRequested():
            time.sleep(0.1)  # Just sleep and wait for signal connections
        # When interruption is requested, print stats and exit
        report = self.generate_statistics_report()
        print(report)