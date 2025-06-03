import sys
import time
import numpy as np
from .visualizer import MainVisualizer
from PySide2.QtCore import QThread, Signal
from PySide2.QtWidgets import QApplication


class DataUpdateThread(QThread): 
    new_data = Signal(np.ndarray)

    def run(self):
        # Simulate continuous point cloud data updates
        while not self.isInterruptionRequested():
            # Generate a simulated point cloud of 1000 points in 3D space
            point_cloud = np.random.uniform(-5, 5, (1000, 3))
            self.new_data.emit(point_cloud)
            time.sleep(0.1)  # update at approximately 10 Hz
            
if __name__ == "__main__":
    app = QApplication(sys.argv) 
    data_thread = DataUpdateThread()
    def on_close(event):
        data_thread.requestInterruption()
        data_thread.wait()

    main_window = MainVisualizer(new_data=data_thread.new_data, on_close=on_close)
    data_thread.start()
    main_window.show()
    sys.exit(app.exec_())