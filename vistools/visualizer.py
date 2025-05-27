import sys
import numpy as np
from PySide2.QtCore import QThread, QObject, Signal, QTimer
from PySide2.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from .plot_3d import Plot3D  # Reuse the Plot3D class from plot_3d.py



class VisualizerWorker(QObject):
    update_cloud = Signal(np.ndarray)
    
    def __init__(self, plot_3d):
        super().__init__()
        self.plot_3d = plot_3d
        self.update_cloud.connect(self._update_scatter)
    
    def _update_scatter(self, point_cloud):
        self.plot_3d.scatter.setData(pos=point_cloud)
        
        QApplication.processEvents()

class MainVisualizer(QMainWindow): 
    
    def __init__(self, parent=None, new_data: Signal = None, tracking_mode='dot', on_close: callable = None): 
        super(MainVisualizer, self).__init__(parent)
        self.setWindowTitle("Interactive Radar Point Cloud Visualizer")
        self.resize(800, 600)

        # Instantiate the Plot3D widget from the existing infrastructure
        self.plot3d = Plot3D()
        self.setCentralWidget(self.plot3d.plot_3d)

        # Connect raw point cloud data to update method
        if new_data is not None:
            new_data.connect(self.update_point_cloud)
        
        # -- Tracking visualization configuration --
        # Choose how to render the tracked location: "dot" or "bbox"
        self.tracking_mode = tracking_mode
        
        # Create a large red dot for dot mode
        self.tracker_dot = gl.GLScatterPlotItem(size=20, color=(1, 0, 0, 1))
        self.tracker_dot.setData(pos=np.zeros((1, 3)))
        self.plot3d.plot_3d.addItem(self.tracker_dot)
        
        # For bbox mode, the bounding box marker will be created on demand.
        self.tracker_box = None
        
        self._on_close = on_close

    def update_point_cloud(self, point_cloud):
        # Update the scatter plot in Plot3D with the new point cloud data
        if point_cloud.shape[1] > 3:
            point_cloud = point_cloud[:, :3]
        self.plot3d.scatter.setData(pos=point_cloud)
        
    def update_tracking(self, location, to_standard=True):
        """
        Slot that receives the tracking result (location) emitted by the tracking thread.
        
        Parameters:
        - location: A numpy array or list of locations; each location is assumed to have at least
        three components (x, y, z). For simplicity, this example uses the first location if multiple are provided.
        """
        # In this example, we only handle the first tracked location
        if location is None:
            return
        if isinstance(location, list) and len(location) > 0:
            loc = location[0]
        elif isinstance(location, np.ndarray):
            loc = location
        else:
            return
        
        # Ensure we have at least three values for x, y, z
        if len(loc) < 3:
            return
        if to_standard:
            # loc[1] -= 160
            pass # TODO: change to standard coordinates
        
        centroid = np.array(loc[:3]).reshape((1, 3))
        
        if self.tracking_mode == "dot":
            self.tracker_dot.setData(pos=centroid, size=20, color=(1, 0, 0, 1))
        elif self.tracking_mode == "bbox":
            # For bounding box mode, define a fixed size or compute dynamically
            bbox_size = np.array([0.5, 0.5, 0.5])  # fixed size (in meters)
            cx, cy, cz = centroid.flatten()
            dx, dy, dz = bbox_size / 2.0
            # Define the 8 vertices of a cube centered at the centroid
            vertices = np.array([
                [cx - dx, cy - dy, cz - dz],
                [cx - dx, cy - dy, cz + dz],
                [cx - dx, cy + dy, cz + dz],
                [cx - dx, cy + dy, cz - dz],
                [cx + dx, cy - dy, cz - dz],
                [cx + dx, cy - dy, cz + dz],
                [cx + dx, cy + dy, cz + dz],
                [cx + dx, cy + dy, cz - dz]
            ])
            # Define edges as pairs of vertex indices
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # left face
                (4, 5), (5, 6), (6, 7), (7, 4),  # right face
                (0, 4), (1, 5), (2, 6), (3, 7)   # connectors
            ]
            # Collect line segments for each edge
            lines = []
            for edge in edges:
                lines.append(vertices[list(edge), :])
            concatenated_lines = np.concatenate(lines, axis=0)
            if self.tracker_box is None:
                self.tracker_box = gl.GLLinePlotItem()
                self.tracker_box.setData(pos=concatenated_lines, color=pg.glColor('r'), width=2, antialias=True, mode='lines')
                self.plot3d.plot_3d.addItem(self.tracker_box)
            else:
                self.tracker_box.setData(pos=concatenated_lines, color=pg.glColor('r'), width=2, antialias=True, mode='lines')
            
    def closeEvent(self, event):
        if self._on_close:
            self._on_close(event)
        event.accept()