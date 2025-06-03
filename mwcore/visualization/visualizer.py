from typing import List, Literal, Optional, Union
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
    
    def __init__(self, parent=None, tracking_mode: Optional[Literal['dot', 'bbox']] = None, on_close: callable = None): 
        super(MainVisualizer, self).__init__(parent)
        self.setWindowTitle("Interactive Radar Point Cloud Visualizer")
        self.resize(800, 600)

        # Instantiate the Plot3D widget from the existing infrastructure
        self.plot3d = Plot3D()
        self.setCentralWidget(self.plot3d.plot_3d)
        
        # -- Tracking visualization configuration --
        self.tracking_mode = tracking_mode
        if self.tracking_mode is not None and self.tracking_mode in ['dot', 'bbox']:
        # Create a large red dot for dot mode
            if self.tracking_mode == "bbox":
                # Create a bounding box for bbox mode
                self.tracker_box = None
            elif self.tracking_mode == "dot":
                self.tracker_dot = gl.GLScatterPlotItem(size=20, color=(1, 0, 0, 1))
                self.tracker_dot.setData(pos=np.zeros((1, 3)))
                self.plot3d.plot_3d.addItem(self.tracker_dot)
        
        self._on_close = on_close

    def update_point_cloud(self, 
                           point_cloud,
                           trans_matrix: Optional[np.ndarray] = None):
        # Update the scatter plot in Plot3D with the new point cloud data
        if point_cloud.shape[1] > 3:
            point_cloud = point_cloud[:, :3]
        if trans_matrix is not None and trans_matrix.shape == (4, 4):
            ones_col = np.ones((point_cloud.shape[0], 1), dtype=float)
            hom_coords = np.hstack([point_cloud, ones_col])  # shape (N, 4)
            transformed_hom = (trans_matrix @ hom_coords.T).T  # shape (N, 4)
            point_cloud = transformed_hom[:, :3]
            
        self.plot3d.scatter.setData(pos=point_cloud)
        
    def update_tracking(self, 
                        location: Union[np.ndarray, List[List[float]]], 
                        trans_matrix: Optional[np.ndarray] = None):
        """
        Slot that receives the tracking result (location) emitted by the tracking thread.
        
        Parameters:
        - location: A numpy array or list of locations; each location is assumed to have at least
        three components (x, y, z). For simplicity, this example uses the first location if multiple are provided.
        """
        # In this example, we only handle the first tracked location
        if location is None:
            return
        
        # Convert `location` into a 2D numpy array `loc_arr` of shape (N, ≥3)
        if isinstance(location, np.ndarray):
            arr = location.astype(float)
            if arr.ndim == 1 and arr.size >= 3:
                # Single point: reshape to (1, ≥3)
                loc_arr = arr.reshape(1, -1)
            elif arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] > 0:
                loc_arr = arr
            else:
                return

        elif isinstance(location, list):
            if len(location) == 0:
                return
            try:
                loc_arr = np.vstack([np.asarray(pt, dtype=float) for pt in location])
            except Exception:
                return
            # Now loc_arr is (N, M). Ensure M ≥ 3 and N ≥ 1
            if loc_arr.ndim != 2 or loc_arr.shape[1] < 3 or loc_arr.shape[0] == 0:
                return
        else:
            return


        if trans_matrix is not None and trans_matrix.shape == (4, 4):
            # Build homogeneous coordinates for all points: shape (N, 4)
            ones_col = np.ones((loc_arr.shape[0], 1), dtype=float)
            hom_coords = np.hstack([loc_arr[:, :3], ones_col])  # (N, 4)

            # Apply transform: (4×4) × (4×N) → (4×N), then transpose to (N, 4)
            transformed_hom = (trans_matrix @ hom_coords.T).T
            # Discard the homogeneous w-component
            loc_arr = transformed_hom[:, :3]

        # Visualize only the first point (centroid) as before
        centroid = loc_arr[0, :3].reshape((1, 3))
        
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