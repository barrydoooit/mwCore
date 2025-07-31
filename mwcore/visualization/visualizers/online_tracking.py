import os

from .online_pointcloud import OnlinePointCloudVisualizer
# os.environ["QT_QPA_PLATFORM"] = "xcb"
from typing import List, Literal, Optional, Union
import numpy as np
# from PySide6.QtCore import QThread, QObject, Signal, QTimer
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from ..plot_3d import Plot3D  # Reuse the Plot3D class from plot_3d.py
from mwcore.registry import VISUALIZERS



@VISUALIZERS.register_module()
class OnlineTrackingVisualizer(OnlinePointCloudVisualizer):    
    def __init__(
        self,
        parent=None,
        tracking_mode: Optional[Literal['dot', 'bbox']] = None,
        receive_polar: bool = False,
        on_close: callable = None
    ):
        super(OnlineTrackingVisualizer, self).__init__(parent=parent, on_close=on_close)
        # -- Tracking visualization configuration --
        self.tracking_mode = tracking_mode
        self.receive_polar = receive_polar
        
        # A small palette of RGBA colors we’ll cycle through:
        self._color_palette = [
            (1.0, 0.0, 0.0, 1.0),   # red
            (0.0, 1.0, 0.0, 1.0),   # green
            (0.0, 0.0, 1.0, 1.0),   # blue
            (1.0, 1.0, 0.0, 1.0),   # yellow
            (1.0, 0.0, 1.0, 1.0),   # magenta
            (0.0, 1.0, 1.0, 1.0),   # cyan
            (1.0, 0.5, 0.0, 1.0),   # orange
            (0.5, 0.0, 1.0, 1.0),   # purple
        ]
        
        if self.tracking_mode == "dot":
            # Create an (initially empty) scatter item; we'll give it all points/colors in update_tracking
            self.tracker_dot = gl.GLScatterPlotItem(
                pos=np.zeros((0, 3)), 
                size=20, 
                color=np.zeros((0, 4))
            )
            self.plot3d.plot_3d.addItem(self.tracker_dot)
        
        elif self.tracking_mode == "bbox":
            # We'll maintain a list of GLLinePlotItems—one per tracked object
            self.tracker_boxes: List[gl.GLLinePlotItem] = []
        
        self._on_close = on_close

    def update_tracking(
        self, 
        location: Union[np.ndarray, List[List[float]]], 
        trans_matrix: Optional[np.ndarray] = None
    ):
        """
        Now plots *all* tracked locations in a single call.
        - If multiple points are given, we reshape into an (N,≥3) array.
        - Each point gets a color from self._color_palette, cycling if needed.
        """
        # Handle None or invalid inputs
        if location is None:
            return
     

        # Convert `location` into a numpy array `loc_arr` of shape (N, ≥3)
        if isinstance(location, np.ndarray):
            arr = location.astype(float)
            if arr.ndim == 1 and arr.size >= 3:
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
            if loc_arr.ndim != 2 or loc_arr.shape[1] < 3 or loc_arr.shape[0] == 0:
                return
        else:
            return

           
        if self.receive_polar:
            cartesian_coords = []
            for p in loc_arr:
                r, theta, r_dot = p[:3]
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = 0.0
                cartesian_coords.append(np.array([x, y, z]))
            loc_arr = np.vstack(cartesian_coords)
            
        # Apply the same transformation (if provided) to all points
        if trans_matrix is not None and trans_matrix.shape == (4, 4):
            ones = np.ones((loc_arr.shape[0], 1), dtype=float)
            hom = np.hstack([loc_arr[:, :3], ones])  # (N,4)
            transformed = (trans_matrix @ hom.T).T  # (N,4)
            loc_arr = transformed[:, :3]

        # At this point, loc_arr is shape (N, 3)
        num_objs = loc_arr.shape[0]
        if num_objs == 0:
            return

        if self.tracking_mode == "dot":
            # For dot mode: send *all* points + an (N,4) array of colors
            centroids = loc_arr[:, :3]  # shape (N,3)

            # Build an (N,4) color array by cycling through the palette
            color_array = np.zeros((num_objs, 4), dtype=float)
            palette = self._color_palette
            P = len(palette)
            for i in range(num_objs):
                color_array[i, :] = palette[i % P]

            # Update the scatter item
            self.tracker_dot.setData(pos=centroids, color=color_array, size=20)

        elif self.tracking_mode == "bbox":
            # First: remove any existing boxes from the scene
            for existing_box in getattr(self, 'tracker_boxes', []):
                self.plot3d.plot_3d.removeItem(existing_box)
            self.tracker_boxes = []

            # For each tracked object, create one GLLinePlotItem
            palette = self._color_palette
            P = len(palette)
            bbox_size = np.array([0.5, 0.5, 0.5])  # fixed size (in meters)

            for idx in range(num_objs):
                cx, cy, cz = loc_arr[idx, :3]
                dx, dy, dz = bbox_size / 2.0

                # 8 vertices of a cube centered at (cx, cy, cz)
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

                # 12 edges as pairs of vertex indices
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # left face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # right face
                    (0, 4), (1, 5), (2, 6), (3, 7)   # connectors
                ]

                # Build a (24,3) array where each pair of rows is one line segment
                lines = []
                for (v0, v1) in edges:
                    lines.append(vertices[v0])
                    lines.append(vertices[v1])
                concatenated_lines = np.vstack(lines)  # shape (24,3)

                # Pick a color from palette
                color = palette[idx % P]

                # Create & add a new box item
                box_item = gl.GLLinePlotItem(
                    pos=concatenated_lines,
                    color=color,
                    width=2,
                    antialias=True,
                    mode='lines'
                )
                self.plot3d.plot_3d.addItem(box_item)
                self.tracker_boxes.append(box_item)


    def closeEvent(self, event):
        if self._on_close:
            self._on_close(event)
        event.accept()