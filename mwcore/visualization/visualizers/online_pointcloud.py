import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
from typing import List, Literal, Optional, Union
import numpy as np
# from PySide6.QtCore import QThread, QObject, Signal, QTimer
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from ..plot_3d import Plot3D  # Reuse the Plot3D class from plot_3d.py
from mwcore.registry import VISUALIZERS



@VISUALIZERS.register_module()
class OnlinePointCloudVisualizer(QMainWindow):    
    def __init__(
        self,
        parent=None,
        on_close: callable = None
    ):
        super(OnlinePointCloudVisualizer, self).__init__(parent)
        self.setWindowTitle("Interactive Radar Point Cloud Visualizer")
        self.resize(800, 600)

        # Instantiate the Plot3D widget from the existing infrastructure
        self.plot3d = Plot3D()
        self.setCentralWidget(self.plot3d.plot_3d)
        
        self._on_close = on_close

    def update_point_cloud(
        self, 
        point_cloud: np.ndarray,
        trans_matrix: Optional[np.ndarray] = None
    ):
        # Update the scatter plot in Plot3D with the new point cloud data
        pts = point_cloud.copy()
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        if trans_matrix is not None and trans_matrix.shape == (4, 4):
            ones_col = np.ones((pts.shape[0], 1), dtype=float)
            hom_coords = np.hstack([pts, ones_col])         # (N,4)
            transformed_hom = (trans_matrix @ hom_coords.T).T
            pts = transformed_hom[:, :3]
        self.plot3d.scatter.setData(pos=pts)

    def closeEvent(self, event):
        if self._on_close:
            self._on_close(event)
        event.accept()