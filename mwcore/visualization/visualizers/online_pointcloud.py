import os
from typing import List, Literal, Optional, Union
import numpy as np
from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import Slot, Qt, QTimer, QElapsedTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from ..plot_3d import Plot3D 
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
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar stretches label to the right
        top_bar = QWidget()
        top_bar.setFixedHeight(24)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(5, 5, 5, 0)
        top_layout.addStretch()

        # Info label with white background
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "background-color: white; padding: 2px; border: 1px solid #ccc;"
        )
        top_layout.addWidget(self.info_label)

        # Assemble: top bar + GL widget
        main_layout.addWidget(top_bar)
        main_layout.addWidget(self.plot3d.plot_3d)

        self.setCentralWidget(container)
        
        self._on_close = on_close

    @Slot(np.ndarray)
    def on_new_cloud(self, points: np.ndarray):
        # Store the newest data, toss whatever was there before.
        self._pending_cloud = points.copy()
        # If no update is scheduled, schedule one ASAP.
        if not getattr(self, "_update_scheduled", False):
            self._update_scheduled = True
            QTimer.singleShot(0, self._flush_pending_cloud)

    def _flush_pending_cloud(self):
        # Called in the GUI thread. Grab & clear the buffer.
        pts = self._pending_cloud
        self._update_scheduled = False
        self.update_point_cloud(pts)

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

    @Slot(str)
    def update_label(self, text: str):
        self.info_label.setText(text)
    
    def closeEvent(self, event):
        if self._on_close:
            self._on_close(event)
        event.accept()