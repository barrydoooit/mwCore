from typing import Iterable, Iterator, List, Literal, Optional, Union
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit)
from PySide6.QtCore import QTimer, Qt
import numpy as np
import sys
import pyqtgraph.opengl as gl
from mwcore.visualization.vis_utils import _draw_bboxes
from mwcore.visualization.plot_3d import Plot3D
import os
from PySide6 import QtCore
import warnings



class PointCloudOfflineVisualizer(QMainWindow):
    def __init__(self, 
                 point_clouds: Union[List[np.ndarray], Iterator[np.ndarray]],
                 tracking_data: Optional[Union[List[np.ndarray], Iterator[np.ndarray]]] = None,
                 total_frames: Optional[int] = None,
                 play_fps: float = 5,
                 tracking_mode: Optional[Literal['dot', 'bbox']] = None,
                 parent=None):
        warnings.warn(
            "`PointCloudOfflineVisualizer` is deprecated and will be moved to `mwcore` in version 0.1.1.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(parent)
        self.setWindowTitle("Offline Point Cloud Visualizer")
        self.resize(800, 600)
        
        # ------------ Handle point clouds ------------
        if isinstance(point_clouds, list):
            self._is_list = True
            self._pc_cache: List[np.ndarray] = point_clouds
            self._pc_iter: Optional[Iterator[np.ndarray]] = None
            self.num_frames = len(point_clouds)
        elif isinstance(point_clouds, Iterable):
            self._is_list = False
            self._pc_cache: List[np.ndarray] = []
            self._pc_iter: Iterator[np.ndarray] = iter(point_clouds)
            self.num_frames = total_frames
        else:
            raise ValueError("point_clouds must be a list or iterable of numpy arrays")
        
        # ------- Handle optional tracking data -------
        self.tracking_mode = tracking_mode
        if tracking_data is not None:
            self._has_tracking = True
            if isinstance(tracking_data, list):
                self._trk_is_list = True
                self._trk_cache: List[np.ndarray] = tracking_data
                self._trk_iter = None
            elif isinstance(tracking_data, Iterable):
                self._trk_is_list = False
                self._trk_cache: List[np.ndarray] = []
                self._trk_iter: Iterator[np.ndarray] = iter(tracking_data)
            else:
                raise TypeError("tracking_data must be a list or iterable of numpy arrays")
        else:
            self._has_tracking = False
            self._trk_cache = []
            self._trk_iter = None
            self._trk_is_list = True
            self.tracking_mode = None

        self.current_frame = 0
        self.play_fps = play_fps
        self.playing = False
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(int(1000 / self.play_fps))
        
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.plot3d = Plot3D()
        layout.addWidget(self.plot3d.plot_3d)
         
        # -- Set up tracking visuals if requested --
        if self._has_tracking and self.tracking_mode == "dot":
            # A scatter plot for tracked centroids
            self._trk_scatter = gl.GLScatterPlotItem(
                pos=np.zeros((0, 3)), size=20, color=np.zeros((0, 4))
            )
            self.plot3d.plot_3d.addItem(self._trk_scatter)
        elif self._has_tracking and self.tracking_mode == "bbox":
            # We'll fill this list with GLLinePlotItem boxes each frame
            self._trk_boxes: List[gl.GLLinePlotItem] = []

        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.play_btn = QPushButton("PLAY")
        self.prev_btn = QPushButton("PREV")
        self.next_btn = QPushButton("NEXT")
             # --- Go-to-frame controls ---
        self.goto_field = QLineEdit()
        self.goto_field.setFixedWidth(60)
        self.goto_field.setPlaceholderText("Frame #")
        self.goto_btn = QPushButton("Go")
    
        # connect
        self.goto_btn.clicked.connect(self.go_to_frame)
        self.goto_field.returnPressed.connect(self.go_to_frame)
    
        controls.addWidget(self.play_btn)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.goto_field)
        controls.addWidget(self.goto_btn)
        controls.addWidget(self.next_btn)
        
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        
        self.play_key = Qt.Key_Space
        self.prev_key = Qt.Key_A
        self.next_key = Qt.Key_D
        
    def initilaize(self):
        self._load_frame(0)
        self.update_display()
    
    def show(self):
        super().show()
        self.initilaize()
        
    def _load_frame(self, index: int) -> None:
        """
        Ensure the frame at `index` is loaded into cache for both 
        point clouds and tracking (if enabled).
        """
        # ---- Load point cloud ----
        if not self._is_list:
            if index < len(self._pc_cache):
                pass
            else:
                try:
                    while len(self._pc_cache) <= index:
                        frame = next(self._pc_iter)
                        self._pc_cache.append(frame)
                except StopIteration:
                    if self.num_frames:
                        # Wrap around if total_frames known
                        self.current_frame = 0
                        return
                    else:
                        # Stop playback if no more frames
                        self.playing = False
                        self.timer.stop()

        # ---- Load tracking (if any) ----
        if self._has_tracking and not self._trk_is_list:
            if index < len(self._trk_cache):
                return
            try:
                while len(self._trk_cache) <= index:
                    trk_frame = next(self._trk_iter)
                    self._trk_cache.append(np.asarray(trk_frame, dtype=float))
            except StopIteration:
                # No more tracking frames
                pass

    def keyPressEvent(self, event):
        if event.key() == self.play_key:
            self.toggle_play()
        elif event.key() == self.prev_key and not self.playing:
            self.prev_frame()
        elif event.key() == self.next_key and not self.playing:
            self.next_frame()
        super().keyPressEvent(event)
    
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("PAUSE")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.timer.start()
        else:
            self.play_btn.setText("PLAY")
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.timer.stop()
    
    def next_frame(self):
        target = self.current_frame + 1
        # For lists, wrap by modulo
        if self._is_list and self.num_frames:
            self.current_frame = target % self.num_frames
        else:
            self.current_frame = target

        self._load_frame(self.current_frame)
        # Clamp if past end of cache and no wrap
        if self.current_frame >= len(self._pc_cache):
            self.current_frame = len(self._pc_cache) - 1
        self.update_display()
    
    def prev_frame(self):
        if self._is_list and self.num_frames:
            self.current_frame = (self.current_frame - 1) % self.num_frames
        else:
            self.current_frame = max(0, self.current_frame - 1)
        self.update_display()
    
    def update_display(self):
        # ----- Draw point cloud for current frame -----
        cloud = self._pc_cache[self.current_frame]
        if isinstance(cloud, np.ndarray) and cloud.ndim == 2 and cloud.shape[1] >= 3:
            pts = cloud[:, :3]
        else:
            pts = np.zeros((0, 3))
        self.plot3d.scatter.setData(pos=pts)

        if self._has_tracking and self.current_frame < len(self._trk_cache):
            locs = self._trk_cache[self.current_frame]
            if locs is None or len(locs) == 0:
                # Clear any previous tracking visuals
                if self.tracking_mode == "dot":
                    self._trk_scatter.setData(pos=np.zeros((0, 3)), color=np.zeros((0, 4)))
                elif self.tracking_mode == "bbox":
                    for box in getattr(self, '_trk_boxes', []):
                        self.plot3d.plot_3d.removeItem(box)
                    self._trk_boxes = []
            else:
                arr = np.asarray(locs, dtype=float)
                # Normalize shape to (N, >=3)
                if arr.ndim == 1 and arr.size >= 3:
                    loc_arr = arr.reshape(1, -1)
                elif arr.ndim == 2 and arr.shape[1] >= 3:
                    loc_arr = arr
                else:
                    loc_arr = np.zeros((0, 3))

                num_objs = loc_arr.shape[0]
                # If in dot‐mode, draw colored centroids
                if self.tracking_mode == "dot":
                    color_array = np.zeros((num_objs, 4), dtype=float)
                    # Reuse the same palette as MainVisualizer (fallback if missing)
                    palette = getattr(self.plot3d, '_color_palette', [
                        (1.0, 0.0, 0.0, 1.0),
                        (0.0, 1.0, 0.0, 1.0),
                        (0.0, 0.0, 1.0, 1.0),
                        (1.0, 1.0, 0.0, 1.0),
                        (1.0, 0.0, 1.0, 1.0),
                        (0.0, 1.0, 1.0, 1.0),
                        (1.0, 0.5, 0.0, 1.0),
                        (0.5, 0.0, 1.0, 1.0),
                    ])
                    P = len(palette)
                    for i in range(num_objs):
                        color_array[i, :] = palette[i % P]
                    centroids = loc_arr[:, :3]
                    self._trk_scatter.setData(pos=centroids, color=color_array, size=20)

                # If in bbox‐mode, use helper to draw 3D boxes
                elif self.tracking_mode == "bbox":
                    # Remove old boxes
                    for box in getattr(self, '_trk_boxes', []):
                        self.plot3d.plot_3d.removeItem(box)
                    self._trk_boxes = []

                    palette = getattr(self.plot3d, '_color_palette', [
                        (1.0, 0.0, 0.0, 1.0),
                        (0.0, 1.0, 0.0, 1.0),
                        (0.0, 0.0, 1.0, 1.0),
                        (1.0, 1.0, 0.0, 1.0),
                        (1.0, 0.0, 1.0, 1.0),
                        (0.0, 1.0, 1.0, 1.0),
                        (1.0, 0.5, 0.0, 1.0),
                        (0.5, 0.0, 1.0, 1.0),
                    ])
                    # Draw boxes via helper
                    self._trk_boxes = _draw_bboxes(loc_arr[:, :3], self.plot3d.plot_3d, palette)

        else:
            # No tracking for this frame: clear previous visuals
            if self._has_tracking:
                if self.tracking_mode == "dot":
                    self._trk_scatter.setData(pos=np.zeros((0, 3)), color=np.zeros((0, 4)))
                elif self.tracking_mode == "bbox":
                    for box in getattr(self, '_trk_boxes', []):
                        self.plot3d.plot_3d.removeItem(box)
                    self._trk_boxes = []

        # Update window title
        title = f"Frame {self.current_frame + 1}"
        if self.num_frames:
            title += f"/{self.num_frames}"
        self.setWindowTitle(title)

    def go_to_frame(self):
        """Jump directly to the frame number in the text field."""
        text = self.goto_field.text()
        try:
            requested = int(text) - 1
            if requested < 0:
                raise ValueError
        except ValueError:
            return  # bad input; ignore

        # wrap if we know total_frames
        if self.num_frames:
            requested = requested % self.num_frames

        # force‐load up through requested index
        self._load_frame(requested)

        # now clamp to however many actually exist in cache
        max_idx = len(self._pc_cache) - 1
        self.current_frame = max(0, min(requested, max_idx))

        self.update_display()