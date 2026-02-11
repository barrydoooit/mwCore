import numpy as np
import logging
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QSlider, QLabel, QPushButton, 
                               QSizePolicy)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QKeyEvent

from mwcore.registry import VISUALIZERS
from mwcore.visualization.plot_3d import Plot3D  # Using your existing 3D plotter

log = logging.getLogger(__name__)

@VISUALIZERS.register_module()
class PointCloudOfflineVisualizerV2(QMainWindow):
    """
    Advanced Visualizer for offline playback.
    Features: 
    - Play/Pause/Seek
    - Frame Buffering
    - Fast-forward triggering on unbuffered seek
    - Keyboard shortcuts (A/D)
    """
    
    # Signal to request the worker to speed up (True) or return to normal speed (False)
    request_fast_forward = Signal(bool)

    def __init__(self, 
                 window_size=(1200, 800), 
                 max_buffer_frames=5000, 
                 on_close=None):
        super().__init__()
        self.setWindowTitle("Offline Radar Replay")
        self.resize(*window_size)
        self._on_close_callback = on_close

        # --- Data State ---
        self.frames = []            # The buffer: List[np.ndarray]
        self.max_buffer = max_buffer_frames
        
        # --- Playback State ---
        self.current_display_idx = 0
        self.is_playing = True
        self.target_seek_idx = None # Used when waiting for buffering
        
        # --- UI Initialization ---
        self._init_ui()

        # --- Timer ---
        self.timer = QTimer()
        self.timer.setInterval(50)  # ~20 FPS refresh rate
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start()

    def _init_ui(self):
        """Setup the widget layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 3D Plot Area
        self.plot3d = Plot3D()
        # Ensure plot expands to fill space
        self.plot3d.plot_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.plot3d.plot_3d)

        # 2. Status Info Bar
        info_layout = QHBoxLayout()
        self.lbl_status = QLabel("State: Waiting...")
        self.lbl_file_info = QLabel("File: N/A")
        self.lbl_frame_counter = QLabel("Frame: 0 / 0")
        
        # Styling labels
        for lbl in [self.lbl_status, self.lbl_file_info, self.lbl_frame_counter]:
            lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #333;")
        
        info_layout.addWidget(self.lbl_status)
        info_layout.addStretch()
        info_layout.addWidget(self.lbl_file_info)
        info_layout.addSpacing(20)
        info_layout.addWidget(self.lbl_frame_counter)
        main_layout.addLayout(info_layout)

        # 3. Controls Area (Slider + Buttons)
        controls_layout = QHBoxLayout()

        # Buttons
        self.btn_prev = QPushButton("<< (A)")
        self.btn_play = QPushButton("PAUSE")
        self.btn_next = QPushButton(">> (D)")
        
        self.btn_prev.setFixedWidth(80)
        self.btn_play.setFixedWidth(80)
        self.btn_next.setFixedWidth(80)

        self.btn_prev.clicked.connect(self._prev_frame)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_next.clicked.connect(self._next_frame)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False) # Disabled until we have data
        
        # Slider Signals
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_moved)

        controls_layout.addWidget(self.btn_prev)
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_next)
        controls_layout.addWidget(self.slider)

        main_layout.addLayout(controls_layout)

    # ---------------------------------------------------------
    # Slots (Receiving Data)
    # ---------------------------------------------------------

    @Slot(object)
    def receive_frame(self, frame_data):
        """
        Called by Worker when a new frame is processed.
        Buffering logic happens here.
        """
        # Extract point cloud from frame_data
        # Adjust 'point_cloud' attribute name based on your actual RadarFrame class
        if hasattr(frame_data, 'point_cloud'):
            pc = frame_data.point_cloud
        elif isinstance(frame_data, np.ndarray):
            pc = frame_data
        elif hasattr(frame_data, 'data'): # Fallback
            pc = frame_data.data
        else:
            log.warning("Received frame with unknown format.")
            return

        self.frames.append(pc)
        current_buffer_size = len(self.frames)
        
        # Enable slider once we have data
        if not self.slider.isEnabled():
            self.slider.setEnabled(True)

        # Update Slider Maximum (add a small buffer zone so it feels like a stream)
        # We only expand the slider range if the buffer is growing
        if current_buffer_size > self.slider.maximum():
            self.slider.setMaximum(current_buffer_size)

        # Check if we were waiting for this frame (Seek Loading State)
        if self.target_seek_idx is not None:
            # Update progress text
            self.lbl_status.setText(f"Loading... ({current_buffer_size}/{self.target_seek_idx})")
            
            if current_buffer_size >= self.target_seek_idx:
                # Target reached!
                self.lbl_status.setText("Buffered.")
                self.request_fast_forward.emit(False) # Disable fast forward
                
                # Snap to target
                self.current_display_idx = self.target_seek_idx
                self.target_seek_idx = None
                
                # Update visual immediately
                self._update_display()
                
                # Resume playback if valid
                if self.is_playing:
                    self.timer.start()

    @Slot(dict)
    def update_status(self, info):
        """Called by Worker when file changes."""
        fname = info.get('current_file', 'Unknown')
        fidx = info.get('file_index', 0)
        total = info.get('total_files', '?')
        self.lbl_file_info.setText(f"File: {fname} ({fidx+1}/{total})")

    # ---------------------------------------------------------
    # Internal Logic (Timer & Display)
    # ---------------------------------------------------------

    def _on_timer_tick(self):
        """Main playback loop driven by QTimer."""
        if not self.is_playing:
            return

        # If we have frames to show
        if self.current_display_idx < len(self.frames) - 1:
            self.current_display_idx += 1
            self._update_display()
            self._sync_slider()
        else:
            # We reached the end of the buffer. 
            # We do NOT stop playing, we just wait for the worker to provide more frames.
            self.lbl_status.setText("Waiting for data...")

    def _update_display(self):
        """Draw the point cloud at current_display_idx."""
        if 0 <= self.current_display_idx < len(self.frames):
            pc = self.frames[self.current_display_idx]
            
            # Rendering
            # Assuming PC is (N, 3) or (N, 4) or (N, 5)
            # Plot3D usually expects (N, 3) for pos
            if isinstance(pc, np.ndarray) and pc.shape[0] > 0:
                self.plot3d.scatter.setData(pos=pc[:, :3])
            else:
                # Clear if empty
                self.plot3d.scatter.setData(pos=np.zeros((0, 3)))
            
            # Update Counter Label
            self.lbl_frame_counter.setText(f"Frame: {self.current_display_idx} / {len(self.frames)}")

    def _sync_slider(self):
        """Quietly update slider position without triggering signals."""
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_display_idx)
        self.slider.blockSignals(False)

    # ---------------------------------------------------------
    # User Interaction (Controls)
    # ---------------------------------------------------------

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("PAUSE")
            self.lbl_status.setText("Playing")
            self.timer.start()
        else:
            self.btn_play.setText("PLAY")
            self.lbl_status.setText("Paused")
            self.timer.stop()

    def _prev_frame(self):
        """Go back one frame."""
        self._manual_seek(self.current_display_idx - 1)

    def _next_frame(self):
        """Go forward one frame."""
        self._manual_seek(self.current_display_idx + 1)

    def _manual_seek(self, idx):
        """Safe seek method that handles boundaries."""
        # Clamp index
        idx = max(0, min(idx, len(self.frames) - 1))
        
        self.current_display_idx = idx
        self._update_display()
        self._sync_slider()

    # ---------------------------------------------------------
    # Slider Logic (The complex part)
    # ---------------------------------------------------------

    def _on_slider_pressed(self):
        """User grabbed the slider. Stop auto-updates."""
        self.timer.stop()

    def _on_slider_released(self):
        """User let go. Resume if we were playing and target is ready."""
        # Note: If we are in 'Loading' state (target_seek_idx is set), 
        # we do NOT resume timer yet. receive_frame will handle it.
        if self.is_playing and self.target_seek_idx is None:
            self.timer.start()

    def _on_slider_moved(self, val):
        """
        Handle dragging. 
        If val is in buffer -> Show immediately.
        If val is NOT in buffer -> Trigger Fast Forward.
        """
        if val < len(self.frames):
            # Buffered: Instant seek
            self.current_display_idx = val
            self._update_display()
            self.lbl_status.setText("Seeking...")
            
            # If we were previously loading, cancel that request
            if self.target_seek_idx is not None:
                self.target_seek_idx = None
                self.request_fast_forward.emit(False)

        else:
            # Unbuffered: Trigger Loading
            self.target_seek_idx = val
            self.lbl_status.setText(f"Loading frame {val}...")
            
            # Signal worker to hurry up
            self.request_fast_forward.emit(True)

    # ---------------------------------------------------------
    # Keyboard Events
    # ---------------------------------------------------------
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle A and D keys for frame stepping."""
        # Only active if paused, per requirements
        if not self.is_playing:
            if event.key() == Qt.Key_A:
                self._prev_frame()
                event.accept()
            elif event.key() == Qt.Key_D:
                self._next_frame()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    
    def closeEvent(self, event):
        self.timer.stop()
        if self._on_close_callback:
            self._on_close_callback(event)
        super().closeEvent(event)