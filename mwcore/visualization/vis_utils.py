from typing import List
import numpy as np
import sys
import pyqtgraph.opengl as gl
import os
from PySide2 import QtCore



def _draw_bboxes(
    loc_arr: np.ndarray,
    plot_widget,
    palette: List[tuple],
    bbox_size: np.ndarray = np.array([0.5, 0.5, 0.5])
) -> List[gl.GLLinePlotItem]:
    """
    Given an (N, >=3) array of centroid locations, draw fixed-size 3D bounding boxes
    for each location on the specified GL plot widget. Returns the list of GLLinePlotItem.
    """
    items: List[gl.GLLinePlotItem] = []
    P = len(palette)
    half_size = bbox_size / 2.0
    for idx in range(loc_arr.shape[0]):
        cx, cy, cz = loc_arr[idx, :3]
        dx, dy, dz = half_size
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
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # left face
            (4, 5), (5, 6), (6, 7), (7, 4),  # right face
            (0, 4), (1, 5), (2, 6), (3, 7)   # connectors
        ]
        lines = []
        for (v0, v1) in edges:
            lines.append(vertices[v0])
            lines.append(vertices[v1])
        concatenated_lines = np.vstack(lines)
        color = palette[idx % P]
        box_item = gl.GLLinePlotItem(
            pos=concatenated_lines,
            color=color,
            width=2,
            antialias=True,
            mode='lines'
        )
        plot_widget.addItem(box_item)
        items.append(box_item)
    return items