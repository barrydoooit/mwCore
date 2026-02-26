from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from ..base import BaseSignalProcess



@ADCPROCESSORS.register_module()
class VirtualAntennaAssembler(BaseSignalProcess):
    """
    Assemble virtual antenna cube from range_fft and perform per-range clutter removal.

    Input:
      range_fft: (Tx,Rx,Loops,Range) complex

    Outputs:
      virt_cube:        (12, Loops, Range) complex64
      virt_cube_cr:     (12, Loops, Range) complex64
      static_mean_ant:  (Range, 12) complex64

    tx_permutation:
      Reorders the TX axis of range_fft into canonical physical order [Tx1,Tx2,Tx3].
      - e.g., IWR6843ISK demo firing: Tx1,Tx2,Tx3 -> tx_permutation=(0,1,2)
      - mmMesh's AWR1843BOOST firing: Tx1,Tx3,Tx2 -> tx_permutation=(0,2,1)
    """
    requires = {"range_fft"}
    provides = {"virt_cube", "virt_cube_cr", "static_mean_ant"}

    def __init__(self, tx_permutation=(0, 1, 2), name: str = "VirtualAntennaAssembler"):
        super().__init__(name)
        self.tx_permutation = tuple(int(x) for x in tx_permutation)

    def _process_generic(self, frame: RadarFrame) -> None:
        rfft = self.read(frame, "range_fft")  # (Tx,Rx,Loops,Range)
        if rfft.ndim != 4:
            raise ValueError(f"[{self.name}] range_fft must be 4D (Tx,Rx,Loops,Range). Got {rfft.shape}")

        tx, rx, loops, rng = rfft.shape
        if tx * rx != 12:
            raise ValueError(f"[{self.name}] Expected Tx*Rx==12. Got Tx={tx}, Rx={rx}")

        if len(self.tx_permutation) != tx:
            raise ValueError(
                f"[{self.name}] tx_permutation length must match Tx ({tx}). Got {self.tx_permutation}"
            )

        # Reorder TX to canonical physical order [Tx1,Tx2,Tx3]
        if self.tx_permutation != tuple(range(tx)):
            rfft = rfft[np.asarray(self.tx_permutation, dtype=np.int32), :, :, :]

        # Virtual order becomes:
        # [Tx1-Rx0..3, Tx2-Rx0..3, Tx3-Rx0..3]
        virt = rfft.reshape(tx * rx, loops, rng).astype(np.complex64)

        mean_ant = virt.mean(axis=1)          # (12, Range)
        static_mean = mean_ant.T.copy()       # (Range, 12)

        virt_cr = virt - mean_ant[:, None, :] # (12, Loops, Range)

        self.write(frame, "virt_cube", virt)
        self.write(frame, "virt_cube_cr", virt_cr)
        self.write(frame, "static_mean_ant", static_mean)