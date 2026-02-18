# mmWave Radar Data Format & Pipeline Structures

This document outlines the data lifecycle of mmWave radar processing within the `mwcore.signal_processing` module. It details how raw analog-to-digital converter (ADC) data is structured on disk, and how it is progressively transformed inside the central `RadarFrame` object by various signal processing modules.

---

## 1. Raw ADC Binary Format (`.bin`)

The raw `.bin` files contain the unprocessed baseband signals captured by the mmWave sensor's ADC. Data is stored as a continuous stream of binary numbers, which are read frame-by-frame by the `OfflineAdcDataReader`.

### Byte-Level Structure
* **Data Type:** 16-bit signed integers (`Int16` / `int16_t`).
* **Byte Order:** Little-endian (typical for Texas Instruments mmWave sensors).
* **IQ Interleaving:** The data consists of Complex numbers (Real/I and Imaginary/Q components). Within the byte stream, they are interleaved. According to the reshaper logic, a block of 4 `Int16` values represents 2 Complex samples:
  * `[I_0, I_1, Q_0, Q_1]` $\rightarrow$ `Sample 0 = I_0 + j*Q_0`, `Sample 1 = I_1 + j*Q_1`

### Frame Size Calculation
Because the reader extracts data sequentially, the exact byte size of a single frame must be known. It is calculated based on the hardware configuration (`RadarConfig`):

$$\text{Bytes per Frame} = \text{Loops} \times \text{Tx} \times \text{Rx} \times \text{Samples} \times 2 \text{ (Complex I\&Q)} \times 2 \text{ (Bytes per Int16)}$$

*Example for standard 3D configuration (128 loops, 3 Tx, 4 Rx, 256 samples):*
$128 \times 3 \times 4 \times 256 \times 2 \times 2 = \mathbf{1,572,864 \text{ bytes per frame}}$

---

## 2. The `RadarFrame` Data Structure

The `RadarFrame` is the central data carrier in the DSP pipeline. As the frame passes through a sequence of processing modules (e.g., `FrameReshaper` $\rightarrow$ `RangeFFT` $\rightarrow$ `Detector` $\rightarrow$ `AoA`), different attributes are populated. 

This design allows algorithms to be swapped modularly (e.g., using a standard CFAR instead of a Top-K Detector) as long as they adhere to the expected attribute shapes.

### Level 0: Raw Data Parsing
* **`raw_bytes`**
  * **Type:** `bytes`
  * **Shape:** 1D byte string (e.g., length 1,572,864).
  * **Description:** The exact byte chunk read from the `.bin` file. Unaltered.
* **`raw_complex`**
  * **Type:** `np.ndarray` (`dtype=np.complex_`)
  * **Shape:** 1D array (e.g., length 393,216).
  * **Description:** The raw bytes parsed into complex numbers. At this stage, the data is still a flat 1D sequence representing the chronological capture order.

### Level 1: Multidimensional Formatting
* **`radar_cube`**
  * **Type:** `np.ndarray` (`dtype=np.complex_`)
  * **Shape:** `(Tx, Rx, Loops, Samples)`
  * **Description:** The parsed complex numbers reshaped into the physical dimensions of the radar capture. It is transposed to an **"antenna-first"** format to make spatial operations (like AoA) mathematically convenient later in the pipeline.

### Level 2: Frequency Domain Transformations
* **`range_fft`**
  * **Type:** `np.ndarray` (`dtype=np.complex_`)
  * **Shape:** `(Tx, Rx, Loops, RangeBins)` *(Same as `radar_cube`)*
  * **Description:** Time-domain ADC samples converted into Range bins via a 1D FFT along the last axis. Modules like `StaticClutterRemoval` operate directly on this array to filter out zero-Doppler signatures.
* **`doppler_fft`**
  * **Type:** `np.ndarray` (`dtype=np.complex_`)
  * **Shape:** `(Tx, Rx, DopplerBins, RangeBins)`
  * **Description:** The Range-Doppler matrix. Created by performing a 1D FFT across the `Loops` axis. This provides velocity information for every spatial antenna combination.

### Level 3: Target Detection & Feature Maps
* **`energy_map`**
  * **Type:** `np.ndarray` (`dtype=float`)
  * **Shape:** `(DopplerBins, RangeBins)`
  * **Description:** A flattened 2D representation of the radar's field of view, typically created by integrating (summing) absolute magnitudes across all `Tx` and `Rx` antennas. Used as the input canvas for detection algorithms.
* **`detected_points`**
  * **Type:** `np.ndarray` (`dtype=float` or `int`)
  * **Shape:** `(N, 3)` where $N$ is the number of detected targets.
  * **Format:**
    * `Column 0:` **Range Index** (Integer index of the range bin)
    * `Column 1:` **Doppler Index** (Integer index of the velocity bin)
    * `Column 2:` **Peak Value / SNR** (Float representing signal strength)
  * **Description:** The output of a Detection module (such as `TopKDetector`, `CA-CFAR`, or `OS-CFAR`). It acts as a sparse list of "Regions of Interest" for the AoA module to process, saving computational power.

### Level 4: Spatial Positioning
* **`point_cloud`**
  * **Type:** `np.ndarray` (`dtype=float`)
  * **Shape:** `(6, N)` where $N$ is the number of valid points (after geometric filtering).
  * **Format:**
    * `Row 0:` **X Coordinate** (Horizontal axis, meters)
    * `Row 1:` **Y Coordinate** (Depth/Forward axis, meters)
    * `Row 2:` **Z Coordinate** (Elevation/Height axis, meters)
    * `Row 3:` **Velocity** (Radial velocity, meters/second)
    * `Row 4:` **SNR / Intensity** (Signal strength)
    * `Row 5:` **Range** (Radial distance, meters)
  * **Description:** The final physical point cloud generated by an Angle of Arrival (AoA) module (e.g., `NaiveAoA`, `CaponAoA`, `MUSIC`). This module uses the `detected_points` indices to look up the complex phases in the `doppler_fft` cube, calculates the spatial angles (Azimuth and Elevation), and projects them into a 3D Cartesian coordinate system.