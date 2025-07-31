## Prepare Dataset
mRI-Walking:
```sh
python tools/create_data.py mri --root-path /parent/of/dataset_release/folder --out-dir data/mri
```
Asterios-PAW:
```sh
python tools/create_data.py paw --root-path /parent/of/kinect/folder --out-dir data/paw
```

## Run online reading with visualization
```sh
python tools/run_app.py configs/apps/read_and_vis.py
```

## Run offline tracking (optional visualization)
```sh
python tools/run_app.py configs/apps/tracking_mri-walking_rkf.py
```

## Tracking details

All tracker threads consume a signal with the following format:

```python
data = np.stack([
    det_obj["x"], 
    det_obj["y"], 
    det_obj["z"], 
    det_obj["doppler"], 
    det_obj["peakVal"]
], axis=-1)
```

### Tracker Output Formats

| Tracker Type         | Output Format         | Description                                 | Example Trackers                |
|----------------------|----------------------|---------------------------------------------|---------------------------------|
| Cartesian Tracker    | `[x, y, z]`          | Centroid in Cartesian coordinates           | `AsteriosTracker`, `GTracker`   |
| Polar Tracker        | `[r, theta, r']`     | Centroid in polar/radial coordinates        | `RKFTracker`                    |

- The output is always a list of `nd.array`s, each representing the centroid of a detected object.
