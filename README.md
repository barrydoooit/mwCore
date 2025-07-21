## Run online tracking with visualization
```sh
python tools/run_app.py configs/apps/read_and_vis.py
```

## Run offline tracking with visualization
```sh
python tools/run_app.py configs/apps/offline_experiment.py
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
