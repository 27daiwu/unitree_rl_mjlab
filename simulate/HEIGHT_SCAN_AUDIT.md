# HEIGHT_SCAN_SIM_AUDIT

Training source: `src/tasks/velocity/velocity_env_cfg.py` and
`src/tasks/velocity/config/g1/stairs_env_cfg.py`.

* Sensor: `RayCastSensorCfg`, attached to G1 `pelvis`, `ray_alignment="yaw"`.
* Pattern: `ForwardGridPatternCfg(size=(1.5,1.0), resolution=0.1,
  x_offset=0.45)`. This produces 16 x 11 = 176 points, flattened x-major
  (`ix * 11 + iy`).
* Distance: maximum 5 m; observation scale is `1 / max_distance`.
* `height_scan` is the sensor-origin-to-hit vertical height difference,
  scaled by 0.2 and clipped by the sensor's finite range. Misses use the
  configured maximum distance.

`src/height_scan.h` reproduces the grid, forward offset, yaw-only transform,
vertical MuJoCo ray query, finite miss handling, and 176-element flattening.
The scanner deliberately queries MuJoCo geometry instead of terrain formulas.

The current `simulate` executable is the viewer/physics process and does not
own the deployment policy observation manager. Therefore policy integration
must call `HeightScan{}(model, data, pelvis_body_id)` from the deployment
observation bridge when a MuJoCo-backed policy loop is enabled.

No training, policy, reward, or real-robot process was started.
