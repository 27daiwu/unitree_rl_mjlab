# HEIGHT_SCAN_SIM_AUDIT

> Historical filtered-only audit. Superseded by
> [HEIGHT_SCAN_TRAINING_PARITY_AUDIT.md](HEIGHT_SCAN_TRAINING_PARITY_AUDIT.md):
> explicit TRAINING_PARITY now passes full 274D parity; DEPLOYMENT_FILTERED
> is preserved. Earlier NO gates below describe the previous stage.

This supersedes the earlier audit, which claimed parity without inspecting the
installed training sensor or running a numerical test. That claim was incorrect.

## Actual training definition

Sources:

- `src/tasks/velocity/velocity_env_cfg.py`: `terrain_scan` and actor `height_scan`.
- `src/tasks/velocity/config/g1/env_cfgs.py`: attachment to `robot/pelvis`.
- `src/tasks/velocity/config/g1/stairs_env_cfg.py`: `ForwardGridPatternCfg`.
- Installed conda `unitree_rl_mjlab` package `mjlab/sensor/raycast_sensor.py`:
  `GridPatternCfg.generate_rays`, `prepare_rays`, `_extract_yaw_rotation`,
  `raycast_kernel`, `postprocess_rays`.
- Installed `mjlab/envs/mdp/observations.py`: `height_scan`.

```text
TRAINING_OBS_DIM = 274
TRAINING_HEIGHT_SCAN_SLICE = 98:274
TRAINING_HEIGHT_SCAN_DIM = 176
TRAINING_SENSOR_TYPE = RayCastSensorCfg
TRAINING_PATTERN = ForwardGridPatternCfg(size=(1.5,1.0), resolution=0.1, x_offset=0.45)
TRAINING_ROWS = 11 (y)
TRAINING_COLS = 16 (x)
TRAINING_X_RANGE = [-0.30, 1.20] meters after forward offset
TRAINING_Y_RANGE = [-0.50, 0.50] meters
TRAINING_FRAME = pelvis position + yaw-aligned axes
TRAINING_YAW_ONLY = YES; projected X-axis norm < 0.1 uses projected Y-axis fallback
TRAINING_OFFSET = z=0; observation offset=0; local x shift=0.45
TRAINING_SCALE = 0.2
TRAINING_CLIP = NONE
TRAINING_FLATTEN_ORDER = meshgrid(indexing="xy").flatten(); i=iy*16+ix (x fastest)
TRAINING_MAX_DISTANCE = 5 meters; distances >5 become misses
TRAINING_MISS_VALUE = raw 5; final 1
TRAINING_INCLUDE_GEOM_GROUPS = (0,1,2)
TRAINING_BODY_EXCLUDE = pelvis only
```

For valid hits, `raw[i] = pelvis_world_z - hit_world_z[i]`;
for misses, `raw[i] = 5`. Play-mode `obs[98+i] = 0.2 * raw[i]`.
Training corruption adds independent uniform noise in `[-0.1,0.1]` to raw
heights before scaling. It is disabled by the existing play configuration.
There is no additional scanner Z offset and no observation clipping.

## Numerical evidence and accepted difference

`tests/legacy_height_scan_evidence.json` records the former scanner's maximum
error **0.6389101744** against a real training sensor. Its 2.5 m origin offset,
transposed flattening, missing range cutoff, unrestricted collision query, and
clipping were not the training definition. The original header also failed to
compile with MuJoCo 3.5 because its `mj_ray` signature has an added normal output.
The evidence uses only a temporary signature adaptation of that old code.

The corrected scanner uses native `mj_ray`, training grid/frame/range/scale,
and **explicitly excludes the entire pelvis subtree**. It filters geom groups
using a private model view, without changing the live model, robot XML, or DDS.
Both bundled MuJoCo 3.3.6 and conda MuJoCo 3.5 builds are supported.

Actual training ray-hit IDs revealed unnamed group-2 robot visual meshes on
hip and wrist bodies. `exclude_parent_body=True` does not exclude these bodies.
The user explicitly chose to **keep full robot filtering and mark height parity
failed**. Therefore the implementation intentionally does not reproduce those
self hits. This discrepancy is not concealed by a relaxed numerical tolerance.

```text
MUJOCO_IMPLEMENTATION = simulate/src/height_scan.h
MUJOCO_RAYCAST_API = mj_ray
SELF_GEOM_FILTERED = YES, entire robot subtree
MISS_HANDLING = distance<0 or distance>5 -> raw 5 -> final 1; nonfinite errors rejected
SIM_HEIGHT_SCAN_DIM = 176
OBS_INTEGRATION = policy_observation::Builder::Build, obs[98:274]
TRAIN_SIM_NUMERIC_COMPARISON = FAIL due to explicitly retained self filtering
RL_TRAINING_STARTED = NO
REAL_ROBOT_STARTED = NO
READY_FOR_POLICY_OBSERVATION_PARITY_TEST = YES (implemented and run)
READY_FOR_CLOSED_LOOP_STAIR_SIM = NO
```

Full scenarios, per-block errors, hit bodies, and snapshots are described in
`OBSERVATION_BRIDGE_AUDIT.md` and `logs/observation/parity/report.json`.
