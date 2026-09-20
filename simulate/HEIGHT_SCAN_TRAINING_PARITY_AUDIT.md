# HEIGHT_SCAN_TRAINING_PARITY_AUDIT

The real mjlab GPU ObservationManager versus native C++ MuJoCo passes all nine
existing fixtures at the unchanged **2e-5** tolerance. No policy was loaded.
This supersedes the previous filtered-only height parity failure; that mode is
still available, unchanged, and is not the training-parity gate.

## Exact ray collision semantics

Verified in the installed `unitree_rl_mjlab` conda environment:

- `mjlab/sensor/raycast_sensor.py`: `RayCastSensorCfg.include_geom_groups=(0,1,2)`,
  `exclude_parent_body=True`; `initialize` excludes the attached pelvis body ID,
  **not its descendants**. `raycast_kernel` calls `mujoco_warp.rays` with
  `flg_static=True`, allowing both static and dynamic geoms.
- `mujoco_warp/_src/ray.py`: `_ray_eliminate` applies group, body, visibility
  (geom/material alpha), and static filters. There is no contact-bit filter.
- The actual sensor uses the BVH path. `ray_mesh_with_bvh` queries the nearest
  triangle two-sided and accepts it only when `dot(local_ray, normal)<0`.
  If the nearest triangle is a backface, the entire mesh is rejected for that
  ray; it does not search for a farther frontface inside that mesh.
- Robot visual meshes in group 2 can be hit. Robot collision geoms in group 3
  are excluded. Collision geoms in allowed groups are eligible just like any
  other geom; the mode does not blindly include every robot geom.

`HeightScan::Mode::TRAINING_PARITY` reproduces those rules with actual geometry
intersections. Native `mj_ray` locates the closest eligible geom. For a mesh,
the compiled mesh triangles determine the nearest face winding. If that face
is a backface, a private geom-group view excludes the mesh and `mj_ray` repeats.
The live model, XML, and hit heights are never edited. This additional rule was
necessary: group/body filtering alone differed at rays 51, 83, and 115 in the
upright fixture (wrist-pitch and waist-yaw backfaces).

`HeightScan::Mode::DEPLOYMENT_FILTERED` retains the entire robot-subtree filter
and original two-sided CPU ray behavior. It remains the default for existing
callers, preventing an implicit behavioral switch. Policy parity tests select
`TRAINING_PARITY` explicitly. The C++ builder and Python diagnostic expose the
mode; snapshots record it. All other observation block calculations are unchanged.

Grid, origin, yaw transform (including singularity handling), output formula,
scale, flatten order, maximum range, and observation slice are unchanged.

## Validation

Every fixture checks every observation block, all ray geom IDs, and self-hit
body/geom identities and unscaled world hit heights. Unnamed visual geoms are
identified by their compiled geom IDs plus body names, not names alone.
`self_hit_identity_audit` records each requested field for each self-hit ray in
`tests/observation_parity_results.json`. The self-hit count below counts ray
samples across fixtures, not unique grid indices.

Both modes' first 98 values are asserted bit-identical. Filtered outputs were
also compared bit-for-bit to the previous saved fixtures, including their height
blocks. Existing miss, maximum range, finite-value and reset checks are retained.
CMake build with bundled MuJoCo 3.3.6 and diagnostic build with conda MuJoCo 3.5
both pass. The GPU/CPU numerical comparison uses the **same compiled training
model** at the same state; it does not certify arbitrary replacement robot XML
or checkpoint normalization. No robot model was modified.

| Case | Height max error | Height mean error | Full obs max error | Full obs mean error |
|---|---:|---:|---:|---:|
| flat_upright | 2.048909664e-07 | 1.657330806e-08 | 2.048909664e-07 | 1.064562927e-08 |
| stair_approach | 2.048909664e-07 | 1.005405093e-08 | 2.048909664e-07 | 6.893146498e-09 |
| yaw | 1.806765795e-07 | 1.246702297e-08 | 1.806765795e-07 | 8.443085342e-09 |
| pitch | 4.470348358e-08 | 8.953396247e-09 | 1.192092896e-07 | 6.186157364e-09 |
| yaw_pitch_roll | 2.346932888e-07 | 1.165211572e-08 | 2.346932888e-07 | 8.02840816e-09 |
| singular_pitch | 3.492459655e-08 | 9.292059566e-09 | 1.192092896e-07 | 6.403692243e-09 |
| moving | 1.415610313e-07 | 1.084779111e-08 | 1.415610313e-07 | 7.674914038e-09 |
| long_episode | 2.048909664e-07 | 1.005405093e-08 | 2.048909664e-07 | 6.458076296e-09 |
| two_riser | 2.048909664e-07 | 1.610764677e-08 | 2.048909664e-07 | 1.078158807e-08 |

Means below aggregate all nine equal-sized fixtures. Gate readiness means the
observation-parity prerequisite has passed, not that policy inference has run.

```text
HEIGHT_SCAN_TRAINING_PARITY_AUDIT
TRAINING_RAY_SEMANTICS = groups 0/1/2; pelvis-only exclusion; static enabled;
                         visibility filtering; nearest-mesh-hit backface rejection
MUJOCO_TRAINING_PARITY_MODE = explicit TRAINING_PARITY; mj_ray + mesh winding audit
ROBOT_VISUAL_HITS_SUPPORTED = YES, eligible group 2 meshes except excluded pelvis
ROBOT_COLLISION_HITS_SUPPORTED = only eligible groups; actual training group 3 excluded
SELF_HIT_RAY_COUNT = 69
SELF_HIT_IDENTITY_PARITY = PASS
FIRST_98_MAX_ERROR = 1.19209289551e-07
HEIGHT_SCAN_MAX_ERROR = 2.34693288803e-07
HEIGHT_SCAN_MEAN_ERROR = 1.17779380323e-08
FULL_274D_MAX_ERROR = 2.34693288803e-07
FULL_274D_MEAN_ERROR = 7.94607747502e-09
FIRST_98_PARITY = PASS
HEIGHT_SCAN_PARITY = PASS
FULL_274D_PARITY = PASS
DEPLOYMENT_FILTERED_MODE_PRESERVED = YES
MODIFIED_FILES = simulate/src/height_scan.h
                 simulate/src/policy_observation.h
                 simulate/src/policy_observation.cc
                 simulate/src/policy_observation_c_api.cc
                 simulate/policy_observation.py
                 simulate/tests/test_observation_parity.py
                 simulate/tests/observation_parity_results.json
                 simulate/HEIGHT_SCAN_AUDIT.md
                 simulate/OBSERVATION_BRIDGE_AUDIT.md
ADDED_FILES = simulate/HEIGHT_SCAN_TRAINING_PARITY_AUDIT.md
POLICY_INFERENCE_STARTED = NO
CLOSED_LOOP_STARTED = NO
RL_TRAINING_STARTED = NO
REAL_ROBOT_STARTED = NO
READY_FOR_POLICY_DRY_RUN = YES
```

## Reproduce

```bash
conda activate unitree_rl_mjlab
python -m simulate.tests.test_observation_parity
python -m simulate.policy_observation \
  --scene logs/observation/training_parity/single_scene.mjb --prefix robot/ \
  --snapshot logs/observation/training_parity/moving/snapshot.npz \
  --ray-mode TRAINING_PARITY --dump logs/observation/training_parity/replay.npz
```

Live results and complete snapshots: `logs/observation/training_parity/`.
The CLI still performs no inference or physics/control stepping.
