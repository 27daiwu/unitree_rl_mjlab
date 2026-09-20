# MUJOCO_OBSERVATION_BRIDGE_AUDIT

> Historical filtered-only audit. Superseded by
> [HEIGHT_SCAN_TRAINING_PARITY_AUDIT.md](HEIGHT_SCAN_TRAINING_PARITY_AUDIT.md):
> explicit TRAINING_PARITY now passes full 274D parity; DEPLOYMENT_FILTERED
> is preserved. Earlier NO gates below describe the previous stage.

The independent C++ builder is implemented and tested without loading a policy.
The first 98 dimensions pass numerical comparison against the actual mjlab GPU
observation manager. Full 274D parity **fails** because the user explicitly
retained full robot filtering, while training height scans include robot visual
meshes. **Not ready for policy dry-run or closed-loop execution.**

## Observation contract

The audit instantiates the existing G1 Phase2C2 single-riser and Phase2C6
two-riser task factories in play mode. Their actor terms inherit
`src/tasks/velocity/velocity_env_cfg.py`. A one-environment deterministic
fixture disables events/curriculum/metrics in memory; no training source,
rewards, policy weights, gains, actions, XML, or DDS settings are modified.
The actual `ObservationManager.active_terms` and `group_obs_term_dim` are
asserted against the layout below, rather than inferring it from 274.

| Slice | Name | Dim | Source / frame / order | Scale | Clip |
|---|---|---:|---|---|---|
| 0:3 | base_ang_vel | 3 | `mdp.builtin_sensor(robot/imu_ang_vel)`; pelvis-aligned gyro xyz, rad/s | 1 | none |
| 3:6 | projected_gravity | 3 | `mdp.projected_gravity`; pelvis `Rᵀ[0,0,-1]`, xyz | 1 | none |
| 6:9 | command | 3 | `generated_commands(twist)`; base-frame vx, vy, angular-z command (yaw rate) | 1 | none |
| 9:11 | phase | 2 | local `mdp.phase`; sin then cos, period 0.6 s | 1 | none |
| 11:40 | joint_pos | 29 | `mdp.joint_pos_rel`; q − configured default q, training joint order | 1 | none |
| 40:69 | joint_vel | 29 | `mdp.joint_vel_rel`; dq − default dq (all zero), same order | 1 | none |
| 69:98 | actions | 29 | `mdp.last_action`; previous raw action-manager input, same order | 1 | none |
| 98:274 | height_scan | 176 | `mdp.height_scan`; pelvis z − ray-hit z; x fastest | 0.2 | none |

Sources for each operation are installed `mjlab/envs/mdp/observations.py`,
`mjlab/entity/data.py`, local `src/tasks/velocity/mdp/observations.py`,
`src/assets/robots/unitree_g1/g1_constants.py` (`HOME_KEYFRAME`), and
`mjlab/managers/observation_manager.py`.

Phase is `p=((episode_step*step_dt) % 0.6)/0.6`, with output
`[sin(2πp), cos(2πp)]`; both values become zero when `norm(command)<0.1`.
The builder uses float32 clock arithmetic like PyTorch, and receives episode
step explicitly rather than substituting global MuJoCo simulation time.
The compiled task's policy `step_dt` is used by the parity tests.

All blocks have history length 1 and no observation-manager normalization.
Training corruption adds uniform noise **before** clip/scale: gyro ±0.2,
gravity ±0.05, joint position ±0.01, joint velocity ±1.5, height ±0.1;
command, phase, and actions have no noise. Existing `play=True` disables
corruption, which is the deployment parity target.

There is a separate **learned policy normalizer**: G1 `rl_cfg.py` sets
`actor.obs_normalization=True`. Installed `rsl_rl/models/mlp_model.py` applies
`obs_normalizer` inside the actor and inside its ONNX/JIT export wrappers.
`EmpiricalNormalization` computes `(x-mean)/(std+eps)` (default eps=0.01),
using checkpoint statistics. This builder returns its 274D input, and never
normalizes it a second time. No checkpoint was loaded here; a future dry-run
must verify that its chosen exported artifact includes that normalizer.

## Implementation and usage

- `src/policy_observation.h/.cc`: independent `Builder`, name-based mapping,
  finite checks, fixed-size output, and static slice/dimension assertions.
- `src/height_scan.h`: `mj_ray` integration at exactly `obs[98:274]`, scaled once.
- `src/policy_observation_c_api.cc`: thin binding for diagnostics only.
- `policy_observation.py`: single-shot, no-policy CLI and reusable Python wrapper.
- `tests/test_observation_parity.py`: actual GPU training sensor/manager versus
  native C++/CPU MuJoCo; no fake training observation reimplementation.

`Builder` is bound to a model (recreate it when the model is reloaded).
Call `mj_forward` before building a snapshot. The caller supplies final velocity
command, episode step, and policy step duration. After accepting a policy output,
call `SetPreviousAction`; `Build` preserves it across calls. Only `Reset` or initial
construction zeros it. No actions, control torques, or DDS messages are sent.

The current deployment XML names its gyro `imu_gyro`; training calls it
`robot/imu_ang_vel`. The builder validates that either sensor is a 3D gyro on a
pelvis-aligned site. Their site translations differ, which does not change angular
velocity. Quaternion convention is MuJoCo **wxyz**. Gravity uses the pelvis
rotation matrix, avoiding quaternion component-order ambiguity.

```bash
cd ~/unitree_workspace/src/unitree_rl_mjlab
conda activate unitree_rl_mjlab
python -m simulate.policy_observation --dump logs/observation/flat.npz
python -m simulate.tests.test_observation_parity
cmake -S simulate -B simulate/build
cmake --build simulate/build --target policy_observation -j2
```

The diagnostic prints each block's min/max/mean and NaN/Inf counts once, then
exits; it does not print at physics frequency. It never loads a policy or connects
to DDS. Python builds its diagnostic library against its own MuJoCo 3.5 ABI;
the CMake target separately compiles against bundled MuJoCo 3.3.6.

Snapshots in `logs/observation/parity/<case>/` contain `training_obs.npy`,
`mujoco_obs.npy`, and `snapshot.npz`. The NPZ stores wall timestamp, simulation
time, full q/dq, base quaternion wxyz, gyro, command, persistent previous action,
raw height values (final scan divided by 0.2), final obs, episode step, and dt.
`single_scene.mjb` and `two_scene.mjb` retain the exact compiled terrain/robot
model for those snapshots; load with the same MuJoCo version.

## Validation status

Fixtures cover flat upright, single riser, two risers, yaw, pitch, combined
roll/pitch/yaw, near-vertical pitch fallback, moving root/joints with nonzero
previous action, and a long-episode phase clock. Additional checks cover
reset/persistence, deployment visual self filtering, all-ray misses, >5 m
range handling, and rejected nonfinite actions.

Flat deployment scan is constant 0.16 at pelvis z=0.8. Positive pitch 0.35 rad
produces gravity `[sin(0.35), 0, -cos(0.35)]`. Static single/two riser scans
have two/three distinct levels. Higher ground makes the **relative-height
observation decrease**. Every snapshot has 274 finite values.

The parity suite distinguishes expected height mismatch from unexpected errors:
all first-98 blocks must pass tolerance 2e-5; every height ray whose training hit
is not a robot body must also pass. The full height mismatch is always reported
as FAIL, even when these regression assertions pass.

See the numerical tables below and `tests/observation_parity_results.json` for
all cases and actual training hit bodies. The preceding scan audit is corrected
in `HEIGHT_SCAN_AUDIT.md`; its old claimed x-major scan and 2.5 m Z offset were
not the training definition.

## Numerical results

| Block | Worst max absolute error across fixtures | Mean absolute error across fixtures |
|---|---:|---:|
| base_ang_vel | 2.980232239e-08 | 1.655684577e-09 |
| projected_gravity | 2.980232239e-08 | 2.207579502e-09 |
| command | 0 | 0 |
| phase | 1.192092896e-07 | 4.635916816e-08 |
| joint_pos | 0 | 0 |
| joint_vel | 0 | 0 |
| previous_action | 0 | 0 |
| height_scan | 0.1580220163 | 0.005276964305 |

Maximum nonrobot-ray error: 1.490116119e-08.

## Joint mapping (verified by names and motor sensor/actuator references)

| Train index | MuJoCo joint index (deployment scene) | Motor index | Joint name |
|---:|---:|---:|---|
| 0 | 1 | 0 | left_hip_pitch_joint |
| 1 | 2 | 1 | left_hip_roll_joint |
| 2 | 3 | 2 | left_hip_yaw_joint |
| 3 | 4 | 3 | left_knee_joint |
| 4 | 5 | 4 | left_ankle_pitch_joint |
| 5 | 6 | 5 | left_ankle_roll_joint |
| 6 | 7 | 6 | right_hip_pitch_joint |
| 7 | 8 | 7 | right_hip_roll_joint |
| 8 | 9 | 8 | right_hip_yaw_joint |
| 9 | 10 | 9 | right_knee_joint |
| 10 | 11 | 10 | right_ankle_pitch_joint |
| 11 | 12 | 11 | right_ankle_roll_joint |
| 12 | 13 | 12 | waist_yaw_joint |
| 13 | 14 | 13 | waist_roll_joint |
| 14 | 15 | 14 | waist_pitch_joint |
| 15 | 16 | 15 | left_shoulder_pitch_joint |
| 16 | 17 | 16 | left_shoulder_roll_joint |
| 17 | 18 | 17 | left_shoulder_yaw_joint |
| 18 | 19 | 18 | left_elbow_joint |
| 19 | 20 | 19 | left_wrist_roll_joint |
| 20 | 21 | 20 | left_wrist_pitch_joint |
| 21 | 22 | 21 | left_wrist_yaw_joint |
| 22 | 23 | 22 | right_shoulder_pitch_joint |
| 23 | 24 | 23 | right_shoulder_roll_joint |
| 24 | 25 | 24 | right_shoulder_yaw_joint |
| 25 | 26 | 25 | right_elbow_joint |
| 26 | 27 | 26 | right_wrist_roll_joint |
| 27 | 28 | 27 | right_wrist_pitch_joint |
| 28 | 29 | 28 | right_wrist_yaw_joint |

## Final audit

```text
MUJOCO_OBSERVATION_BRIDGE_AUDIT
TRAINING_OBS_DIM = 274
OBS_LAYOUT = 0:3 gyro; 3:6 gravity; 6:9 command; 9:11 phase;
             11:40 joint_pos; 40:69 joint_vel; 69:98 previous_action; 98:274 height_scan
BASE_ANG_VEL_SOURCE = named pelvis-aligned MuJoCo gyro sensordata
PROJECTED_GRAVITY_SOURCE = pelvis xmat transpose * [0,0,-1]
COMMAND_SOURCE = caller-supplied final [vx,vy,yaw_rate], training twist convention
JOINT_POS_SOURCE = named MuJoCo hinge qpos - audited training HOME_KEYFRAME defaults
JOINT_VEL_SOURCE = named MuJoCo hinge qvel - zero defaults
PREVIOUS_ACTION_SOURCE = persistent Builder state, explicit SetPreviousAction / Reset
HEIGHT_SCAN_SOURCE = simulate/src/height_scan.h
JOINT_ORDER_PARITY = PASS, all 29 joints and motor sensor indices
QUATERNION_ORDER = wxyz
FRAME_PARITY = PASS (gyro, gravity, grid alignment; includes pitch singularity)
SCALE_PARITY = PASS (height 0.2 exactly once, all other blocks 1)
CLIP_PARITY = PASS (none)
OBS_DIM = 274
OBS_NAN_COUNT = 0
OBS_INF_COUNT = 0
HEIGHT_SCAN_DIM = 176
HEIGHT_SCAN_SLICE = 98:274
SNAPSHOT_DUMP_IMPLEMENTED = YES
TRAINING_MUJOCO_PARITY_TEST = FAIL_HEIGHT_SCAN_SELF_FILTER_DIFFERENCE
BLOCKWISE_ERROR = numerical table above; complete per-case JSON in tests/
MODIFIED_FILES = simulate/CMakeLists.txt; simulate/src/height_scan.h;
                 simulate/HEIGHT_SCAN_AUDIT.md
ADDED_FILES = simulate/src/policy_observation.h; simulate/src/policy_observation.cc;
              simulate/src/policy_observation_c_api.cc; simulate/policy_observation.py;
              simulate/tests/test_observation_parity.py;
              simulate/tests/legacy_height_scan_evidence.json;
              simulate/tests/observation_parity_results.json;
              simulate/OBSERVATION_BRIDGE_AUDIT.md
POLICY_INFERENCE_STARTED = NO
CLOSED_LOOP_STARTED = NO
RL_TRAINING_STARTED = NO
REAL_ROBOT_STARTED = NO
READY_FOR_POLICY_DRY_RUN = NO
```

The remaining mismatch is an explicitly accepted observation semantics difference,
not an unimplemented bridge or an unmeasured result. Training definitions remain
unchanged. A future policy dry-run must resolve or explicitly account for it.
