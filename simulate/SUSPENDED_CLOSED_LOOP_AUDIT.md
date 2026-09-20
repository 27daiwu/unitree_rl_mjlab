# SUSPENDED_CLOSED_LOOP_AUDIT

**Stage A and Stage B PASS** in a short, flat-only suspended native MuJoCo
closed-loop test. This verifies controller bring-up under substantial support,
not unsupported balance, stair capability, or DDS control. All frozen observation,
policy, joint-mapping, action-scale/offset and robot XML files remain unchanged.

## Training control contract

Sources: frozen run
`logs/rsl_rl/g1_velocity/2026-09-15_15-49-46_phase2c8g-second-velocity-control-pilot/params/env.yaml`,
`src/assets/robots/unitree_g1/g1_constants.py`, installed
`mjlab/actuator/builtin_actuator.py`, `mjlab/utils/spec.py:create_position_actuator`,
and `mjlab/envs/mdp/actions/actions.py`.

```text
physics_dt = 0.005 s
policy_dt = 0.020 s
decimation = 4
integrator = implicitfast
encoder_bias = 0
q_target = default_joint_pos + action_scale * raw_action
```

The independent flat scene is built through the existing training `Scene` and
`get_g1_robot_cfg()` factories, with plane terrain only. This invokes the real
training actuator construction rather than modifying the deployment XML. The
frozen run's simulator options are applied to the compiled model. Every actuator's
Kp/Kd/effort/armature/frictionloss is asserted equal to the saved run config.
There is no terrain generator, stairs task, domain randomization or training
runner. The control contract is **nominal**, not a randomized training episode.

Native position actuators take q_target in radians as ctrl. Gear is 1, gain is Kp,
bias is `-Kp*q - Kd*dq`, dynamics type is none. They enforce force limits but do
not clamp position setpoints (`ctrllimited=False`). The MuJoCo-computed force is
checked every physics substep against:

```text
tau = clip(Kp*(q_target-q) - Kd*dq, -effort_limit, effort_limit)
```

This is the existing training actuator force limiter, not a new raw-action clamp.
No additional deployment clamp is introduced. Training `BuiltinPositionActuator`
has **no configured hard velocity limiter**; the source motor speed ratings are
used as abort thresholds only. Training joint damping and frictionloss are zero;
armature is the reflected inertia from the training config. Joint ranges remain
those compiled from the original robot definition.

The generic `scene_g1.xml` / LowCmd path is not equivalent: it contains torque
motors (gain=1, bias=0), torque ctrl-range limits, armature 0.01, frictionloss 0.2,
and joint damping 0.05. `unitree_sdk2_bridge.h` writes a separate external PD torque
into ctrl. This test therefore bypasses LowCmd and DDS completely. The full
read-only comparison is saved in `logs/suspended_control/generic_lowcmd_model_audit.json`.

## Per-joint control and watchdog values

All vectors use the frozen 29-joint training order. Detailed vectors, including
joint position ranges, are in `logs/suspended_control/control_contract.json`.

| Joint | Kp | Kd | Effort Nm | Rated speed rad/s | Armature | Error abort rad | Target-step abort rad |
|---|---:|---:|---:|---:|---:|---:|---:|
| left_hip_pitch_joint | 40.1792386 | 2.55788978 | 88 | 32 | 0.01017752 | 2.19018585 | 0.64 |
| left_hip_roll_joint | 99.0984278 | 6.30880185 | 139 | 20 | 0.025101925 | 1.40264587 | 0.4 |
| left_hip_yaw_joint | 40.1792386 | 2.55788978 | 88 | 32 | 0.01017752 | 2.19018585 | 0.64 |
| left_knee_joint | 99.0984278 | 6.30880185 | 139 | 20 | 0.025101925 | 1.40264587 | 0.4 |
| left_ankle_pitch_joint | 28.5012462 | 1.81444569 | 50 | 37 | 0.00721945 | 1.75430926 | 0.74 |
| left_ankle_roll_joint | 28.5012462 | 1.81444569 | 50 | 37 | 0.00721945 | 1.75430926 | 0.74 |
| right_hip_pitch_joint | 40.1792386 | 2.55788978 | 88 | 32 | 0.01017752 | 2.19018585 | 0.64 |
| right_hip_roll_joint | 99.0984278 | 6.30880185 | 139 | 20 | 0.025101925 | 1.40264587 | 0.4 |
| right_hip_yaw_joint | 40.1792386 | 2.55788978 | 88 | 32 | 0.01017752 | 2.19018585 | 0.64 |
| right_knee_joint | 99.0984278 | 6.30880185 | 139 | 20 | 0.025101925 | 1.40264587 | 0.4 |
| right_ankle_pitch_joint | 28.5012462 | 1.81444569 | 50 | 37 | 0.00721945 | 1.75430926 | 0.74 |
| right_ankle_roll_joint | 28.5012462 | 1.81444569 | 50 | 37 | 0.00721945 | 1.75430926 | 0.74 |
| waist_yaw_joint | 40.1792386 | 2.55788978 | 88 | 32 | 0.01017752 | 2.19018585 | 0.64 |
| waist_roll_joint | 28.5012462 | 1.81444569 | 50 | 37 | 0.00721945 | 1.75430926 | 0.74 |
| waist_pitch_joint | 28.5012462 | 1.81444569 | 50 | 37 | 0.00721945 | 1.75430926 | 0.74 |
| left_shoulder_pitch_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| left_shoulder_roll_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| left_shoulder_yaw_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| left_elbow_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| left_wrist_roll_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| left_wrist_pitch_joint | 16.7783275 | 1.0681415 | 5 | 22 | 0.00425 | 0.298003481 | 0.44 |
| left_wrist_yaw_joint | 16.7783275 | 1.0681415 | 5 | 22 | 0.00425 | 0.298003481 | 0.44 |
| right_shoulder_pitch_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| right_shoulder_roll_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| right_shoulder_yaw_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| right_elbow_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| right_wrist_roll_joint | 14.2506231 | 0.907222843 | 25 | 37 | 0.003609725 | 1.75430926 | 0.74 |
| right_wrist_pitch_joint | 16.7783275 | 1.0681415 | 5 | 22 | 0.00425 | 0.298003481 | 0.44 |
| right_wrist_yaw_joint | 16.7783275 | 1.0681415 | 5 | 22 | 0.00425 | 0.298003481 | 0.44 |

Watchdogs reject nonfinite obs/action/target/q/dq/ctrl/actuator force, actual joint
positions beyond original joint limits, speed above the corresponding motor
rating, force above its configured bound, and any MuJoCo warning. Additional
limits are derived before running:

- `abs(raw_action) <= 6`: ceiling of historical model_9400 maximum magnitude
  5.84973526 from the prior audit's 1500 raw action samples.
- `abs(q_target-q) <= effort_limit/Kp`: static deflection at rated actuator force;
  checked both at target acceptance and every physics substep.
- `abs(new_target-old_target) <= rated_velocity*policy_dt`: per-policy-step
  setpoint-jump diagnostic. Targets outside joint ranges are not silently clipped;
  this preserves training's position-actuator contract.

On a guard or controller exception, policy execution stops, ctrl is zeroed,
actuation is disabled on this private model, and the simulation is frozen with
the suspension force retained. **Zero position ctrl alone is not a zero-torque
stop**; actuation disable is essential. Nonfinite state never re-enters the solver.
A completed short episode also disables actuation without releasing suspension.
No automatic retry, threshold relaxation, or continuation after an abort occurs.

19 injected guard checks passed on initial-state copies without physics stepping,
including NaN/Inf, action/error/jump bounds, joint position/velocity, effort,
no restart after abort, zero actuator force and retained upward suspension force.
Results: `logs/suspended_control/guard_tests.json`.

## Controller and startup

`simulate/policy/controller.py` composes the frozen native observation Builder
(`TRAINING_PARITY`), the existing normalized JIT wrapper, action transform and
native position actuators. `simulate/tests/test_suspended_control.py` owns bounded
stage sequencing and logging; nothing is inserted into the GUI loop.

Startup follows LOAD → WAIT_SIM_READY → ENABLE_SUSPENSION → SET_INITIAL_POSE →
SETTLE → RESET_POLICY_STATE → POLICY_HOLD_ZERO_COMMAND → bounded closed loop.
HOME pose is placed before forward kinematics and checked against saved offsets:
initial max joint error=0, max dq=0, root z=0.8 m. Settlement holds HOME for 20
physics steps (0.1 s), separate from policy-step counts. The first policy episode
then resets previous action and episode step once. Stage B uses a fresh episode
and 5 zero-command handover steps before the forward command; its clock continues
through that command transition.

The always-on band uses the existing simulator's k=200 N/m, c=100 Ns/m and
zero rest length. Root position/linear velocity drive it and the force is applied
to `robot/torso_link`, matching the existing attachment convention. Its independent
anchor is chosen to provide **one robot weight** at HOME root height, rather than
releasing or weakening support. The existing simulator's elastic-band code is not
changed. No rotational stabilizer or artificial joint lock was added.

Robot weight: 327.076603 N. Anchor: [0.0, 0.0, 2.4353830151].

Stage A minimum vertical support: 308.761025 N;
Stage B minimum vertical support: 310.544429 N.
Both remain above 94% of robot weight during policy substeps.

The frozen training command range is vx=[0.3,0.5], vy=0, yaw_rate=0. The smallest
positive in-range Stage B command is **0.3 m/s**, not an invented 0.05/0.1 command.
Zero command for Stage A is the explicitly requested diagnostic, even though the
frozen stair run sampled forward commands only. No command convention changed.

## Results

| Metric | Stage A: zero | Stage B: forward after zero handover |
|---|---:|---:|
| Completed policy steps | 40 | 45 |
| Inference count | 40 | 45 |
| Policy physics substeps | 160 | 180 |
| Max raw action abs | 1.671519518 | 2.35152483 |
| Max target step rad | 0.5553141804 | 0.5553141804 |
| Max target-position error rad | 0.5920656409 | 0.5920656409 |
| Max joint velocity rad/s | 3.321114964 | 5.845024168 |
| Max ctrl (position target rad) | 0.8823064517 | 1.042562088 |
| Max torque Nm | 14.77805384 | 22.9555738 |

Stage A runs 40 policy steps (0.8 s). Stage B runs 5 zero-command steps plus 40
forward-command steps (0.9 s). All observation/action/target/control values are
finite. Neither stage triggers an abort or a MuJoCo warning. The largest initial
target transition is about 0.555 rad; it is recorded, not hidden, and remains
within that joint's derived target-step and position-error limits. This short
supported result does not establish long-duration or unsupported stability.

Every policy step verifies previous action feedback; first step is zero, all
subsequent `[69:98]` equal prior raw action exactly. Phase uses episode policy
steps, remains zero under zero command, and follows sin/cos when vx=0.3. Logs are
independently checked against the phase formula. Target transform is rechecked
across all joints from logged raw action/scale/offset. Each target is held for
exactly four physics steps; timestamps, counts and held ctrl values are asserted.

## Logs and reproducibility

```bash
conda activate unitree_rl_mjlab
python -m simulate.tests.test_controller_guards
python -m simulate.tests.test_suspended_control
```

The stage runner executes Stage B only after Stage A passes. It checks the
verified JIT artifact hash before control. `logs/suspended_control/` contains:

- `flat_nominal.mjb`: exact initial compiled flat model.
- `control_contract.json`: options, mapping, gains, ratings and guard thresholds.
- `stage_a/policy_steps.npz`, `stage_b/policy_steps.npz`: each policy-step timestamp,
  command, complete obs/raw action/q_target/q/dq/root pose, summary min/max/L2,
  height range, ctrl, all four substep torque and suspension-force samples,
  target delta, max position error and speed.
- Stage result files and aggregate `report.json`; diagnostic guard results and
  generic LowCmd model comparison. Abort paths additionally save attempted
  obs/action/target and state when available.

Terminal output is limited to a summary every 25 policy steps plus stage results.
A reviewable result copy is `simulate/tests/suspended_control_results.json`.

## Final audit

```text
SUSPENDED_CLOSED_LOOP_AUDIT
CHECKPOINT = model_9400.pt
PHYSICS_DT = 0.005
POLICY_DT = 0.02
DECIMATION = 4
ACTION_FORMULA = default_joint_pos + action_scale * raw_action
ENCODER_BIAS = 0
PD_KP = per-joint table above / control_contract.json
PD_KD = per-joint table above / control_contract.json
SUSPENSION_ENABLED = YES, never released
SUSPENSION_BODY = robot/torso_link
INITIAL_POSE_SOURCE = training HOME_KEYFRAME; root z=0.8
ZERO_COMMAND_TEST = PASS
ZERO_COMMAND_POLICY_STEPS = 40
TINY_FORWARD_TEST = PASS
TINY_FORWARD_COMMAND = [0.3,0,0] (training range lower bound)
TINY_FORWARD_POLICY_STEPS = 40 forward + 5 zero-command handover
OBS_NAN_COUNT = 0
OBS_INF_COUNT = 0
ACTION_NAN_COUNT = 0
ACTION_INF_COUNT = 0
TARGET_NAN_COUNT = 0
TARGET_INF_COUNT = 0
CTRL_NAN_COUNT = 0
CTRL_INF_COUNT = 0
MAX_RAW_ACTION_ABS = 2.35152482986
MAX_Q_TARGET_STEP = 0.555314180433
MAX_JOINT_ERROR = 0.592065640928
MAX_JOINT_VELOCITY = 5.84502416783
MAX_CTRL = 1.04256208816
MAX_TORQUE = 22.9555737971
PREVIOUS_ACTION_ONLINE_PARITY = PASS
POLICY_RATE_PARITY = PASS
ACTION_TRANSFORM_PARITY = PASS
SAFETY_ABORT = NO (both stages; injected guard failures are separate unit checks)
POLICY_INFERENCE_STARTED = YES
PHYSICS_CONTROL_STARTED = YES
CLOSED_LOOP_STARTED = YES
RL_TRAINING_STARTED = NO
REAL_ROBOT_STARTED = NO
READY_FOR_FLAT_GROUND_CLOSED_LOOP = YES
READY_FOR_STAIRS = NO
```

Added files: `simulate/policy/controller.py`,
`simulate/tests/test_suspended_control.py`, `simulate/tests/test_controller_guards.py`,
`simulate/tests/suspended_control_results.json`, and this report. Frozen components,
policy artifact, XML, training configuration, reward, PPO, joystick, DDS and existing
elastic-band implementation were not changed. The next-stage gate is recorded;
unsupported flat-ground testing has not been started.
