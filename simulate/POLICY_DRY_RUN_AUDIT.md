# POLICY_DRY_RUN_AUDIT

Nine frozen TRAINING_PARITY observation snapshots pass deterministic raw-action
parity against the real installed RSL-RL actor. No physics loop, control write,
training, DDS connection, or real robot was started. Observation code is unchanged.

## Checkpoint and inference path

Selected the original Phase2C8g second-velocity-control pilot `model_9400.pt`,
the frozen reference used by existing Phase2C context audits, rather than a
same-named copy in subsequent intervention runs.

Checkpoint: `/home/hebe/unitree_workspace/src/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-09-15_15-49-46_phase2c8g-second-velocity-control-pilot/model_9400.pt`

SHA256: `1a7243a1dd229ae5567a0d117b8bada9171ea685f53a5e01b804f1ae02614232`

Format is a PyTorch checkpoint dictionary with `actor_state_dict`, critic,
optimizer, iteration and environment metadata. The audit loads only actor
parameters into the installed `rsl_rl.models.MLPModel` with strict loading.
Architecture and activation come from the run's saved `params/agent.yaml`:
274 → 512 → 256 → 128 → 29, ELU, deterministic Gaussian mean output.
No network or weight parser was reimplemented.

The checkpoint contains `obs_normalizer._mean`, `_var`, `_std`, and `count`;
mean/std shapes are `(1,274)`. Epsilon is **0.01**, read from the installed
`EmpiricalNormalization` implementation (it is not a checkpoint tensor).
The exact operation is `(raw_obs - mean) / (std + eps)`.

Installed `MLPModel.forward → get_latent → obs_normalizer → mlp →
distribution.deterministic_output` is the reference. There is no runner-external
normalization. The official `OnPolicyRunner.export_policy_to_jit` uses
`actor.as_jit()` then `torch.jit.script(...).save(...)`; this test uses that exact
export path without constructing a training runner. Its `_TorchMLPModel` copies
the normalizer, MLP and deterministic-output module. The official ONNX wrapper
also includes normalization; JIT was chosen for same-backend verification.

The inference-only deployment wrapper `simulate/policy/inference.py` accepts
finite raw `(274,)` observations and returns finite raw `(29,)` actions. It
loads the JIT in CPU/eval mode, applies **no external normalization**, action
scaling, clamp, offset, PD, or output reordering. Mean/std/variance/count and
normalizer outputs were compared exactly against the training actor.

## Action semantics and ordering

Saved `params/env.yaml` defines `JointPositionAction`, all 29 actuated joints,
`preserve_order=False`, `use_default_offset=True`, and no action clip. Saved
agent config also has `clip_actions: null`. Installed `BaseAction.process_actions`
retains raw actions and computes scale/offset; `JointPositionAction.apply_actions`
then subtracts the entity's encoder bias:

```text
processed_action[i] = default_joint_pos[i] + scale[i] * raw_action[i]
q_target[i] = processed_action[i] - encoder_bias[i]
```

The configured scalar offset 0 is overridden by default joint positions.
Encoder bias is randomized in training; at zero bias this reduces to the usual
`default_q + scale * raw_action`. These formulas were audited only, never applied.
`ActionManager.process_action` stores the unscaled input; `mdp.last_action`
reads that buffer. No clip is configured in either the wrapper or action term.

`Entity.find_joints_by_actuator_names` resolves actuated joints in natural entity
joint order; this is the previously audited 29-name list. Neither official JIT
export nor the deployment wrapper permutes output components. Every fixture
includes `joint_actions.csv` with index/name/reference/deployment/absolute error.

The offline sequence builds a frozen moving-state observation, infers action_t,
calls the existing `Builder.set_previous_action(action_t)`, and builds the next
observation. Its `[69:98]` exactly equals raw action_t. Before/after assertions
verify identical ctrl, q, dq, and simulation time. `mj_forward` is used once to
populate snapshot kinematics; there is no `mj_step` or actuator command.

## Results

Tolerance **1e-6 absolute**, chosen before testing for identical CPU float32
PyTorch/JIT operations. Actual errors were zero for every element. All 261
fixture action values are finite. No tolerance relaxation or clamping occurred.

| Fixture | Action min | Action max | Action mean | Action L2 | Max error | Mean error | L2 error |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat_upright | -0.678747892 | 0.691573143 | -0.0242963787 | 1.74589264 | 0 | 0 | 0 |
| stair_approach | -1.20695496 | 0.878123403 | 0.0166568477 | 2.4885571 | 0 | 0 | 0 |
| yaw | -1.19570994 | 0.858934581 | 0.0161672179 | 2.48263884 | 0 | 0 | 0 |
| pitch | -1.27010393 | 2.06733894 | 0.117310286 | 3.59842396 | 0 | 0 | 0 |
| yaw_pitch_roll | -1.09949017 | 1.77984846 | 0.13075082 | 3.18735433 | 0 | 0 | 0 |
| singular_pitch | -2.42497206 | 1.83768177 | 0.0504826605 | 5.20547771 | 0 | 0 | 0 |
| moving | -1.03981984 | 1.96751797 | 0.137629583 | 3.2978549 | 0 | 0 | 0 |
| long_episode | -1.04114532 | 0.9384951 | 0.00740243774 | 2.55055428 | 0 | 0 | 0 |
| two_riser | -1.6380769 | 1.18370366 | 0.0859057605 | 3.49978185 | 0 | 0 | 0 |

Historical sanity: `doc/g1_phase2c8k_context_evidence.json`, checkpoint
`model_9400`, contains 1500 saved raw 29D action rows.
Their range is [-4.378300189971924, 5.849735260009766]; current fixture range is
[-2.4249721, 2.0673389], consistent in magnitude. This is a scale sanity check,
not a locomotion performance or stability claim.

Resolved per-joint scale and offset (in audited training order):

```text
ACTION_SCALE = [0.5475464629911068, 0.35066146637882434, 0.5475464629911068, 0.35066146637882434, 0.43857731392336724, 0.43857731392336724, 0.5475464629911068, 0.35066146637882434, 0.5475464629911068, 0.35066146637882434, 0.43857731392336724, 0.43857731392336724, 0.5475464629911068, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.07450087032950714, 0.07450087032950714, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.07450087032950714, 0.07450087032950714]
ACTION_OFFSET = [-0.10000000149011612, 0.0, 0.0, 0.30000001192092896, -0.20000000298023224, 0.0, -0.10000000149011612, 0.0, 0.0, 0.30000001192092896, -0.20000000298023224, 0.0, 0.0, 0.0, 0.0, 0.3499999940395355, 0.18000000715255737, 0.0, 0.8700000047683716, 0.0, 0.0, 0.0, 0.3499999940395355, -0.18000000715255737, 0.0, 0.8700000047683716, 0.0, 0.0, 0.0]
```

## Final gate

```text
POLICY_DRY_RUN_AUDIT
CHECKPOINT = logs/rsl_rl/g1_velocity/2026-09-15_15-49-46_phase2c8g-second-velocity-control-pilot/model_9400.pt
CHECKPOINT_ITERATION = 9400
POLICY_LOAD = PASS
ACTOR_INPUT_DIM = 274
ACTOR_OUTPUT_DIM = 29
OBS_NORMALIZATION_ENABLED = YES
OBS_NORMALIZER_PRESENT_IN_CHECKPOINT = YES
NORMALIZER_SOURCE = checkpoint actor_state_dict.obs_normalizer buffers
NORMALIZER_LOCATION = INSIDE_JIT
NORMALIZER_EPS = 0.01
NORMALIZER_PARITY = PASS
DEPLOYMENT_ARTIFACT = logs/policy_parity/model_9400_normalized.jit
DEPLOYMENT_RUNTIME = torch.jit CPU float32, eval / inference_mode
RAW_ACTION_DIM = 29
ACTION_ORDER_PARITY = PASS
ACTION_SCALE = per-joint vector above, from saved training run configuration
ACTION_OFFSET = default_joint_pos vector above
ACTION_TARGET_FORMULA = default_joint_pos + scale * raw_action - encoder_bias
FIXTURES_TESTED = 9
WORST_ACTION_MAX_ABS_ERROR = 0
WORST_ACTION_MEAN_ABS_ERROR = 0
WORST_ACTION_L2_ERROR = 0
ACTION_NAN_COUNT = 0
ACTION_INF_COUNT = 0
PREVIOUS_ACTION_SEMANTICS = unscaled raw policy output; no configured clip; exact offline PASS
FULL_274D_PARITY = PASS (frozen prerequisite)
POLICY_INFERENCE_STARTED = YES
PHYSICS_CONTROL_STARTED = NO
CLOSED_LOOP_STARTED = NO
RL_TRAINING_STARTED = NO
REAL_ROBOT_STARTED = NO
POLICY_ACTION_PARITY = PASS
READY_FOR_SUSPENDED_CLOSED_LOOP = YES
```

This gate records only the requested prerequisites. Suspended closed-loop control
has not been implemented or started, and arbitrary replacement XML/exported
policies are outside these verified snapshot results.

## Reproduce and outputs

```bash
conda activate unitree_rl_mjlab
python -m simulate.tests.test_policy_parity
```

`logs/policy_parity/<fixture>/` contains `obs.npy`, `training_raw_action.npy`,
`deployment_action.npy`, and `joint_actions.csv`. The root contains the JIT,
`report.json` (checkpoint/artifact hashes, scale/offset and per-fixture statistics),
and `previous_action_sequence.npz`. A reviewable report copy is stored at
`simulate/tests/policy_parity_results.json`.

Added files: `simulate/policy/inference.py`,
`simulate/tests/test_policy_parity.py`, `simulate/tests/policy_parity_results.json`,
and this report. No frozen observation implementation was changed.
