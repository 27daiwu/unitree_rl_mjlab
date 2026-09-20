# STABLE_CONTACT_POLICY_HANDOVER_AUDIT

Classification **B: POLICY_TARGET_ASYMMETRY_FROM_STABLE_CONTACT**. Policy OFF preserves bilateral contact; the frozen policy disrupts it from the same contact-settled state. This is a bounded suspended-state result, not a general claim that the checkpoint cannot stand unsupported.

## Shared state and experiment controls

Settled simulation time: 2.04000000 s. The preceding 0.4 s has 100% bilateral contact, no non-foot contact, finite state, and peak root speed 0.04966090 m/s. Mean actual support is 92.32046%; target remains 92.5%.

Settling uses actor OFF, HOME PD, and the existing L0 slow rest-length adjustment law. A 15 s diagnostic timeout is used; no root height is selected or edited. On qualification, the rest length is held fixed in both branches, as in the existing formal L0 window. Actual force changes with motion; this is not a lower-support stage.

The saved `mjSTATE_INTEGRATION` includes simulation time, qpos, qvel, act, solver warmstart, controls and applied forces. Band anchor/rest length/force, root pose and contact snapshot are also saved. Both branches restore this state with exact array equality and reproduce the same initial observation. Policy history is reset to zero, episode_step=0, command=0. Both branches run 40 policy-step equivalents (0.8 s), with 5 ms telemetry.

At the saved instant: Fz L/R = 13.901149299/14.235584027 N; clearance L/R = -0.000050153471/-0.000048667246 m. Small negative clearances are compliant contact penetration.

## OFF / ON results

| Metric (whole 0.8 s) | OFF | ON |
|---|---:|---:|
| bilateral_contact_ratio | 1.000000000 | 0.125000000 |
| left_contact_ratio | 1.000000000 | 0.200000000 |
| right_contact_ratio | 1.000000000 | 0.125000000 |
| left_foot_fz_mean | 13.857616235 | 27.058668041 |
| right_foot_fz_mean | 14.107328996 | 10.802392136 |
| max_abs_roll | 0.000900153 | 0.027305048 |
| max_abs_pitch | 0.071141521 | 0.064065787 |
| root_speed_max | 0.047907760 | 0.488579222 |
| support_mean | 0.913967456 | 0.878740992 |
| max_nonfoot_contact_count | 0.000000000 | 0.000000000 |

Neither branch safety-aborted. The ON peak root speed is a diagnostic result; the pre-handover 0.1 m/s qualification is not silently reused as a new post-handover abort rule.

## First action and mirror convention

Angular joint axes transform as axial vectors under sagittal reflection: `axis_mirror = diag(-1,+1,-1) @ axis_right`. Verified on compiled-model world axes at zero joint coordinates, including XML body rotations. Pitch/knee mirror with the same sign; roll/yaw with the opposite sign.

`delta = q_target - current_q`; signed asymmetry is `delta_left - mirror_sign * delta_right` (radians).

| Joint | L raw | R raw | L delta | R delta | Mirror sign | L-R mirrored delta |
|---|---:|---:|---:|---:|---:|---:|
| hip_pitch | 0.147116125 | -0.000906236 | 0.061292173 | -0.019606954 | 1 | 0.080899127 |
| hip_roll | -0.164025277 | 0.260064095 | -0.057808536 | 0.091342331 | -1 | 0.033533795 |
| hip_yaw | -0.107661016 | -0.161723390 | -0.058700876 | -0.088819423 | -1 | -0.147520299 |
| knee | -0.103319801 | -0.095100090 | -0.026122629 | -0.023126779 | 1 | -0.002995851 |
| ankle_pitch | 1.430991650 | 1.210550070 | 0.680850680 | 0.585601613 | 1 | 0.095249068 |
| ankle_roll | 0.120173536 | 0.138256520 | 0.053298568 | 0.060953114 | -1 | 0.114251683 |

Largest asymmetries: hip_yaw (0.147520 rad), ankle_roll (0.114252 rad), ankle_pitch (0.095249 rad).

Both ankle-pitch targets also move strongly in the same direction: +0.680851/+0.585602 rad relative to current q. Ranking the asymmetric components does not establish that hip yaw or any single joint causes unloading: no action interventions were performed.

All 29 raw actions, targets, current positions and deltas are in `logs/settled_handover/first_action_29.csv` and JSON; all 40 policy ticks (obs/action/target/q/dq/substep torque) are in `policy_on_steps.npz`. Post-step poses/Fz/contact/torque are in the branch physics JSONs.

## Contact chronology

| Policy step | Relative time s | L Fz N | R Fz N | L/R contact | Roll rad | Pitch rad | Root xyz m |
|---|---:|---:|---:|---|---:|---:|---|
| 1 | 0.020 | 131.266423 | 107.685279 | [True, True] | -0.0009517 | 0.0636570 | [-0.0093239, 0.0001365, 0.7841608] |
| 2 | 0.040 | 170.434334 | 130.663664 | [True, True] | -0.0029476 | 0.0571694 | [-0.0095834, 4.47e-05, 0.7878823] |
| 3 | 0.060 | 167.645106 | 96.540030 | [True, True] | -0.0040468 | 0.0486796 | [-0.0107205, -5.19e-05, 0.7943785] |
| 5 | 0.100 | 81.728633 | 0.000000 | [True, False] | -0.0056995 | 0.0340431 | [-0.0126135, -0.0005705, 0.8125192] |
| 10 | 0.200 | 0.000000 | 0.000000 | [False, False] | 0.0120243 | 0.0448946 | [-0.0127241, -0.0047616, 0.8496243] |
| 20 | 0.400 | 0.000000 | 0.000000 | [False, False] | 0.0035511 | -0.0030764 | [-0.0021471, -0.0111201, 0.8662188] |
| 40 | 0.800 | 193.081225 | 26.381104 | [True, True] | -0.0236530 | 0.0070531 | [0.0260046, -0.0043439, 0.7907091] |

At t=0 targets are applied. At 5 ms, measured joint motion exceeds 1e-5 rad and both feet briefly have zero force. At 10 ms both recover contact and the change in L-R force difference exceeds 1 N. The right foot subsequently loses contact while the left remains loaded at 0.085 s; this begins a right-contact absence lasting at least 50 ms. The right force first falls below 90% of its initial value at 5 ms, but this first event is bilateral unloading, not selective right unloading.

The targets precede the observed loss. Joint motion and the first unloading share the 5 ms measurement bin, so their sub-bin ordering is unresolved. Later both feet are airborne and both recover by step 40: the behavior is a transient contact disruption and load asymmetry, not permanent right-foot lift.

## Offline inference and OOD

Offline saved-observation inference exactly matches online first action (max error 0; tolerance 1e-6). Checkpoint normalizer mean/std/var/count are exactly equal to the JIT buffers. Diagnostic z uses `(obs-mean)/(std+eps)`, eps=0.01.

| Block | Max abs z | Count >3 | Count >5 |
|---|---:|---:|---:|
| base_ang_vel | 0.044517264 | 0 | 0 |
| projected_gravity | 1.088032246 | 0 | 0 |
| command | 5.904021740 | 1 | 1 |
| phase | 0.010158075 | 0 | 0 |
| joint_pos | 1.311855912 | 0 | 0 |
| joint_vel | 0.016399812 | 0 | 0 |
| previous_action | 0.591437578 | 0 | 0 |
| height_scan | 2.406420231 | 0 | 0 |

| Index | Block | Value | Mean | Std | z |
|---:|---|---:|---:|---:|---:|
| 6 | command | 0.000000000 | 0.400013357 | 0.057752691 | -5.904021740 |
| 150 | height_scan | 0.038517494 | 0.147617489 | 0.035337053 | -2.406420231 |
| 214 | height_scan | 0.038451072 | 0.146314085 | 0.037339959 | -2.278476954 |
| 198 | height_scan | 0.156591073 | 0.047721416 | 0.042006079 | 2.093402386 |
| 166 | height_scan | 0.156591073 | 0.054930914 | 0.047993269 | 1.752964854 |
| 27 | joint_pos | -0.047027543 | 0.036907237 | 0.053981710 | -1.311855912 |
| 34 | joint_pos | 0.045821384 | -0.038246572 | 0.056640942 | 1.261506081 |
| 3 | projected_gravity | 0.064046100 | -0.020553445 | 0.067754634 | 1.088032246 |
| 149 | height_scan | 0.019160245 | 0.104584016 | 0.072257087 | -1.038497448 |
| 213 | height_scan | 0.018950827 | 0.102637976 | 0.072821401 | -1.010453224 |
| 25 | joint_pos | -0.049024902 | -0.007286280 | 0.058368608 | -0.610493958 |
| 92 | previous_action | 0.000000000 | -0.317897528 | 0.527499735 | 0.591437578 |
| 85 | previous_action | 0.000000000 | 0.307329059 | 0.528337002 | -0.570886016 |
| 18 | joint_pos | -0.000147874 | 0.031435210 | 0.049086541 | -0.534522533 |
| 12 | joint_pos | 0.000291192 | -0.029999394 | 0.050309032 | 0.502256215 |
| 70 | previous_action | 0.000000000 | 0.243443668 | 0.510578156 | -0.467640966 |
| 76 | previous_action | 0.000000000 | -0.241186678 | 0.517198682 | 0.457487255 |
| 36 | joint_pos | 0.029152691 | -0.003672892 | 0.069265060 | 0.414124250 |
| 14 | joint_pos | -0.010107636 | 0.062842831 | 0.167782500 | -0.410335481 |
| 71 | previous_action | 0.000000000 | -0.149293229 | 0.364737719 | 0.398393929 |

Only command vx exceeds 3 and 5. Normalizer statistics are marginal moments, not proof of causal attribution or full multivariate in-distribution status. Full block values, including 176 height values, are saved in report JSON.

## Training-environment reference

Existing Phase2C8G rollout reports provide aggregate/trajectory evidence, not enough paired zero-command raw actions and joint states. Saved training config has vx=[0.3,0.5] and rel_standing_envs=0.0. Thus zero-command standing is excluded by that run command sampler; inherited checkpoint history is not established by this audit.

A separate actual mjlab GPU Phase2C8G environment sampled 20 environments for 200 steps with the same frozen normalized JIT, deterministic mean actions, seed 42, and zero commands. This evaluation-only instance retains training observation noise, terrain, physics and randomization; no optimizer or training was run. This is an unsuspended reference, not another support ladder branch. Of 4000 frames, 562 have speed<0.1 m/s and episode age>=0.4 s; 520 additionally have bilateral sensor contact. Samples are correlated, not independent trials.

| Joint | Settled abs asymmetry rad | Reference median | Reference P90 | Reference P95 | Settled percentile |
|---|---:|---:|---:|---:|---:|
| hip_pitch | 0.080899 | 0.206844 | 0.512924 | 0.619464 | 21.92% |
| hip_roll | 0.033534 | 0.128689 | 0.262959 | 0.298793 | 13.85% |
| hip_yaw | 0.147520 | 0.104703 | 0.228238 | 0.256238 | 66.15% |
| knee | 0.002996 | 0.101516 | 0.284542 | 0.347608 | 0.96% |
| ankle_pitch | 0.095249 | 0.227919 | 0.555601 | 0.653053 | 22.69% |
| ankle_roll | 0.114252 | 0.061894 | 0.148591 | 0.173189 | 78.85% |

The current first-frame asymmetries are not unusually large relative to this evaluator reference (all below P90). This cannot establish that zero-command behavior is within the original training distribution. Raw action ranges for all 29 joints and all sampled obs/actions/q/targets are preserved in reference JSON/NPZ.

## Final gate

```text
STABLE_CONTACT_POLICY_HANDOVER_AUDIT
CONTACT_SETTLED_STATE = /home/hebe/unitree_workspace/src/unitree_rl_mjlab/logs/settled_handover/contact_settled_state.npz
STABLE_CONTACT_ESTABLISHED_BEFORE_POLICY = YES
SETTLED_LEFT_FZ = 13.901149299354302
SETTLED_RIGHT_FZ = 14.235584027395404
SETTLED_BILATERAL_CONTACT = 1.0
POLICY_OFF_BILATERAL = 1.0
POLICY_ON_BILATERAL = 0.125
POLICY_INTRODUCED_RIGHT_UNLOADING_FROM_STABLE_CONTACT = YES
FIRST_RAW_ACTION = [0.14711612462997437, -0.16402527689933777, -0.10766101628541946, -0.10331980139017105, 1.4309916496276855, 0.12017353624105453, -0.000906236469745636, 0.26006409525871277, -0.16172339022159576, -0.09510008990764618, 1.21055006980896, 0.1382565200328827, 0.018092282116413116, 0.04438934847712517, -0.1069987416267395, -0.13284577429294586, 1.504037857055664, 1.0086922645568848, -0.17909923195838928, 0.1860082447528839, 0.040382057428359985, 0.7148014307022095, -0.15937303006649017, -1.3219528198242188, -0.8936541080474854, -0.30196577310562134, -0.002704988233745098, -0.045906491577625275, -0.491655558347702]
FIRST_Q_TARGET = [-0.019447087800014806, -0.057517344120714485, -0.058949408669109374, 0.2637697388594827, 0.42760047096024634, 0.05270538672927413, -0.10049620806375889, 0.09119445699590248, -0.0885510702987653, 0.2666520749411557, 0.33091979500632607, 0.060636173188413904, 0.009906365080279258, 0.01946816122190587, -0.04692722069583579, 0.2917368511840654, 0.8396368905390879, 0.44238954396463703, 0.7914511446903231, 0.0815789963513201, 0.0030084984241089577, 0.053253328700091494, 0.2801025986011462, -0.7597785240044843, -0.39193641828404874, 0.7375646671029152, -0.0011863464737502385, -0.0034200735763072723, -0.036628766999243584]
TOP_ASYMMETRIC_LEG_TARGETS = [{"joint": "hip_yaw", "magnitude": 0.14752029910179462}, {"joint": "ankle_roll", "magnitude": 0.11425168251785492}, {"joint": "ankle_pitch", "magnitude": 0.09524906774494224}, {"joint": "hip_pitch", "magnitude": 0.08089912658439327}, {"joint": "hip_roll", "magnitude": 0.03353379531400252}, {"joint": "knee", "magnitude": 0.002995850705486802}]
RIGHT_FZ_DECLINE_TIME = 0.004999999999999893
RIGHT_CONTACT_LOSS_TIME = 0.004999999999999893
RIGHT_SELECTIVE_CONTACT_LOSS_TIME = 0.08499999999999819
RIGHT_CONTACT_LOSS_AT_LEAST_50MS_TIME = 0.08499999999999819
FIRST_ACTION_OFFLINE_ONLINE_PARITY = PASS
FIRST_ACTION_MAX_ERROR = 0.0
RIGHT_UNLOAD_TEMPORALLY_FOLLOWS_POLICY_TARGETS = YES; motion/unload share first 5ms bin; selective loss at 85ms
NORMALIZED_OBS_MAX_ABS = 5.904021739959717
OOD_3SIGMA_COUNT = 1
OOD_5SIGMA_COUNT = 1
TRAINING_ZERO_COMMAND_REFERENCE = 20 environments x 200 steps; 520 low-speed bilateral samples; frozen JIT in actual mjlab
ZERO_COMMAND_ASYMMETRY_WITHIN_TRAINING_DISTRIBUTION = UNRESOLVED: original run sampled vx 0.3-0.5, standing fraction 0
WITHIN_EVALUATOR_ZERO_COMMAND_REFERENCE = YES: all six target asymmetries below reference 90th percentile
FAILURE_PRIMARY = POLICY_TARGET_ASYMMETRY_FROM_STABLE_CONTACT
FAILURE_CLASS = B
FAILURE_SECONDARY = large bilateral ankle-pitch target deltas; high suspension support; zero-command distribution mismatch (causality not isolated)
CONTACT_GEOM_PARITY = NOT_EXACT; existing 1 mm lateral mismatch unchanged
READY_TO_RETRY_L0 = NO
READY_FOR_LOWER_SUPPORT = NO
FROZEN_COMPONENTS_MODIFIED = NO
RL_TRAINING_STARTED = NO
REAL_ROBOT_STARTED = NO
```

The OFF control is nearly load-symmetric, so the existing 1 mm collision mismatch was not investigated further. No frozen component, contact gate, action, PD, observation or model was changed. No corrective L0 retry or lower-support stage was run.

Reproduce in the existing conda environment:

```bash
python -m simulate.tests.audit_settled_handover
python -m simulate.tests.audit_settled_training_reference
python -m simulate.tests.report_settled_handover
```
