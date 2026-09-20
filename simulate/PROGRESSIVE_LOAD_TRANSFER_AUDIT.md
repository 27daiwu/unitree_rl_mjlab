# PROGRESSIVE_LOAD_TRANSFER_AUDIT

**Stopped at L0; no further unloading or velocity tests were run.** Actual support
reached the 90–95% window, but stable bilateral foot contact was not established.
This is a standing/contact gate failure, not a NaN, fall, effort explosion or
controller watchdog abort. No lower support level is verified by this experiment.
The preceding suspended pipeline PASS remains valid for its narrower purpose.

## Frozen contract and experiment

Checkpoint `model_9400.pt`, normalized JIT, TRAINING_PARITY ray mode, observation
builder, joint order, action transform, PD, 0.005/0.02 s periods, decimation 4,
phase and previous action are unchanged. The existing controller/flat-scene
factory is reused unchanged. File hashes before and after the experiment verify
that observation/policy/controller sources, artifact, frozen run config and robot
XML files were not modified. No training task or terrain generation was started.

New composition layer: `simulate/policy/load_transfer.py`. It provides an
adjustable band, read-only contact/load measurements, and additional termination
guards around the existing controller. `simulate/tests/test_load_transfer.py`
owns independent stage resets and advancement gates.

Each planned trial starts from the same HOME q/root pose, zero dq and rest length
0, holds HOME for 20 physics steps, then resets policy state **once**. All subsequent
adjust/settle/formal steps preserve episode clock and raw previous action. Trial
length adjustment is independent, not a continuation from the previous stage.

## Suspension metric and adjustment

```text
robot_weight = total_mass * 9.81 = 327.07660302 N
vertical_support_ratio = max(0, actual_band_force_z) / robot_weight
left_foot_Fz, right_foot_Fz = summed world vertical external contact force ON each foot
total_ground_Fz = left_foot_Fz + right_foot_Fz
ground_load_ratio = total_ground_Fz / robot_weight
```

Contacts use `mj_contactForce`, transformed by `contact.frame.T` into world
coordinates with sign chosen for the robot side (force is on geom2). Self contacts
are excluded from external ground load. Foot bodies are the two ankle-roll bodies;
all their active external foot contacts are accumulated. Active external contacts
on any other robot body count as non-foot contacts. A 1e-6 N force-norm numerical
floor distinguishes active contact from a geometric near-contact. Ground load is
not inferred from height or penetration.

A separate known one-kg, two-foot static fixture verifies the sign/coordinate
conversion: each foot reads 4.905 N and their sum is 9.81 N. This metric test does
not run the G1 policy or change the experimental scene.

The inherited band has k=200 N/m, c=100 Ns/m, anchor
`[0,0,2.4353830151]`, body `robot/torso_link`. Anchor, stiffness and damping stay
fixed. Rest length starts at zero and is the only adjusted control parameter.
A tension-only slack extension prevents compression/downward force when the band
becomes slack; length-zero positive-tension behavior is verified exactly equal
to the previous band. Neither band disable nor a k→0 change is used.

For taut band, force magnitude is
`max(0, k*(distance-length) - c*dot(root_linear_velocity,direction))`;
for slack band it is zero. It follows the same root-position/torso-force attachment
convention as the previous test.

Rest length changes continuously at no more than 0.081769150755 m/s: equivalent
to **5 percentage points of robot weight per second** at fixed geometry.
The initial goal `(1-target_ratio)*weight/k` is only a starting estimate.
Every 20 policy steps the actual previous-0.4-second support mean can update that
goal; it is never accepted merely because a particular length was reached.

## Predeclared gates

Planned levels and support windows:

```text
L0 target 92.5%: 90–95%
L1 target 75%:   70–80%
L2 target 50%:   45–55%
L3 target 25%:   20–30%
L4 target 7.5%:   5–10%
L5 target 0%:     0–2%
```

Before a formal window starts, the last 0.4 s must satisfy the measured support
window, root height span <1 cm, root speed <0.1 m/s, bilateral contact fraction
≥90%, and no continuous nonbilateral interval longer than 0.1 s. These are explicit
experimental settle criteria, not claimed training terminations. The settle
budget is the nominal ramp duration plus 4 s; L0 had 274 policy steps (5.48 s).
The intended formal window is 60 policy steps (1.2 s), with ≥90% bilateral contact,
no nonbilateral interval >0.2 s and zero-command root speed ≤0.3 m/s. No thresholds
were loosened after observing the result.

Additional aborts:

- Exact frozen training orientation termination: total tilt >70°, with separate
  roll/pitch checks at the same bound. Source: saved `params/env.yaml` fell_over
  and installed `mdp.terminations.bad_orientation`.
- Root z <0.4376144083 m: an extra geometric collapse diagnostic derived from the
  HOME pelvis-to-knee vertical separation, not an invented training termination.
- Any active external non-foot body contact.
- All existing joint position/velocity, target-jump/error, effort and finite-value
  watchdogs remain enabled and unchanged.

A settle/contact FAIL stops the ladder just as a safety abort does. On termination,
actuation is disabled, simulation stops, and band support remains enabled.
No attempt is made to see whether further unloading might improve contact.

## Results

The following measured L0 metrics describe the **last settle window**, not a
formal standing PASS. The root and dynamics columns for lower stages are N/A.

| Support stage | Actual support | Zero command | vx test | Max roll rad | Max pitch rad | dq max rad/s | Torque max Nm | Ground load |
|---|---:|---|---|---:|---:|---:|---:|---:|
| L0 90–95% | 92.699% | FAIL: bilateral contact unsettled | Not run | 0.025971 | 0.029197 | 1.369504 | 4.819660 | 7.974% |
| 75% | N/A | NOT RUN: L0 failed | Not run | N/A | N/A | N/A | N/A | N/A |
| 50% | N/A | NOT RUN: L0 failed | Not run | N/A | N/A | N/A | N/A | N/A |
| 25% | N/A | NOT RUN: L0 failed | Not run | N/A | N/A | N/A | N/A | N/A |
| 5–10% | N/A | NOT RUN: L0 failed | Not run | N/A | N/A | N/A | N/A | N/A |
| 0% | N/A | NOT RUN: L0 failed | Not run | N/A | N/A | N/A | N/A | N/A |

Final settle window (0.4 s):

- Suspension mean 92.699% (range 92.076–93.467%).
- Left foot Fz mean 22.049842 N; right foot 4.032631 N.
- Ground mean 7.974%; support+ground mean 100.673%.
- Left contact fraction 98.75%; right 47.50%.
- Bilateral contact fraction **46.25%**, below 90%.
- Longest nonbilateral interval 0.105 s, above 0.1 s.
- Root speed max 0.033234 m/s; root height minimum 0.806899 m.
- Final rest length 0.113949925 m. No non-foot contact or safety abort.

Entire recorded startup/adjustment interval:

- Bilateral fraction 11.201%; longest nonbilateral interval 1.965 s.
- Support+ground mean 100.034%, consistent with average weight support.
- Max height drop from initial HOME 0.000186432 m.
- Max roll 0.037598629 rad (2.154°), pitch 0.039491242 rad (2.263°).
- Max joint speed 3.346851165 rad/s; torque 14.778167520 Nm.

274 policy steps completed; no formal-window steps were admitted. The last-window
signed height-drop field in raw JSON is negative because the root remained above
its initial height; the overall maximum positive drop is reported above.

No lowest stable suspension percentage can be assigned. In particular, the old
≥94% pipeline test must not be relabeled a normal-load standing validation. The
experiment does not show that free-load standing necessarily fails; it shows that
this protocol's initial contact gate failed, so lower loads remain untested.

## Reproduce and artifacts

```bash
conda activate unitree_rl_mjlab
python -m simulate.tests.test_load_metrics
python -m simulate.tests.test_load_transfer
```

`logs/load_transfer/support_95/load_samples.npz` records every physics substep's
root pose/roll/pitch/linear and angular velocity, foot Fz/contact flags, non-foot
contacts, suspension vector/ratio, ground ratio, rest length, max dq and torque.
`policy_steps.npz` contains the existing full obs/action/target/state/force records,
including raw action L2/max and q_target changes. Aggregate and last-window metrics
are in `result.json`; the stage progression and frozen-file hashes are in
`logs/load_transfer/report.json`. A reviewable copy is
`simulate/tests/load_transfer_results.json`.

Added files only: `simulate/policy/load_transfer.py`,
`simulate/tests/test_load_transfer.py`, `simulate/tests/test_load_metrics.py`,
`simulate/tests/load_transfer_results.json`, and this audit. No preexisting frozen
implementation was edited. Metric unit tests are separate from the single formal
ladder attempt and do not constitute retries at another support level.

## Final audit

```text
PROGRESSIVE_LOAD_TRANSFER_AUDIT
CHECKPOINT = model_9400.pt
ROBOT_WEIGHT_N = 327.07660302
SUPPORT_STAGES = 92.5%,75%,50%,25%,7.5%,0%; advance only after PASS
STAGE_95_ZERO = FAIL_CONTACT_SETTLE_GATE (actual final mean 92.6987%)
STAGE_75_ZERO = NOT_RUN
STAGE_50_ZERO = NOT_RUN
STAGE_25_ZERO = NOT_RUN
STAGE_10_ZERO = NOT_RUN
STAGE_0_ZERO = NOT_RUN
LOWEST_STABLE_SUSPENSION_RATIO = NOT_ESTABLISHED
MAX_ROOT_HEIGHT_DROP = 0.000186432115438
MAX_ABS_ROLL = 0.0375986291041
MAX_ABS_PITCH = 0.0394912415915
MAX_JOINT_VELOCITY = 3.34685116518
MAX_TORQUE = 14.7781675201
MIN_BILATERAL_CONTACT_RATIO = 0.112007168459
MAX_NONFOOT_CONTACT_COUNT = 0
OBS_FINITE = YES
ACTION_FINITE = YES
CTRL_FINITE = YES
SAFETY_ABORT = NO
PROTOCOL_STOP = YES (failed standing-contact gate)
FREE_LOAD_ZERO_COMMAND = NOT_TESTED
FREE_LOAD_VX_01 = NOT_RUN
FREE_LOAD_VX_02 = NOT_RUN
FREE_LOAD_VX_03 = NOT_RUN
FROZEN_OBSERVATION_MODIFIED = NO
POLICY_MODIFIED = NO
PD_MODIFIED = NO
RL_TRAINING_STARTED = NO
REAL_ROBOT_STARTED = NO
READY_FOR_FLAT_GROUND_FREE_WALK = NO
READY_FOR_RISER_TEST = NO
READY_FOR_STAIRS = NO
```

`MIN_BILATERAL_CONTACT_RATIO` above is the minimum whole-trial aggregate among
executed trials (only L0), not an instantaneous minimum or a formal-window result.
