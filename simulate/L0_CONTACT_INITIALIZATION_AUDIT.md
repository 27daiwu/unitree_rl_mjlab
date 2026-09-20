# L0_CONTACT_INITIALIZATION_AUDIT

**The pronounced right-foot unloading is introduced by the policy after takeover,
not by an initial left/right sole-height mismatch.** Both feet initially float
15.79782 mm above the plane. With the same initial state, frozen HOME PD and
identical recorded band rest-length history, actor-OFF settles into bilateral
contact; actor-ON becomes strongly asymmetric. This isolates the effect of policy
commands under this suspension condition, not the suitability of the checkpoint
for unsupported walking.

No support stage below L0, altered root initialization, gate change, policy change,
PD change, XML edit or corrective retry was performed.

## HOME static geometry (before policy)

Ground is the existing world plane z=0. Capsule lowest z is computed from exact
FK as `geom_world_z - half_length*abs(world_axis_z) - radius`, not from mesh
visual bounds or body-origin height. All seven collision capsules per foot are
included.

```text
LEFT_SOLE_LOWEST_Z = 0.015797817293 m
RIGHT_SOLE_LOWEST_Z = 0.015797817293 m
GROUND_Z = 0 m
LEFT_SOLE_CLEARANCE = 0.015797817293 m
RIGHT_SOLE_CLEARANCE = 0.015797817293 m
LEFT_RIGHT_SOLE_Z_DIFF = 0.000000000000 m
PELVIS_XYZ = [0.0, 0.0, 0.8]
PELVIS_ROLL_PITCH_YAW = [0.0, -0.0, 0.0] rad
LEFT_ANKLE_PITCH_ROLL_Q = [-0.20000000298023224, 0.0] rad
RIGHT_ANKLE_PITCH_ROLL_Q = [-0.20000000298023224, 0.0] rad
```

robot/left_ankle_roll_link: world position `[-0.02600160389750885, 0.11850645499999998, 0.05079781827647041]`, quaternion wxyz `[1.0, 0.0, 3.7252902707063384e-09, 0.0]`.

robot/right_ankle_roll_link: world position `[-0.02600160389750885, -0.11850645499999998, 0.05079781827647041]`, quaternion wxyz `[1.0, 0.0, 3.7252902707063384e-09, 0.0]`.

The vertical geometry is symmetric to the requested 1e-5 m precision. Startup is
not within ±5 mm of the contact boundary; it has approximately 15.8 mm clearance.
No new root height is selected.

## Collision geometry audit

Fields below are from the compiled model used in the experiment. All foot geoms
are capsules. Positions are body-local, sizes are `[radius, half_length, unused]`.
World positions, FK minima, body poses, and all solver/contact fields are also
stored per geom in `logs/l0_contact/report.json`.

| Side | ID | Name suffix | Local position m | Size m |
|---|---:|---|---|---|
| left | 14 | left_foot1_collision | [0.07500000000000001, -0.0265, -0.025] | [0.01, 0.02500499950009998, 0.0] |
| left | 15 | left_foot2_collision | [0.0395, -0.018, -0.025] | [0.01, 0.08349999999999999, 0.0] |
| left | 16 | left_foot3_collision | [0.03900000000000001, -0.01, -0.025] | [0.01, 0.091, 0.0] |
| left | 17 | left_foot4_collision | [0.03900000000000001, 0.0, -0.025] | [0.01, 0.093, 0.0] |
| left | 18 | left_foot5_collision | [0.03900000000000001, 0.01, -0.025] | [0.01, 0.091, 0.0] |
| left | 19 | left_foot6_collision | [0.0395, 0.018, -0.025] | [0.01, 0.08349999999999999, 0.0] |
| left | 20 | left_foot7_collision | [0.07500000000000001, 0.026, -0.025] | [0.01, 0.025, 0.0] |
| right | 31 | right_foot1_collision | [0.07500000000000001, -0.026, -0.025] | [0.01, 0.025, 0.0] |
| right | 32 | right_foot2_collision | [0.0395, -0.018, -0.025] | [0.01, 0.08349999999999999, 0.0] |
| right | 33 | right_foot3_collision | [0.03900000000000001, -0.01, -0.025] | [0.01, 0.091, 0.0] |
| right | 34 | right_foot4_collision | [0.03900000000000001, 0.0, -0.025] | [0.01, 0.093, 0.0] |
| right | 35 | right_foot5_collision | [0.03900000000000001, 0.01, -0.025] | [0.01, 0.091, 0.0] |
| right | 36 | right_foot6_collision | [0.0395, 0.018, -0.025] | [0.01, 0.08349999999999999, 0.0] |
| right | 37 | right_foot7_collision | [0.07500000000000001, 0.026, -0.025] | [0.01, 0.025, 0.0] |

Common fields (verified equal across all 14 foot collision geoms):

```text
margin = 0.0
gap = 0.0
solref = [0.02, 1.0]
solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
friction = [0.6, 0.005, 0.0001]
contype = 1
conaffinity = 1
```

**Strict full collision-geometry mirror parity fails**, although sole heights pass.
The correct mirror pairing reverses capsule indices: left 1 ↔ right 7, left 2 ↔
right 6, etc. Six pairs are exact mirrors. One pair differs:

```text
g1.xml:98  left_foot1:  (0.1,-0.026,-0.025) → (0.05,-0.027,-0.025)
g1.xml:146 right_foot7: (0.1,+0.026,-0.025) → (0.05,+0.026,-0.025)
```

The second endpoint has a 1 mm lateral mismatch; half-lengths differ by about
0.000005 m. Radius, z, margin/gap, solref/solimp, friction and collision masks are
identical. This preexisting XML asymmetry is reported, not modified. It can
contribute a small lateral load bias; this experiment does not isolate its
individual causal magnitude. It cannot explain a HOME left/right sole z difference
because that difference is exactly zero.

## Matched policy OFF / ON diagnostic

Both arms reuse the existing frozen controller factory and HOME initial state.
Actor-OFF holds the HOME position target through the same frozen native PD;
joints/root remain dynamic, not artificially locked. Initial dq is zero; dq is
not overwritten each step. The actor is never invoked in the OFF arm.

Each arm runs 0.1 s HOME startup plus 4.9 s recorded diagnostic (5 s total).
They replay the exact original L0 rest-length trace, continuously rate-limited by
the existing monitor. No new target support or adaptive correction is chosen.
Using identical length inputs controls the external input; actual support ratios
can differ as the robot's state changes. The actor-ON replay is **bit-identical**
to the original L0 first 245 policy steps for obs/raw_action/target/q/dq.

| Metric | Actor OFF | Actor ON |
|---|---:|---:|
| Full 4.9 s left contact | 88.8776% | 76.1224% |
| Full 4.9 s right contact | 88.9796% | 9.5918% |
| Full 4.9 s bilateral contact | 88.8776% | 8.7755% |
| Last 0.4 s left contact | 100.0000% | 97.5000% |
| Last 0.4 s right contact | 100.0000% | 7.5000% |
| Last 0.4 s bilateral contact | 100.0000% | 7.5000% |
| Last 0.4 s support | 94.0860% | 92.3006% |
| Last 0.4 s Left Fz mean | 9.490392048 N | 24.643455587 N |
| Last 0.4 s Right Fz mean | 9.739015082 N | 0.283331405 N |

Neither diagnostic arm aborted. OFF's full-interval contact fraction includes the
initial aerial interval; its final bilateral fraction is 100%. ON's last-window
7.5% is at simulation time 4.6–5.0 s. It does not replace the original L0 result
46.25%, whose last window ends later at 5.58 s. All percentages are explicitly
windowed to avoid comparing different times as if they were the same sample.

## Force-line audit (no suspension change)

The existing spring direction is computed from root position to anchor, while
`xfrc_applied` applies force at the torso **inertial COM** (`xipos`), not at its body
frame origin (`xpos`). The relevant diagnostic moment is therefore
`cross(torso_xipos - robot_subtree_COM, applied_force)`.


HOME:

```text
anchor = [0.0, 0.0, 2.4353830151]
torso body origin = [-0.0039635, 0.0, 0.8440000000000001]
force application position = [-0.0019319200000000002, 0.000339683, 1.0285680000000001]
robot COM = [0.0064131950457178245, 8.226090168117387e-05, 0.704484886570613]
force = [0.0, 0.0, 327.07660302000005] N
r_cross_F = [0.08419674546040212, 2.7294918809644786, -0.0] Nm
```

Before policy, t=0.1 s:

```text
anchor = [0.0, 0.0, 2.4353830151]
torso body origin = [-0.0001410681611001864, 6.851156624099217e-05, 0.8446602790989316]
force application position = [-0.0025112291033578794, 0.00028929562465770497, 1.0292244221965123]
robot COM = [0.006388686985658213, 8.183270875987261e-05, 0.7043953046581889]
force = [-0.6533650650597567, -0.01218723903465929, 326.3654771283152] N
r_cross_F = [0.07166750363428458, 2.692393363279976, 0.0002440144263083232] Nm
```

OFF after 5 s:

```text
anchor = [0.0, 0.0, 2.4353830151]
torso body origin = [-0.022615501765263897, 0.00038472196079992614, 0.8282202773386611]
force application position = [-0.019536566611294756, 0.0004251021780794415, 1.0127740848138596]
robot COM = [-0.0217355455071908, 0.000449324583432575, 0.6875714747749864]
force = [3.846835658478972, -0.07502642113000597, 307.9456270161795] N
r_cross_F = [0.016939604169042846, 0.573835061635915, -7.180190405292638e-05] Nm
```

HOME has a small roll moment (~0.0842 Nm) and a larger pitch moment (~2.7295 Nm).
Thus the band is not moment-free. The final OFF state still has a small lateral
load difference (about 0.249 N) and small roll, but maintains bilateral contact.
The force-line offset is a secondary contributor, not evidence that it alone
causes the large ON right-foot unloading. No anchor or attachment adjustment
was made.

## Takeover chronology

Rows are **after** the indicated number of complete policy steps (first action
at t=0.1 s, first resulting state at t=0.12 s). Before takeover both feet are
already airborne; it would be inaccurate to say the policy initially lifts a
previously grounded right foot.

| Event | Time s | Left sole z m | Right sole z m | Left Fz N | Right Fz N | Roll rad | Pitch rad |
|---|---:|---:|---:|---:|---:|---:|---:|
| before_policy | 0.100 | 0.014754816 | 0.014774196 | 0.000000 | 0.000000 | -0.000071609 | 0.007607773 |
| 1 | 0.120 | 0.011304718 | 0.012866419 | 0.000000 | 0.000000 | -0.000259033 | 0.007444366 |
| 5 | 0.200 | -0.002154130 | 0.004467678 | 51.278500 | 0.000000 | -0.000501962 | 0.017270048 |
| 20 | 0.500 | 0.029716363 | 0.033550734 | 0.000000 | 0.000000 | 0.018418433 | 0.033767865 |

After the first action, left/right hip-pitch targets are about -0.025302/-0.217875
rad; knee targets 0.144001/0.062161 rad; ankle-pitch targets -0.062455/-0.335830 rad.
These are not mirror-equivalent sagittal targets. By policy step 5 the left sole
has penetrated/contacted ground while the right sole remains +4.46768 mm clear.
By step 20 both are airborne again. This demonstrates policy-induced asymmetric
contact timing, not a permanent geometric right-foot offset or a simple monotonic
right-foot lift. Full raw actions/targets/actual q for all 29 joints are in
`takeover_joint_actions.csv` and event JSON, including hip and ankle components.

The matched OFF/ON comparison supports causality at the **policy-command vs HOME
hold** level. It does not prove why the learned actor selects these outputs, nor
attribute a particular network feature, ray self-hit or joint action in isolation.
No claim of a policy bug is made; this is a highly unloaded, initially aerial pose.

## Offline root-height sweep (no policy, no stepping)

Original root orientation, HOME joint q, zero dq, frozen PD, same anchor and band
are retained. `mj_forward` computes contacts at the seven requested heights; no
new height is used to initialize a dynamics trial.

| Root z offset | Left clearance m | Right clearance m | Left/right contact | Fz N | Penetration |
|---|---:|---:|---|---|---|
| -5 mm | 0.010797817 | 0.010797817 | none / none | 0 / 0 | none |
| -2 mm | 0.013797817 | 0.013797817 | none / none | 0 / 0 | none |
| -1 mm | 0.014797817 | 0.014797817 | none / none | 0 / 0 | none |
| +0 mm | 0.015797817 | 0.015797817 | none / none | 0 / 0 | none |
| +1 mm | 0.016797817 | 0.016797817 | none / none | 0 / 0 | none |
| +2 mm | 0.017797817 | 0.017797817 | none / none | 0 / 0 | none |
| +5 mm | 0.020797817 | 0.020797817 | none / none | 0 / 0 | none |

The entire ±5 mm range remains airborne. Sensitivity is exactly 1 mm clearance
per 1 mm root translation on both sides. Contact would require a larger downward
translation at this frozen joint pose, but none was tested or selected.

## Conclusion and gate

```text
L0_CONTACT_INITIALIZATION_AUDIT
POLICY_OFF_BILATERAL_CONTACT = 100% final 0.4 s; 88.877551% across diagnostic 4.9 s
POLICY_ON_BILATERAL_CONTACT = 7.5% matched final 0.4 s; original L0 final window 46.25%
LEFT_SOLE_CLEARANCE = 0.015797817293 m (initial HOME)
RIGHT_SOLE_CLEARANCE = 0.015797817293 m (initial HOME)
SOLE_CLEARANCE_DIFF = 0.000000000000 m
LEFT_STATIC_FZ = 9.49039204753 N (OFF final-window mean)
RIGHT_STATIC_FZ = 9.73901508229 N (OFF final-window mean)
ROOT_ROLL = 0.000488918268942 rad (OFF final state)
ROOT_PITCH = 0.0491133843278 rad (OFF final state)
BAND_FORCE_VECTOR = [0.0, 0.0, 327.07660302000005] N (HOME)
BAND_TORQUE_ABOUT_COM = [0.08419674546040212, 2.7294918809644786, -0.0] Nm (HOME)
CONTACT_GEOM_PARITY = NOT_EXACT; one capsule endpoint has 1 mm lateral mismatch
HOME_POSE_GEOMETRY_PARITY = SOLE_Z_PASS; full lateral collision mirror not exact
ASYMMETRY_PRESENT_BEFORE_POLICY = small force-line/XML bias; no HOME sole-z mismatch
ASYMMETRY_INTRODUCED_BY_POLICY = YES for pronounced right unloading/contact loss
ROOT_HEIGHT_SENSITIVITY = +/-5 mm remains airborne on both sides
PRIMARY_CAUSE = asymmetric policy targets during highly supported, initially aerial takeover
SECONDARY_CAUSE = initial bilateral clearance; off-COM band moment; minor lateral XML asymmetry
FROZEN_COMPONENTS_MODIFIED = NO
RL_TRAINING_STARTED = NO
REAL_ROBOT_STARTED = NO
READY_TO_RETRY_L0 = NO (diagnosis only; no corrective change selected or authorized)
```

No lower-support stage or locomotion test was run. The bilateral gate remains
unchanged. Small pre-policy asymmetries are measured secondary factors; their
individual causal contributions were not isolated through model modifications.

Reproduce with `python -m simulate.tests.audit_l0_contact` in the existing conda
environment. Script outputs: `logs/l0_contact/report.json`, both arms' full sample
JSON and policy-ON NPZ. The takeover CSV, replay-equality check and reviewable
summary `simulate/tests/l0_contact_results.json` were produced by subsequent
offline analysis. Added diagnostic script, results and this report; existing
control/perception/scene sources were not edited.
