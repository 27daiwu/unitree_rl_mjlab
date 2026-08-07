"""Unitree G1 flat tracking environment configurations."""

from src.assets.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.tracking import mdp
from src.tasks.tracking.mdp import MotionCommandCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


G1_29DOF_MIMIC_ACTION_SCALE = {
  r".*_hip_pitch_joint": 0.55,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.55,
  r".*_knee_joint": 0.35,
  r".*_ankle_pitch_joint": 0.44,
  r".*_ankle_roll_joint": 0.44,
  "waist_yaw_joint": 0.55,
  "waist_roll_joint": 0.44,
  "waist_pitch_joint": 0.44,
  r".*_shoulder_pitch_joint": 0.44,
  r".*_shoulder_roll_joint": 0.44,
  r".*_shoulder_yaw_joint": 0.44,
  r".*_elbow_joint": 0.44,
  r".*_wrist_roll_joint": 0.44,
  r".*_wrist_pitch_joint": 0.07,
  r".*_wrist_yaw_joint": 0.07,
}


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  training_stage: str = "stage1",
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg(training_stage=training_stage)

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  foot_geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision$",
      entity="robot",
      exclude=foot_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  torso_pelvis_ground_cfg = ContactSensorCfg(
    name="torso_pelvis_ground_touch",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(pelvis_collision|torso_collision)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)
  if training_stage != "legacy":
    cfg.scene.sensors += (nonfoot_ground_cfg, torso_pelvis_ground_cfg)
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = 64

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_29DOF_MIMIC_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
      r"^(left|right)_foot[1-7]_collision$"
    )
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  if training_stage != "legacy":
    cfg.rewards["undesired_contacts"] = RewardTermCfg(
      func=mdp.undesired_contact_cost,
      weight=-1.0,
      params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
    )
    cfg.terminations["torso_pelvis_ground_contact"] = TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": torso_pelvis_ground_cfg.name, "force_threshold": 10.0},
    )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    expected_actor_terms = (
      "motion_command",
      "motion_anchor_ori_b",
      "base_ang_vel",
      "joint_pos_rel",
      "joint_vel_rel",
      "last_action",
    )
    if tuple(new_actor_terms.keys()) != expected_actor_terms:
      raise ValueError(
        "Unexpected No-State-Estimation actor observation order: "
        f"{tuple(new_actor_terms.keys())}; expected {expected_actor_terms}"
      )
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events = {}

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

    motion_cmd.sampling_mode = "start"

  return cfg
