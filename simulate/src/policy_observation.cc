#include "policy_observation.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace policy_observation {
const std::array<const char*, kJointDim> kJointNames = {
 "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
 "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
 "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
 "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"};
const Action kDefaultJointPos = {-.1f,0,0,.3f,-.2f,0, -.1f,0,0,.3f,-.2f,0, 0,0,0, .35f,.18f,0,.87f,0,0,0, .35f,-.18f,0,.87f,0,0,0};

Builder::Builder(const mjModel* model, const std::string& prefix, HeightScan::Mode mode) : model_(model), mode_(mode) {
  if (!model_) throw std::invalid_argument("Null model");
  auto id = [&](mjtObj kind, const std::string& name) {
    int value = mj_name2id(model_, kind, (prefix + name).c_str());
    if (value < 0) throw std::invalid_argument("Missing model object: " + prefix + name);
    return value;
  };
  pelvis_ = id(mjOBJ_BODY, "pelvis");
  gyro_ = mj_name2id(model_, mjOBJ_SENSOR, (prefix + "imu_ang_vel").c_str());
  // Deployment scene calls its pelvis-aligned gyro imu_gyro. Site translation
  // does not change angular velocity; validate the body and rotation below.
  if (gyro_ < 0) gyro_ = id(mjOBJ_SENSOR, "imu_gyro");
  if (model_->sensor_type[gyro_] != mjSENS_GYRO || model_->sensor_dim[gyro_] != 3 ||
      model_->sensor_objtype[gyro_] != mjOBJ_SITE)
    throw std::invalid_argument("imu_ang_vel must be a 3D site gyro");
  const int site = model_->sensor_objid[gyro_];
  if (model_->site_bodyid[site] != pelvis_ ||
      std::abs(model_->site_quat[4*site]) < 1 - 1e-8)
    throw std::invalid_argument("Expected pelvis-aligned training IMU site");
  std::set<int> unique;
  for (int i=0; i<kJointDim; ++i) {
    joints_[i] = id(mjOBJ_JOINT, kJointNames[i]);
    if (model_->jnt_type[joints_[i]] != mjJNT_HINGE || !unique.insert(joints_[i]).second)
      throw std::invalid_argument("Joint mapping is not 29 unique hinges");
  }
}

void Builder::SetPreviousAction(const Action& policy_output) {
  for (float x : policy_output) if (!std::isfinite(x))
    throw std::invalid_argument("Non-finite previous action");
  previous_action_ = policy_output;  // raw policy output, before PD/action scaling
}

Observation Builder::Build(const mjData* d, const Command& command,
                          unsigned long long episode_step, double step_dt) const {
  if (!d || !std::isfinite(step_dt) || step_dt <= 0)
    throw std::invalid_argument("Invalid observation state or policy step_dt");
  Observation obs{};
  const int adr = model_->sensor_adr[gyro_];
  for (int i=0; i<3; ++i) {
    obs[i] = static_cast<float>(d->sensordata[adr+i]);
    obs[3+i] = static_cast<float>(-d->xmat[9*pelvis_+6+i]); // R^T * [0,0,-1]
    obs[6+i] = command[i];
  }
  const float norm = std::sqrt(command[0]*command[0]+command[1]*command[1]+command[2]*command[2]);
  if (norm >= .1f) {
    // Training converts episode_length_buf to float32 for the clock arithmetic.
    const float phase = std::fmod(static_cast<float>(episode_step) * static_cast<float>(step_dt), .6f) / .6f;
    const float angle = phase * static_cast<float>(std::acos(-1.0)) * 2.0f;
    obs[9] = std::sin(angle);
    obs[10] = std::cos(angle);
  }
  for (int i=0; i<kJointDim; ++i) {
    obs[11+i] = static_cast<float>(d->qpos[model_->jnt_qposadr[joints_[i]]]) - kDefaultJointPos[i];
    obs[40+i] = static_cast<float>(d->qvel[model_->jnt_dofadr[joints_[i]]]);
    obs[69+i] = previous_action_[i];
  }
  const auto height = HeightScan{}(model_, d, pelvis_, mode_);
  std::copy(height.begin(), height.end(), obs.begin()+kHeightStart);
  for (float x : obs) if (!std::isfinite(x)) throw std::runtime_error("Non-finite observation");
  return obs;
}
}  // namespace policy_observation
