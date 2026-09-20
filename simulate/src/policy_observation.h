#pragma once
#include "height_scan.h"
#include <array>
#include <string>

namespace policy_observation {
inline constexpr int kObsDim = 274, kJointDim = 29, kHeightStart = 98;
static_assert(HeightScan::kDim == 176 && kHeightStart + HeightScan::kDim == kObsDim);
using Observation = std::array<float, kObsDim>;
using Action = std::array<float, kJointDim>;
using Command = std::array<float, 3>;  // base-frame vx, vy, angular-z command (training twist convention)
extern const std::array<const char*, kJointDim> kJointNames;
extern const Action kDefaultJointPos;

// State must have current forward kinematics and sensordata (mj_forward).
// Output follows the play-mode observation layout and selected ray mode, BEFORE the policy's
// learned normalizer. The normalizer belongs to the exported policy.
class Builder {
 public:
  explicit Builder(const mjModel* model, const std::string& prefix = "",
                   HeightScan::Mode mode = HeightScan::Mode::DEPLOYMENT_FILTERED);
  void Reset() { previous_action_.fill(0); }
  void SetPreviousAction(const Action& policy_output);
  Observation Build(const mjData* data, const Command& command,
                    unsigned long long episode_step, double step_dt) const;
  HeightScan::Hits RayHits(const mjData* data) const {
    HeightScan::Hits hits; HeightScan{}(model_, data, pelvis_, mode_, &hits); return hits;
  }
  const Action& PreviousAction() const { return previous_action_; }
  const std::array<int, kJointDim>& JointIds() const { return joints_; }
 private:
  const mjModel* model_;
  HeightScan::Mode mode_;
  int pelvis_, gyro_;
  std::array<int, kJointDim> joints_{};
  Action previous_action_{};
};
}  // namespace policy_observation
