#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include "Types.h"
#include "isaaclab/envs/mdp/terminations.h"

class SafetyGuard {
public:
  struct Config {
    // 单位：rad。G1 正常行走时建议不要低于 0.38，否则大动作转向可能误触发。
    float soft_roll_pitch = 0.45f;       // 约 25.8 deg，持续一段时间才退出
    float hard_roll_pitch = 0.75f;       // 约 43.0 deg，立即退出
    float mjlab_bad_orientation = 0.75f; // 替代原来的 1.0 rad

    // 单位：rad/s。IMU 角速度异常大时退出。
    float gyro_norm = 4.5f;

    // 单位：rad/s。关节速度异常大时退出，防止摔倒后 policy 继续乱甩。
    float motor_dq = 26.0f;

    // 连续触发时间。FSM 是 1 kHz，不要只看单帧。
    float soft_hold_s = 0.12f;
    float gyro_hold_s = 0.08f;
    float motor_dq_hold_s = 0.08f;
  };

  SafetyGuard() = default;

  void reset() {
    has_last_time_ = false;
    soft_tilt_time_ = 0.0f;
    gyro_time_ = 0.0f;
    motor_dq_time_ = 0.0f;
    last_reason_.clear();
  }

  const std::string& lastReason() const {
    return last_reason_;
  }

  template <typename EnvT>
  bool shouldExitToPassive(const LowState_t* lowstate, EnvT* env) {
    const float dt = updateDt();

    if (lowstate == nullptr) {
      last_reason_ = "lowstate is null";
      return true;
    }

    // 1) 用 mjlab 自带 bad_orientation，但阈值从 1.0 rad 提前到 0.75 rad。
    if (env != nullptr && isaaclab::mdp::bad_orientation(env, cfg_.mjlab_bad_orientation)) {
      last_reason_ = "mjlab bad_orientation";
      return true;
    }

    const auto& imu = lowstate->msg_.imu_state();

    // 2) IMU 姿态角保护。
    // 如果你的 SDK 版本没有 imu_state().rpy()，需要改成由 quaternion() 转 roll/pitch。
    const float roll = imu.rpy()[0];
    const float pitch = imu.rpy()[1];
    const float abs_rp = std::max(std::abs(roll), std::abs(pitch));

    if (abs_rp > cfg_.hard_roll_pitch) {
      last_reason_ = "hard roll/pitch limit";
      return true;
    }

    accumulate(abs_rp > cfg_.soft_roll_pitch, soft_tilt_time_, dt);
    if (soft_tilt_time_ > cfg_.soft_hold_s) {
      last_reason_ = "soft roll/pitch timeout";
      return true;
    }

    // 3) IMU 角速度保护。
    const float wx = imu.gyroscope()[0];
    const float wy = imu.gyroscope()[1];
    const float wz = imu.gyroscope()[2];
    const float gyro_norm = std::sqrt(wx * wx + wy * wy + wz * wz);

    accumulate(gyro_norm > cfg_.gyro_norm, gyro_time_, dt);
    if (gyro_time_ > cfg_.gyro_hold_s) {
      last_reason_ = "high imu gyro norm";
      return true;
    }

    // 4) 关节速度保护。摔倒后原地扑腾通常会伴随若干关节 dq 异常。
    float max_abs_dq = 0.0f;
    for (size_t i = 0; i < lowstate->msg_.motor_state().size(); ++i) {
      max_abs_dq = std::max(max_abs_dq, std::abs(lowstate->msg_.motor_state()[i].dq()));
    }

    accumulate(max_abs_dq > cfg_.motor_dq, motor_dq_time_, dt);
    if (motor_dq_time_ > cfg_.motor_dq_hold_s) {
      last_reason_ = "high motor dq";
      return true;
    }

    last_reason_.clear();
    return false;
  }

private:
  float updateDt() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();

    if (!has_last_time_) {
      last_time_ = now;
      has_last_time_ = true;
      return 0.001f;
    }

    float dt = std::chrono::duration<float>(now - last_time_).count();
    last_time_ = now;

    // 防止系统调度抖动导致计时器一次性跳太大。
    return std::clamp(dt, 0.0005f, 0.02f);
  }

  static void accumulate(bool condition, float& timer, float dt) {
    if (condition) {
      timer += dt;
    } else {
      timer = 0.0f;
    }
  }

  Config cfg_;
  bool has_last_time_ = false;
  std::chrono::steady_clock::time_point last_time_;

  float soft_tilt_time_ = 0.0f;
  float gyro_time_ = 0.0f;
  float motor_dq_time_ = 0.0f;
  std::string last_reason_;
};