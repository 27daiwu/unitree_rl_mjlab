#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}


REGISTER_OBSERVATION(gait_phase_my)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    auto cmd = isaaclab::mdp::velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}


}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
    std::make_pair(
            [this]() -> bool {
            if (safety_exit_latched_) {
                return true;
            }

            if (safety_guard_.shouldExitToPassive(FSMState::lowstate.get(), env.get())) {
                safety_exit_latched_ = true;
                safety_exit_reason_ = safety_guard_.lastReason();
                spdlog::warn("[SafetyGuard] Exit RL state '{}' to Passive: {}",
                            this->getStateString(), safety_exit_reason_);
                return true;
            }

            return false;
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run() {
  if (!safety_exit_latched_ &&
      safety_guard_.shouldExitToPassive(FSMState::lowstate.get(), env.get())) {
    safety_exit_latched_ = true;
    safety_exit_reason_ = safety_guard_.lastReason();

    spdlog::warn("[SafetyGuard] Immediate damping before FSM transition: {}",
                 safety_exit_reason_);
  }

  // 关键点：安全触发后，本周期就不要再发布 policy action。
  // 先发送 kp=0、kd=Passive.kd、dq=0、tau=0，下一轮 FSM 会切到 Passive。
  if (safety_exit_latched_) {
    static const auto passive_kd =
      param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();

    const size_t n = std::min(lowcmd->msg_.motor_cmd().size(),
                              lowstate->msg_.motor_state().size());

    for (size_t i = 0; i < n; ++i) {
      auto& motor = lowcmd->msg_.motor_cmd()[i];
      motor.q() = lowstate->msg_.motor_state()[i].q();
      motor.kp() = 0.0f;
      motor.kd() = i < passive_kd.size() ? passive_kd[i] : 3.0f;
      motor.dq() = 0.0f;
      motor.tau() = 0.0f;
    }

    return;
  }

  auto action = env->action_manager->processed_actions();
  for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
    lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
  }
}