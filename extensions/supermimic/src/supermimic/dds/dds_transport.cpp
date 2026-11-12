// Local-only Unitree SDK2 C ABI for SuperMimic; no robot-specific control law.
#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace {
constexpr int kMotors = 29;
constexpr char kLowCmdTopic[] = "rt/lowcmd";
constexpr char kLowStateTopic[] = "rt/lowstate";
constexpr char kRootStateTopic[] = "rt/sportmodestate";

std::mutex g_error_mutex;
std::string g_error;

void set_error(const std::string& value) {
  std::lock_guard<std::mutex> lock(g_error_mutex);
  g_error = value;
}

double monotonic_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

extern "C" {

typedef struct {
  uint32_t tick;
  uint8_t mode_machine;
  float q[kMotors];
  float dq[kMotors];
  float tau_est[kMotors];
  float root_quat_wxyz[4];
  float omega_b[3];
  float root_pos_w[3];
  float root_lin_vel_w[3];
  double received_monotonic;
  uint64_t sequence;
  int32_t root_valid;
} SmDdsState;

typedef struct {
  float q[kMotors];
  float dq[kMotors];
  float kp[kMotors];
  float kd[kMotors];
  float tau[kMotors];
  uint8_t motor_mode;
  uint8_t mode_machine;
} SmDdsCommand;

struct SmDdsHandle {
  std::mutex mutex;
  std::condition_variable condition;
  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> publisher;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> root;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowCmd_> other_commands;
  SmDdsState state{};
  uint64_t delivered_sequence = 0;
  uint64_t other_command_count = 0;
  bool opened = false;
  bool read_only = false;
};

const char* sm_dds_last_error() {
  std::lock_guard<std::mutex> lock(g_error_mutex);
  return g_error.c_str();
}

uint32_t sm_dds_state_size() { return sizeof(SmDdsState); }
uint32_t sm_dds_command_size() { return sizeof(SmDdsCommand); }
uint32_t sm_dds_lowcmd_size() { return sizeof(unitree_hg::msg::dds_::LowCmd_); }

void* sm_dds_create() { return new SmDdsHandle(); }

int sm_dds_close(void* opaque) {
  if (!opaque) return 0;
  auto* handle = static_cast<SmDdsHandle*>(opaque);
  {
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->opened = false;
  }
  handle->condition.notify_all();
  if (handle->other_commands) handle->other_commands->CloseChannel();
  if (handle->root) handle->root->CloseChannel();
  if (handle->lowstate) handle->lowstate->CloseChannel();
  handle->other_commands.reset();
  handle->root.reset();
  handle->lowstate.reset();
  handle->publisher.reset();
  return 0;
}

int sm_dds_open_internal(void* opaque, const char* network_interface, int domain_id, bool read_only) {
  if (!opaque || !network_interface) {
    set_error("Invalid DDS handle or interface");
    return -1;
  }
  auto* handle = static_cast<SmDdsHandle*>(opaque);
  if (handle->opened) {
    set_error("DDS handle is already open");
    return -2;
  }
  try {
    handle->read_only = read_only;
    unitree::robot::ChannelFactory::Instance()->Init(domain_id, network_interface);
    if (!read_only) {
      handle->publisher =
          std::make_shared<unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>>(
              kLowCmdTopic);
      handle->publisher->InitChannel();
    }
    handle->lowstate =
        std::make_shared<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>>(
            kLowStateTopic);
    handle->lowstate->InitChannel(
        [handle](const void* message) {
          const auto& source = *static_cast<const unitree_hg::msg::dds_::LowState_*>(message);
          std::lock_guard<std::mutex> lock(handle->mutex);
          handle->state.tick = source.tick();
          handle->state.mode_machine = source.mode_machine();
          for (int i = 0; i < kMotors; ++i) {
            handle->state.q[i] = source.motor_state()[i].q();
            handle->state.dq[i] = source.motor_state()[i].dq();
            handle->state.tau_est[i] = source.motor_state()[i].tau_est();
          }
          for (int i = 0; i < 4; ++i) {
            handle->state.root_quat_wxyz[i] = source.imu_state().quaternion()[i];
          }
          for (int i = 0; i < 3; ++i) {
            handle->state.omega_b[i] = source.imu_state().gyroscope()[i];
          }
          handle->state.received_monotonic = monotonic_seconds();
          ++handle->state.sequence;
          handle->condition.notify_all();
        },
        1);
    handle->root =
        std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(
            kRootStateTopic);
    handle->root->InitChannel(
        [handle](const void* message) {
          const auto& source = *static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
          std::lock_guard<std::mutex> lock(handle->mutex);
          for (int i = 0; i < 3; ++i) {
            handle->state.root_pos_w[i] = source.position()[i];
            handle->state.root_lin_vel_w[i] = source.velocity()[i];
          }
          handle->state.root_valid = 1;
        },
        1);
    if (!read_only) {
      handle->other_commands =
          std::make_shared<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_>>(
              kLowCmdTopic);
      handle->other_commands->InitChannel(
          [handle](const void*) {
            std::lock_guard<std::mutex> lock(handle->mutex);
            ++handle->other_command_count;
          },
          1);
    }
    handle->opened = true;
    if (!read_only) {
      // Detect an existing writer before this process is allowed to publish.
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (handle->other_command_count != 0) {
        set_error("Another rt/lowcmd writer is active; refusing dual control");
        sm_dds_close(opaque);
        return -3;
      }
    }
    set_error("");
    return 0;
  } catch (const std::exception& error) {
    set_error(error.what());
    sm_dds_close(opaque);
    return -4;
  }
}

int sm_dds_open(void* opaque, const char* network_interface, int domain_id) {
  return sm_dds_open_internal(opaque, network_interface, domain_id, false);
}

int sm_dds_open_read_only(void* opaque, const char* network_interface, int domain_id) {
  return sm_dds_open_internal(opaque, network_interface, domain_id, true);
}

int sm_dds_read_state(void* opaque, SmDdsState* output, uint32_t timeout_ms) {
  if (!opaque || !output) {
    set_error("Invalid state output or DDS handle");
    return -1;
  }
  auto* handle = static_cast<SmDdsHandle*>(opaque);
  std::unique_lock<std::mutex> lock(handle->mutex);
  if (!handle->opened) {
    set_error("DDS handle is not open");
    return -2;
  }
  const bool updated = handle->condition.wait_for(
      lock, std::chrono::milliseconds(timeout_ms),
      [handle] { return handle->state.sequence != handle->delivered_sequence || !handle->opened; });
  if (!handle->opened) {
    set_error("DDS handle closed while waiting for state");
    return -3;
  }
  if (!updated) return 0;
  *output = handle->state;
  handle->delivered_sequence = handle->state.sequence;
  return 1;
}

int sm_dds_encode_command(const SmDdsCommand* command, void* output, uint32_t output_size) {
  if (!command || !output) {
    set_error("Invalid offline command or output buffer");
    return -1;
  }
  if (output_size != sizeof(unitree_hg::msg::dds_::LowCmd_)) {
    set_error("LowCmd output buffer size mismatch");
    return -2;
  }
  if (command->mode_machine == 0) {
    set_error("mode_machine=0 is not accepted for this G1 simulation");
    return -3;
  }
  auto* message = static_cast<unitree_hg::msg::dds_::LowCmd_*>(output);
  *message = unitree_hg::msg::dds_::LowCmd_{};
  message->mode_pr() = 0;
  message->mode_machine() = command->mode_machine;
  for (int i = 0; i < kMotors; ++i) {
    auto& motor = message->motor_cmd()[i];
    motor.mode() = command->motor_mode;
    motor.q() = command->q[i];
    motor.dq() = command->dq[i];
    motor.kp() = command->kp[i];
    motor.kd() = command->kd[i];
    motor.tau() = command->tau[i];
  }
  message->crc() = crc32_core(reinterpret_cast<uint32_t*>(message),
                            (sizeof(*message) >> 2) - 1);
  return static_cast<int>(sizeof(*message));
}

int sm_dds_write_command(void* opaque, const SmDdsCommand* command) {
  if (!opaque || !command) {
    set_error("Invalid command or DDS handle");
    return -1;
  }
  auto* handle = static_cast<SmDdsHandle*>(opaque);
  if (!handle->opened) {
    set_error("DDS handle is not open");
    return -2;
  }
  if (handle->read_only) {
    set_error("Read-only DDS handle cannot publish commands");
    return -4;
  }
  unitree_hg::msg::dds_::LowCmd_ message{};
  const int encoded = sm_dds_encode_command(command, &message, sizeof(message));
  if (encoded < 0) return encoded;
  return handle->publisher->Write(message, 0) ? 1 : 0;
}

void sm_dds_destroy(void* opaque) {
  if (!opaque) return;
  sm_dds_close(opaque);
  delete static_cast<SmDdsHandle*>(opaque);
}
}  // extern "C"
