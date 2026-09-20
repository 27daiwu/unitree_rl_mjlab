// Thin diagnostic binding. Production C++ callers use Builder directly.
#include "policy_observation.h"
#include <exception>
#include <algorithm>
#include <string>
namespace { thread_local std::string error; }
extern "C" {
const char* observation_joint_name(int i) { return i >= 0 && i < 29 ? policy_observation::kJointNames[i] : nullptr; }
float observation_default_q(int i) { return i >= 0 && i < 29 ? policy_observation::kDefaultJointPos[i] : 0; }
const char* observation_error() { return error.c_str(); }
void* observation_create(const mjModel* m, const char* prefix, int mode) {
  try { return new policy_observation::Builder(m, prefix, static_cast<HeightScan::Mode>(mode)); }
  catch (const std::exception& e) { error=e.what(); return nullptr; }
}
void observation_destroy(void* p) { delete static_cast<policy_observation::Builder*>(p); }
void observation_reset(void* p) { static_cast<policy_observation::Builder*>(p)->Reset(); }
int observation_set_action(void* p, const float* a) {
  try {
    policy_observation::Action action; std::copy_n(a, action.size(), action.begin());
    static_cast<policy_observation::Builder*>(p)->SetPreviousAction(action); return 0;
  } catch (const std::exception& e) { error=e.what(); return -1; }
}
int observation_ray_hits(void* p, const mjData* d, int* geoms, double* heights) {
  try {
    const auto hits=static_cast<policy_observation::Builder*>(p)->RayHits(d);
    for (int i=0;i<HeightScan::kDim;++i) { geoms[i]=hits[i].geom; heights[i]=hits[i].z; }
    return 0;
  } catch (const std::exception& e) { error=e.what(); return -1; }
}
int observation_build(void* p, const mjData* d, const float* cmd,
                      unsigned long long step, double dt, float* out) {
  try {
    policy_observation::Command command; std::copy_n(cmd,3,command.begin());
    auto obs=static_cast<policy_observation::Builder*>(p)->Build(d,command,step,dt);
    std::copy(obs.begin(),obs.end(),out); return 0;
  } catch (const std::exception& e) { error=e.what(); return -1; }
}
}
