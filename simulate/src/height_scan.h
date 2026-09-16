#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <cmath>
#include <algorithm>

// Training-compatible G1 stairs scan: 16 x 11 points, pelvis yaw frame.
struct HeightScan {
  static constexpr int kRows = 16, kCols = 11, kDim = 176;
  static constexpr double kXMin = -0.75, kXMax = 0.75;
  static constexpr double kYMin = -0.5, kYMax = 0.5;
  static constexpr double kResolution = 0.1, kForwardOffset = 0.45;
  static constexpr double kMaxDistance = 5.0;
  std::array<float, kDim> operator()(const mjModel* m, const mjData* d,
                                      int body_id, int terrain_group = 0) const {
    std::array<float, kDim> out{};
    const mjtNum* p = d->xpos + 3 * body_id;
    const mjtNum* R = d->xmat + 9 * body_id;
    for (int ix=0; ix<kRows; ++ix) for (int iy=0; iy<kCols; ++iy) {
      const double x = kXMin + ix*kResolution + kForwardOffset;
      const double y = kYMin + iy*kResolution;
      // yaw-only attachment: rotate XY by pelvis yaw, keeping scanner horizontal.
      const double yaw = std::atan2(R[3], R[0]);
      const double ox = p[0] + std::cos(yaw)*x - std::sin(yaw)*y;
      const double oy = p[1] + std::sin(yaw)*x + std::cos(yaw)*y;
      mjtNum origin[3] = {ox, oy, p[2] + 2.5};
      mjtNum dir[3] = {0,0,-1}; int geom = -1;
      const mjtNum dist = mj_ray(m, d, origin, dir, nullptr, 1, -1, &geom);
      const double hit = (dist >= 0 && std::isfinite(dist)) ? origin[2]-dist : origin[2]-kMaxDistance;
      // Raw training height_scan is sensor-origin minus hit height, then scaled by 1/max_distance.
      out[ix*kCols+iy] = static_cast<float>(std::clamp((origin[2]-hit)/kMaxDistance, -1.0, 1.0));
    }
    return out;
  }
};
