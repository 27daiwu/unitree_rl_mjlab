#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// G1 grid/frames/scale follow mjlab; collision semantics are explicitly selected.
// Parity corrections are documented in ../HEIGHT_SCAN_AUDIT.md.
struct HeightScan {
  enum class Mode { TRAINING_PARITY = 0, DEPLOYMENT_FILTERED = 1 };
  struct Hit { int geom = -1; double z = 0; };
  using Hits = std::array<Hit, 176>;
  static constexpr int kRows = 11, kCols = 16, kDim = kRows * kCols;
  static constexpr double kXMin = -0.75, kXMax = 0.75;
  static constexpr double kYMin = -0.5, kYMax = 0.5;
  static constexpr double kResolution = 0.1, kForwardOffset = 0.45;
  static constexpr double kMaxDistance = 5.0;

  // Warp ray_mesh_with_bvh queries the closest triangle (two-sided), then
  // rejects the WHOLE mesh when that triangle faces away from the ray.
  // CPU mj_ray is two-sided. Recover the closest face winding from the actual
  // compiled mesh triangles; no body-specific corrections or height overrides.
  static bool MeshFrontFace(const mjModel* m, const mjData* d, int geom,
                            const mjtNum* origin, const mjtNum* direction) {
    const int mesh = m->geom_dataid[geom];
    mjtNum delta[3], p[3], v[3];
    mju_sub3(delta, origin, d->geom_xpos + 3*geom);
    mju_mulMatTVec3(p, d->geom_xmat + 9*geom, delta);
    mju_mulMatTVec3(v, d->geom_xmat + 9*geom, direction);
    double closest = mjMAXVAL, facing = 0;
    for (int f=0; f<m->mesh_facenum[mesh]; ++f) {
      const int* face = m->mesh_face + 3*(m->mesh_faceadr[mesh]+f);
      const float* a = m->mesh_vert + 3*(m->mesh_vertadr[mesh]+face[0]);
      const float* b = m->mesh_vert + 3*(m->mesh_vertadr[mesh]+face[1]);
      const float* c = m->mesh_vert + 3*(m->mesh_vertadr[mesh]+face[2]);
      mjtNum e1[3],e2[3],h[3],s[3],q[3];
      for (int k=0;k<3;++k) {e1[k]=b[k]-a[k];e2[k]=c[k]-a[k];s[k]=p[k]-a[k];}
      mju_cross(h,v,e2);
      const double det=mju_dot3(e1,h);
      if (std::abs(det)<1e-15) continue;
      const double u=mju_dot3(s,h)/det;
      if (u<0 || u>1) continue;
      mju_cross(q,s,e1);
      const double w=mju_dot3(v,q)/det;
      if (w<0 || u+w>1) continue;
      const double t=mju_dot3(e2,q)/det;
      if (t>=0 && t<closest) {closest=t;facing=-det;}
    }
    return closest<mjMAXVAL && facing<0;
  }

  std::array<float, kDim> operator()(const mjModel* m, const mjData* d,
                                    int body_id, Mode mode = Mode::DEPLOYMENT_FILTERED,
                                    Hits* hits = nullptr) const {
    if (!m || !d || body_id <= 0 || body_id >= m->nbody)
      throw std::invalid_argument("HeightScan requires a valid robot root body");
    // mjlab: groups (0,1,2), static geoms enabled, only parent body excluded.
    // Filtered mode additionally excludes the robot subtree without mutation.
    if (mode != Mode::TRAINING_PARITY && mode != Mode::DEPLOYMENT_FILTERED)
      throw std::invalid_argument("Unknown height scan mode");
    std::vector<int> groups(m->geom_group, m->geom_group + m->ngeom);
    if (mode == Mode::DEPLOYMENT_FILTERED) for (int g = 0; g < m->ngeom; ++g) {
      int b = m->geom_bodyid[g];
      while (b > 0 && b != body_id) b = m->body_parentid[b];
      if (b == body_id) groups[g] = 5;
    }
    mjModel ray_model = *m;
    ray_model.geom_group = groups.data();
    const mjtByte mask[6] = {1, 1, 1, 0, 0, 0};
    const mjtNum* p = d->xpos + 3 * body_id;
    const mjtNum* R = d->xmat + 9 * body_id;
    double c = R[0], s = R[3], norm = std::hypot(c, s);
    // Match mjlab's near-vertical X-axis fallback, including threshold.
    if (norm < 0.1) { c = R[4]; s = -R[1]; norm = std::hypot(c, s); }
    c /= std::max(norm, 1e-6); s /= std::max(norm, 1e-6);
    std::array<float, kDim> out{};
    for (int iy = 0; iy < kRows; ++iy) for (int ix = 0; ix < kCols; ++ix) {
      const double x = kXMin + ix * kResolution + kForwardOffset;
      const double y = kYMin + iy * kResolution;
      mjtNum origin[3] = {p[0] + c*x - s*y, p[1] + s*x + c*y, p[2]};
      const mjtNum dir[3] = {0, 0, -1};
      int geom = -1;
      double dist = -1;
      std::vector<int> culled;
      for (;;) {
#if mjVERSION_HEADER >= 3005000
      dist = mj_ray(&ray_model, d, origin, dir, mask, 1, body_id, &geom, nullptr);
#else
      dist = mj_ray(&ray_model, d, origin, dir, mask, 1, body_id, &geom);
#endif
        if (mode != Mode::TRAINING_PARITY || geom<0 ||
            m->geom_type[geom] != mjGEOM_MESH || MeshFrontFace(m,d,geom,origin,dir)) break;
        groups[geom]=5;
        culled.push_back(geom);
      }
      for (int g : culled) groups[g]=m->geom_group[g];
      if (!std::isfinite(dist)) throw std::runtime_error("Non-finite ray distance");
      const bool miss = dist < 0 || dist > kMaxDistance;
      const double raw = miss ? kMaxDistance : dist;
      if (hits) (*hits)[iy*kCols + ix] = {miss ? -1 : geom, miss ? p[2] : origin[2]-dist};
      // meshgrid(indexing="xy").flatten(): x changes fastest. No obs clip.
      out[iy*kCols + ix] = static_cast<float>(raw * 0.2);
    }
    return out;
  }
};
