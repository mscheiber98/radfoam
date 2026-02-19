#pragma once

#include "../utils/geometry.h"
#include "camera.h"

namespace radfoam {

template <int block_size, int chunk_size, typename CellFunctor>
__forceinline__ __device__ uint32_t
trace(const Ray &ray,
      const Vec3f *__restrict__ points,
      const uint32_t *__restrict__ point_adjacency,
      const uint32_t *__restrict__ point_adjacency_offsets,
      const Vec4h *__restrict__ adjacent_points,
      uint32_t start_point,
      uint32_t max_steps,
      CellFunctor cell_functor) {
    float t_0 = 0.0f;
    uint32_t n = 0;

    uint32_t current_point_idx = start_point;
    Vec3f primal_point = points[current_point_idx];

    for (;;) {
        n++;
        if (n > max_steps) {
            break;
        }

        // Outer loop iterates through Voronoi cells
        uint32_t point_adjacency_begin =
            point_adjacency_offsets[current_point_idx];
        uint32_t point_adjacency_end =
            point_adjacency_offsets[current_point_idx + 1];

        uint32_t num_faces = point_adjacency_end - point_adjacency_begin;
        float t_1 = std::numeric_limits<float>::infinity();

        uint32_t next_face = UINT32_MAX;
        Vec3f next_point = Vec3f::Zero();

        half2 chunk[chunk_size * 2];
        for (uint32_t i = 0; i < num_faces; i += chunk_size) {
#pragma unroll
            for (uint32_t j = 0; j < chunk_size; ++j) {
                chunk[2 * j] = reinterpret_cast<const half2 *>(
                    adjacent_points + point_adjacency_begin + i + j)[0];
                chunk[2 * j + 1] = reinterpret_cast<const half2 *>(
                    adjacent_points + point_adjacency_begin + i + j)[1];
            }

#pragma unroll
            for (uint32_t j = 0; j < chunk_size; ++j) {
                Vec3f offset(__half2float(chunk[2 * j].x),
                             __half2float(chunk[2 * j].y),
                             __half2float(chunk[2 * j + 1].x));
                Vec3f face_origin = primal_point + offset / 2.0f;
                Vec3f face_normal = offset;
                float dp = face_normal.dot(ray.direction);
                float t = (face_origin - ray.origin).dot(face_normal) / dp;

                if (dp > 0.0f && t < t_1 && (i + j) < num_faces) {
                    t_1 = t;
                    next_face = i + j;
                }
            }
        }

        if (next_face == UINT32_MAX) {
            break;
        }

        uint32_t next_point_idx =
            point_adjacency[point_adjacency_begin + next_face];
        next_point = points[next_point_idx];

        if (t_1 > t_0) {
            if (!cell_functor(
                    current_point_idx, t_0, t_1, primal_point, next_point)) {
                break;
            }
        }
        t_0 = fmaxf(t_0, t_1);
        current_point_idx = next_point_idx;
        primal_point = next_point;
    }

    return n;
}

__forceinline__ __device__ Vec3f cell_intersection_grad(
    const Vec3f &primal_point, const Vec3f &opposite_point, const Ray &ray) {
    Vec3f face_origin = (primal_point + opposite_point) / 2.0f;
    Vec3f face_normal = (opposite_point - primal_point);

    float num = (face_origin - ray.origin).dot(face_normal);
    float dp = face_normal.dot(ray.direction);

    Vec3f grad = num * ray.direction + dp * (ray.origin - primal_point);
    grad /= dp * dp;

    return grad;
}

inline RADFOAM_HD uint32_t make_rgba8(float r, float g, float b, float a) {
    r = std::max(0.0f, std::min(1.0f, r));
    g = std::max(0.0f, std::min(1.0f, g));
    b = std::max(0.0f, std::min(1.0f, b));
    a = std::max(0.0f, std::min(1.0f, a));
    int ri = static_cast<int>(r * 255.0f);
    int gi = static_cast<int>(g * 255.0f);
    int bi = static_cast<int>(b * 255.0f);
    int ai = static_cast<int>(a * 255.0f);
    return (ai << 24) | (bi << 16) | (gi << 8) | ri;
}

inline __device__ Vec3f colormap(float v,
                                 ColorMap map,
                                 const CMapTable &cmap_table) {
    int map_len = cmap_table.sizes[map];
    const Vec3f *map_vals =
        reinterpret_cast<const Vec3f *>(cmap_table.data[map]);

    int i0 = static_cast<int>(v * (map_len - 1));
    int i1 = i0 + 1;
    float t = v * (map_len - 1) - i0;
    i0 = max(0, min(i0, map_len - 1));
    i1 = max(0, min(i1, map_len - 1));
    return map_vals[i0] * (1.0f - t) + map_vals[i1] * t;
}
template <typename scalar>
__device__ void write_density_grad_to_sggx(const Vec3f &dir, //viewing direction i.e. ray direction (normalized)
                                    float dL_ds_primal, // Loss gradient w.r.t. cell density
                                    scalar *sggx_grad) {
    //sxx
    atomicAdd(sggx_grad + 0, (scalar)(dir[0]*dir[0] * dL_ds_primal));
    //sxy
    atomicAdd(sggx_grad + 1, (scalar)(2.0*dir[0]*dir[1] * dL_ds_primal));
    //sxz
    atomicAdd(sggx_grad + 2, (scalar)(2.0*dir[0]*dir[2] * dL_ds_primal));
    //syy
    atomicAdd(sggx_grad + 3, (scalar)(dir[1]*dir[1] * dL_ds_primal));
    //syz
    atomicAdd(sggx_grad + 4, (scalar)(2.0*dir[1]*dir[2] * dL_ds_primal));
    //szz
    atomicAdd(sggx_grad + 5, (scalar)(dir[2]*dir[2] * dL_ds_primal));

}

__device__ float sigmoid(float x) {
    if (x >= 0) {
        float exp_neg_x = expf(-x);
        return 1.0f / (1.0f + exp_neg_x);
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

__device__ float dsigmoid(float x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

__device__ float softplus(float x, float beta = 10.0f){ 
    float threshold = 20.0;
    float bx = beta *x;

    if (bx > threshold) {
        return x;  // log(1 + exp(bx)) / beta ≈ x
    } else if (bx < -20.0f) {
        return 0.0f;  // exp(bx) ≈ 0
    }
    
    return (1.0f / beta) * log1pf(expf(bx));
}

__device__ float dsoftplus (float x, float beta = 10.0f){
    float bx = beta *x;
    return sigmoid(bx);
}

} // namespace radfoam