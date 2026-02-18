#include "../aabb_tree/aabb_tree.h"
#include "../delaunay/triangulation_ops.h"
#include "../utils/cuda_array.h"
#include "../utils/cuda_helpers.h"
#include "../utils/geometry.h"
#include "pipeline.h"

#include "../utils/common_kernels.cuh"
#include "sh_utils.cuh"
#include "tracing_utils.cuh"

namespace radfoam {
    

template <typename attr_scalar, int sh_degree, int block_size>
// __restrict__ is a compiler hint that the memory is not aliased with other memory and allows the compiler to optimize the code
// attr_scalar is the type of the attributes, sh_degree is the degree of the spherical harmonics, block_size is the number of threads per block
__global__ void forward(TraceSettings settings,
                        const Vec3f *__restrict__ points, // input: coordinates of primal points [x,y,z]
                        const attr_scalar *__restrict__ attributes, //input: [sh-coefficients, density, sggx]
                        const uint32_t *__restrict__ point_adjacency, // input: list of point adjacency
                        const uint32_t *__restrict__ point_adjacency_offsets, // input: index offsets for point adjacency list so we can extract the adjacency for a specific point
                        const Vec4h *__restrict__ adjacent_diff,
                        const Ray *__restrict__ rays,
                        uint32_t num_rays,
                        const uint32_t *__restrict__ start_point_index,
                        uint32_t num_depth_quantiles, // input: how many depth quantiles are used
                        const float *__restrict__ depth_quantiles, // input: quantiles for the rays (e.g. ray 1 has quantiles [0.4,0.7])
                        attr_scalar *__restrict__ ray_rgba,
                        float *__restrict__ quantile_depths, // output: the depths along the rays where the input quantiles lie
                        uint32_t *__restrict__ quantile_point_indices,// output: the points/cells in which the quantiles lie
                        uint32_t *__restrict__ num_intersections,
                        attr_scalar *__restrict__ point_contribution // holds an array of the contributions of the points to the final image)
)
                        {

    //each thread works on one ray
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    // every color gets one set of spherical harmonics coefficients that describe the view dependent color
    // the number of coefficients per degree is calculated by (1 + degree)²
    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    // spherical harmonics + opacity + view dependent opacity (sggx matrix)
    constexpr int attr_memory_size = sh_dim + 1 + 6;

    //normalize ray direction
    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    //depth quantiles is the pointer to the beginning of the depth_quantiles array
    // by adding thread_idx * num_depth_quantiles it now points to the depth quantiles of the current ray
    const float *ray_depth_quantiles =
        depth_quantiles + thread_idx * num_depth_quantiles;
    
    //calculate the shperical harmonics coefficients given the view direction
    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    //lambda function to load vertex attributes
    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s, float &vod) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        s = (float)attr_ptr[attr_memory_size - 7];
        //if density is larger than threshold, load color using spherical harmonics
        if (s > 1e-6f) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            rgb = Vec3f::Zero();
        }
        //VOD Parameters - stored row-major: row1=[1,2,3], row2=[4,5,6], row3=[7,8,9]
        const float sxx = (float)attr_ptr[attr_memory_size -6];
        const float sxy = (float)attr_ptr[attr_memory_size -5];
        const float sxz = (float)attr_ptr[attr_memory_size -4];
        const float syy = (float)attr_ptr[attr_memory_size -3];
        const float syz = (float)attr_ptr[attr_memory_size -2];
        const float szz = (float)attr_ptr[attr_memory_size -1];
        Mat3f sggx;
        // Initialize row-major: assign row by row
        sggx.row(0) << sxx, sxy, sxz;
        sggx.row(1) << sxy, syy, syz;
        sggx.row(2) << sxz, syz, szz;
        //calculate the view dependent density for the point and the ray view direction
        // w^T * S * w
        vod = ray.direction.transpose() * sggx * ray.direction;
    };

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();

    uint32_t current_quantile_idx = 0;
    float current_quantile;
    if (depth_quantiles) {
        current_quantile = ray_depth_quantiles[current_quantile_idx];
    }

    // calculate Volume Rendering equation for one constant piece, i.e. one cell
    auto functor = [&](uint32_t point_idx,
                       float t_0, //entry point in the current voronoi cell (encoded in distance along the ray)
                       float t_1, //exit point from the current voronoi cell
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s;
        // view dependent density
        float vod;

        load_attributes(point_idx, rgb_primal, s, vod);
        
        //SEE VOLUME RENDERING EQUATION PIECEWISE CONSTANT IN RADFOAM PAPER
        //Transmittance is variable T in the formula
        //delta
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        //NEW VIEW DEPENDENT DENSITY
        // = old density + vod
        float s_primal = sigmoid(s + vod);
        
        // 1 - exp(-rho*delta)
        //float alpha = 1 - expf(-s_primal * delta_t);
        float alpha = 1 - expf(-s_primal * delta_t);
        
        //T * (1 - exp(-rho*delta)) --> this is the contribution of this cell to the final pixel color
        float weight = transmittance * alpha;        
        // per point sum off rendering weights across all rays
        // multiple rays may pass through the same point/cell, hence we use atomicAdd
        if (point_contribution) {
            atomicAdd(point_contribution + point_idx, (attr_scalar)weight);
        }
        accumulated_rgb += weight * rgb_primal;
        // exp(-rho*delta)
        float next_transmittance = transmittance * (1 - alpha);
        //if next transmittance drops below the transmittance of the current depth quantile threshold
        while (current_quantile_idx < num_depth_quantiles &&
               next_transmittance < current_quantile) {
                //then calculate the depth along the ray of that quantile
                // transmittance = transmittance BEFORE current cell
                // next_transmittance = transmittance AFTER current cell
                // s_primal = density of current cell
            quantile_depths[thread_idx * num_depth_quantiles +
                            current_quantile_idx] =
                t_0 + logf(transmittance / current_quantile) / s_primal;
                //save the point index where the quantile lies in
            quantile_point_indices[thread_idx * num_depth_quantiles +
                                   current_quantile_idx] = point_idx;
            //update so the next quantile can be worked on
            current_quantile_idx++;
            if (current_quantile_idx < num_depth_quantiles) {
                current_quantile = ray_depth_quantiles[current_quantile_idx];
            }
        }

        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold;
    };

    // i guess this is the 3d point nearest to the camera based in nearest neighbor
    // for the current ray
    uint32_t start_point = start_point_index[thread_idx];

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      start_point,
                                      settings.max_intersections,
                                      functor);

    // fill any quantiles that haven't been reached by the ray marching
    while (current_quantile_idx < num_depth_quantiles) {
        quantile_depths[thread_idx * num_depth_quantiles +
                        current_quantile_idx] = -1.0f;
        quantile_point_indices[thread_idx * num_depth_quantiles +
                               current_quantile_idx] = UINT32_MAX;
        current_quantile_idx++;
    }

    for (uint32_t i = 0; i < 3; ++i) {
        ray_rgba[thread_idx * 4 + i] = attr_scalar(accumulated_rgb[i]);
    }
    ray_rgba[thread_idx * 4 + 3] = attr_scalar(1 - transmittance);

    if (num_intersections)
        num_intersections[thread_idx] = n;
}

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void backward(TraceSettings settings,
                         const Vec3f *__restrict__ points,
                         const attr_scalar *__restrict__ attributes,
                         const uint32_t *__restrict__ point_adjacency,
                         const uint32_t *__restrict__ point_adjacency_offsets,
                         const Vec4h *__restrict__ adjacent_diff,
                         const Ray *__restrict__ rays,
                         uint32_t num_rays,
                         const uint32_t *__restrict__ start_point_index,
                         uint32_t num_depth_quantiles,
                         const float *__restrict__ depth_quantiles,
                         const uint32_t *__restrict__ quantile_point_indices,
                         const attr_scalar *__restrict__ ray_rgba,
                         const attr_scalar *__restrict__ ray_rgba_grad, // input: ∂loss/∂RGBA
                         const float *__restrict__ depth_grad, //input: Gradient w.r.t. ∂loss/∂depth
                         const attr_scalar *__restrict__ ray_error, // input: Optional per-ray error for point error accumulation
                         Ray *__restrict__ ray_grad,
                         Vec3f *__restrict__ points_grad, // output: Gradient w.r.t. point positions
                         attr_scalar *__restrict__ attribute_grad, //output: Gradient w.r.t. SH coefficients and density
                         attr_scalar *__restrict__ point_error) // output: accumulated error per point
                         {

    //each thread works on one ray
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    // every color gets one set of spherical harmonics coefficients that describe the view dependent color
    // the number of coefficients per degree is calculated by (1 + degree)²
    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    // spherical harmonics + density + view dependent density (sggx matrix)
    constexpr int attr_memory_size = sh_dim + 1 + 6;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    // ray_depth_grad[0] = ∂L/∂quantile_depth[0] for this ray
    // ray_depth_grad[1] = ∂L/∂quantile_depth[1] for this ray
    const float *ray_depth_grad = depth_grad + thread_idx * num_depth_quantiles;
    //depth quantiles is the pointer to the beginning of the depth_quantiles array
    // by adding thread_idx * num_depth_quantiles it now points to the depth quantiles of the current ray
    const float *ray_depth_quantiles =
        depth_quantiles + thread_idx * num_depth_quantiles;
    
    //calculate the spherical harmonics coefficients given the view direction
    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    //lambda function to load vertex attributes
    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s, float &vod) {
            const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
            s = (float)attr_ptr[attr_memory_size - 7];
            //if density is larger than threshold, load color using spherical harmonics
            if (s > 1e-6f) {
                rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
            } else {
                rgb = Vec3f::Zero();
            }
        //VOD Parameters - stored row-major: row1=[1,2,3], row2=[4,5,6], row3=[7,8,9]
        const float sxx = (float)attr_ptr[attr_memory_size -6];
        const float sxy = (float)attr_ptr[attr_memory_size -5];
        const float sxz = (float)attr_ptr[attr_memory_size -4];
        const float syy = (float)attr_ptr[attr_memory_size -3];
        const float syz = (float)attr_ptr[attr_memory_size -2];
        const float szz = (float)attr_ptr[attr_memory_size -1];
        Mat3f sggx;
        // Initialize row-major: assign row by row
        sggx.row(0) << sxx, sxy, sxz;
        sggx.row(1) << sxy, syy, syz;
        sggx.row(2) << sxz, syz, szz;
        //calculate the view dependent density for the point and the ray view direction
        // w^T * S * w
        vod = ray.direction.transpose() * sggx * ray.direction;
    };

    Vec4f rgba_grad, rgba;
#pragma unroll
    // get rgba of current ray
    for (uint32_t i = 0; i < 4; ++i) {
        rgba_grad[i] = (float)ray_rgba_grad[thread_idx * 4 + i];
        rgba[i] = (float)ray_rgba[thread_idx * 4 + i];
    }

    // get error of current ray
    float error;
    if (ray_error) {
        error = (float)ray_error[thread_idx];
    }

    uint32_t current_quantile_idx = 0;
    float current_quantile;
    if (depth_quantiles) {
        current_quantile = ray_depth_quantiles[current_quantile_idx];
    }
    float current_depth_grad = 0.0f;
    // read out all the quantile depth gradients for the current ray
    // NORMALIZE each quantile depth gradient by the density of the cell where the quantile appears
    // add the normalized gradients up in current_depth_grad
    for (uint32_t i = 0; i < num_depth_quantiles; ++i) {
        if (quantile_point_indices[thread_idx * num_depth_quantiles + i] !=
            UINT32_MAX) {
            uint32_t point_idx =
                quantile_point_indices[thread_idx * num_depth_quantiles + i];

            float s = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 7];

            //VOD Parameters - stored row-major: row1=[1,2,3], row2=[4,5,6], row3=[7,8,9]
            const float sxx = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 6];
            const float sxy = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 5];
            const float sxz = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 4];
            const float syy = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 3];
            const float syz = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 2];
            const float szz = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 1];
            Mat3f sggx;
            // Initialize row-major: assign row by row
            sggx.row(0) << sxx, sxy, sxz;
            sggx.row(1) << sxy, syy, syz;
            sggx.row(2) << sxz, syz, szz;
            //calculate the view dependent density for the point and the ray view direction
            // w^T * S * w
            float vod = ray.direction.transpose() * sggx * ray.direction;

            current_depth_grad += ray_depth_grad[i] / sigmoid(s + vod);
        }
    }

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();

    uint32_t prev_point_idx = UINT32_MAX;
    Vec3f prev_point = Vec3f::Zero();
    Vec3f prev_point_grad = Vec3f::Zero();

    Vec3f current_point_grad = Vec3f::Zero();
    Vec3f next_point_grad = Vec3f::Zero();

    // calculate gradients in backward pass for one single cell
    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s;        
        //vod parameters
        float vod;

        load_attributes(point_idx, rgb_primal, s, vod);
        
        // this is our new formula inlcuding vied dependent density
        float s_primal = sigmoid(s + vod);

        // calculate weight of the cell like in forward pass
        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);
        float weight = transmittance * alpha;
        float dalpha_ds_primal = delta_t * (1 - alpha);
        float dalpha_ddelta_t = 0.0f;
        if (delta_t > 0.0f) {
            dalpha_ddelta_t = s_primal * (1 - alpha);
        }

        // add cell contribution of color to final color like in forward
        accumulated_rgb += weight * rgb_primal;
        // track the error of the point
        if (point_error) {
            atomicAdd(point_error + point_idx, (attr_scalar)(weight * error));
        }

        ////////RGBA LOSS
        // extract the first 3 components of thr rgba gradient --> [dL/dR, dL/dG, dL/dB]
        // for the current point
        // weight = contribution of the point to the final color
        // use weight to propagate rgb loss to current cell
        // dL/dRGB_final is given in rgba_grad
        // dRGB_final/dRGB_point = weight
        Vec3f dL_drgb_primal = rgba_grad.template head<3>() * weight;
        
        // the color that would be accumulated after the current cell
        // the remainding color is theoretically calculated by the remainding transmittance * the rgb color of the following cells
        Vec3f rgb_rest = rgba.template head<3>() - accumulated_rgb;
        // this divides out the transmittance so we get alpha*color of the following cells
        rgb_rest /= (transmittance * (1 - alpha + 1e-6f));

        // how changing alpha affects the RGB color
        float dL_dalpha =
            transmittance *
            (rgb_primal - rgb_rest).dot(rgba_grad.template head<3>());
        //how changing alpha affects the final opacity
        dL_dalpha += (1 - rgba[3]) * rgba_grad[3] / (1 - alpha + 1e-6f);

        float dL_ds_primal = dL_dalpha * dalpha_ds_primal;
        float dL_ddelta_t = dL_dalpha * dalpha_ddelta_t;

        //////DEPTH QUANTILE LOSS
        float dL_dt0 = 0.0f;

        // transmittance = transmittance before current cell
        // next tansmittance = transmittance after current cell
        float next_transmittance = transmittance * (1 - alpha);
        // check if we cross quantile threshold in the current cell
        // so we can differentiate the exact formula that calculates the quantile depth
        // we take in account every quantile that is affected by the current cell
        while (current_quantile_idx < num_depth_quantiles &&
               next_transmittance < current_quantile) {

            float depth_grad_i =
                ray_depth_grad[current_quantile_idx] / s_primal;
            dL_dt0 += depth_grad_i;

            // effectively, the s_primal is squared because it is already contained in dept_grad_i
            // this is correct according to differentiating the formula of the depth quantiles
            dL_ds_primal += -depth_grad_i *
                            logf(transmittance / current_quantile) / s_primal;

            //we update the depth gradients for the quantiles that still lie ahead
            current_depth_grad -= depth_grad_i;

            current_quantile_idx++;
            if (current_quantile_idx < num_depth_quantiles) {
                current_quantile = ray_depth_quantiles[current_quantile_idx];
            }
        }

        // The codes accumulates the gradients for ALL following quantiles that are still ahead, i.e. come after the current cell
        // by using current_depth_grad, which accumulates ALL depth gradients "dL/ddepthq" for the quantiles that lie ahead
        if (current_quantile_idx < num_depth_quantiles) {
            // - delta_cell * (dL/ddepthq / rho_q)
            dL_ds_primal += -delta_t * current_depth_grad;
            // - rho_cell * (dL/ddepthq / rho_q)
            dL_ddelta_t += -s_primal * current_depth_grad;
        }

        dL_dt0 += -dL_ddelta_t;
        float dL_dt1 = dL_ddelta_t;

        Vec3f dt0_dprev_point;
        if (prev_point_idx != UINT32_MAX) {
            dt0_dprev_point =
                cell_intersection_grad(prev_point, current_point, ray);
        } else {
            dt0_dprev_point = Vec3f::Zero();
        }

        Vec3f dt1_dcurrent_point =
            cell_intersection_grad(current_point, next_point, ray);
        Vec3f dt0_dcurrent_point =
            cell_intersection_grad(current_point, prev_point, ray);

        Vec3f dt1_dnext_point =
            cell_intersection_grad(next_point, current_point, ray);

        prev_point_grad += dL_dt0 * dt0_dprev_point;
        current_point_grad +=
            dL_dt0 * dt0_dcurrent_point + dL_dt1 * dt1_dcurrent_point;
        next_point_grad += dL_dt1 * dt1_dnext_point;

        if (prev_point_idx != UINT32_MAX) {
            atomic_add_vec(points_grad + prev_point_idx, prev_point_grad);
        }
        prev_point = current_point;
        prev_point_idx = point_idx;
        prev_point_grad = current_point_grad;

        current_point_grad = next_point_grad;
        next_point_grad = Vec3f::Zero();

        transmittance = next_transmittance;

        for (uint32_t i = 0; i < 3; ++i) {
            if (rgb_primal[i] == 0.0f) {
                dL_drgb_primal[i] = 0.0f;
            }
        }
        write_rgb_grad_to_sh<attr_scalar, sh_degree>(
            sh_coeffs,
            dL_drgb_primal,
            attribute_grad + point_idx * attr_memory_size);
        
        float ds_primal_ddensity = s_primal * (1.0 - s_primal);
        float dL_ddensity = dL_ds_primal * ds_primal_ddensity;
        // ddensity_ds = 1.0;
        // ddensity_dvod = 1.0;

        // writing density grad to primal density
        atomicAdd(attribute_grad + point_idx * attr_memory_size +
                      (attr_memory_size - 7),
                  (attr_scalar)dL_ddensity);

        write_density_grad_to_sggx<attr_scalar>(ray.direction,
        dL_ddensity,
        attribute_grad + point_idx * attr_memory_size + (attr_memory_size - 6));

        return transmittance > settings.weight_threshold;
    };

    uint32_t start_point = start_point_index[thread_idx];

    trace<block_size, 2>(ray,
                         points,
                         point_adjacency,
                         point_adjacency_offsets,
                         adjacent_diff,
                         start_point,
                         settings.max_intersections,
                         functor);
}

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void
visualization(TraceSettings settings,
              const Vec3f *__restrict__ points,
              const attr_scalar *__restrict__ attributes,
              const uint32_t *__restrict__ point_adjacency,
              const uint32_t *__restrict__ point_adjacency_offsets,
              const Vec4h *__restrict__ adjacent_diff,
              VisualizationSettings vis_settings,
              CMapTable cmap_table,
              Camera camera,
              cudaSurfaceObject_t output_rgba,
              uint32_t start_point_index) {

    // every kernel works on one pixel
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pix_i = thread_idx % camera.width;
    uint32_t pix_j = thread_idx / camera.width;

    if (pix_i >= camera.width || pix_j >= camera.height)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim + 6;

    Ray ray = cast_ray(camera, pix_i, pix_j);
    if (ray.direction.norm() < 0.1f) {
        surf2Dwrite(0, output_rgba, 4 * pix_i, camera.height - 1 - pix_j);
        return;
    }

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    //lambda function to load vertex attributes
    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s, float &vod) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        s = (float)attr_ptr[attr_memory_size - 7];
        //if density is larger than threshold, load color using spherical harmonics
        if (s > 1e-6f) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            rgb = Vec3f::Zero();
        }
        //VOD Parameters - stored row-major: row1=[1,2,3], row2=[4,5,6], row3=[7,8,9]
        const float sxx = (float)attr_ptr[attr_memory_size -6];
        const float sxy = (float)attr_ptr[attr_memory_size -5];
        const float sxz = (float)attr_ptr[attr_memory_size -4];
        const float syy = (float)attr_ptr[attr_memory_size -3];
        const float syz = (float)attr_ptr[attr_memory_size -2];
        const float szz = (float)attr_ptr[attr_memory_size -1];
        Mat3f sggx;
        // Initialize row-major: assign row by row
        sggx.row(0) << sxx, sxy, sxz;
        sggx.row(1) << sxy, syy, syz;
        sggx.row(2) << sxz, syz, szz;
        //calculate the view dependent density for the point and the ray view direction
        // w^T * S * w
        vod = ray.direction.transpose() * sggx * ray.direction;
};

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();
    float depth = 0.0f;
    bool depth_quantile_passed = false;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s;
        float vod;

        load_attributes(point_idx, rgb_primal, s, vod);

        float s_primal = sigmoid(s + vod);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);

        accumulated_rgb += transmittance * alpha * rgb_primal;

        float next_transmittance = transmittance * (1 - alpha);
        if (!depth_quantile_passed &&
            next_transmittance < vis_settings.depth_quantile) {
            depth = t_0 + logf(transmittance / vis_settings.depth_quantile) /
                              s_primal;
            depth_quantile_passed = true;
        }

        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold;
    };

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      start_point_index,
                                      settings.max_intersections,
                                      functor);

    uint32_t out;

    if (vis_settings.mode == VisualizationMode::RGB) {
        Vec3f color = accumulated_rgb;

        Vec3f bg_color;
        if (vis_settings.checker_bg) {
            int is = 2 * ((pix_i / 20) % 2) - 1;
            int js = 2 * ((pix_j / 20) % 2) - 1;
            if (is * js > 0) {
                bg_color = Vec3f(0.3f, 0.3f, 0.3f);
            } else {
                bg_color = Vec3f(0.5f, 0.5f, 0.5f);
            }
        } else {
            bg_color = *vis_settings.bg_color;
        }

        color += transmittance * bg_color;

        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    } else if (vis_settings.mode == VisualizationMode::Depth) {
        float val = depth / vis_settings.max_depth;

        Vec3f color = colormap(val, vis_settings.color_map, cmap_table);

        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    } else if (vis_settings.mode == VisualizationMode::Alpha) {
        out = make_rgba8(1.0f - transmittance,
                         1.0f - transmittance,
                         1.0f - transmittance,
                         1.0f);
    } else if (vis_settings.mode == VisualizationMode::Intersections) {
        float val = float(n - 1) / float(settings.max_intersections);

        Vec3f color = colormap(val, vis_settings.color_map, cmap_table);

        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    }

    surf2Dwrite(out, output_rgba, 4 * pix_i, camera.height - 1 - pix_j);
}

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void benchmark(TraceSettings settings,
                          const Vec3f *__restrict__ points,
                          const attr_scalar *__restrict__ attributes,
                          const uint32_t *__restrict__ point_adjacency,
                          const uint32_t *__restrict__ point_adjacency_offsets,
                          const Vec4h *__restrict__ adjacent_diff,
                          Camera camera,
                          const uint32_t *__restrict__ start_point_index,
                          uint32_t *__restrict__ output_rgba) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pix_i = thread_idx % camera.width;
    uint32_t pix_j = thread_idx / camera.width;

    if (pix_i >= camera.width || pix_j >= camera.height)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim + 6;

    Ray ray = cast_ray(camera, pix_i, pix_j);
    if (ray.direction.norm() < 0.1f) {
        output_rgba[thread_idx] = 0;
        return;
    }

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    //lambda function to load vertex attributes
    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s, float &vod) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        s = (float)attr_ptr[attr_memory_size - 7];
        //if density is larger than threshold, load color using spherical harmonics
        if (s > 1e-6f) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            rgb = Vec3f::Zero();
        }
        //VOD Parameters - stored row-major: row1=[1,2,3], row2=[4,5,6], row3=[7,8,9]
        const float sxx = (float)attr_ptr[attr_memory_size -6];
        const float sxy = (float)attr_ptr[attr_memory_size -5];
        const float sxz = (float)attr_ptr[attr_memory_size -4];
        const float syy = (float)attr_ptr[attr_memory_size -3];
        const float syz = (float)attr_ptr[attr_memory_size -2];
        const float szz = (float)attr_ptr[attr_memory_size -1];
        Mat3f sggx;
        // Initialize row-major: assign row by row
        sggx.row(0) << sxx, sxy, sxz;
        sggx.row(1) << sxy, syy, syz;
        sggx.row(2) << sxz, syz, szz;
        //calculate the view dependent density for the point and the ray view direction
        // w^T * S * w
        vod = ray.direction.transpose() * sggx * ray.direction;
};

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s;
        float vod;

        load_attributes(point_idx, rgb_primal, s, vod);

        float s_primal = sigmoid(s + vod);
        //float s_primal = vod;
        

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);

        accumulated_rgb += transmittance * alpha * rgb_primal;
        transmittance = transmittance * (1 - alpha);

        return transmittance > settings.weight_threshold;
    };

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      *start_point_index,
                                      settings.max_intersections,
                                      functor);

    output_rgba[thread_idx] = make_rgba8(
        accumulated_rgb[0], accumulated_rgb[1], accumulated_rgb[2], 1.0f);
}

__global__ void prefetch_adjacent_diff_kernel(
    const Vec3f *__restrict__ points,
    uint32_t num_points,
    uint32_t point_adjacency_size,
    const uint32_t *__restrict__ point_adjacency,
    const uint32_t *__restrict__ point_adjacency_offsets,
    Vec4h *__restrict__ adjacent_diff) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    Vec3f p = points[i];
    uint32_t offset_start = point_adjacency_offsets[i];
    uint32_t offset_end = point_adjacency_offsets[i + 1];
    uint32_t num_adjacent = offset_end - offset_start;

    for (uint32_t j = 0; j < num_adjacent; ++j) {
        uint32_t adjacent_idx = point_adjacency[offset_start + j];
        Vec3f q = points[adjacent_idx];
        Vec3f diff = q - p;
        adjacent_diff[offset_start + j] = Vec4h(diff[0], diff[1], diff[2], 0);
    }
}

void prefetch_adjacent_diff(const Vec3f *points,
                            uint32_t num_points,
                            uint32_t point_adjacency_size,
                            const uint32_t *point_adjacency,
                            const uint32_t *point_adjacency_offsets,
                            Vec4h *adjacent_diff,
                            const void *stream) {
    launch_kernel_1d<256>(prefetch_adjacent_diff_kernel,
                          num_points,
                          stream,
                          points,
                          num_points,
                          point_adjacency_size,
                          point_adjacency,
                          point_adjacency_offsets,
                          adjacent_diff);
}

template <typename attr_scalar, int sh_degree>
class CUDATracingPipeline : public Pipeline {
  public:
    CUDATracingPipeline() = default;

    virtual ~CUDATracingPipeline() {}

    void trace_forward(const TraceSettings &settings,
                       uint32_t num_points,
                       const Vec3f *points,
                       const void *attributes,
                       uint32_t point_adjacency_size,
                       const uint32_t *point_adjacency,
                       const uint32_t *point_adjacency_offsets,
                       uint32_t num_rays,
                       const Ray *rays,
                       const uint32_t *start_point_index,
                       uint32_t num_depth_quantiles,
                       const float *depth_quantiles,
                       void *ray_rgba,
                       float *quantile_dpeths,
                       uint32_t *quantile_point_indices,
                       uint32_t *num_intersections,
                       void *point_contribution) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        launch_kernel_1d<block_size>(
            forward<attr_scalar, sh_degree, block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            reinterpret_cast<const attr_scalar *>(attributes),
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff.begin(),
            rays,
            num_rays,
            start_point_index,
            num_depth_quantiles,
            depth_quantiles,
            static_cast<attr_scalar *>(ray_rgba),
            quantile_dpeths,
            quantile_point_indices,
            num_intersections,
            static_cast<attr_scalar *>(point_contribution));
    }

    void trace_backward(const TraceSettings &settings,
                        uint32_t num_points,
                        const Vec3f *points,
                        const void *attributes,
                        uint32_t point_adjacency_size,
                        const uint32_t *point_adjacency,
                        const uint32_t *point_adjacency_offsets,
                        uint32_t num_rays,
                        const Ray *rays,
                        const uint32_t *start_point_index,
                        uint32_t num_depth_quantiles,
                        const float *depth_quantiles,
                        const uint32_t *quantile_point_indices,
                        const void *ray_rgba,
                        const void *ray_rgba_grad,
                        const float *depth_grad,
                        const void *ray_error,
                        Ray *ray_grad,
                        Vec3f *points_grad,
                        void *attribute_grad,
                        void *point_error) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        launch_kernel_1d<block_size>(
            backward<attr_scalar, sh_degree, block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            reinterpret_cast<const attr_scalar *>(attributes),
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff.begin(),
            rays,
            num_rays,
            start_point_index,
            num_depth_quantiles,
            depth_quantiles,
            quantile_point_indices,
            static_cast<const attr_scalar *>(ray_rgba),
            static_cast<const attr_scalar *>(ray_rgba_grad),
            depth_grad,
            static_cast<const attr_scalar *>(ray_error),
            ray_grad,
            points_grad,
            static_cast<attr_scalar *>(attribute_grad),
            static_cast<attr_scalar *>(point_error));
    }

    void trace_visualization(const TraceSettings &settings,
                             const VisualizationSettings &vis_settings,
                             const Camera &camera,
                             CMapTable cmap_table,
                             uint32_t num_points,
                             uint32_t num_tets,
                             const void *points,
                             const void *attributes,
                             const void *point_adjacency,
                             const void *point_adjacency_offsets,
                             const void *adjacent_diff,
                             uint32_t start_index,
                             uint64_t output_surface,
                             const void *stream) override {

        uint32_t num_rays = camera.width * camera.height;
        constexpr uint32_t block_size = 128;

        launch_kernel_1d<block_size>(
            visualization<attr_scalar, sh_degree, block_size>,
            num_rays,
            stream,
            settings,
            reinterpret_cast<const Vec3f *>(points),
            reinterpret_cast<const attr_scalar *>(attributes),
            reinterpret_cast<const uint32_t *>(point_adjacency),
            reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
            reinterpret_cast<const Vec4h *>(adjacent_diff),
            vis_settings,
            cmap_table,
            camera,
            output_surface,
            start_index);
    }

    void trace_benchmark(const TraceSettings &settings,
                         uint32_t num_points,
                         const Vec3f *points,
                         const void *attributes,
                         const uint32_t *point_adjacency,
                         const uint32_t *point_adjacency_offsets,
                         const Vec4h *adjacent_diff,
                         Camera camera,
                         const uint32_t *start_point_index,
                         uint32_t *ray_rgba) override {

        uint32_t num_rays = camera.width * camera.height;

        constexpr uint32_t block_size = 512;
        launch_kernel_1d<block_size>(
            benchmark<attr_scalar, sh_degree, block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            reinterpret_cast<const attr_scalar *>(attributes),
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff,
            camera,
            start_point_index,
            ray_rgba);
    }

    uint32_t attribute_dim() const override {
        // here we also have to add the new VOD parameters
        return 1 + 3 * (1 + sh_degree) * (1 + sh_degree) + 6;
    }

    ScalarType attribute_type() const override {
        return scalar_code<attr_scalar>();
    }
};

std::shared_ptr<Pipeline> create_pipeline(int sh_degree, ScalarType attr_type) {

    if (attr_type == ScalarType::Float32) {
        if (sh_degree == 0) {
            return std::make_shared<CUDATracingPipeline<float, 0>>();
        } else if (sh_degree == 1) {
            return std::make_shared<CUDATracingPipeline<float, 1>>();
        } else if (sh_degree == 2) {
            return std::make_shared<CUDATracingPipeline<float, 2>>();
        } else if (sh_degree == 3) {
            return std::make_shared<CUDATracingPipeline<float, 3>>();
        } else {
            throw std::runtime_error("Unsupported SH degree");
        }
    } else if (attr_type == ScalarType::Float16) {
        if (sh_degree == 0) {
            return std::make_shared<CUDATracingPipeline<__half, 0>>();
        } else if (sh_degree == 1) {
            return std::make_shared<CUDATracingPipeline<__half, 1>>();
        } else if (sh_degree == 2) {
            return std::make_shared<CUDATracingPipeline<__half, 2>>();
        } else if (sh_degree == 3) {
            return std::make_shared<CUDATracingPipeline<__half, 3>>();
        } else {
            throw std::runtime_error("Unsupported SH degree");
        }
    } else {
        throw std::runtime_error("Unsupported attribute type");
    }
}

} // namespace radfoam