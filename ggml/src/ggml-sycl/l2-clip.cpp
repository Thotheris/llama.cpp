// GGML_OP_L2_CLIP for the SYCL backend, see ggml_l2_clip in ggml.h and ggml_compute_forward_l2_clip_f32 in
// ggml-cpu/ops.cpp (the reference): per row (along ne[0]) of x, with the optional per-channel scale s broadcast
// to x (s->ne[0] == x->ne[0], its rows repeat over x's rows)
//   xs = x*s,  y = xs * clamp(threshold / (sqrt(sum(xs*xs)) + 1e-6), 0, 1)
// One sub-group per row: strided per-lane partial sums, a sub-group reduction, then the scaled write. The rows of
// the model are short (128 values of a head), so a sub-group per row keeps every lane busy
// without local memory, and rows of up to 8 values per lane stay in registers between the two passes.

#include "l2-clip.hpp"

#include <algorithm>

#define SYCL_L2_CLIP_WG_SIZE 256 // sub-groups (rows) per work-group: SYCL_L2_CLIP_WG_SIZE / WARP_SIZE

struct l2_clip_params {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1, ne2;            // x rows: i1 + ne1*(i2 + ne2*i3)
    int64_t sx1, sx2, sx3;       // x strides (elements)
    int64_t sd1, sd2, sd3;       // dst strides (elements)
    int64_t sne1, sne2, sne3;    // s shape (broadcast by modulo)
    int64_t ss1, ss2, ss3;       // s strides (elements)
    float   threshold;
};

// npl > 0: the row (at most npl*WARP_SIZE values) is held in registers, so the loads of both passes are issued
// at once and x and s are read once; npl == 0: any row length, two strided passes over memory
template <bool has_scale, int npl>
static void l2_clip_f32_sycl(const float * x, const float * s, float * dst, const l2_clip_params p,
                             dpct::queue_ptr stream) {
    constexpr int rows_per_wg = SYCL_L2_CLIP_WG_SIZE / WARP_SIZE;
    const int64_t n_wg        = (p.nrows + rows_per_wg - 1) / rows_per_wg;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(n_wg * SYCL_L2_CLIP_WG_SIZE), sycl::range<1>(SYCL_L2_CLIP_WG_SIZE)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const auto    sg   = it.get_sub_group();
            const int64_t row  = it.get_group(0) * rows_per_wg + sg.get_group_linear_id();
            const int     lane = sg.get_local_linear_id();
            if (row >= p.nrows) {
                return;
            }

            const int64_t i3 = row / (p.ne1 * p.ne2);
            const int64_t r  = row - i3 * p.ne1 * p.ne2;
            const int64_t i2 = r / p.ne1;
            const int64_t i1 = r - i2 * p.ne1;

            const float * xr = x + i1 * p.sx1 + i2 * p.sx2 + i3 * p.sx3;
            float *       yr = dst + i1 * p.sd1 + i2 * p.sd2 + i3 * p.sd3;
            const float * sr = nullptr;
            if constexpr (has_scale) {
                sr = s + (i1 % p.sne1) * p.ss1 + (i2 % p.sne2) * p.ss2 + (i3 % p.sne3) * p.ss3;
            }

            float acc = 0.0f;
            if constexpr (npl > 0) {
                float v[npl];
#pragma unroll
                for (int j = 0; j < npl; ++j) {
                    const int64_t i = lane + j * WARP_SIZE;
                    float         xi = 0.0f;
                    if (i < p.ncols) {
                        xi = xr[i];
                        if constexpr (has_scale) {
                            xi = xi * sr[i];
                        }
                    }
                    v[j] = xi;
                    acc += xi * xi;
                }
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());

                const float norm   = sycl::sqrt(acc) + 1e-6f;
                const float factor = sycl::clamp(p.threshold / norm, 0.0f, 1.0f);
#pragma unroll
                for (int j = 0; j < npl; ++j) {
                    const int64_t i = lane + j * WARP_SIZE;
                    if (i < p.ncols) {
                        yr[i] = v[j] * factor;
                    }
                }
            } else {
                for (int64_t i = lane; i < p.ncols; i += WARP_SIZE) {
                    float xi = xr[i];
                    if constexpr (has_scale) {
                        xi = xi * sr[i];
                    }
                    acc += xi * xi;
                }
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());

                const float norm   = sycl::sqrt(acc) + 1e-6f;
                const float factor = sycl::clamp(p.threshold / norm, 0.0f, 1.0f);

                for (int64_t i = lane; i < p.ncols; i += WARP_SIZE) {
                    float xi = xr[i];
                    if constexpr (has_scale) {
                        xi = xi * sr[i];
                    }
                    yr[i] = xi * factor;
                }
            }
        });
}

template <bool has_scale>
static void l2_clip_f32_sycl_dispatch(const float * x, const float * s, float * dst, const l2_clip_params p,
                                      dpct::queue_ptr stream) {
    if (p.ncols <= 4 * WARP_SIZE) {
        l2_clip_f32_sycl<has_scale, 4>(x, s, dst, p, stream);
    } else if (p.ncols <= 8 * WARP_SIZE) {
        l2_clip_f32_sycl<has_scale, 8>(x, s, dst, p, stream);
    } else {
        l2_clip_f32_sycl<has_scale, 0>(x, s, dst, p, stream);
    }
}

bool ggml_sycl_l2_clip_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * s = op->src[1];

    if (op->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, op)) {
        return false;
    }
    if (x->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (x->nb[i] % sizeof(float) != 0 || op->nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    if (s) {
        if (s->type != GGML_TYPE_F32 || s->nb[0] != sizeof(float) || s->ne[0] != x->ne[0] || !ggml_can_repeat(s, x)) {
            return false;
        }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (s->nb[i] % sizeof(float) != 0) {
                return false;
            }
        }
    }
    return true;
}

void ggml_sycl_l2_clip(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * s = dst->src[1];

    GGML_ASSERT(ggml_sycl_l2_clip_supported(dst));

    constexpr size_t ts = sizeof(float);

    l2_clip_params p;
    p.ncols     = x->ne[0];
    p.nrows     = ggml_nrows(x);
    p.ne1       = x->ne[1];
    p.ne2       = x->ne[2];
    p.sx1       = x->nb[1] / ts;
    p.sx2       = x->nb[2] / ts;
    p.sx3       = x->nb[3] / ts;
    p.sd1       = dst->nb[1] / ts;
    p.sd2       = dst->nb[2] / ts;
    p.sd3       = dst->nb[3] / ts;
    p.sne1      = s ? s->ne[1] : 1;
    p.sne2      = s ? s->ne[2] : 1;
    p.sne3      = s ? s->ne[3] : 1;
    p.ss1       = s ? s->nb[1] / ts : 0;
    p.ss2       = s ? s->nb[2] / ts : 0;
    p.ss3       = s ? s->nb[3] / ts : 0;
    p.threshold = ggml_get_op_params_f32(dst, 0);

    if (p.nrows == 0 || p.ncols == 0) {
        return;
    }

    dpct::queue_ptr stream = ctx.stream();

    if (s) {
        l2_clip_f32_sycl_dispatch<true>((const float *) x->data, (const float *) s->data, (float *) dst->data, p, stream);
    } else {
        l2_clip_f32_sycl_dispatch<false>((const float *) x->data, nullptr, (float *) dst->data, p, stream);
    }
}
