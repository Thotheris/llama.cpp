// GGML_OP_GATE_MIX_MV for the SYCL backend, see ggml_gate_mix_mv in ggml.h and ggml_compute_forward_gate_mix_mv in
// ggml-cpu/ops.cpp (the reference). Per token t (a row of x, m, z and y) and element i < n:
//   gain = dot(w[:, i], z[:, t]), forget = dot(w[:, n + i], z[:, t])   (w: [k, 2n], or [k, 2n, n_tokens], one per token)
//   y[i, t] = (x - tanh(forget)*x) + tanh(gain)*m
// The model builds it for small batches (up to 8 tokens), where the k*2n weight read dominates. One sub-group per
// (i, t) computes both dots (lanes stride over k, float4 loads when k and the strides allow, then a sub-group
// reduction), so the op is one launch that reads every weight row once per token.

#include "gate-mix-mv.hpp"

#include <cstdint>

#define SYCL_GATE_MIX_MV_WG_SIZE 256 // SYCL_GATE_MIX_MV_WG_SIZE / WARP_SIZE elements per work-group

struct gate_mix_mv_params {
    int64_t n;    // elements per row of x, m, y
    int64_t k;    // the gate input width (columns of w, values per row of z)
    int64_t sx;   // token strides (elements); sx can be 0 (one x row for all tokens)
    int64_t sm;
    int64_t sz;
    int64_t sd;
    int64_t sw1;  // the stride of w between its 2n rows (elements)
    int64_t sw2;  // the stride of w between tokens: 0 for one weight for all tokens
};

template <bool vec4>
static void gate_mix_mv_f32_sycl(const float * x, const float * m, const float * w, const float * z, float * dst,
                                 const int64_t n_tokens, const gate_mix_mv_params p, dpct::queue_ptr stream) {
    constexpr int elems_per_wg = SYCL_GATE_MIX_MV_WG_SIZE / WARP_SIZE;
    const int64_t n_wg         = (p.n + elems_per_wg - 1) / elems_per_wg;

    const sycl::nd_range<2> range(sycl::range<2>(n_tokens, n_wg * SYCL_GATE_MIX_MV_WG_SIZE),
                                  sycl::range<2>(1, SYCL_GATE_MIX_MV_WG_SIZE));

    stream->parallel_for(range, [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        const auto    sg   = it.get_sub_group();
        const int     lane = sg.get_local_linear_id();
        const int64_t t    = it.get_global_id(0);
        const int64_t i    = it.get_group(1) * elems_per_wg + sg.get_group_linear_id();
        if (i >= p.n) {
            return;
        }

        const float * wt = w + t * p.sw2;
        const float * wg = wt + i * p.sw1;          // the gain row
        const float * wf = wt + (p.n + i) * p.sw1;  // the forget row
        const float * zt = z + t * p.sz;

        float gain   = 0.0f;
        float forget = 0.0f;
        if constexpr (vec4) {
            const sycl::float4 * wg4 = (const sycl::float4 *) wg;
            const sycl::float4 * wf4 = (const sycl::float4 *) wf;
            const sycl::float4 * zt4 = (const sycl::float4 *) zt;
            for (int64_t c = lane; c < p.k / 4; c += WARP_SIZE) {
                const sycl::float4 zv = zt4[c];
                const sycl::float4 gv = wg4[c] * zv;
                const sycl::float4 fv = wf4[c] * zv;
                gain   += (gv.x() + gv.y()) + (gv.z() + gv.w());
                forget += (fv.x() + fv.y()) + (fv.z() + fv.w());
            }
        } else {
            for (int64_t c = lane; c < p.k; c += WARP_SIZE) {
                const float zv = zt[c];
                gain   += wg[c] * zv;
                forget += wf[c] * zv;
            }
        }
        gain   = sycl::reduce_over_group(sg, gain, sycl::plus<float>());
        forget = sycl::reduce_over_group(sg, forget, sycl::plus<float>());

        if (lane == 0) {
            gain   = sycl::tanh(gain);
            forget = sycl::tanh(forget);
            const float xv = x[t * p.sx + i];
            const float mv = m[t * p.sm + i];
            const float fx = forget * xv;
            const float a  = xv - fx;
            const float gm = gain * mv;
            dst[t * p.sd + i] = a + gm;
        }
    });
}

bool ggml_sycl_gate_mix_mv_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * m = op->src[1];
    const ggml_tensor * w = op->src[2];
    const ggml_tensor * z = op->src[3];

    for (const ggml_tensor * t : { x, m, w, z, (const ggml_tensor *) op }) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float)) {
            return false;
        }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % sizeof(float) != 0) {
                return false;
            }
        }
    }
    // x, m, z, y: one row per token (ne[1] tokens); w: [k, 2n] or [k, 2n, n_tokens]
    const int64_t n_tokens = op->ne[1];
    for (const ggml_tensor * t : { x, m, z, (const ggml_tensor *) op }) {
        if (t->ne[1] != n_tokens || t->ne[2] != 1 || t->ne[3] != 1) {
            return false;
        }
    }
    return x->ne[0] == op->ne[0] && m->ne[0] == op->ne[0] && w->ne[0] == z->ne[0] && w->ne[1] == 2 * op->ne[0] &&
           (w->ne[2] == 1 || w->ne[2] == n_tokens) && w->ne[3] == 1;
}

void ggml_sycl_gate_mix_mv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);

    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * m = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * z = dst->src[3];

    GGML_ASSERT(ggml_sycl_gate_mix_mv_supported(dst));

    constexpr size_t ts = sizeof(float);

    gate_mix_mv_params p;
    p.n   = dst->ne[0];
    p.k   = w->ne[0];
    p.sx  = x->nb[1] / ts;
    p.sm  = m->nb[1] / ts;
    p.sz  = z->nb[1] / ts;
    p.sd  = dst->nb[1] / ts;
    p.sw1 = w->nb[1] / ts;
    p.sw2 = w->ne[2] > 1 ? w->nb[2] / ts : 0;

    const int64_t n_tokens = dst->ne[1];
    if (n_tokens == 0 || p.n == 0) {
        return;
    }

    // float4 loads of the w rows and z need 16-byte aligned rows (the pointers are device addresses, not read here)
    const bool vec4 = p.k % 4 == 0 && p.sw1 % 4 == 0 && p.sw2 % 4 == 0 && p.sz % 4 == 0 &&
                      (uintptr_t) w->data % 16 == 0 && (uintptr_t) z->data % 16 == 0;

    const float * xd = (const float *) x->data;
    const float * md = (const float *) m->data;
    const float * wd = (const float *) w->data;
    const float * zd = (const float *) z->data;
    float *       dd = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    if (vec4) {
        gate_mix_mv_f32_sycl<true>(xd, md, wd, zd, dd, n_tokens, p, stream);
    } else {
        gate_mix_mv_f32_sycl<false>(xd, md, wd, zd, dd, n_tokens, p, stream);
    }
}
