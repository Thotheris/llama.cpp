// GGML_OP_GATE_MIX for the SYCL backend, see ggml_gate_mix in ggml.h and ggml_compute_forward_gate_mix in
// ggml-cpu/ops.cpp (the reference). Per element i of a row of x and m (n values) and the gate row g (2n values):
//   gain = g[i], forget = g[n + i] (both through tanh when tanh_gates), y = (x - forget*x) + gain*m
// One work-item per output element over a 3-D range (i2 + ne2*i3, i1, i0); any row strides of x, m, g and y.

#include "gate-mix.hpp"

#define SYCL_GATE_MIX_BLOCK_SIZE 256

struct gate_mix_params {
    int64_t n;                    // elements per row of x, m, y
    int64_t ne1, ne2;             // rows of x
    int64_t sx1, sx2, sx3;        // x strides (elements)
    int64_t sm1, sm2, sm3;        // m strides
    int64_t sg1, sg2, sg3;        // g strides
    int64_t sd1, sd2, sd3;        // y strides
};

template <bool tanh_gates>
static void gate_mix_f32_sycl(const float * x, const float * m, const float * g, float * dst, const int64_t ne3,
                              const gate_mix_params p, dpct::queue_ptr stream) {
    // (i2 + ne2*i3, i1, i0): the row indices come from the range, one 32-bit division for i3
    const int64_t           ncb = (p.n + SYCL_GATE_MIX_BLOCK_SIZE - 1) / SYCL_GATE_MIX_BLOCK_SIZE;
    const sycl::nd_range<3> range(sycl::range<3>(p.ne2 * ne3, p.ne1, ncb * SYCL_GATE_MIX_BLOCK_SIZE),
                                  sycl::range<3>(1, 1, SYCL_GATE_MIX_BLOCK_SIZE));

    stream->parallel_for(range, [=](sycl::nd_item<3> it) {
        const int64_t i = it.get_global_id(2);
        if (i >= p.n) {
            return;
        }
        const uint32_t i23 = (uint32_t) it.get_global_id(0);
        const int64_t  i3  = i23 / (uint32_t) p.ne2;
        const int64_t  i2  = i23 - (uint32_t) i3 * (uint32_t) p.ne2;
        const int64_t  i1  = it.get_global_id(1);

        const float * gr = g + i1 * p.sg1 + i2 * p.sg2 + i3 * p.sg3;
        const float   xv = x[i1 * p.sx1 + i2 * p.sx2 + i3 * p.sx3 + i];
        const float   mv = m[i1 * p.sm1 + i2 * p.sm2 + i3 * p.sm3 + i];
        float         gain   = gr[i];
        float         forget = gr[p.n + i];
        if constexpr (tanh_gates) {
            gain   = sycl::tanh(gain);
            forget = sycl::tanh(forget);
        }
        const float fx = forget * xv;
        const float a  = xv - fx;
        const float gm = gain * mv;
        dst[i1 * p.sd1 + i2 * p.sd2 + i3 * p.sd3 + i] = a + gm;
    });
}

bool ggml_sycl_gate_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * m = op->src[1];
    const ggml_tensor * g = op->src[2];

    if (op->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || m->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(x, op) || !ggml_are_same_shape(x, m) || g->ne[0] != 2 * x->ne[0]) {
        return false;
    }
    for (const ggml_tensor * t : { x, m, g, (const ggml_tensor *) op }) {
        if (t->nb[0] != sizeof(float)) {
            return false;
        }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % sizeof(float) != 0) {
                return false;
            }
        }
    }
    return true;
}

void ggml_sycl_gate_mix(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * m = dst->src[1];
    const ggml_tensor * g = dst->src[2];

    GGML_ASSERT(ggml_sycl_gate_mix_supported(dst));

    constexpr size_t ts = sizeof(float);

    gate_mix_params p;
    p.n   = x->ne[0];
    p.ne1 = x->ne[1];
    p.ne2 = x->ne[2];
    p.sx1 = x->nb[1] / ts;   p.sx2 = x->nb[2] / ts;   p.sx3 = x->nb[3] / ts;
    p.sm1 = m->nb[1] / ts;   p.sm2 = m->nb[2] / ts;   p.sm3 = m->nb[3] / ts;
    p.sg1 = g->nb[1] / ts;   p.sg2 = g->nb[2] / ts;   p.sg3 = g->nb[3] / ts;
    p.sd1 = dst->nb[1] / ts; p.sd2 = dst->nb[2] / ts; p.sd3 = dst->nb[3] / ts;

    if (ggml_nrows(x) == 0 || p.n == 0) {
        return;
    }

    const bool tanh_gates = ggml_get_op_params_i32(dst, 0) != 0;

    const float * xd = (const float *) x->data;
    const float * md = (const float *) m->data;
    const float * gd = (const float *) g->data;
    float *       dd = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    if (tanh_gates) {
        gate_mix_f32_sycl<true>(xd, md, gd, dd, x->ne[3], p, stream);
    } else {
        gate_mix_f32_sycl<false>(xd, md, gd, dd, x->ne[3], p, stream);
    }
}
