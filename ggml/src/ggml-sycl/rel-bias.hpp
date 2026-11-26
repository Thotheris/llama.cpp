#pragma once

#include "common.hpp"

// GGML_OP_REL_BIAS (ggml_rel_bias), see ggml.h
bool ggml_sycl_rel_bias_supported(const ggml_tensor * op);
void ggml_sycl_rel_bias(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// the bias of one cell: the f32 dot of its table column c and v in k order (rel_bias_kernel and fattn_dec_kernel)
template <int NV> static inline float ggml_sycl_rel_bias_dot(const float (&c)[NV], const float * v) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        acc += c[k]*v[k];
    }
    return acc;
}

// Flash attention with the bias computed in the kernel (GGML_SYCL_FA_REL_INKERNEL, default on): a REL_BIAS node in
// the table form (pos NULL, f16, n_rel 16) whose only reader is a rel-bias flash attention that takes the prefill
// kernel is not computed; in its place ggml_sycl_rel_bias_fa_pack packs the inputs of the bias into the node's buffer
// (at the node, while those inputs are alive) and the flash attention computes bias = dot(table[:, pos], v) itself.
// The dot takes f16 table rows and f16 r with f32 sums and is not rounded to f16, so the scores differ from those of
// the op in rounding only; row-exact flash attention (the same row in every batch) keeps the op.
//
// The packed buffer (byte offsets from rb->data, all 16-byte aligned):
//   hdr: int xc, float x16, float x17, int c     xc: table rows n_rel, n_rel + 1 are (x16, x17) in every column 0..c
//   v:   [token][head] 3 x uint4: the 16 clipped r of the token and head as f16 pairs (the B operand of an f16 DPAS),
//        then { half2(t_hi, t_lo), float t, 0, 0 } with t_hi + t_lo = t
//   t2:  [c + 9][8] uint4: row d + 8 (d = -8 .. c), column pair cp: dword g = the f16 pair cp of table column
//        clamp(d + 2 g, 0, c), the A operand of 8 consecutive distances in one 16-byte load per lane
//   x2:  [c + 9][2] uint4: the same for (row n_rel, row n_rel) and (row n_rel + 1, 0)
//   tf:  [n_cols][16] f16: table rows 0..15 of every column;  tx: [n_cols] float2: rows n_rel, n_rel + 1
// c is the clamp column (distances past it read column c), the column before masked_pos.
struct ggml_sycl_rel_pack_layout {
    size_t hdr, v, t2, x2, tf, tx, end;
    int    c, n_cols;
};

bool ggml_sycl_rel_bias_fa_pack_layout(const ggml_tensor * rb, int masked_pos, ggml_sycl_rel_pack_layout & L);
void ggml_sycl_rel_bias_fa_pack(ggml_backend_sycl_context & ctx, const ggml_tensor * rb, const ggml_sycl_rel_pack_layout & L);

// The decode form (pos set, GGML_SYCL_FA_REL_DEC, default on): a REL_BIAS node whose only reader is a flash attention
// that takes fattn_dec_kernel (rb as its per-head mask) is replaced by a pack of
//   v:   [token][head][nvp] f32, the v of rel_bias_kernel;   pos: [token][n_kv] int32, the pos rows clamped to the table
// and the kernel computes f16(dot(table[:, pos], v)) for its cells: the bits of the mask the op would have written.
struct ggml_sycl_rel_dec_layout {
    size_t v, pos, end;
    int    nvp, n_kv, n_head;
};

bool ggml_sycl_rel_bias_fa_dec_layout(const ggml_tensor * rb, ggml_sycl_rel_dec_layout & L);

// graph compute: true when the REL_BIAS node i was packed in place of its op because its flash attention computes the
// bias; the flash attention asks ggml_sycl_rel_bias_fa_take(rb, fa) (once) whether that happened
bool ggml_sycl_rel_bias_fa_pack_node(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i);
bool ggml_sycl_rel_bias_fa_take(const ggml_tensor * rb, const ggml_tensor * fa);
