#ifndef GGML_SYCL_FATTN_REL_XE_HPP
#define GGML_SYCL_FATTN_REL_XE_HPP

#include "common.hpp"

// The rel-bias prefill flash attention with the bias computed in the kernel, restructured the way Intel's Xe2
// flash attention (sycl-tla) is built: one ESIMD thread owns 16 query rows of one head over all 128 output dims, so
// the softmax tile P stays in registers as the A operand of the P V DPAS; K and V come straight from global memory
// (L1) as DPAS B operands by 2D block loads (K transposed, V VNNI-transformed), with a 2D prefetch of the next tile;
// the threads of a work-group share K and V through L1 and are held together by one split barrier per tile. Called
// by fattn_rel_prefill (fattn-dpas.cpp) after the tile flags (fattn_rel_tiles_aff); GGML_SYCL_FA_XE=0 keeps the
// SIMT kernel.
struct ggml_sycl_fa_xe_params {
    const char *       Q;
    const char *       K;
    const char *       V;
    const int32_t *    pos;
    const float *      sinks;
    float *            dst;
    float              scale;
    int                neq1, neq2, nek1, gqa;
    int                n_rb64;   // 64-row blocks of the tile flags
    int                n_tiles;  // 32-cell tiles
    int                n_words;  // flag words per block
    int64_t            nbq1, nbq2, nbq3;  // bytes
    int64_t            nbk1, nbk2, nbk3;  // bytes
    int64_t            nbv1, nbv2, nbv3;  // bytes
    int64_t            nbp1;              // int32
    int                masked_pos;
    const char *       pk;                // the packed REL_BIAS node (rel-bias.hpp)
    int64_t            pk_hdr, pk_v, pk_tf, pk_tx;
    int                pk_c, pk_ncols, pk_nhead;
    const uint32_t *   tl;                // tile flags: live, non-uniform, affine words per 64-row block
    const sycl::int2 * bw;                // (B, W) per (stream, 64-row block, tile)
};

// the kernel can take the op (env switch, alignment of the 2D surfaces)
bool ggml_sycl_fa_rel_xe_ok(const ggml_sycl_fa_xe_params & p);
void ggml_sycl_fa_rel_xe(const ggml_sycl_fa_xe_params & p, int ne3, dpct::queue_ptr stream);

#endif // GGML_SYCL_FATTN_REL_XE_HPP
