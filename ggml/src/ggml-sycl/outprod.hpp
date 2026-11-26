#ifndef GGML_SYCL_OUTPROD_HPP
#define GGML_SYCL_OUTPROD_HPP

#include "common.hpp"

void ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

// true when dst runs as one direct kernel (short contraction): no oneMKL call, no host wait
bool ggml_sycl_out_prod_direct(const ggml_tensor * dst);

// The MoE combine's second half (see ggml_sycl_moe_block_fused): dst [ne0, 1, n_tokens] f32, contiguous, is the sum
// over the n_slots half rows h[token][slot][ne0] of a token, in slot order in f32 (the rows already carry the
// router weights). ne0 % 8 == 0.
void ggml_sycl_moe_combine_half(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const sycl::half * h, int n_slots);


#endif // GGML_SYCL_OUTPROD_HPP

