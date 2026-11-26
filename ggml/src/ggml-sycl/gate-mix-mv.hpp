#ifndef GGML_SYCL_GATE_MIX_MV_HPP
#define GGML_SYCL_GATE_MIX_MV_HPP

#include "common.hpp"

// GGML_OP_GATE_MIX_MV: [gain | forget] = w*z per token, y = (x - tanh(forget)*x) + tanh(gain)*m
void ggml_sycl_gate_mix_mv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// the forms ggml_sycl_gate_mix_mv handles (for supports_op)
bool ggml_sycl_gate_mix_mv_supported(const ggml_tensor * op);

#endif // GGML_SYCL_GATE_MIX_MV_HPP
