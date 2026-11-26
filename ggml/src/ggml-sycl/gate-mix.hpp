#ifndef GGML_SYCL_GATE_MIX_HPP
#define GGML_SYCL_GATE_MIX_HPP

#include "common.hpp"

// GGML_OP_GATE_MIX: y = (x - forget*x) + gain*m with g = [gain | forget] per row (tanh on both when tanh_gates)
void ggml_sycl_gate_mix(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// the forms ggml_sycl_gate_mix handles (for supports_op)
bool ggml_sycl_gate_mix_supported(const ggml_tensor * op);

#endif // GGML_SYCL_GATE_MIX_HPP
