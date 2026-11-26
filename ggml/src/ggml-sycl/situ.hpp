#ifndef GGML_SYCL_SITU_HPP
#define GGML_SYCL_SITU_HPP

#include "common.hpp"

// GGML_GLU_OP_SITU: situ(a, b) = beta*tanh(a/beta)*sigmoid(a) * lb*tanh(b/lb), b passes through when lb <= 0
void ggml_sycl_situ(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// the forms ggml_sycl_situ handles (for supports_op)
bool ggml_sycl_situ_supported(const ggml_tensor * op);

#endif // GGML_SYCL_SITU_HPP
