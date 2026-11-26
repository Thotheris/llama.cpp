#ifndef GGML_SYCL_L2_CLIP_HPP
#define GGML_SYCL_L2_CLIP_HPP

#include "common.hpp"

// GGML_OP_L2_CLIP: y = x*s * clamp(threshold / (||x*s||_2 + 1e-6), 0, 1) per row, s optional
void ggml_sycl_l2_clip(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// the forms ggml_sycl_l2_clip handles (for supports_op)
bool ggml_sycl_l2_clip_supported(const ggml_tensor * op);

#endif // GGML_SYCL_L2_CLIP_HPP
