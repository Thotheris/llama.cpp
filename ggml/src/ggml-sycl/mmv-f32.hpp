//
// MIT license
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MMV_F32_HPP
#define GGML_SYCL_MMV_F32_HPP

#include "common.hpp"

// f32 weights times a few f32 columns (decode of small router / control projections): one work-group per output
// row splits K, the columns share each weight load. The sum order is fixed, so the result is deterministic.

// the most columns (src1->ne[1]) the kernel takes
#define GGML_SYCL_MMV_F32_MAX_COLS 16

// dst = src0^T * src1 for f32 src0 [K, M, ne02], f32 src1 [K, N, ne12] (ne12 a multiple of ne02), f32 dst [M, N, ne12],
// rows contiguous, N <= GGML_SYCL_MMV_F32_MAX_COLS, no 4th dimension; the caller rules out split buffers
bool ggml_sycl_mmv_f32_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_sycl_mmv_f32(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       ggml_tensor * dst);

#endif  // GGML_SYCL_MMV_F32_HPP
