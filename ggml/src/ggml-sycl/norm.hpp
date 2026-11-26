//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_NORM_HPP
#define GGML_SYCL_NORM_HPP

#include "common.hpp"

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

// GATE_MIX -> RMS_NORM -> MUL(weight) as one kernel (the GATE_MIX row computed in registers, never written);
// false (nothing queued) when the shapes or alignment do not qualify. f16_for_dense: a dense oneDNN matmul reads
// the result, so the kernel also writes its f16 copy into the activation cache (dense-dnn.hpp).
// gates_h: the gains|forgets of the mix as dense f16 rows [2*ne0] with the tanh applied (the mix must apply it),
// read instead of src[2]; check_only: whether the call would qualify, nothing queued.
// pin_until >= 0 (GGML_SYCL_DROP_F32): only the f16 copy is written, pinned in the cache until node pin_until, the
// last reader (the f32 result too when the cache has no free slot)
bool ggml_sycl_op_gate_mix_rms_norm_mul(ggml_backend_sycl_context& ctx, ggml_tensor* gate_mix, ggml_tensor* rms,
                                        ggml_tensor* mul, bool f16_for_dense, const sycl::half * gates_h = nullptr,
                                        bool check_only = false, int pin_until = -1);

// RMS_NORM -> MUL(weight) that also writes the f16 copy of the result for the dense oneDNN matmuls (as above);
// false (nothing queued) when it does not qualify, ggml_sycl_op_rms_norm_fused() then runs the pair
bool ggml_sycl_op_rms_norm_mul_f16(ggml_backend_sycl_context& ctx, ggml_tensor* rms, ggml_tensor* mul, int pin_until = -1);

void ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

void ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor);

bool ggml_sycl_op_add_rms_norm_fused(ggml_backend_sycl_context & ctx,
                                     ggml_tensor *               add,
                                     ggml_tensor *               rms,
                                     ggml_tensor *               mul);

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#define GGML_SYCL_L2_BATCH_MAX 8
void ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, int count);

#endif // GGML_SYCL_NORM_HPP
