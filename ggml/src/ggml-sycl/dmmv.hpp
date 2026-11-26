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

#ifndef GGML_SYCL_DMMV_HPP
#define GGML_SYCL_DMMV_HPP

#include "common.hpp"


void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
bool ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    enum ggml_type src0_type, enum ggml_glu_op glu_op,
    const void * vx, const void * vgate, const float * y,
    float * dst, int ncols, int nrows, dpct::queue_ptr stream);

// One-column products of up to GGML_SYCL_DMMV_MULTI_MAX reordered Q4_K / Q6_K weights that share the activation y
// and the row length (a multiple of 768), in one launch. A work-group owns two rows of one weight; each of its
// ncols/768 threads keeps its 768-float slice of y in registers and reads the three contiguous blocks of each row.
constexpr int GGML_SYCL_DMMV_MULTI_MAX = 4;

struct ggml_sycl_dmmv_multi_seg {
    const void *  w;      // reordered weight
    float *       dst;    // [nrows]
    int           nrows;
    int           type;   // GGML_TYPE_Q4_K or GGML_TYPE_Q6_K
};

struct ggml_sycl_dmmv_multi_args {
    ggml_sycl_dmmv_multi_seg seg[GGML_SYCL_DMMV_MULTI_MAX];
    int                      nseg;
    // columns (0 or 1: one): column c reads y + c * y_stride and writes dst + c * dst_stride (floats) of each
    // segment, through the work-groups of the one-column launch
    int                      ncol;
    size_t                   y_stride;
    size_t                   dst_stride;
};

// A request for the next one-column reordered ESIMD DMMV (Q4_K, Q5_K, Q6_K, row pairs) whose activation and result
// are y0 and dst0: run it over ncol columns at y0 + c * y_stride and dst0 + c * dst_stride (floats) in one launch,
// each column as the one-column launch computes it, and set used. Set by the row-exact verify (ggml-sycl.cpp) around
// a one-column ggml_sycl_mul_mat, so the column takes the dispatch of the one-token decode.
struct ggml_sycl_dmmv_nc_req {
    const float * y0;
    float *       dst0;
    int           ncol;
    size_t        y_stride;
    size_t        dst_stride;
    bool          used;
};
extern thread_local ggml_sycl_dmmv_nc_req g_ggml_sycl_dmmv_nc;

// every segment of one launch has the same type (the kernel is specialised on it)
// false if the shape or a type is not covered (nothing is launched then)
bool ggml_sycl_dmmv_multi(const ggml_sycl_dmmv_multi_args & args, const float * y, int ncols, dpct::queue_ptr stream);
bool ggml_sycl_dmmv_multi_supported(int type, int64_t ncols);

#endif // GGML_SYCL_DMMV_HPP
