//
// MIT license
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MMVQ_ID_MULTI_HPP
#define GGML_SYCL_MMVQ_ID_MULTI_HPP

#include "common.hpp"

// Multi-column MoE GEMV for MUL_MAT_ID at a few tokens (2..~16): each touched expert's weights are read once
// for all the tokens routed to it, the way mmvq-id reads them once for one token. A one-work-group device
// kernel first groups the routes by expert (in route order, so the result is deterministic and the graph
// stays recordable), then one kernel runs over (active expert slot, row block).

// the largest expert count and route count (n_ids * n_tokens) the schedule kernel takes
#define GGML_SYCL_MMID_MC_MAX_EXPERTS 1024
#define GGML_SYCL_MMID_MC_MAX_ROUTES  2048

// true when src0_type has a kernel in the layout it is in (reordered or standard)
bool ggml_sycl_mmid_mc_supports_type(enum ggml_type src0_type, bool reordered);

// int32 elements of the device schedule buffer for n_as experts and n_routes routes
size_t ggml_sycl_mmid_mc_sched_ints(int n_as, int n_routes);

// vx_base:      stacked expert weights, expert_stride bytes apart (reordered: each slice a self-contained SoA)
// vy:           src1 quantized to q8_1 (quantize_and_reorder_q8_1_soa when reordered), one row per
//               (token, slot) or one row per token when ne11 == 1, q8_row_bytes apart, row = token*ne11 + slot
// ids:          int32 [n_ids, n_tokens], ids_nb1 bytes between tokens
// dst:          f32, the row of (token, slot) at dst + slot*nb1 + token*nb2, nrows floats; skipped slots (-1) are zeroed
// sched:        device scratch of ggml_sycl_mmid_mc_sched_ints(n_as, n_ids*n_tokens) int32; sched_ready: it already
//               holds the schedule of these ids (an earlier call with the same ids built it)
void ggml_sycl_mmid_mc(enum ggml_type src0_type, bool reordered, const void * vx_base, size_t expert_stride,
                       int ncols, int nrows, const void * vy, size_t q8_row_bytes, int ne11, const int32_t * ids,
                       size_t ids_nb1, int n_ids, int n_tokens, int n_as, float * dst, size_t nb1, size_t nb2,
                       int32_t * sched, bool sched_ready, dpct::queue_ptr stream);

// The XMX (DPAS) variant (reordered Q4_K/Q5_K/Q6_K with K a multiple of QK_K, reordered Q8_0 or standard Q5_0 with
// K a multiple of 128): the routes' src1 rows
// (f32, src1 + token*nb12 + slot*nb11, or token rows only when ne11 == 1) are quantized into q8_packed in schedule
// order, q8_row_bytes (a multiple of 64) apart, n_ids*n_tokens rows; q8_packed must be 64-byte aligned.
bool ggml_sycl_mmid_mc_xmx_supported(enum ggml_type src0_type, bool reordered, int ncols, size_t expert_stride,
                                     const void * vx_base);
void ggml_sycl_mmid_mc_xmx(enum ggml_type src0_type, const void * vx_base, size_t expert_stride, int ncols, int nrows,
                           const char * src1, size_t nb11, size_t nb12, int ne11, char * q8_packed,
                           size_t q8_row_bytes, const int32_t * ids, size_t ids_nb1, int n_ids, int n_tokens, int n_as,
                           float * dst, size_t nb1, size_t nb2, int32_t * sched, bool sched_ready,
                           dpct::queue_ptr stream);

#endif // GGML_SYCL_MMVQ_ID_MULTI_HPP
