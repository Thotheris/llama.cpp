//
// MIT license
// SPDX-License-Identifier: MIT
//

// Dense (non-expert) MUL_MAT at prefill as a oneDNN matmul.
//
// A MUL_MAT whose src0 is a model weight (any type that dequantizes to f32: the k-quants, q8_0, F32, F16, BF16)
// and whose src1 has at least GGML_SYCL_DENSE_DNN_MIN_COLS columns runs as one oneDNN f16 (or bf16) matmul with
// f32 accumulation and f32 output, against a persistent f16 copy of the weight (plain {K, N} layout by default;
// GGML_SYCL_DENSE_DNN_PLAIN=0 lets oneDNN pick a blocked layout, which measured no faster on the B70).
// The copy is made at the first such call (dequantized once, honouring the reorder layout of the whole weight, so
// a row view of a reordered stacked weight is read correctly). src1 is converted once per input: the weights that
// read the same activation (q, gate, r and sink; k and its conv projection; ...) share one converted copy, which
// stays valid until a later node of the graph writes its bytes.
//
// oneDNN 3.11 cannot be recorded into a SYCL graph, so a graph with such a node runs eagerly;
// ggml_sycl_dense_dnn_eligible() is what the graph compatibility check asks. Decode (one column, or any batch
// below the threshold) keeps the vector and XMX kernels and stays recordable.

#pragma once

#include "common.hpp"

#if GGML_SYCL_DNNL

extern int g_ggml_sycl_dense_dnn;          // GGML_SYCL_DENSE_DNN: 0 off, 1 on
extern int g_ggml_sycl_dense_dnn_min_cols; // GGML_SYCL_DENSE_DNN_MIN_COLS: fewest src1 columns taken
extern int g_ggml_sycl_dense_dnn_bf16;     // GGML_SYCL_DENSE_DNN_BF16: 1 runs bf16 instead of f16
extern int g_ggml_sycl_dense_dnn_max_mb;   // GGML_SYCL_DENSE_DNN_MAX_MB: VRAM budget of the weight copies
extern int g_ggml_sycl_dense_dnn_max_weight_mb; // GGML_SYCL_DENSE_DNN_MAX_WEIGHT_MB: largest weight copied
extern int g_ggml_sycl_dense_dnn_act_cache;     // GGML_SYCL_DENSE_DNN_ACT_CACHE: 0 converts src1 on every call
extern int g_ggml_sycl_dense_dnn_plain;    // GGML_SYCL_DENSE_DNN_PLAIN: 0 lets oneDNN block the weight copy

void ggml_sycl_dense_dnn_init_env();

// true when ggml_sycl_mul_mat() hands dst to the dense oneDNN path; cheap, no allocation
bool ggml_sycl_dense_dnn_eligible(int device, const ggml_tensor * dst);

// runs dst = src0 * src1; false (nothing queued) when the weight copy cannot be made, the caller falls back
bool ggml_sycl_dense_dnn_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                 ggml_tensor * dst);

// CONCAT(a, b, dim 0) -> RESHAPE -> MUL_MAT(F32 weight) at node i, the concat used only there: two f32 matmuls over
// the halves of the weight rows, accumulating into the MUL_MAT result, so the concat is never written. Returns the
// nodes consumed after i (2), or 0 when the pattern does not apply (nothing queued).
// GGML_SYCL_CTRL_F16 (default 1, 0: the f32 matmuls above): one f16 matmul per half instead, the halves read from
// the activation cache (written by their producer, or converted once and shared by the matmuls of the same rows)
int ggml_sycl_dense_dnn_concat_mul_mat(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i);
// node i is such a concat and the f16 path reads its halves from the activation cache
bool ggml_sycl_dense_dnn_concat_reads_f16(int device, const ggml_cgraph * cgraph, int i);
// and the f16 copies of the weight halves exist, so the f16 path is sure to run
bool ggml_sycl_dense_dnn_concat_ready(int device, const ggml_cgraph * cgraph, int i);

// the converted src1 copies are only valid inside one graph, and only until a node writes their source bytes
void ggml_sycl_dense_dnn_graph_begin(ggml_backend_sycl_context & ctx);
void ggml_sycl_dense_dnn_node_writes(ggml_backend_sycl_context & ctx, const ggml_tensor * node);

// ggml_sycl_dense_dnn_eligible() and the dense path converts src1 to f16/bf16 (not an f32-direct weight); the
// caller checks the split buffer and the Hadamard hint as ggml_sycl_mul_mat() does
bool ggml_sycl_dense_dnn_converts_src1(int device, const ggml_tensor * dst);

// for a producer that writes the f16 copy of t itself: an activation slot registered as the copy of t's bytes (as
// dense_dnn_get_act() would make it), or nullptr (bf16 mode, cache off, not 2D rows). The caller must write all of
// it before any later node reads it, and must call it only after the node_writes() checks of the nodes that write t.
// pin_until >= 0: the producer does not write the f32 bytes of t (GGML_SYCL_DROP_F32), the slot is never evicted while
// the graph has not passed node pin_until (its last reader, see ggml_sycl_dense_dnn_set_node), and nullptr is returned
// when every slot is pinned (the producer then writes the f32 result instead)
sycl::half * ggml_sycl_dense_dnn_act_reserve(ggml_backend_sycl_context & ctx, const ggml_tensor * t, int pin_until = -1);

// GGML_SYCL_DROP_F32 (default 1, 0: f32 and f16 both written): producers whose f32 result only the dense oneDNN matmuls read
// write the f16 copy alone
bool ggml_sycl_dense_dnn_drop_f32();
// the index of the graph node being computed, for the pins above
void ggml_sycl_dense_dnn_set_node(ggml_backend_sycl_context & ctx, int i);
// the f16 copy of the weight src0 exists (made by an earlier call), so a dense call of it cannot fail on the copy
bool ggml_sycl_dense_dnn_weight_ready(int device, const ggml_tensor * src0);
// the valid f16 copy of the M rows of K floats at src (row stride ld floats), as M x K dense, or nullptr
const sycl::half * ggml_sycl_dense_dnn_act_lookup(ggml_backend_sycl_context & ctx, const void * src, int64_t K, int64_t M,
                                                  int64_t ld);

// the MUL_MAT dst (dense-eligible, converting src1) computed as dense f16 rows into out (ne1 x ne0) instead of
// dst->data, with tanh applied on the f32 accumulator when asked; false (nothing queued) when it cannot run
bool ggml_sycl_dense_dnn_mul_mat_f16(ggml_backend_sycl_context & ctx, const ggml_tensor * dst, sycl::half * out, bool tanh);

// drops the weight copies made from the weights of this buffer (the buffer is being freed)
void ggml_sycl_dense_dnn_drop_buffer(ggml_backend_buffer_t buffer);

// per-context state (converted activations, primitive caches), owned by ggml_backend_sycl_context
struct ggml_sycl_dense_dnn_ctx;
void ggml_sycl_dense_dnn_ctx_free(ggml_sycl_dense_dnn_ctx * state);

#endif // GGML_SYCL_DNNL
