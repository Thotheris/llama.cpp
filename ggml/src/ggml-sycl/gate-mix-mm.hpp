#pragma once

#include "common.hpp"

// MUL_MAT(gain|forget weight, z) -> GATE_MIX(x, m, ., tanh) -> RMS_NORM -> MUL(w) at prefill as one ESIMD kernel
// (GGML_SYCL_GATE_MIX_FUSED, default 1; 0: the unfused path, a oneDNN matmul writing tanh(gains|forgets) as f16 and the
// fused mix + norm reading them).
//
// The loop controllers' gain|forget projection has K = 64 (the gate rank) and N = 2*n_embd outputs: one dpas k
// chain of 4 per 16 outputs, so the matmul is nearly free in FLOPs, and the cost of the old path was the 2*n_embd
// f16 gates per token written by oneDNN and read back by the norm. Here a work-group takes 8 tokens and all the
// n_embd channels: each thread owns 9 tiles of 16 channels (256 registers; 3 tiles and 128 registers when n_embd is
// not a multiple of 144), computes the gain and forget tiles with dpas (A = the tokens' z as f16, from the dense
// activation cache when a producer wrote it, else converted from f32; B = a persistent VNNI f16 copy of the
// weight), applies tanh (rounded to f16, as the old path stored them), the blend and the squares in registers, then
// the row norms are summed across the threads in a fixed order and the result is written once: f32 (unless
// GGML_SYCL_DROP_F32 dropped it) and the f16 copy the dense oneDNN matmuls read. x, m and the results move as 2D
// block messages of 8 rows x 16 channels when the rows are 64-byte aligned (GGML_SYCL_GATE_MIX_FUSED_2D=0: per row).
//
// At node i (the MUL_MAT), in this order:
// - match(): the pattern (gate rank 64, tanh gates, shapes, strides and 16-byte alignment) and the
//   GGML_SYCL_GATE_MIX_FUSED switch, nothing that depends on where the allocator placed the tensors (a result over z,
//   x or m is handled by run()); nothing queued;
// - ready(): the packed weight exists. prepare() makes the copies of every matching weight of the graph at the start of
//   ggml_backend_sycl_graph_compute(), before the graph runs or records, so the path is the same in the first graph,
//   in a recorded one and in every later one (false only when the copy could not be allocated, then for good);
// - z16(): the f16 copy of z in the dense activation cache, or nullptr (z is then read as f32); the caller copies it
//   out before any slot is reserved, so the kernel never reads a slot it also writes;
// - run(): the kernel; f16_for_dense / pin_until as for ggml_sycl_op_gate_mix_rms_norm_mul, called after the
//   node_writes() checks of nodes i..i+3. When the f32 result overlaps z, z is copied out first; when it overlaps x
//   or m, the kernel writes it to a scratch buffer and a copy moves it (the values are the same either way).
// The fused path is thus a function of the graph and the weights alone: a first graph, a recorded one and every later
// one, in any memory layout (base or speculative contexts), take the same path and give the same bits.
bool         ggml_sycl_gate_mix_mm_match(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i);
void         ggml_sycl_gate_mix_mm_prepare(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph);
bool         ggml_sycl_gate_mix_mm_ready(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i);
const void * ggml_sycl_gate_mix_mm_z16(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i);
void         ggml_sycl_gate_mix_mm_run(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i, const void * z16,
                                       bool f16_for_dense, int pin_until);

// drops the packed weight copies made from the weights of this buffer (the buffer is being freed)
void ggml_sycl_gate_mix_mm_drop_buffer(ggml_backend_buffer_t buffer);
