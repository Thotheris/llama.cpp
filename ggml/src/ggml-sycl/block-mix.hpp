#pragma once

#include "common.hpp"

// GGML_OP_BLOCK_MIX (ggml_block_mix and ggml_block_mix_norm), see ggml.h
bool ggml_sycl_block_mix_supported(const ggml_tensor * op);
// f16_for_dense: n_query flags, query j's plane-0 rows [n, n_tokens] (row stride nb[2]) are the B of a dense oneDNN
// matmul, which then reads an f16 copy the kernel writes beside them (nullptr: none)
// pin_until (n_query entries, -1: none): GGML_SYCL_DROP_F32, query j's plane-0 f32 rows are not written when its f16
// copy can be pinned in the cache until node pin_until[j], their last reader
void ggml_sycl_block_mix(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const bool * f16_for_dense = nullptr,
                         const int * pin_until = nullptr);
