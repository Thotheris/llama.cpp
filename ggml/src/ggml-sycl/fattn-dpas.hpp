#ifndef GGML_SYCL_FATTN_DPAS_HPP
#define GGML_SYCL_FATTN_DPAS_HPP

#include "common.hpp"

// q4_0 / q8_0 K/V flash attention for decode and small batches on Xe2 XMX (s8 x s4 / s8 x s8 DPAS)
bool ggml_sycl_flash_attn_ext_dpas_supported(int device, const ggml_tensor * dst);
void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fa reads the REL_BIAS node rb and takes a kernel that can compute the bias itself (see rel-bias.hpp)
bool ggml_sycl_flash_attn_ext_dpas_rel_inkernel_ok(const ggml_tensor * fa, const ggml_tensor * rb);
// the same for the decode kernel, rb (pos set) being the per-head mask of fa
bool ggml_sycl_flash_attn_ext_dpas_dec_rel_ok(const ggml_tensor * fa, const ggml_tensor * rb);

#endif // GGML_SYCL_FATTN_DPAS_HPP
