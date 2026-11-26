#pragma once

#include "common.hpp"

// GGML_OP_SSM_CONV_HIST (ggml_ssm_conv_hist and ggml_ssm_conv_hist_post), see ggml.h
bool ggml_sycl_ssm_conv_hist_supported(const ggml_tensor * op);
void ggml_sycl_ssm_conv_hist(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
