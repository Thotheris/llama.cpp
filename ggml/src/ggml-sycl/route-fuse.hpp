#ifndef GGML_SYCL_ROUTE_FUSE_HPP
#define GGML_SYCL_ROUTE_FUSE_HPP

#include "common.hpp"

// The biased sigmoid MoE routing chain of one to four tokens starting at node i (SIGMOID ... SCALE) in one launch;
// returns the number of following nodes it consumed (0: not applied). GGML_SYCL_ROUTE_FUSE=0 disables it.
int ggml_sycl_route_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

#endif // GGML_SYCL_ROUTE_FUSE_HPP
