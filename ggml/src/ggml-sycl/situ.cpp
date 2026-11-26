// GGML_GLU_OP_SITU for the SYCL backend, see ggml_situ in ggml.h and ggml_compute_forward_situ_impl in
// ggml-cpu/ops.cpp (the reference): per element of a row
//   y = (tanh(x/beta)*beta) * sigmoid(x) * (lb > 0 ? tanh(g/lb)*lb : g)
// x and g are rows of two tensors (the split form, e.g. the strided views of a fused gate_up) or the two halves of
// one row (the single-tensor form, swapped picks which half is x). One work-item per output element, a 2-D range
// (rows, columns) so no 64-bit division per element.

#include "situ.hpp"

#define SYCL_SITU_BLOCK_SIZE 256

template <typename T>
static void situ_sycl(const T * x, const T * g, T * dst, const int64_t nrows, const int64_t nc, const int64_t sx,
                      const int64_t sg, const int64_t sd, const float beta, const float lb, dpct::queue_ptr stream) {
    const float inv_beta = 1.0f / beta;
    const float inv_lb   = lb > 0.0f ? 1.0f / lb : 0.0f;

    const int64_t           ncb = (nc + SYCL_SITU_BLOCK_SIZE - 1) / SYCL_SITU_BLOCK_SIZE;
    const sycl::nd_range<2> range(sycl::range<2>(nrows, ncb * SYCL_SITU_BLOCK_SIZE),
                                  sycl::range<2>(1, SYCL_SITU_BLOCK_SIZE));

    stream->parallel_for(range, [=](sycl::nd_item<2> it) {
        const int64_t row = it.get_global_id(0);
        const int64_t col = it.get_global_id(1);
        if (col >= nc) {
            return;
        }
        const float xv  = (float) x[row * sx + col];
        float       gv  = (float) g[row * sg + col];
        const float act = (sycl::tanh(xv * inv_beta) * beta) * (1.0f / (1.0f + sycl::exp(-xv)));
        if (lb > 0.0f) {
            gv = sycl::tanh(gv * inv_lb) * lb;
        }
        dst[row * sd + col] = (T) (act * gv);
    });
}

bool ggml_sycl_situ_supported(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) {
        return false;
    }
    if (src0->type != op->type || (src1 && src1->type != op->type)) {
        return false;
    }
    // rows of every tensor are addressed by row index * nb[1]
    if (!ggml_is_contiguous_1(src0) || !ggml_is_contiguous_1(op) || (src1 && !ggml_is_contiguous_1(src1))) {
        return false;
    }
    const size_t ts = ggml_type_size(op->type);
    if (src0->nb[1] % ts != 0 || op->nb[1] % ts != 0 || (src1 && src1->nb[1] % ts != 0)) {
        return false;
    }
    if (src1 && !ggml_are_same_shape(src0, src1)) {
        return false;
    }
    return true;
}

void ggml_sycl_situ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_sycl_situ_supported(dst));

    const int64_t nc    = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nrows = ggml_nrows(src0);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nrows);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float   beta    = ggml_get_op_params_f32(dst, 2);
    const float   lb      = ggml_get_op_params_f32(dst, 3);

    const size_t  ts = ggml_type_size(dst->type);
    const int64_t sx = src0->nb[1] / ts;
    const int64_t sg = src1 ? src1->nb[1] / ts : sx;
    const int64_t sd = dst->nb[1] / ts;

    dpct::queue_ptr stream = ctx.stream();

    if (nrows == 0 || nc == 0) {
        return;
    }

    auto run = [&](auto * type_tag) {
        using T     = std::remove_pointer_t<decltype(type_tag)>;
        const T * x = (const T *) src0->data;
        const T * g = (const T *) (src1 ? src1->data : src0->data);
        if (!src1) {
            x += swapped ? nc : 0;
            g += swapped ? 0 : nc;
        }
        situ_sycl<T>(x, g, (T *) dst->data, nrows, nc, sx, sg, sd, beta, lb, stream);
    };

    if (dst->type == GGML_TYPE_F16) {
        run((sycl::half *) nullptr);
    } else {
        run((float *) nullptr);
    }
}
