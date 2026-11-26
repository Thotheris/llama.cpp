//
// MIT license
// SPDX-License-Identifier: MIT
//

#include "mmv-f32.hpp"

namespace {

constexpr int MMV_F32_SG = 16;

// one work-group of wg work-items per (output row, batch); element strides, all multiples of 4
template <int NC>
void mmv_f32_kernel(const float * x, const float * y, float * dst, const int K4, const int ncols, const int r2,
                    const size_t sx1, const size_t sx2, const size_t sy1, const size_t sy2, const size_t sd1,
                    const size_t sd2, float * slm, const sycl::nd_item<2> & it) {
    const int row = (int) it.get_group(1);
    const int b   = (int) it.get_group(0);
    const int lid = (int) it.get_local_id(1);
    const int wg  = (int) it.get_local_range(1);

    const sycl::float4 * xr = (const sycl::float4 *) (x + (size_t) (b / r2) * sx2 + (size_t) row * sx1);
    const float *        yb = y + (size_t) b * sy2;

    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = 0.0f;
    }
    for (int k = lid; k < K4; k += wg) {
        const sycl::float4 w = xr[k];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (j < ncols) {
                const sycl::float4 v = ((const sycl::float4 *) (yb + (size_t) j * sy1))[k];
                acc[j] += w.x() * v.x() + w.y() * v.y() + w.z() * v.z() + w.w() * v.w();
            }
        }
    }

    const auto sg  = it.get_sub_group();
    const int  sgi = lid / MMV_F32_SG;
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = sycl::reduce_over_group(sg, acc[j], sycl::plus<float>());
    }
    if (sg.get_local_linear_id() == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            slm[sgi * NC + j] = acc[j];
        }
    }
    sycl::group_barrier(it.get_group());
    if (lid < ncols) {
        const int nsg = wg / MMV_F32_SG;
        float     s   = 0.0f;
        for (int i = 0; i < nsg; ++i) {
            s += slm[i * NC + lid];
        }
        dst[(size_t) b * sd2 + (size_t) lid * sd1 + row] = s;
    }
}

template <int NC>
void mmv_f32_launch(const float * x, const float * y, float * dst, int K, int M, int nb, int ncols, int r2, size_t sx1,
                    size_t sx2, size_t sy1, size_t sy2, size_t sd1, size_t sd2, dpct::queue_ptr stream) {
    // enough sub-groups in flight to cover the memory latency with few rows
    const int rows = M * nb;
    const int wg   = rows <= 512 ? 256 : (rows <= 2048 ? 128 : 64);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) (wg / MMV_F32_SG) * NC), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) nb, (size_t) M * wg), sycl::range<2>(1, wg)),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(MMV_F32_SG)]] {
                             mmv_f32_kernel<NC>(x, y, dst, K / 4, ncols, r2, sx1, sx2, sy1, sy2, sd1, sd2,
                                                slm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
                         });
    });
}

}  // namespace

bool ggml_sycl_mmv_f32_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[1] < 1 || src1->ne[1] > GGML_SYCL_MMV_F32_MAX_COLS || src0->ne[3] != 1 || src1->ne[3] != 1 ||
        src0->ne[0] != src1->ne[0] || src0->ne[0] % 4 != 0 || src1->ne[2] % src0->ne[2] != 0 ||
        dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2]) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    // float4 loads: 16 byte aligned bases and row strides
    return (uintptr_t) src0->data % 16 == 0 && (uintptr_t) src1->data % 16 == 0 && src0->nb[1] % 16 == 0 &&
           src0->nb[2] % 16 == 0 && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && dst->nb[1] % 4 == 0 &&
           dst->nb[2] % 4 == 0;
}

void ggml_sycl_mmv_f32(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       ggml_tensor * dst) {
    const int       K      = (int) src0->ne[0];
    const int       M      = (int) src0->ne[1];
    const int       nb     = (int) src1->ne[2];
    const int       ncols  = (int) src1->ne[1];
    const int       r2     = (int) (src1->ne[2] / src0->ne[2]);
    const size_t    sx1    = src0->nb[1] / sizeof(float);
    const size_t    sx2    = src0->nb[2] / sizeof(float);
    const size_t    sy1    = src1->nb[1] / sizeof(float);
    const size_t    sy2    = src1->nb[2] / sizeof(float);
    const size_t    sd1    = dst->nb[1] / sizeof(float);
    const size_t    sd2    = dst->nb[2] / sizeof(float);
    const float *   x      = (const float *) src0->data;
    const float *   y      = (const float *) src1->data;
    float *         d      = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();
    if (ncols <= 1) {
        mmv_f32_launch<1>(x, y, d, K, M, nb, ncols, r2, sx1, sx2, sy1, sy2, sd1, sd2, stream);
    } else if (ncols <= 2) {
        mmv_f32_launch<2>(x, y, d, K, M, nb, ncols, r2, sx1, sx2, sy1, sy2, sd1, sd2, stream);
    } else if (ncols <= 4) {
        mmv_f32_launch<4>(x, y, d, K, M, nb, ncols, r2, sx1, sx2, sy1, sy2, sd1, sd2, stream);
    } else if (ncols <= 8) {
        mmv_f32_launch<8>(x, y, d, K, M, nb, ncols, r2, sx1, sx2, sy1, sy2, sd1, sd2, stream);
    } else {
        mmv_f32_launch<16>(x, y, d, K, M, nb, ncols, r2, sx1, sx2, sy1, sy2, sd1, sd2, stream);
    }
}
