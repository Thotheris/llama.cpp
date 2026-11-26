#include "outprod.hpp"
#include "convert.hpp"

// The contracted dimension (ne01 == ne11) is short and every plane is small, as in the MoE combine
// out[i, 0, t] = sum_k experts[i, k, t] * weights[0, k, t] (k = the experts used, a handful): one
// launch over all planes instead of one library GEMM per plane. Bounded so the per-element loop
// stays short; a wide contraction or a large plane is left to oneMKL.
static constexpr int64_t OUT_PROD_DIRECT_MAX_K  = 64;
static constexpr int64_t OUT_PROD_DIRECT_MAX_NE1 = 16;

bool ggml_sycl_out_prod_direct(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[1] <= OUT_PROD_DIRECT_MAX_K && dst->ne[1] <= OUT_PROD_DIRECT_MAX_NE1 &&
           src0->nb[0] == sizeof(float) && src1->nb[0] % sizeof(float) == 0 && src1->nb[1] % sizeof(float) == 0 &&
           src1->nb[2] % sizeof(float) == 0 && src1->nb[3] % sizeof(float) == 0 &&
           src0->nb[1] % sizeof(float) == 0 && src0->nb[2] % sizeof(float) == 0 && src0->nb[3] % sizeof(float) == 0 &&
           ggml_is_contiguous(dst) && ggml_nelements(dst) <= INT32_MAX;
}

// dst[i0, i1, i2, i3] = sum_k src0[i0, k, i2 / r2, i3 / r3] * src1[i1, k, i2, i3], summed in f32 in
// ascending k: deterministic, and no host wait, so it can be recorded in a SYCL graph.
// A work-item owns one i0 of one (i1, i2, i3) plane; consecutive work-items read consecutive i0 of
// src0 and every work-item of a work-group reads the same src1 values.
static void out_prod_direct(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const float * x = (const float *) src0->data;
    const float * y = (const float *) src1->data;
    float *       d = (float *) dst->data;

    const int64_t r2 = ne2 / ne02;
    const int64_t r3 = ne3 / ne03;
    // strides in floats
    const int64_t s01 = nb01 / 4, s02 = nb02 / 4, s03 = nb03 / 4;
    const int64_t s10 = nb10 / 4, s11 = nb11 / 4, s12 = nb12 / 4, s13 = nb13 / 4;
    const int     K   = (int) ne01;

    constexpr int WG     = 256;
    const int64_t planes = ne1 * ne2 * ne3;
    const int64_t groups = (ne0 + WG - 1) / WG;
    ctx.stream()->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(planes, groups * WG), sycl::range<2>(1, WG)), [=](sycl::nd_item<2> it) {
            const int64_t i0 = it.get_global_id(1);
            if (i0 >= ne0) {
                return;
            }
            const int64_t p  = it.get_group(0);
            const int64_t i1 = p % ne1;
            const int64_t i2 = (p / ne1) % ne2;
            const int64_t i3 = p / (ne1 * ne2);
            const float * xr = x + i0 + (i2 / r2) * s02 + (i3 / r3) * s03;
            const float * yr = y + i1 * s10 + i2 * s12 + i3 * s13;
            float         sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += xr[k * s01] * yr[k * s11];
            }
            d[i0 + ne0 * p] = sum;
        });
}

// A work-item sums 8 consecutive values of one token over its slots: one 16-byte load a slot, two float4 stores.
// Graph-recordable, and the order of the sum is fixed (slot 0 first), so the result is deterministic.
// NS > 0 fixes the slot count at compile time so all the loads are issued before the first add.
template <int NS>
static void moe_combine_half_launch(dpct::queue_ptr q, float * d, const sycl::half * h, const int64_t n,
                                    const int64_t c8, const int64_t ne0, const int n_slots_rt) {
    constexpr int WG = 256;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((n + WG - 1) / WG * WG)), sycl::range<1>(WG)),
                    [=](sycl::nd_item<1> it) {
                        const int64_t i = it.get_global_id(0);
                        if (i >= n) {
                            return;
                        }
                        const int          ns = NS > 0 ? NS : n_slots_rt;
                        const int64_t      t  = i / c8;
                        const int64_t      c  = i - t * c8;
                        const sycl::half * r  = h + (t * ns * c8 + c) * 8;
                        sycl::float4       s0(0.0f), s1(0.0f);
                        if constexpr (NS > 0) {
                            sycl::vec<sycl::half, 8> v[NS];
#pragma unroll
                            for (int k = 0; k < NS; ++k) {
                                v[k] = *(const sycl::vec<sycl::half, 8> *) (r + (size_t) k * ne0);
                            }
#pragma unroll
                            for (int k = 0; k < NS; ++k) {
                                s0 += sycl::float4(v[k][0], v[k][1], v[k][2], v[k][3]);
                                s1 += sycl::float4(v[k][4], v[k][5], v[k][6], v[k][7]);
                            }
                        } else {
                            for (int k = 0; k < ns; ++k) {
                                const sycl::vec<sycl::half, 8> v =
                                    *(const sycl::vec<sycl::half, 8> *) (r + (size_t) k * ne0);
                                s0 += sycl::float4(v[0], v[1], v[2], v[3]);
                                s1 += sycl::float4(v[4], v[5], v[6], v[7]);
                            }
                        }
                        *(sycl::float4 *) (d + i * 8)     = s0;
                        *(sycl::float4 *) (d + i * 8 + 4) = s1;
                    });
}

void ggml_sycl_moe_combine_half(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const sycl::half * h,
                                const int n_slots) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const int64_t ne0 = dst->ne[0];
    const int64_t n   = ggml_nelements(dst) / 8;
    GGML_ASSERT(ne0 % 8 == 0 && ggml_is_contiguous(dst));
    float * d = (float *) dst->data;
    if (n_slots == 8) {
        moe_combine_half_launch<8>(ctx.stream(), d, h, n, ne0 / 8, ne0, n_slots);
    } else {
        moe_combine_half_launch<0>(ctx.stream(), d, h, n, ne0 / 8, ne0, n_slots);
    }
}

void ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];

    if (ggml_sycl_out_prod_direct(dst)) {
        GGML_ASSERT(src0->ne[1] == src1->ne[1] && dst->ne[0] == src0->ne[0] && dst->ne[1] == src1->ne[0] &&
                    dst->ne[2] == src1->ne[2] && dst->ne[3] == src1->ne[3] &&
                    dst->ne[2] % src0->ne[2] == 0 && dst->ne[3] % src0->ne[3] == 0);
        out_prod_direct(ctx, dst);
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    // Get SYCL queue
    dpct::queue_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // Get data pointers
    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float * dst_d = (float *) dst->data;

    ggml_sycl_pool_alloc<float> src0_as_f32(ctx.pool());
    int64_t src0_nb02 = nb02;
    int64_t src0_nb03 = nb03;
    if (src0->type == GGML_TYPE_Q1_0) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                             " : converting src0 Q1_0 to fp32");
        src0_d = src0_as_f32.alloc(ne00 * ne01 * ne02 * ne03);
        const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
        GGML_ASSERT(to_fp32_sycl != nullptr);
        to_fp32_sycl(src0->data, const_cast<float *>(src0_d), ne00 * ne01 * ne02 * ne03, stream);

        // Dequantized src0 buffer is contiguous fp32 [ne00, ne01, ne02, ne03].
        src0_nb02 = ne00 * ne01 * (int64_t) sizeof(float);
        src0_nb03 = ne00 * ne01 * ne02 * (int64_t) sizeof(float);
    }

    // GEMM parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Handle transposition of src1
    const bool src1_T = ggml_is_transposed(src1);
    const oneapi::mkl::transpose src1_op = src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);

    const int64_t r2 = ne2 / ne02;
    const int64_t r3 = ne3 / ne03;

    try {
        // OUT_PROD applies independently to each (i2, i3) destination plane.
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const int64_t i03 = i3 / r3;
                const int64_t i02 = i2 / r2;

                const float * src0_plane = (const float *) ((const char *) src0_d + i02 * src0_nb02 + i03 * src0_nb03);
                const float * src1_plane = (const float *) ((const char *) src1_d + i2 * nb12 + i3 * nb13);
                float * dst_plane = (float *) ((char *) dst_d + i2 * nb2 + i3 * nb3);

                // Perform matrix multiplication using oneMKL GEMM
                oneapi::mkl::blas::column_major::gemm(*stream, oneapi::mkl::transpose::nontrans, src1_op,
                                                      ne0, ne1, ne01, alpha, src0_plane, ne00,
                                                      src1_plane, ldb, beta, dst_plane, ne0);
            }
        }
    } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << std::endl;
        GGML_ASSERT(false);
    }
}
