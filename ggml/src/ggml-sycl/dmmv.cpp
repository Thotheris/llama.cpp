#include "convert.hpp"
#include "dmmv.hpp"
#include "dequantize.hpp"
#include "presets.hpp"

#if defined(__INTEL_LLVM_COMPILER)
    #if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #include <sycl/ext/oneapi/bfloat16.hpp>
        #define GGML_SYCL_DMMV_HAS_BF16
    #endif
    #include <sycl/ext/intel/esimd.hpp>
    #include "esimd.hpp"
#endif

static void convert_f16(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const sycl::half *x = (const sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#ifdef GGML_SYCL_DMMV_HAS_BF16
static void convert_bf16(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const sycl::ext::oneapi::bfloat16 *x = (const sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#endif

static void convert_f32(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const float * x = (const float *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);

    const int iter_stride = 2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_SYCL_F16

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        if (col >= ncols) {
            continue;
        }
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#else
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

template <int qk, int qr, int qbytes, dequantize_kernel_t_reorder dequantize_kernel_reorder>
static void dequantize_mul_mat_vec_reorder(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    // qbytes = bytes of the quants region per block (not qk/qr for bit-packed types)
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);


    const int ncols_left = ncols % (qk*WARP_SIZE);
    const int ncols_align = ncols - ncols_left;
    const int iter_stride = 8*2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_SYCL_F16
    const char *d_ptr = (const char*)vx + (size_t) ncols * nrows / qk * qbytes;
    int i=0;
    for (i = 0; i < ncols_align; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((const void *)d_ptr, ib, (const void *)vx, ib * (qk/qr) +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    for (; i < ncols; i += iter_stride) {
        if (tid>=ncols_left/qk) continue;
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((const void *)d_ptr, ib, (const void *)vx, ib * (qk/qr) +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#else
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

static void convert_mul_mat_vec_f16_sycl(const void *vx, const dfloat *y,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, item_ct1);
            });
    }
}

#ifdef GGML_SYCL_DMMV_HAS_BF16
static void convert_mul_mat_vec_bf16_sycl(const void *vx, const dfloat *y,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, item_ct1);
            });
    }
}
#endif

static void dequantize_mul_mat_vec_q2_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int in = tid % step;                           // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int s_offset = 8*im;
            const int y_offset = 128*im + l0;

            const float   * y = yy + i * QK_K + y_offset;
            const uint8_t * q = x[i].qs + q_offset;

            const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
                sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                      + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                      + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                      + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                      + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                      + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                      + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                      +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
                sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                      + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

            }
            tmp += dall * sum1 - dmin * sum2;
        }

    }
#else
    const int tid = item_ct1.get_local_id(2) /
                    (2 * K_QUANTS_PER_ITERATION); // 0...15 or 0...7
    const int ix = item_ct1.get_local_id(2) %
                   (2 * K_QUANTS_PER_ITERATION); // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    const uint8_t * d = (const uint8_t *)uaux;


    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint32_t * s = (const uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        const sycl::float2 dall =
            x[i].dm.convert<float, sycl::rounding_mode::automatic>();

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q2_k_reorder(const void *__restrict__ vx,
                                                const float *__restrict__ yy,
                                                float *__restrict__ dst,
                                                const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t     * qs_base     = (const uint8_t *)vx;
    const uint8_t     * scales_base = qs_base + (size_t)nb * (QK_K / 4);
    const sycl::half2 * dm_base     = (const sycl::half2 *)(scales_base + (size_t)nb * (QK_K / 16));

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int in = tid % step;                           // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const sycl::half2 dm_val = dm_base[bi];
        const float dall = dm_val[0];
        const float dmin = dm_val[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int s_offset = 8*im;
            const int y_offset = 128*im + l0;

            const float   * y = yy + i * QK_K + y_offset;
            const uint8_t * q = qs_base + bi * (QK_K / 4) + q_offset;

            const uint32_t * a = (const uint32_t *)(scales_base + bi * (QK_K / 16) + s_offset);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
                sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                      + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                      + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                      + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                      + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                      + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                      + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                      +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
                sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                      + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

            }
            tmp += dall * sum1 - dmin * sum2;
        }
    }
#else
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(ncols);
    GGML_UNUSED(item_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q3_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int in = tid % step;                           // 0...15 or 0...7

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const uint8_t * h = x[i].hmask + l0;
        const float d = x[i].d;

        for (int im = 0; im < 2; ++im) {
            const int q_offset =  32*im + l0;
            const int y_offset = 128*im + l0;
            const uint16_t s_shift = 4*im;
            const uint8_t m = 1 << (4*im);

            const float   * y  = yy + i * QK_K + y_offset;
            const uint8_t * q = x[i].qs + q_offset;

            const uint16_t * a = (const uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

            float sum = 0;
            for (int l = 0; l < n; ++l) {
                sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * sum;
        }

    }
#else

    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    const int in = offset/8;                                 // 0 or 1
    const int im = offset%8;                                 // 0...7

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint8_t * s = x[i].scales;

        const float dall = (float)x[i].d;

        float sum = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t hl = x[i].hmask[im+l] >> in;
            const uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q3_k_reorder(const void *__restrict__ vx,
                                                const float *__restrict__ yy,
                                                float *__restrict__ dst,
                                                const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t   * qs_base     = (const uint8_t *)vx;
    const uint8_t   * hmask_base  = qs_base + (size_t)nb * (QK_K / 4);
    const uint8_t   * scales_base = hmask_base + (size_t)nb * (QK_K / 8);
    const sycl::half * d_base     = (const sycl::half *)(scales_base + (size_t)nb * 12);

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int in = tid % step;                           // 0...15 or 0...7

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const uint8_t * h  = hmask_base + bi * (QK_K / 8) + l0;

        const float d = d_base[bi];

        for (int im = 0; im < 2; ++im) {
            const int q_offset =  32*im + l0;
            const int y_offset = 128*im + l0;
            const uint16_t s_shift = 4*im;
            const uint8_t m = 1 << (4*im);

            const float   * y  = yy + i * QK_K + y_offset;
            const uint8_t * q  = qs_base + bi * (QK_K / 4) + q_offset;

            const uint16_t * a = (const uint16_t *)(scales_base + bi * 12);
            utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

            float sum = 0;
            for (int l = 0; l < n; ++l) {
                sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * sum;
        }
    }
#else
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(ncols);
    GGML_UNUSED(item_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q4_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il_base = tid/step;                        // 0 or 1 (was 0...3)
    const int ir  = tid - step*il_base;                  // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int in = il_base%2;

    const int l0 = n*(2*ir + in);

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int y_offset = 64*im + l0;

            const float   * y1 = yy + i*QK_K + y_offset;
            const float   * y2 = y1 + 128;

            const uint16_t * a = (const uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
            const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
            const uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            for (int l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#else
            const uint16_t * q1 = (const uint16_t *)(x[i].qs + q_offset);
            const uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            for (int l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#endif
        }

    }
#else
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const float d = (float)x[i].dm[0];
        const float m = (float)x[i].dm[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q4_k_reorder(const void *__restrict__ vx,
                                                const float *__restrict__ yy,
                                                float *__restrict__ dst,
                                                const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t     * qs_base     = (const uint8_t *)vx;
    const uint8_t     * scales_base = qs_base + (size_t)nb * (QK_K / 2);
    const sycl::half2 * dm_base     = (const sycl::half2 *)(scales_base + (size_t)nb * K_SCALE_SIZE);

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il_base = tid/step;                        // 0 or 1 (was 0...3)
    const int ir  = tid - step*il_base;                  // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int in = il_base%2;

    const int l0 = n*(2*ir + in);

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const sycl::half2 dm_val = dm_base[bi];
        const float dall = dm_val[0];
        const float dmin = dm_val[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int y_offset = 64*im + l0;

            const float   * y1 = yy + i*QK_K + y_offset;
            const float   * y2 = y1 + 128;

            const uint16_t * a = (const uint16_t *)(scales_base + bi * K_SCALE_SIZE);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
            const uint32_t * q1 = (const uint32_t *)(qs_base + bi * (QK_K / 2) + q_offset);
            const uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            for (int l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#else
            const uint16_t * q1 = (const uint16_t *)(qs_base + bi * (QK_K / 2) + q_offset);
            const uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            for (int l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#endif
        }

    }
#else
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const uint8_t * q = qs_base + bi * (QK_K / 2) + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)(scales_base + bi * K_SCALE_SIZE);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const sycl::half2 dm_val = dm_base[bi];
        const float d = (float)dm_val[0];
        const float m = (float)dm_val[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q5_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = item_ct1.get_local_id(2) / 2; // 0...7
    const int ix = item_ct1.get_local_id(2) % 2;

    const int il_base = tid/4;     // 0 or 1 (was 0...3)
    const int ir  = tid - 4*il_base;// 0...3
    const int n   = 2;

    const int in = il_base%2;

    const int l0 = n*(2*ir + in);

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * qh  = x[i].qh + l0;
        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int y_offset = 64*im + l0;

            const uint8_t hm1  = 1 << (2*im);
            const uint8_t hm2  = hm1 << 4;

            const uint8_t * ql1 = x[i].qs + q_offset;
            const float   * y1  = yy + i*QK_K + y_offset;
            const float   * y2  = y1 + 128;

            const uint16_t * a = (const uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            const uint16_t * q1 = (const uint16_t *)ql1;
            const uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            for (int l = 0; l < n; ++l) {
                sum.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                sum.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                sum.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                sum.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                           sum.w() * sc[5]) -
                   dmin * smin;
        }
    }

#else
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);
    const int step = tid * K_QUANTS_PER_ITERATION;
    const int im = step/8;
    const int in = step%8;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const int8_t  * s = x[i].scales;
        const float   * y = yy + i*QK_K + step;
        const float     d = x[i].d;
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q5_k_reorder(const void *__restrict__ vx,
                                                const float *__restrict__ yy,
                                                float *__restrict__ dst,
                                                const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t     * qs_base     = (const uint8_t *)vx;
    const uint8_t     * qh_base     = qs_base + (size_t)nb * (QK_K / 2);
    const uint8_t     * scales_base = qh_base + (size_t)nb * (QK_K / 8);
    const sycl::half2 * dm_base     = (const sycl::half2 *)(scales_base + (size_t)nb * K_SCALE_SIZE);

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = item_ct1.get_local_id(2) / 2; // 0...15
    const int ix = item_ct1.get_local_id(2) % 2;

    const int il_base = tid/4;      // 0...3
    const int ir  = tid - 4*il_base;// 0...3
    const int n   = 2;

    const int in = il_base%2;

    const int l0 = n*(2*ir + in);

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {
        const int bi = ib0 + i;

        const uint8_t * qh  = qh_base + bi * (QK_K / 8) + l0;
        const sycl::half2 dm_val = dm_base[bi];
        const float dall = dm_val[0];
        const float dmin = dm_val[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int y_offset = 64*im + l0;

            const uint8_t hm1  = 1 << (2*im);
            const uint8_t hm2  = hm1 << 4;

            const uint8_t * ql1 = qs_base + bi * (QK_K / 2) + q_offset;
            const float   * y1  = yy + i*QK_K + y_offset;
            const float   * y2  = y1 + 128;

            const uint16_t * a = (const uint16_t *)(scales_base + bi * K_SCALE_SIZE);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            const uint16_t * q1 = (const uint16_t *)ql1;
            const uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            for (int l = 0; l < n; ++l) {
                sum.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                sum.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                sum.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                sum.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                           sum.w() * sc[5]) -
                   dmin * smin;
        }
    }
#else
    // The reordered Q5_K layout is only produced for QK_K == 256.
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q6_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

#if QK_K == 256

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int in = tid % step;                           // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float d = x[i].d;

        for (int im = 0; im < 2; ++im) {
            const int ql_offset = 64*im + l0;
            const int qh_offset = 32*im + l0;
            const int s_offset  =  8*im + is;
            const int y_offset = 128*im + l0;

            const float   * y  = yy + i * QK_K + y_offset;
            const uint8_t * ql = x[i].ql + ql_offset;
            const uint8_t * qh = x[i].qh + qh_offset;
            const int8_t  * s  = x[i].scales + s_offset;

#if K_QUANTS_PER_ITERATION == 1
            float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += sum;
#else
            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += sum;
#endif
        }

    }

#else

    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q6_k_reorder(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t   * ql_base     = (const uint8_t *)vx;
    const uint8_t   * qh_base     = ql_base + (size_t)nb * (QK_K / 2);
    const int8_t    * scales_base = (const int8_t *)(qh_base + (size_t)nb * (QK_K / 4));
    const sycl::half * d_base     = (const sycl::half *)((const uint8_t *)scales_base + (size_t)nb * (QK_K / 16));

#if QK_K == 256

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int in = tid % step;                           // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const float d = d_base[bi];

        for (int im = 0; im < 2; ++im) {
            const int ql_offset = 64*im + l0;
            const int qh_offset = 32*im + l0;
            const int s_offset  =  8*im + is;
            const int y_offset = 128*im + l0;

            const float   * y  = yy + i * QK_K + y_offset;
            const uint8_t * ql = ql_base + bi * (QK_K / 2) + ql_offset;
            const uint8_t * qh = qh_base + bi * (QK_K / 4) + qh_offset;
            const int8_t  * s  = scales_base + bi * (QK_K / 16) + s_offset;

#if K_QUANTS_PER_ITERATION == 1
            float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += sum;
#else
            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += sum;
#endif
        }

    }

#else

    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = ql_base + bi * (QK_K / 2) + step;
        const uint8_t * qh = qh_base + bi * (QK_K / 4) + step;
        const int8_t  * s  = scales_base + bi * (QK_K / 16);

        const float d = d_base[bi];

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static void launch_dequantize_mul_mat_vec(const void *vx, const dfloat *y, float *dst,
                                          const int ncols, const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<qk, qr, dequantize_kernel>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

template <int qk, int qr, int qbytes, dequantize_kernel_t_reorder dequantize_kernel>
static void launch_dequantize_mul_mat_vec_reorder(const void *vx, const dfloat *y, float *dst,
                                                  const int ncols, const int nrows,
                                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_reorder<qk, qr, qbytes, dequantize_kernel>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

template <int ny, auto kernel>
static void launch_dequantize_mul_mat_vec_k(const void *vx, const float *y, float *dst,
                                            const int ncols, const int nrows,
                                            dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            kernel(vx, y, dst, ncols, nrows, item_ct1);
        });
}

thread_local ggml_sycl_dmmv_nc_req g_ggml_sycl_dmmv_nc = {};

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
using ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_SIZE;

// policies for dequantize_mul_mat_vec_reorder_esimd: row mapping and what to do with the two sums
// plain mat-vec: a work-group owns two consecutive rows of one matrix
struct esimd_dmmv_row_pair_policy {
    const void * vx;
    float *      dst;

    ESIMD_INLINE const void * w_a() const { return vx; }
    ESIMD_INLINE const void * w_b() const { return vx; }

    ESIMD_INLINE int  row_a(int g) const { return g * 2; }
    ESIMD_INLINE int  row_b(int g) const { return g * 2 + 1; }
    ESIMD_INLINE bool has_b(int g, int nrows) const { return row_b(g) < nrows; }

    ESIMD_INLINE void epilogue(int g, int nrows, float sum_a, float sum_b) const {
        dst[row_a(g)] = sum_a;
        if (has_b(g, nrows)) {
            dst[row_b(g)] = sum_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
struct esimd_dmmv_glu_policy {
    const void * vgate;
    const void * vup;
    float *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE const void * w_a() const { return vgate; }
    ESIMD_INLINE const void * w_b() const { return vup; }

    ESIMD_INLINE int  row_a(int g) const { return g; }
    ESIMD_INLINE int  row_b(int g) const { return g; }
    ESIMD_INLINE bool has_b(int, int) const { return true; } // up always contributes

    ESIMD_INLINE void epilogue(int g, int, float gate, float up) const {
        using namespace sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        const float GELU_COEF_A    = 0.044715f;
        const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        const float NEG_LOG2E      = -1.44269504088896341f;

        const float arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<float, 16> t = arg * NEG_LOG2E;
        simd<float, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * up;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
template <ggml_type T, typename Policy>
ESIMD_INLINE void dequantize_mul_mat_vec_reorder_esimd(
        const Policy & policy, const float * y,
        const int ncols, const int nrows,
        sycl::local_accessor<float, 1> lmem,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    using traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    const int    num_blocks_per_row = ncols / QK_K;
    const size_t nb = (size_t) nrows * num_blocks_per_row;
    const auto   pa = traits::make_ptrs(policy.w_a(), nb);
    const auto   pb = traits::make_ptrs(policy.w_b(), nb);

    const int  tid   = it.get_local_id(0);
    const int  g     = it.get_group(0);
    const int  row_a = policy.row_a(g);
    const int  row_b = policy.row_b(g);
    const bool has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<float, 32> acc_a = 0.0f;
    simd<float, 32> acc_b = 0.0f;

    for (int ib = tid; ib < num_blocks_per_row; ib += GGML_SYCL_DMMV_ESIMD_WG_SIZE) {
        simd<float, 256> y_vec = block_load<float, 256>(y + (size_t) ib * QK_K);

        const size_t bi_a = (size_t) row_a * num_blocks_per_row + ib;
        const size_t bi_b = (size_t) row_b * num_blocks_per_row + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<float>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<float>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        for (int p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_SIZE; ++p) {
            sum_a += lmem[p * 2 + 0];
            sum_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, sum_a, sum_b);
    }
}

// The row pair of dequantize_mul_mat_vec_reorder_esimd over ncol <= NC columns: per column the accumulators, sums
// and reduction order of the one-column work-group, the weight blocks of the later columns from the cache.
template <ggml_type T, int NC>
ESIMD_INLINE void dequantize_mul_mat_vec_reorder_esimd_nc(
        const void * vx, const float * y, const size_t ys, float * dst, const size_t ds, const int ncol,
        const int ncols, const int nrows, sycl::local_accessor<float, 1> lmem, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    using traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    const int    num_blocks_per_row = ncols / QK_K;
    const size_t nb = (size_t) nrows * num_blocks_per_row;
    const auto   pa = traits::make_ptrs(vx, nb);

    const int  tid   = it.get_local_id(0);
    const int  g     = it.get_group(0);
    const int  row_a = g * 2;
    const int  row_b = g * 2 + 1;
    const bool has_b = row_b < nrows;

    // the 32-wide accumulators of column c at 32 * c (one simd each: an array of simd objects crashed the JIT)
    simd<float, 32 * NC> acc_a = 0.0f;
    simd<float, 32 * NC> acc_b = 0.0f;

    for (int ib = tid; ib < num_blocks_per_row; ib += GGML_SYCL_DMMV_ESIMD_WG_SIZE) {
        const size_t bi_a = (size_t) row_a * num_blocks_per_row + ib;
        const size_t bi_b = (size_t) row_b * num_blocks_per_row + ib;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < ncol) {
                simd<float, 256> y_vec = block_load<float, 256>(y + c * ys + (size_t) ib * QK_K);
                simd<float, 32>  ca    = acc_a.template select<32, 1>(32 * c);
                simd<float, 32>  cb    = acc_b.template select<32, 1>(32 * c);
                traits::mac_pair(pa, bi_a, pa, bi_b, has_b, y_vec, ca, cb);
                acc_a.template select<32, 1>(32 * c) = ca;
                acc_b.template select<32, 1>(32 * c) = cb;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (c < ncol) {
            simd<float, 32> ca = acc_a.template select<32, 1>(32 * c);
            simd<float, 32> cb = acc_b.template select<32, 1>(32 * c);
            lmem[(c * GGML_SYCL_DMMV_ESIMD_WG_SIZE + tid) * 2 + 0] = reduce<float>(ca, std::plus<>{});
            lmem[(c * GGML_SYCL_DMMV_ESIMD_WG_SIZE + tid) * 2 + 1] = reduce<float>(cb, std::plus<>{});
        }
    }
    it.barrier(sycl::access::fence_space::local_space);

    for (int c = tid; c < ncol; c += GGML_SYCL_DMMV_ESIMD_WG_SIZE) {
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        for (int p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_SIZE; ++p) {
            sum_a += lmem[(c * GGML_SYCL_DMMV_ESIMD_WG_SIZE + p) * 2 + 0];
            sum_b += lmem[(c * GGML_SYCL_DMMV_ESIMD_WG_SIZE + p) * 2 + 1];
        }
        dst[c * ds + row_a] = sum_a;
        if (has_b) {
            dst[c * ds + row_b] = sum_b;
        }
    }
}

template <ggml_type T, int NC>
static void dmmv_reorder_esimd_nc_launch(const void * vx, const float * y, const size_t ys, float * dst,
                                         const size_t ds, const int ncol, const int ncols, const int nrows,
                                         dpct::queue_ptr stream) {
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2 * NC), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE),
                                         sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
                       [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                           dequantize_mul_mat_vec_reorder_esimd_nc<T, NC>(vx, y, ys, dst, ds, ncol, ncols, nrows,
                                                                          lmem, it);
                       });
    });
}

// the request g_ggml_sycl_dmmv_nc, taken by the one-column row-pair launch over y and dst
template <ggml_type T>
static bool dmmv_reorder_esimd_nc_try(const void * vx, const float * y, float * dst, const int ncols, const int nrows,
                                      dpct::queue_ptr stream) {
    ggml_sycl_dmmv_nc_req & r = g_ggml_sycl_dmmv_nc;
    if (r.ncol <= 1 || r.ncol > 8 || r.used || y != r.y0 || dst != r.dst0) {
        return false;
    }
    if (r.ncol <= 4) {
        dmmv_reorder_esimd_nc_launch<T, 4>(vx, y, r.y_stride, dst, r.dst_stride, r.ncol, ncols, nrows, stream);
    } else {
        dmmv_reorder_esimd_nc_launch<T, 8>(vx, y, r.y_stride, dst, r.dst_stride, r.ncol, ncols, nrows, stream);
    }
    r.used = true;
    return true;
}

static void dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    if (dmmv_reorder_esimd_nc_try<GGML_TYPE_Q4_K>(vx, y, dst, ncols, nrows, stream)) {
        return;
    }
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    if (dmmv_reorder_esimd_nc_try<GGML_TYPE_Q5_K>(vx, y, dst, ncols, nrows, stream)) {
        return;
    }
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    if (dmmv_reorder_esimd_nc_try<GGML_TYPE_Q6_K>(vx, y, dst, ncols, nrows, stream)) {
        return;
    }
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
template <int NBLK>
ESIMD_INLINE void q8_0_mac_stripe(
        const int8_t * qs_a, const int8_t * qs_b,
        const sycl::half * d_a, const sycl::half * d_b, bool has_b,
        sycl::ext::intel::esimd::simd<float, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<float, 32> & acc_a,
        sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
    using namespace sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = block_load<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = block_load<sycl::half, NBLK>(d_a, element_aligned_tag{});
    simd<sycl::half, NBLK>  db = 0;
    if (has_b) {
        qb = block_load<int8_t, 32 * NBLK>(qs_b);
        db = block_load<sycl::half, NBLK>(d_b, element_aligned_tag{});
    }

    simd<float, NBLK> da_f = convert<float>(da);
    simd<float, NBLK> db_f = convert<float>(db);

#pragma unroll
    for (int s = 0; s < NBLK; ++s) {
        simd<float, 32>  y_s  = y_vec.template select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.template select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.template select<32, 1>(s * 32);
        const float sa = da_f[s];
        const float sb = db_f[s];
        acc_a += y_s * (convert<float>(qa_s) * sa);
        acc_b += y_s * (convert<float>(qb_s) * sb);
    }
}

template <int WG, typename Policy>
ESIMD_INLINE void dequantize_mul_mat_vec_q8_0_reorder_esimd(
        const Policy & policy, const float * y,
        const int ncols, const int nrows,
        sycl::local_accessor<float, 1> lmem,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;

    constexpr int STRIPE = 8;

    const int          nblk_row = ncols / QK8_0;
    const size_t       nb       = (size_t) nrows * nblk_row;
    const int8_t *     qs_a     = (const int8_t *) policy.w_a();
    const int8_t *     qs_b     = (const int8_t *) policy.w_b();
    const sycl::half * d_a      = (const sycl::half *) (qs_a + nb * QK8_0);
    const sycl::half * d_b      = (const sycl::half *) (qs_b + nb * QK8_0);

    const int  tid   = it.get_local_id(0);
    const int  g     = it.get_group(0);
    const bool has_b = policy.has_b(g, nrows);

    const size_t base_a = (size_t) policy.row_a(g) * nblk_row;
    const size_t base_b = (size_t) policy.row_b(g) * nblk_row;

    simd<float, 32> acc_a = 0.0f;
    simd<float, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    int ib = 0;
    for (; ib + WG * STRIPE <= nblk_row; ib += WG * STRIPE) {
        const int b = ib + tid * STRIPE;
        simd<float, 256> y_vec = block_load<float, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (base_a + b) * QK8_0, qs_b + (base_b + b) * QK8_0,
                                d_a + base_a + b, d_b + base_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    for (int b = ib + tid; b < nblk_row; b += WG) {
        simd<float, 32> y_vec = block_load<float, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (base_a + b) * QK8_0, qs_b + (base_b + b) * QK8_0,
                           d_a + base_a + b, d_b + base_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<float>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<float>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        for (int p = 0; p < WG; ++p) {
            sum_a += lmem[p * 2 + 0];
            sum_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, sum_a, sum_b);
    }
}

template <int WG, typename Policy>
static void q8_0_esimd_launch(const Policy & policy, const float * y, const int ncols, const int nrows,
                              const int workgroups, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
template <typename Policy>
static void q8_0_esimd_launch_scaled(const Policy & policy, const float * y, const int ncols, const int nrows,
                                     const int workgroups, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    const int nblk_row = ncols / QK8_0;
    if (nblk_row >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } else if (nblk_row >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } else if (nblk_row >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } else {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
template <ggml_type T>
static void dequantize_mul_mat_vec_glu_reorder_esimd(const void * vx, const void * vgate,
                                                     const float * y, float * dst,
                                                     const int ncols, const int nrows,
                                                     const ggml_glu_op glu_op,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

bool ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(enum ggml_type src0_type, enum ggml_glu_op glu_op,
                                                        const void * vx, const void * vgate, const float * y,
                                                        float * dst, int ncols, int nrows, dpct::queue_ptr stream) {
    switch (src0_type) {
        case GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            return true;
        case GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            return true;
        case GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            return true;
        case GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            return true;
        case GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            return true;
        case GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            return true;
        default:
            return false;
    }
}


// ---------------------------------------------------------------------------------------------------------------
// One-column products of several reordered Q4_K / Q6_K weights sharing y, in one launch (ggml_sycl_dmmv_multi).
// A work-group owns DMMV_MULTI_R rows of one segment; its ncols / (256 * DMMV_MULTI_KT) threads each keep a slice of
// DMMV_MULTI_KT blocks of y in registers and read the DMMV_MULTI_KT contiguous blocks of each row (one 384-byte quant
// load per Q4_K row), so y is read once per work-group instead of once per block and row pair.
// ---------------------------------------------------------------------------------------------------------------
namespace {
constexpr int DMMV_MULTI_KT      = 3;   // blocks per thread
constexpr int DMMV_MULTI_R       = 2;   // rows per work-group (at most 16 threads per work-group: ncols <= 12288)

using sycl::ext::intel::esimd::simd;

static ESIMD_INLINE float dmmv_multi_row_q4_K(const uint8_t * base, size_t nb, size_t b0,
                                              simd<float, 256 * DMMV_MULTI_KT> & yv) {
    using namespace sycl::ext::intel::esimd;
    constexpr int KT = DMMV_MULTI_KT;
    const uint8_t * qs_p = base;
    const uint8_t * sc_p = qs_p + nb * (QK_K / 2);
    const uint8_t * dm_p = sc_p + nb * K_SCALE_SIZE;

    simd<uint8_t, 128 * KT> qs   = block_load<uint8_t, 128 * KT>(qs_p + b0 * (QK_K / 2), overaligned_tag<16>{});
    simd<uint32_t, 3 * KT>  sc32 = block_load<uint32_t, 3 * KT>((const uint32_t *) (sc_p + b0 * K_SCALE_SIZE), overaligned_tag<4>{});
    simd<uint32_t, KT>      dm32 = block_load<uint32_t, KT>((const uint32_t *) (dm_p + b0 * 4), overaligned_tag<4>{});
    simd<uint8_t, 12 * KT>    sc  = sc32.template bit_cast_view<uint8_t>();
    simd<sycl::half, 2 * KT>  dmh = dm32.template bit_cast_view<sycl::half>();
    simd<float, 2 * KT>       dm  = dmh;

    simd<float, 32> acc = 0.0f;
#pragma unroll
    for (int k = 0; k < KT; ++k) {
        simd<float, 8> scale_f, min_f;
        ggml_sycl_esimd::unpack_scale_min_k4(sc.template select<12, 1>(12 * k), (float) dm[2 * k], (float) dm[2 * k + 1],
                                             scale_f, min_f);
        simd<uint8_t, 128> q  = qs.template select<128, 1>(128 * k);
        simd<uint8_t, 128> lo = q & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> hi = q >> simd<uint8_t, 128>(4);
#pragma unroll
        for (int sb = 0; sb < 8; sb += 2) {
            simd<uint8_t, 32> l8 = lo.template select<32, 1>(sb * 16);
            simd<uint8_t, 32> h8 = hi.template select<32, 1>(sb * 16);
            simd<float, 32> d_lo = convert<float>(l8) * (float) scale_f[sb] + (float) min_f[sb];
            simd<float, 32> d_hi = convert<float>(h8) * (float) scale_f[sb + 1] + (float) min_f[sb + 1];
            acc += yv.template select<32, 1>(256 * k + sb * 32) * d_lo;
            acc += yv.template select<32, 1>(256 * k + (sb + 1) * 32) * d_hi;
        }
    }
    return reduce<float>(acc, std::plus<>{});
}

static ESIMD_INLINE float dmmv_multi_row_q6_K(const uint8_t * base, size_t nb, size_t b0,
                                              simd<float, 256 * DMMV_MULTI_KT> & yv) {
    using namespace sycl::ext::intel::esimd;
    constexpr int KT = DMMV_MULTI_KT;
    const uint8_t *    ql_p = base;
    const uint8_t *    qh_p = ql_p + nb * (QK_K / 2);
    const int8_t *     sc_p = (const int8_t *) (qh_p + nb * (QK_K / 4));
    const sycl::half * d_p  = (const sycl::half *) ((const uint8_t *) sc_p + nb * (QK_K / 16));

    simd<uint8_t, 128 * KT> ql = block_load<uint8_t, 128 * KT>(ql_p + b0 * (QK_K / 2), overaligned_tag<16>{});
    simd<uint8_t, 64 * KT>  qh = block_load<uint8_t, 64 * KT>(qh_p + b0 * (QK_K / 4), overaligned_tag<16>{});
    simd<int8_t, 16 * KT>   sc = block_load<int8_t, 16 * KT>(sc_p + b0 * (QK_K / 16), overaligned_tag<16>{});

    simd<float, 32> acc = 0.0f;
#pragma unroll
    for (int k = 0; k < KT; ++k) {
        const float     d  = (float) d_p[b0 + k];
        simd<int8_t, 16> s8 = sc.template select<16, 1>(16 * k);
        simd<float, 16>  sf = convert<float>(s8);
#pragma unroll
        for (int im = 0; im < 2; ++im) {
            simd<uint8_t, 32> ql_lo   = ql.template select<32, 1>(128 * k + 64 * im);
            simd<uint8_t, 32> ql_hi   = ql.template select<32, 1>(128 * k + 64 * im + 32);
            simd<uint8_t, 32> qh_bits = qh.template select<32, 1>(64 * k + 32 * im);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                simd<uint8_t, 32> q;
                switch (g) {
                    case 0:  q = (ql_lo & simd<uint8_t, 32>(0x0F)) | ((qh_bits & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4)); break;
                    case 1:  q = (ql_hi & simd<uint8_t, 32>(0x0F)) | ((qh_bits & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2)); break;
                    case 2:  q = (ql_lo >> simd<uint8_t, 32>(4)) | (qh_bits & simd<uint8_t, 32>(0x30)); break;
                    default: q = (ql_hi >> simd<uint8_t, 32>(4)) | ((qh_bits & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2)); break;
                }
                const float scale_lo = (float) sf[8 * im + 2 * g + 0] * d;
                const float scale_hi = (float) sf[8 * im + 2 * g + 1] * d;
                simd<float, 32> scale_vec = ggml_sycl_esimd::splat_lo_hi(scale_lo, scale_hi);
                simd<float, 32> deq = (convert<float>(q) - 32.0f) * scale_vec;
                acc += yv.template select<32, 1>(256 * k + 32 * (4 * im + g)) * deq;
            }
        }
    }
    return reduce<float>(acc, std::plus<>{});
}
} // namespace

namespace {
using sycl::ext::intel::esimd::simd;
using sycl::ext::intel::esimd::block_load;
using sycl::ext::intel::esimd::overaligned_tag;

// TYPE: GGML_TYPE_Q4_K or GGML_TYPE_Q6_K for every segment; NTH = ncols / 768 threads per work-group. The segment of
// a work-group is picked from scalar captures (a struct of segments indexed in the kernel made it twice as slow).
template <int TYPE, int NTH, int NC>
void dmmv_multi_launch_nc(const ggml_sycl_dmmv_multi_args & a, const int * ends, const float * y, int total,
                          dpct::queue_ptr stream);

template <int TYPE, int NTH>
void dmmv_multi_launch(const ggml_sycl_dmmv_multi_args & a, const int * ends, const float * y, int total,
                       dpct::queue_ptr stream) {
    if (a.ncol > 1) {
        dmmv_multi_launch_nc<TYPE, NTH, 8>(a, ends, y, total, stream);
        return;
    }
    constexpr int KT   = DMMV_MULTI_KT;
    constexpr int R    = DMMV_MULTI_R;
    constexpr int nbpr = NTH * KT;
    const int e0 = ends[0], e1 = ends[1], e2 = ends[2];
    const uint8_t * w0 = (const uint8_t *) a.seg[0].w, * w1 = (const uint8_t *) a.seg[1].w,
                  * w2 = (const uint8_t *) a.seg[2].w, * w3 = (const uint8_t *) a.seg[3].w;
    float * d0 = a.seg[0].dst, * d1 = a.seg[1].dst, * d2 = a.seg[2].dst, * d3 = a.seg[3].dst;
    const int n0 = a.seg[0].nrows, n1 = a.seg[1].nrows, n2 = a.seg[2].nrows, n3 = a.seg[3].nrows;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(NTH * R), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) total * NTH), sycl::range<1>(NTH)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                const int tid = (int) it.get_local_id(0);
                const int g   = (int) it.get_group(0);
                const int s   = (g >= e0) + (g >= e1) + (g >= e2);
                const uint8_t * w = s == 0 ? w0 : s == 1 ? w1 : s == 2 ? w2 : w3;
                const int nrows   = s == 0 ? n0 : s == 1 ? n1 : s == 2 ? n2 : n3;
                const int base    = s == 0 ? 0 : s == 1 ? e0 : s == 2 ? e1 : e2;
                const int row0    = (g - base) * R;

                simd<float, 256 * KT> yv = block_load<float, 256 * KT>(y + (size_t) tid * KT * QK_K, overaligned_tag<16>{});
                const size_t nb = (size_t) nrows * nbpr;

                simd<float, R> part;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int row = row0 + r < nrows ? row0 + r : nrows - 1;
                    if constexpr (TYPE == GGML_TYPE_Q4_K) {
                        part[r] = dmmv_multi_row_q4_K(w, nb, (size_t) row * nbpr + tid * KT, yv);
                    } else {
                        part[r] = dmmv_multi_row_q6_K(w, nb, (size_t) row * nbpr + tid * KT, yv);
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    lmem[tid * R + r] = part[r];
                }
                it.barrier(sycl::access::fence_space::local_space);

                if (tid < R) {
                    const int row = row0 + tid;
                    if (row < nrows) {
                        float x = 0.0f;
#pragma unroll
                        for (int p = 0; p < NTH; ++p) {
                            x += lmem[p * R + tid];
                        }
                        float * dst = s == 0 ? d0 : s == 1 ? d1 : s == 2 ? d2 : d3;
                        dst[row] = x;
                    }
                }
            });
    });
}

// the launch above over a.ncol <= NC columns (row-exact verify): a work-group per (row group, column), the column
// fastest so that the work-groups of a row group run together and read its blocks once from memory; column c reads
// y + c * y_stride and writes dst + c * dst_stride through the one-column work-group. (A loop over the columns in one
// work-group crashed the JIT.)
template <int TYPE, int NTH, int NC>
void dmmv_multi_launch_nc(const ggml_sycl_dmmv_multi_args & a, const int * ends, const float * y, int total,
                          dpct::queue_ptr stream) {
    constexpr int KT   = DMMV_MULTI_KT;
    constexpr int R    = DMMV_MULTI_R;
    constexpr int nbpr = NTH * KT;
    const int e0 = ends[0], e1 = ends[1], e2 = ends[2];
    const uint8_t * w0 = (const uint8_t *) a.seg[0].w, * w1 = (const uint8_t *) a.seg[1].w,
                  * w2 = (const uint8_t *) a.seg[2].w, * w3 = (const uint8_t *) a.seg[3].w;
    float * d0 = a.seg[0].dst, * d1 = a.seg[1].dst, * d2 = a.seg[2].dst, * d3 = a.seg[3].dst;
    const int n0 = a.seg[0].nrows, n1 = a.seg[1].nrows, n2 = a.seg[2].nrows, n3 = a.seg[3].nrows;
    const int    ncol = a.ncol;
    const size_t ys   = a.y_stride;
    const size_t ds   = a.dst_stride;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(NTH * R), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) total * ncol * NTH), sycl::range<1>(NTH)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                const int tid = (int) it.get_local_id(0);
                const int G   = (int) it.get_group(0);
                const int g   = G / ncol;
                const int c   = G - g * ncol;
                const int s   = (g >= e0) + (g >= e1) + (g >= e2);
                const uint8_t * w = s == 0 ? w0 : s == 1 ? w1 : s == 2 ? w2 : w3;
                const int nrows   = s == 0 ? n0 : s == 1 ? n1 : s == 2 ? n2 : n3;
                const int base    = s == 0 ? 0 : s == 1 ? e0 : s == 2 ? e1 : e2;
                const int row0    = (g - base) * R;

                simd<float, 256 * KT> yv = block_load<float, 256 * KT>(y + c * ys + (size_t) tid * KT * QK_K, overaligned_tag<16>{});
                const size_t nb = (size_t) nrows * nbpr;

                simd<float, R> part;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int row = row0 + r < nrows ? row0 + r : nrows - 1;
                    if constexpr (TYPE == GGML_TYPE_Q4_K) {
                        part[r] = dmmv_multi_row_q4_K(w, nb, (size_t) row * nbpr + tid * KT, yv);
                    } else {
                        part[r] = dmmv_multi_row_q6_K(w, nb, (size_t) row * nbpr + tid * KT, yv);
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    lmem[tid * R + r] = part[r];
                }
                it.barrier(sycl::access::fence_space::local_space);

                if (tid < R) {
                    const int row = row0 + tid;
                    if (row < nrows) {
                        float x = 0.0f;
#pragma unroll
                        for (int p = 0; p < NTH; ++p) {
                            x += lmem[p * R + tid];
                        }
                        float * dst = s == 0 ? d0 : s == 1 ? d1 : s == 2 ? d2 : d3;
                        dst[c * ds + row] = x;
                    }
                }
            });
    });
}

template <int TYPE>
bool dmmv_multi_nth(const ggml_sycl_dmmv_multi_args & a, const int * ends, const float * y, int nth, int total,
                    dpct::queue_ptr stream) {
    switch (nth) {
        case 3: dmmv_multi_launch<TYPE, 3>(a, ends, y, total, stream); return true;
        case 4: dmmv_multi_launch<TYPE, 4>(a, ends, y, total, stream); return true;
        case 6: dmmv_multi_launch<TYPE, 6>(a, ends, y, total, stream); return true;
        case 8: dmmv_multi_launch<TYPE, 8>(a, ends, y, total, stream); return true;
        default: return false;
    }
}
} // namespace

bool ggml_sycl_dmmv_multi(const ggml_sycl_dmmv_multi_args & args, const float * y, int ncols, dpct::queue_ptr stream) {
    constexpr int KT = DMMV_MULTI_KT;
    constexpr int R  = DMMV_MULTI_R;
    if (ncols <= 0 || ncols % (QK_K * KT) != 0 || args.nseg < 1 || args.nseg > GGML_SYCL_DMMV_MULTI_MAX) {
        return false;
    }
    const int nth = ncols / (QK_K * KT);
    if (!ggml_sycl_dmmv_multi_supported(args.seg[0].type, ncols)) {
        return false;
    }
    int  ends[GGML_SYCL_DMMV_MULTI_MAX] = {};
    int  total = 0;
    for (int s = 0; s < args.nseg; ++s) {
        const auto & sg = args.seg[s];
        if (sg.type != args.seg[0].type || sg.nrows <= 0) {
            return false;
        }
        total += (sg.nrows + R - 1) / R;
        ends[s] = total;
    }
    for (int s = args.nseg; s < GGML_SYCL_DMMV_MULTI_MAX; ++s) {
        ends[s] = total;
    }
    if (args.seg[0].type == GGML_TYPE_Q4_K) {
        return dmmv_multi_nth<GGML_TYPE_Q4_K>(args, ends, y, nth, total, stream);
    }
    return dmmv_multi_nth<GGML_TYPE_Q6_K>(args, ends, y, nth, total, stream);
}

bool ggml_sycl_dmmv_multi_supported(int type, int64_t ncols) {
    if ((type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q6_K) || ncols <= 0 || ncols % (QK_K * DMMV_MULTI_KT) != 0) {
        return false;
    }
    const int64_t nth = ncols / (QK_K * DMMV_MULTI_KT);
    return nth == 3 || nth == 4 || nth == 6 || nth == 8;
}

#endif // GGML_SYCL_DMMV_HAS_ESIMD

#ifndef GGML_SYCL_DMMV_HAS_ESIMD
bool ggml_sycl_dmmv_multi(const ggml_sycl_dmmv_multi_args &, const float *, int, dpct::queue_ptr) {
    return false;
}
bool ggml_sycl_dmmv_multi_supported(int, int64_t) {
    return false;
}
#endif

void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = nullptr; // dfloat == half

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    const bool q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#else
    const bool q8_0_esimd = false;
#endif

    bool src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    if (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#else
    const dfloat * src1_dfloat = (const dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_SYCL_F16

    switch (src0->type) {
        case GGML_TYPE_Q1_0:
            if ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                launch_dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, QK1_0/8, dequantize_q1_0_reorder>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } else {
                launch_dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q4_0:
            if ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                launch_dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, QK4_0/2, dequantize_q4_0_reorder>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } else {
                launch_dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q4_1:
            launch_dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_0:
            launch_dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_1:
            launch_dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q8_0:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                } else
#endif
                {
                    launch_dequantize_mul_mat_vec_reorder<QK8_0, QR8_0, QK8_0, dequantize_q8_0_reorder>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                launch_dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q2_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q2_k_reorder>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                launch_dequantize_mul_mat_vec_k<2, dequantize_mul_mat_vec_q2_k>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q3_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q3_k_reorder>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q3_k>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q4_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q4_k_reorder>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q4_k>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q5_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q5_k_reorder>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q5_k>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q6_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q6_k_reorder>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                launch_dequantize_mul_mat_vec_k<2/K_QUANTS_PER_ITERATION, dequantize_mul_mat_vec_q6_k>(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
#ifdef GGML_SYCL_DMMV_HAS_BF16
        case GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
#endif
        default:
            printf("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(ctx);
}
