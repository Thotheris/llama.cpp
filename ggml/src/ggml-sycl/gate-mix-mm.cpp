// MUL_MAT(gain|forget) -> GATE_MIX -> RMS_NORM -> MUL as one ESIMD kernel at prefill, see gate-mix-mm.hpp

#include "gate-mix-mm.hpp"
#if GGML_SYCL_DNNL
#include "dense-dnn.hpp"
#endif

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>

namespace {

constexpr int GMM_K   = 64; // the gate rank: K of the gain|forget projection
constexpr int GMM_KS  = GMM_K / 16;
constexpr int GMM_MAX_THREADS = 64;

// the VNNI f16 copy of a gain|forget weight [K][2n] (row r < n gains, n + r forgets): per 16-channel tile c, gain /
// forget h and k step s, the dpas B operand [8 k pairs][16 channels][2], 512 bytes
struct gmm_weight {
    ggml_backend_buffer_t buffer = nullptr;
    sycl::half *          data   = nullptr;
    int                   device = 0;
    bool                  failed = false; // the copy could not be made: the old path, for good
};
using gmm_wkey = std::tuple<int, const void *, int64_t>;
std::mutex                      g_gmm_mutex;
std::map<gmm_wkey, gmm_weight>  g_gmm_weights;

int gmm_env() {
    static const int v = ggml_sycl_get_env("GGML_SYCL_GATE_MIX_FUSED", 1);
    return v;
}
// the x and m tiles as 2D block loads and the results as 2D block stores, 8 rows x 16 channels a message, when the
// rows allow it (GGML_SYCL_GATE_MIX_FUSED_2D=0: per-row block loads and stores)
int gmm_2d_env() {
    static const int v = ggml_sycl_get_env("GGML_SYCL_GATE_MIX_FUSED_2D", 1);
    return v;
}

// 16-channel tiles per thread: 9 (256 registers: 16 threads a work-group for 2304 channels, two work-groups an
// Xe-core), else 3 (128 registers)
int gmm_tpt(int64_t n) {
    return n % (16 * 9) == 0 && n / (16 * 9) <= GMM_MAX_THREADS ? 9 : 3;
}

// rows of a 2D-able f32 tensor: M rows of ne[0] floats at a constant stride ld (floats)
bool gmm_rows(const ggml_tensor * t, int64_t & M, int64_t & ld) {
    if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->ne[3] != 1) {
        return false;
    }
    if (t->ne[2] == 1) {
        M = t->ne[1]; ld = t->nb[1] / sizeof(float);
        return t->nb[1] % sizeof(float) == 0;
    }
    if (t->ne[1] == 1) {
        M = t->ne[2]; ld = t->nb[2] / sizeof(float);
        return t->nb[2] % sizeof(float) == 0;
    }
    return false;
}

bool gmm_al16(const void * p) {
    return ((uintptr_t) p % 16) == 0;
}

// dst[M][ldd] = src[M][lds], K floats a row
void gmm_copy_rows(queue_ptr q, float * dst, int64_t ldd, const float * src, int64_t lds, int64_t K, int64_t M) {
    q->parallel_for(sycl::range<2>((size_t) M, (size_t) K), [=](sycl::id<2> id) {
        dst[(int64_t) id[0] * ldd + (int64_t) id[1]] = src[(int64_t) id[0] * lds + (int64_t) id[1]];
    });
}

// whether the bytes of M rows of K floats at stride ld overlap those of the other rows
bool gmm_overlap(const void * a, int64_t Ma, int64_t lda, int64_t Ka, const void * b, int64_t Mb, int64_t ldb, int64_t Kb) {
    const char * a0 = (const char *) a;
    const char * a1 = a0 + ((Ma - 1) * lda + Ka) * sizeof(float);
    const char * b0 = (const char *) b;
    const char * b1 = b0 + ((Mb - 1) * ldb + Kb) * sizeof(float);
    return a0 < b1 && b0 < a1;
}

template <int TPT, int NG, bool BIG, bool L2D> struct gate_mix_mm_kernel {
    static constexpr int TB   = 8 * NG;
    static constexpr int ZOFF = 0;                          // z as f16 [KS][TB][16]
    static constexpr int SOFF = GMM_KS * TB * 16 * 2;      // the threads' row sums [thread][TB]
    static constexpr int SLM  = SOFF + GMM_MAX_THREADS * TB * 4;

    const float *      z32; int64_t ldz;  // z rows (f32), or
    const sycl::half * z16;               // z as dense f16 rows [T][64]
    const uint32_t *   wp;
    const float *      x;   int64_t ldx;
    const float *      m;   int64_t ldm;
    const float *      w;
    float *            dst; int64_t ldd;
    sycl::half *       dst_h;             // dense f16 [T][n] or nullptr
    int                n, T;
    float              eps;
    bool               no_f32;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        namespace xmx = sycl::ext::intel::esimd::xmx;
        namespace esimdx = sycl::ext::intel::experimental::esimd;

        slm_init<SLM>();
        const int thr = (int) it.get_local_id(0);
        const int nth = (int) it.get_local_range(0);
        const int t0  = (int) it.get_group(0) * TB;

        // the z rows of the block in SLM as f16, laid out per k step: rows of 16 halves, 8 tokens a dpas A
        if (thr < TB) {
            const int t = t0 + thr;
            simd<sycl::half, GMM_K> zh = 0;
            if (t < T) {
                if (z16) {
                    zh = block_load<sycl::half, GMM_K>(z16 + (size_t) t * GMM_K, overaligned_tag<16>{});
                } else {
                    simd<float, GMM_K> zf = block_load<float, GMM_K>(z32 + (size_t) t * ldz, overaligned_tag<16>{});
                    zh = zf;
                }
            }
            simd<uint32_t, GMM_K / 2> zu = zh.template bit_cast_view<uint32_t>();
#pragma unroll
            for (int s = 0; s < GMM_KS; ++s) {
                slm_block_store<uint32_t, 8>(ZOFF + (s * TB + thr) * 32, zu.template select<8, 1>(s * 8));
            }
        }
        barrier();

        simd<float, 128> yk[TPT][NG];
        simd<float, TB>  ss = 0.0f;
#pragma unroll
        for (int j = 0; j < TPT; ++j) {
            const int c = thr * TPT + j;
            // the x and m rows of the tile first: their loads are in flight during the dpas
            simd<float, 128> xv[NG];
            simd<float, 128> mv[NG];
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                xv[g] = 0.0f;
                mv[g] = 0.0f;
                if constexpr (L2D) {
                    // rows past T read as zeros
                    xv[g] = load_2d<float, 16, 8>(x, (unsigned) (n * 4 - 1), (unsigned) (T - 1), (unsigned) (ldx * 4 - 1), c * 16, t0 + g * 8);
                    mv[g] = load_2d<float, 16, 8>(m, (unsigned) (n * 4 - 1), (unsigned) (T - 1), (unsigned) (ldm * 4 - 1), c * 16, t0 + g * 8);
                } else {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int t = t0 + g * 8 + r;
                        if (t < T) {
                            xv[g].template select<16, 1>(r * 16) = block_load<float, 16>(x + (size_t) t * ldx + c * 16, overaligned_tag<16>{});
                            mv[g].template select<16, 1>(r * 16) = block_load<float, 16>(m + (size_t) t * ldm + c * 16, overaligned_tag<16>{});
                        }
                    }
                }
            }
            simd<float, 128> ag[NG];
            simd<float, 128> af[NG];
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                ag[g] = 0.0f;
                af[g] = 0.0f;
            }
#pragma unroll
            for (int s = 0; s < GMM_KS; ++s) {
                simd<uint32_t, 128> bg = block_load<uint32_t, 128>(wp + ((size_t) (c * 2 + 0) * GMM_KS + s) * 128, overaligned_tag<16>{});
                simd<uint32_t, 128> bf = block_load<uint32_t, 128>(wp + ((size_t) (c * 2 + 1) * GMM_KS + s) * 128, overaligned_tag<16>{});
                simd<sycl::half, 256> bgh = bg.template bit_cast_view<sycl::half>();
                simd<sycl::half, 256> bfh = bf.template bit_cast_view<sycl::half>();
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    simd<uint32_t, 64> au = slm_block_load<uint32_t, 64>(ZOFF + (s * TB + g * 8) * 32);
                    simd<sycl::half, 128> a = au.template bit_cast_view<sycl::half>();
                    ag[g] = xmx::dpas<8, 8, float, float>(ag[g], bgh, a);
                    af[g] = xmx::dpas<8, 8, float, float>(af[g], bfh, a);
                }
            }
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                // tanh on the f32 accumulator, rounded to f16 as the old path stored the gates
                simd<sycl::half, 128> gh = esimdx::tanh(ag[g]);
                simd<sycl::half, 128> fh = esimdx::tanh(af[g]);
                simd<float, 128> gain   = gh;
                simd<float, 128> forget = fh;
                // the arithmetic of gate_mix_f32_sycl: (x - forget*x) + gain*m
                simd<float, 128> fx = forget * xv[g];
                simd<float, 128> av = xv[g] - fx;
                simd<float, 128> gm = gain * mv[g];
                simd<float, 128> y  = av + gm;
                yk[j][g] = y;
                simd<float, 128> yy = y * y;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    simd<float, 16> row = yy.template select<16, 1>(r * 16);
                    ss[g * 8 + r] = ss[g * 8 + r] + reduce<float>(row, std::plus<>());
                }
            }
        }
        slm_block_store<float, TB>(SOFF + thr * TB * 4, ss);
        barrier();
        // every thread adds the partials of all threads in the same order
        simd<float, TB> tot = 0.0f;
        for (int i = 0; i < nth; ++i) {
            tot += slm_block_load<float, TB>(SOFF + i * TB * 4);
        }
        simd<float, TB> mean = tot / (float) n;
        simd<float, TB> sc   = rsqrt(mean + eps);

#pragma unroll
        for (int j = 0; j < TPT; ++j) {
            const int c = thr * TPT + j;
            simd<float, 16> wv = block_load<float, 16>(w + c * 16, overaligned_tag<16>{});
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                if constexpr (L2D) {
                    // the tile as one 2D block store per output (rows past T are dropped)
                    simd<float, 128> o;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        // the float order of the norm kernel: scale*x, then *mul
                        simd<float, 16> v = sc[g * 8 + r] * yk[j][g].template select<16, 1>(r * 16);
                        o.template select<16, 1>(r * 16) = v * wv;
                    }
                    if (!no_f32) {
                        store_2d<float, 16, 8>(dst, (unsigned) (n * 4 - 1), (unsigned) (T - 1), (unsigned) (ldd * 4 - 1), c * 16, t0 + g * 8, o);
                    }
                    if (dst_h) {
                        simd<sycl::half, 128> oh = o;
                        store_2d<sycl::half, 16, 8>(dst_h, (unsigned) (n * 2 - 1), (unsigned) (T - 1), (unsigned) (n * 2 - 1), c * 16, t0 + g * 8, oh);
                    }
                } else {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int t = t0 + g * 8 + r;
                        if (t < T) {
                            // the float order of the norm kernel: scale*x, then *mul
                            simd<float, 16> o = sc[g * 8 + r] * yk[j][g].template select<16, 1>(r * 16);
                            o = o * wv;
                            if (!no_f32) {
                                block_store<float, 16>(dst + (size_t) t * ldd + c * 16, o, overaligned_tag<16>{});
                            }
                            if (dst_h) {
                                simd<sycl::half, 16> oh = o;
                                block_store<sycl::half, 16>(dst_h + (size_t) t * n + c * 16, oh, overaligned_tag<16>{});
                            }
                        }
                    }
                }
            }
        }
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<BIG ? 256 : 128> };
    }
};

// the packed copy of the weight rows [2n][ldw] (K = 64): made by ggml_sycl_gate_mix_mm_prepare() before the graph
// runs (or records), looked up while it runs; nullptr when it could not be made (then never retried, so the path a
// weight takes does not depend on when it was first seen)
const uint32_t * gmm_get_weight(ggml_backend_sycl_context & ctx, const ggml_tensor * wt, int64_t n, bool make) {
    const gmm_wkey key{ ctx.device, wt->data, n };
    std::lock_guard<std::mutex> lock(g_gmm_mutex);
    auto it = g_gmm_weights.find(key);
    if (it != g_gmm_weights.end()) {
        return it->second.failed ? nullptr : (const uint32_t *) it->second.data;
    }
    if (!make) {
        return nullptr;
    }
    const ggml_tensor * base = wt->view_src ? wt->view_src : wt;
    gmm_weight & w = g_gmm_weights[key];
    w.buffer = base->buffer;
    w.device = ctx.device;
    w.failed = true;
    queue_ptr q = ctx.stream();
    const int64_t nel = 2 * n * GMM_K;
    sycl::half * data = (sycl::half *) ggml_sycl_malloc_device(nel * sizeof(sycl::half), *q, GGML_SYCL_MEM_BUFFER);
    if (!data) {
        return nullptr;
    }
    const float * src = (const float *) wt->data;
    const int64_t ldw = wt->nb[1] / sizeof(float);
    q->parallel_for(sycl::range<1>((size_t) nel), [=](sycl::id<1> id) {
        const int64_t i   = id[0];
        const int     jj  = (int) (i % 2);
        const int     col = (int) ((i / 2) % 16);
        const int     kp  = (int) ((i / 32) % 8);
        const int     s   = (int) ((i / 256) % GMM_KS);
        const int64_t ch  = i / (256 * GMM_KS); // c*2 + h
        const int64_t c   = ch / 2;
        const int64_t h   = ch % 2;
        const int64_t row = h * n + c * 16 + col;
        const int     k   = s * 16 + 2 * kp + jj;
        data[i] = (sycl::half) src[row * ldw + k];
    });
    q->wait();
    w.data   = data;
    w.failed = false;
    return (const uint32_t *) data;
}

} // namespace

void ggml_sycl_gate_mix_mm_drop_buffer(ggml_backend_buffer_t buffer) {
    std::lock_guard<std::mutex> lock(g_gmm_mutex);
    for (auto it = g_gmm_weights.begin(); it != g_gmm_weights.end();) {
        if (it->second.buffer == buffer) {
            sycl::queue & q = dpct::get_device(it->second.device).default_queue();
            if (it->second.data) {
                q.wait();
                ggml_sycl_free_device(it->second.data, q);
            }
            it = g_gmm_weights.erase(it);
        } else {
            ++it;
        }
    }
}

bool ggml_sycl_gate_mix_mm_match(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!gmm_env() || i + 3 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * mm  = cgraph->nodes[i];
    const ggml_tensor * gm  = cgraph->nodes[i + 1];
    const ggml_tensor * rms = cgraph->nodes[i + 2];
    const ggml_tensor * mul = cgraph->nodes[i + 3];
    if (mm->op != GGML_OP_MUL_MAT || gm->op != GGML_OP_GATE_MIX || rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_tensor * wt = mm->src[0];
    const ggml_tensor * z  = mm->src[1];
    const ggml_tensor * x  = gm->src[0];
    const ggml_tensor * m  = gm->src[1];
    const int64_t n = x->ne[0];
    if (gm->src[2] != mm || ggml_get_op_params_i32(gm, 0) == 0 || rms->src[0] != gm || mm->type != GGML_TYPE_F32) {
        return false;
    }
    if (wt->type != GGML_TYPE_F32 || wt->ne[0] != GMM_K || wt->ne[1] != 2 * n || wt->ne[2] != 1 || wt->ne[3] != 1 ||
        wt->nb[0] != sizeof(float) || wt->nb[1] % sizeof(float) != 0 || !wt->buffer ||
        ggml_backend_buffer_get_usage((wt->view_src ? wt->view_src : wt)->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if ((mul->src[0] != rms && mul->src[1] != rms) || w->type != GGML_TYPE_F32 || w->ne[0] != n || ggml_nrows(w) != 1 ||
        w->nb[0] != sizeof(float) || !ggml_are_same_shape(mul, rms)) {
        return false;
    }
    int64_t Mz, ldz, Mx, ldx, Mm, ldm, Md, ldd;
    if (!gmm_rows(z, Mz, ldz) || !gmm_rows(x, Mx, ldx) || !gmm_rows(m, Mm, ldm) || !gmm_rows(mul, Md, ldd) ||
        z->ne[0] != GMM_K || m->ne[0] != n || mul->ne[0] != n || Mz != Mx || Mm != Mx || Md != Mx || Mx < 16) {
        return false;
    }
    const int64_t nth = n / (16 * gmm_tpt(n));
    if (n % (16 * gmm_tpt(n)) != 0 || nth > GMM_MAX_THREADS || nth < 1) {
        return false;
    }
    if (!gmm_al16(z->data) || !gmm_al16(x->data) || !gmm_al16(m->data) || !gmm_al16(w->data) || !gmm_al16(mul->data) ||
        ldz % 4 || ldx % 4 || ldm % 4 || ldd % 4) {
        return false;
    }
    // (where the allocator put z, x, m and the result does not decide the path: run() copies what overlaps)
    GGML_UNUSED(ctx);
    return true;
}

void ggml_sycl_gate_mix_mm_prepare(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph) {
    if (!gmm_env()) {
        return;
    }
#ifdef GGML_SYCL_GRAPH
    // a queue that is recording cannot be waited on (the callers prepare before they record)
    if (ctx.stream()->ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording) {
        return;
    }
#endif
    for (int i = 0; i + 3 < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT && cgraph->nodes[i + 1]->op == GGML_OP_GATE_MIX &&
            ggml_sycl_gate_mix_mm_match(ctx, cgraph, i)) {
            gmm_get_weight(ctx, cgraph->nodes[i]->src[0], cgraph->nodes[i + 1]->src[0]->ne[0], true);
        }
    }
}

bool ggml_sycl_gate_mix_mm_ready(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) {
    return gmm_get_weight(ctx, cgraph->nodes[i]->src[0], cgraph->nodes[i + 1]->src[0]->ne[0], false) != nullptr;
}

const void * ggml_sycl_gate_mix_mm_z16(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) {
#if GGML_SYCL_DNNL
    const ggml_tensor * z = cgraph->nodes[i]->src[1];
    int64_t M, ld;
    gmm_rows(z, M, ld);
    return ggml_sycl_dense_dnn_act_lookup(ctx, z->data, GMM_K, M, ld);
#else
    GGML_UNUSED(ctx); GGML_UNUSED(cgraph); GGML_UNUSED(i);
    return nullptr;
#endif
}

void ggml_sycl_gate_mix_mm_run(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i, const void * z16,
                               bool f16_for_dense, int pin_until) {
    const ggml_tensor * mm  = cgraph->nodes[i];
    const ggml_tensor * gm  = cgraph->nodes[i + 1];
    const ggml_tensor * rms = cgraph->nodes[i + 2];
    ggml_tensor *       mul = cgraph->nodes[i + 3];
    const ggml_tensor * z   = mm->src[1];
    const ggml_tensor * x   = gm->src[0];
    const ggml_tensor * m   = gm->src[1];
    const ggml_tensor * w   = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t n = x->ne[0];

    int64_t T, ldz, ldx, ldm, ldd;
    gmm_rows(z, T, ldz);
    gmm_rows(x, T, ldx);
    gmm_rows(m, T, ldm);
    gmm_rows(mul, T, ldd);

    const uint32_t * wp = gmm_get_weight(ctx, mm->src[0], n, false);
    GGML_ASSERT(wp);

    sycl::half * dst_h  = nullptr;
    bool         no_f32 = false;
#if GGML_SYCL_DNNL
    if (f16_for_dense) {
        dst_h = ggml_sycl_dense_dnn_act_reserve(ctx, mul, pin_until);
        if (!dst_h && pin_until >= 0) {
            dst_h = ggml_sycl_dense_dnn_act_reserve(ctx, mul);
        } else {
            no_f32 = dst_h && pin_until >= 0;
        }
    }
#else
    GGML_UNUSED(f16_for_dense);
    GGML_UNUSED(pin_until);
#endif
    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));

    // Every work-group reads its tokens' z, x and m rows and writes their results, all of them as one kernel, so a
    // result that the allocator placed over those inputs (their last reader is this chain) is written elsewhere
    // first: z (64 floats a token) is copied out, the result goes through a scratch buffer when it overlaps x or m.
    // The path then never depends on where the tensors are (a fall back to the old path did: it rounds differently).
    queue_ptr stream = ctx.stream();
    const float * zp  = (const float *) z->data;
    float *       dst = (float *) mul->data;
    ggml_sycl_pool_alloc<float> zcopy(ctx.pool());
    ggml_sycl_pool_alloc<float> dcopy(ctx.pool());
    if (!no_f32) {
        if (!z16 && gmm_overlap(zp, T, ldz, GMM_K, dst, T, ldd, n)) {
            float * zc = zcopy.alloc((size_t) T * GMM_K);
            gmm_copy_rows(stream, zc, GMM_K, zp, ldz, GMM_K, T);
            zp  = zc;
            ldz = GMM_K;
        }
        if (gmm_overlap(x->data, T, ldx, n, dst, T, ldd, n) || gmm_overlap(m->data, T, ldm, n, dst, T, ldd, n)) {
            dst = dcopy.alloc((size_t) T * ldd);
        }
    }

    const int tpt = gmm_tpt(n);
    const int nth = (int) (n / (16 * tpt));
    // 2D block messages: 64-byte aligned bases, pitches a multiple of 64 bytes
    const bool l2d = gmm_2d_env() && ((uintptr_t) x->data % 64) == 0 && ((uintptr_t) m->data % 64) == 0 &&
                     ((uintptr_t) dst % 64) == 0 && ((uintptr_t) dst_h % 64) == 0 && (ldx * 4) % 64 == 0 &&
                     (ldm * 4) % 64 == 0 && (ldd * 4) % 64 == 0 && (n * 2) % 64 == 0 && T < (1 << 24);
    auto launch = [&](auto tptc, auto l2c) {
        constexpr int  TPT = decltype(tptc)::value;
        constexpr bool L2  = decltype(l2c)::value;
        using K = gate_mix_mm_kernel<TPT, 1, (TPT > 3), L2>;
        K k{ zp, ldz, (const sycl::half *) z16, wp, (const float *) x->data, ldx,
             (const float *) m->data, ldm, (const float *) w->data, dst, ldd, dst_h,
             (int) n, (int) T, eps, no_f32 };
        const size_t n_wg = (size_t) ((T + K::TB - 1) / K::TB);
        stream->parallel_for(sycl::nd_range<1>(n_wg * nth, (size_t) nth), k);
    };
    using T3 = std::integral_constant<int, 3>;
    using T9 = std::integral_constant<int, 9>;
    if (tpt == 9) {
        l2d ? launch(T9{}, std::true_type{}) : launch(T9{}, std::false_type{});
    } else {
        l2d ? launch(T3{}, std::true_type{}) : launch(T3{}, std::false_type{});
    }
    if (dst != (float *) mul->data) {
        gmm_copy_rows(stream, (float *) mul->data, ldd, dst, ldd, n, T);
    }
}
