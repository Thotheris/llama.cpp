// GGML_OP_REL_BIAS: the relative-position attention bias (ggml_rel_bias), per token i, head h and kv cell j
//   v = [l2_clip(r[:, h, i], s[:, h], threshold)*tau[i]; t[h, i]; 1]
//   y[j, i % n_tps, h, i / n_tps] = dot(table[:, pos[j, i]], v)      (pos NULL: pos[j, i] = j)
// The rounding follows ggml_compute_forward_rel_bias (ggml-cpu/ops.cpp): the clip, then the MUL by tau, then one
// f32 dot of n_rel + 2 terms in k order (the CPU sums the squares of the clip in double, here in f32).
//
// One work-group of WG threads handles a block of WG kv cells for TT consecutive tokens: the v of the TT tokens
// and all heads go to local memory first, then each thread reads the table column of its cell once (in registers
// for n_rel == 16) and writes its cell of every (token, head) row, so a row of the output is written by
// consecutive work-items.

#include "rel-bias.hpp"

#include <cstdio>

namespace {

constexpr int REL_BIAS_WG = 256; // kv cells per work-group
constexpr int REL_BIAS_TT = 8;   // tokens per work-group

struct rel_bias_params {
    const char * r;     int64_t nbr1, nbr2;
    const char * s;     int64_t nbs1;
    const char * t;     int64_t nbt1, nbt2;
    const char * tau;   int64_t nbu2;
    const char * table; int64_t nbc1;
    const char * pos;   int64_t nbp1;
    char       * dst;   int64_t nbd1, nbd2, nbd3; // in elements of the dst type
    int   n_rel, n_head, n_tokens, n_cols, n_kv, n_tps, nvp, n_blk;
    float threshold;
};

template <typename T> static inline T to_dst(float v);
template <> inline float      to_dst<float>(float v)      { return v; }
template <> inline sycl::half to_dst<sycl::half>(float v) { return sycl::half(v); }

// v of (token i, head h) into y[0 .. nvp): [l2_clip(r*s)*tau; t; 1; 0 ...] (rel_bias_kernel and the decode pack of
// a flash attention that computes the bias itself: the same arithmetic)
static inline void rel_bias_v(const rel_bias_params & p, const int n_rel, const int64_t i, const int h, float * y) {
    const float * x  = (const float *) (p.r + h*p.nbr1 + i*p.nbr2);
    const float * sc = p.s ? (const float *) (p.s + h*p.nbs1) : nullptr;

    float sum = 0.0f;
    for (int k = 0; k < n_rel; ++k) {
        const float xk = sc ? x[k]*sc[k] : x[k];
        y[k] = xk;
        sum += xk*xk;
    }
    const float norm   = sycl::sqrt(sum) + 1e-6f;
    const float factor = sycl::clamp(p.threshold / norm, 0.0f, 1.0f);
    const float tau_i  = p.tau ? *(const float *) (p.tau + i*p.nbu2) : 1.0f;
    for (int k = 0; k < n_rel; ++k) {
        float yk = y[k]*factor;
        if (p.tau) {
            yk *= tau_i;
        }
        y[k] = yk;
    }
    y[n_rel]     = *(const float *) (p.t + h*p.nbt1 + i*p.nbt2);
    y[n_rel + 1] = 1.0f;
    for (int k = n_rel + 2; k < p.nvp; ++k) {
        y[k] = 0.0f;
    }
}

// NREL > 0: n_rel as a compile-time constant (the table column in registers), 0: any n_rel
template <int NREL, typename T>
static void rel_bias_kernel(const sycl::nd_item<1> & it, float * sv, const rel_bias_params p) {
    const int    tid  = (int) it.get_local_id(0);
    const size_t g    = it.get_group(0);
    const int    blk  = (int) (g % p.n_blk);
    const int    i0   = (int) (g / p.n_blk) * REL_BIAS_TT;
    const int    ntt  = sycl::min(REL_BIAS_TT, p.n_tokens - i0);
    const int    n_rel = NREL > 0 ? NREL : p.n_rel;
    const int    nv    = n_rel + 2;

    // v of (token i0 + tt, head h) at sv[(tt*n_head + h)*nvp]
    for (int task = tid; task < ntt*p.n_head; task += REL_BIAS_WG) {
        const int tt = task / p.n_head;
        const int h  = task % p.n_head;
        rel_bias_v(p, n_rel, i0 + tt, h, sv + (size_t) task*p.nvp);
    }

    it.barrier(sycl::access::fence_space::local_space);

    const int j = blk*REL_BIAS_WG + tid;
    if (j >= p.n_kv) {
        return;
    }

    constexpr int NC = NREL > 0 ? NREL + 2 : 1;
    [[maybe_unused]] float c[NC]; // the table column (NREL > 0), loaded once for all tokens when pos is NULL
    for (int tt = 0; tt < ntt; ++tt) {
        const int64_t i  = i0 + tt;
        int d = p.pos ? ((const int32_t *) (p.pos + i*p.nbp1))[j] : j;
        // the CPU op asserts 0 <= d < n_cols; a bad index must not read outside the table
        d = sycl::clamp(d, 0, p.n_cols - 1);
        const float * col = (const float *) (p.table + (int64_t) d*p.nbc1);

        const int64_t ti = i % p.n_tps;
        const int64_t si = i / p.n_tps;
        T * out = (T *) p.dst + j + ti*p.nbd1 + si*p.nbd3;

        if constexpr (NREL > 0) {
            constexpr int NV = NREL + 2;
            if (tt == 0 || p.pos) {
#pragma unroll
                for (int k = 0; k < NV; ++k) {
                    c[k] = col[k];
                }
            }
            for (int h = 0; h < p.n_head; ++h) {
                const float * v = sv + (size_t) (tt*p.n_head + h)*p.nvp;
                out[h*p.nbd2] = to_dst<T>(ggml_sycl_rel_bias_dot<NV>(c, v));
            }
        } else {
            for (int h = 0; h < p.n_head; ++h) {
                const float * v = sv + (size_t) (tt*p.n_head + h)*p.nvp;
                float acc = 0.0f;
                for (int k = 0; k < nv; ++k) {
                    acc += col[k]*v[k];
                }
                out[h*p.nbd2] = to_dst<T>(acc);
            }
        }
    }
}

template <int NREL, typename T>
static void rel_bias_launch(sycl::queue * q, const rel_bias_params & p) {
    const size_t n_groups = (size_t) p.n_blk * ((p.n_tokens + REL_BIAS_TT - 1) / REL_BIAS_TT);
    const size_t n_local  = (size_t) REL_BIAS_TT * p.n_head * p.nvp;
    q->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sv(sycl::range<1>(n_local), cgh);
        cgh.parallel_for(sycl::nd_range<1>(n_groups*REL_BIAS_WG, REL_BIAS_WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                rel_bias_kernel<NREL, T>(it, sv.get_multi_ptr<sycl::access::decorated::no>().get(), p);
            });
    });
}

// Tiled form for an f16 output with dense rows (nbd1 == n_kv), pos NULL and n_rel == 16 (the prefill of the
// model): the rows of TT consecutive tokens of one head are one contiguous region (TT*n_kv halves, 16-byte
// aligned when the head and sequence strides are), built in local memory and copied out with 16-byte stores
// (a row of 1026 halves starts at a different offset in a 64-byte line for every token). One work-group per
// (head, TT tokens); each work-item computes CPI consecutive cells per pass, the group is at most 256 wide
// (a wider one lets fewer groups share an Xe core, which measured slower) and the v of token tt is read from
// local memory by lane k = component k, then broadcast over the sub-group.
// The v and the dots are those of rel_bias_kernel<16, half>: the same values in the same order (same bits).
constexpr int RT_VP = 20;  // padded v (n_rel + 2 = 18)
constexpr int RT_TT = 8;   // tokens per work-group

template <int TT, int CPI>
static void rel_bias_tiled_kernel(const sycl::nd_item<1> & it, float * sv, sycl::half * tile, const rel_bias_params p) {
    constexpr int NREL = 16;
    constexpr int NV   = NREL + 2;
    const int tid = (int) it.get_local_id(0);
    const int WG  = (int) it.get_local_range(0);
    const int g   = (int) it.get_group(0);
    const int h   = g % p.n_head;
    const int i0  = (g / p.n_head) * TT;
    const int ntt = sycl::min(TT, p.n_tokens - i0);

    if (tid < ntt) {
        const int64_t i  = i0 + tid;
        const float * x  = (const float *) (p.r + h*p.nbr1 + i*p.nbr2);
        const float * sc = p.s ? (const float *) (p.s + h*p.nbs1) : nullptr;
        float * y = sv + tid*RT_VP;

        float sum = 0.0f;
        for (int k = 0; k < NREL; ++k) {
            const float xk = sc ? x[k]*sc[k] : x[k];
            y[k] = xk;
            sum += xk*xk;
        }
        const float norm   = sycl::sqrt(sum) + 1e-6f;
        const float factor = sycl::clamp(p.threshold / norm, 0.0f, 1.0f);
        const float tau_i  = p.tau ? *(const float *) (p.tau + i*p.nbu2) : 1.0f;
        for (int k = 0; k < NREL; ++k) {
            float yk = y[k]*factor;
            if (p.tau) {
                yk *= tau_i;
            }
            y[k] = yk;
        }
        y[NREL]     = *(const float *) (p.t + h*p.nbt1 + i*p.nbt2);
        y[NREL + 1] = 1.0f;
    }

    it.barrier(sycl::access::fence_space::local_space);

    const auto sg   = it.get_sub_group();
    const int  lane = (int) sg.get_local_linear_id();
    // the bound of the pass loop is the same for the whole group: every work-item of a sub-group stays in it for
    // the broadcasts, the ones past the row only skip their stores
    for (int jb = 0; jb < p.n_kv; jb += WG*CPI) {
    const int j0 = jb + tid*CPI;
    float c[CPI][NV];
#pragma unroll
    for (int q = 0; q < CPI; ++q) {
        const float * col = (const float *) (p.table + (int64_t) sycl::min(j0 + q, p.n_cols - 1)*p.nbc1);
#pragma unroll
        for (int k = 0; k < NV; ++k) {
            c[q][k] = col[k];
        }
    }
    for (int tt = 0; tt < ntt; ++tt) {
        // v[k] from lane k of the sub-group: two 64-byte local loads per token
        const float va = sv[tt*RT_VP + lane];
        const float vb = sv[tt*RT_VP + WARP_SIZE + (lane & 1)];
        float a[CPI];
#pragma unroll
        for (int q = 0; q < CPI; ++q) {
            a[q] = 0.0f;
        }
#pragma unroll
        for (int k = 0; k < NV; ++k) {
            const float vk = k < WARP_SIZE ? sycl::group_broadcast(sg, va, k) : sycl::group_broadcast(sg, vb, k - WARP_SIZE);
#pragma unroll
            for (int q = 0; q < CPI; ++q) {
                a[q] += c[q][k]*vk;
            }
        }
        sycl::half * o = tile + tt*p.n_kv + j0;
#pragma unroll
        for (int q = 0; q < CPI; ++q) {
            if (j0 + q < p.n_kv) {
                o[q] = to_dst<sycl::half>(a[q]);
            }
        }
    }
    }

    it.barrier(sycl::access::fence_space::local_space);

    const int64_t ti = i0 % p.n_tps;
    const int64_t si = i0 / p.n_tps;
    sycl::half * out = (sycl::half *) p.dst + h*p.nbd2 + si*p.nbd3 + ti*p.nbd1;
    const int n_el  = ntt*p.n_kv;
    const int n_u4  = n_el / 8;
    const sycl::uint4 * t4 = (const sycl::uint4 *) tile;
    sycl::uint4 *       o4 = (sycl::uint4 *) out;
    for (int u = tid; u < n_u4; u += WG) {
        o4[u] = t4[u];
    }
    for (int e = 8*n_u4 + tid; e < n_el; e += WG) {
        out[e] = tile[e];
    }
}

template <int TT, int CPI>
static void rel_bias_tiled_launch(sycl::queue * q, const rel_bias_params & p) {
    // the fewest passes over the row that keep the group at most 256 wide (several groups per Xe core)
    const int    ncu      = (p.n_kv + CPI - 1) / CPI;
    const int    passes   = (ncu + 255) / 256;
    const int    WG       = (int) GGML_PAD((ncu + passes - 1) / passes, WARP_SIZE);
    const size_t n_groups = (size_t) p.n_head * ((p.n_tokens + TT - 1) / TT);
    const size_t n_units  = ((size_t) TT*p.n_kv*sizeof(sycl::half) + 15) / 16;
    q->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::float4, 1> sv(sycl::range<1>(TT*RT_VP/4), cgh);
        sycl::local_accessor<sycl::uint4, 1>  tile(sycl::range<1>(n_units), cgh);
        cgh.parallel_for(sycl::nd_range<1>(n_groups*WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                rel_bias_tiled_kernel<TT, CPI>(it, (float *) sv.get_multi_ptr<sycl::access::decorated::no>().get(),
                                               (sycl::half *) tile.get_multi_ptr<sycl::access::decorated::no>().get(), p);
            });
    });
}

// the tiled kernel applies: see above; a block of tokens never spans two sequences
static bool rel_bias_tiled_ok(const rel_bias_params & p, int64_t ne3) {
    static const bool disabled = getenv("GGML_SYCL_REL_BIAS_TILED") && atoi(getenv("GGML_SYCL_REL_BIAS_TILED")) == 0;
    const size_t a = 16 / sizeof(sycl::half);
    return !disabled && p.n_rel == 16 && !p.pos && p.nbd1 == p.n_kv && ((uintptr_t) p.dst % 16) == 0 &&
           p.nbd2 % a == 0 && p.nbd3 % a == 0 && (ne3 == 1 || p.n_tps % RT_TT == 0) &&
           ((int64_t) RT_TT*p.n_kv) % a == 0 && RT_TT*p.n_kv*sizeof(sycl::half) <= 32*1024 && p.n_tokens >= RT_TT;
}

// 8 tokens and 4 cells per work-item measured best on the B70 (n_kv 1026: 144 work-items, 2 passes, 309 us
// for 2048 tokens x 24 heads against 400 us for rel_bias_kernel); groups wider than 256 cost Xe-core occupancy
static void rel_bias_tiled(sycl::queue * q, const rel_bias_params & p) {
    rel_bias_tiled_launch<RT_TT, 4>(q, p);
}

} // namespace

// the local memory holds the v of REL_BIAS_TT tokens and all heads
static int rel_bias_nvp(int64_t n_rel) {
    return (int) ((n_rel + 2 + 3) & ~3);
}

bool ggml_sycl_rel_bias_supported(const ggml_tensor * op) {
    const ggml_tensor * r     = op->src[0];
    const ggml_tensor * s     = op->src[1];
    const ggml_tensor * t     = op->src[2];
    const ggml_tensor * table = op->src[3];
    const ggml_tensor * pos   = op->src[4];
    const ggml_tensor * tau   = op->src[5];

    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) {
        return false;
    }
    if (r->type != GGML_TYPE_F32 || t->type != GGML_TYPE_F32 || table->type != GGML_TYPE_F32 ||
        (s && s->type != GGML_TYPE_F32) || (tau && tau->type != GGML_TYPE_F32) || (pos && pos->type != GGML_TYPE_I32)) {
        return false;
    }
    // element strides of the output, float rows of the inputs (the asserts of ggml_rel_bias)
    if (op->nb[0] != ggml_type_size(op->type) || op->nb[1] % op->nb[0] || op->nb[2] % op->nb[0] || op->nb[3] % op->nb[0]) {
        return false;
    }
    if (r->nb[0] != sizeof(float) || t->nb[0] != sizeof(float) || table->nb[0] != sizeof(float) || (s && s->nb[0] != sizeof(float)) ||
        (pos && pos->nb[0] != sizeof(int32_t))) {
        return false;
    }
    // the kernel indexes kv cells, tokens and table columns with int
    if (op->ne[0] > INT32_MAX || r->ne[2] > INT32_MAX || table->ne[1] > INT32_MAX) {
        return false;
    }
    // local memory: REL_BIAS_TT tokens x n_head x the padded v
    return (int64_t) REL_BIAS_TT * r->ne[1] * rel_bias_nvp(r->ne[0]) * (int64_t) sizeof(float) <= 32*1024;
}

static rel_bias_params rel_bias_make_params(const ggml_tensor * dst) {
    const ggml_tensor * r     = dst->src[0];
    const ggml_tensor * s     = dst->src[1];
    const ggml_tensor * t     = dst->src[2];
    const ggml_tensor * table = dst->src[3];
    const ggml_tensor * pos   = dst->src[4];
    const ggml_tensor * tau   = dst->src[5];

    const size_t td = ggml_type_size(dst->type);

    rel_bias_params p;
    p.r     = (const char *) r->data;     p.nbr1 = r->nb[1]; p.nbr2 = r->nb[2];
    p.s     = s ? (const char *) s->data : nullptr; p.nbs1 = s ? s->nb[1] : 0;
    p.t     = (const char *) t->data;     p.nbt1 = t->nb[1]; p.nbt2 = t->nb[2];
    p.tau   = tau ? (const char *) tau->data : nullptr; p.nbu2 = tau ? tau->nb[2] : 0;
    p.table = (const char *) table->data; p.nbc1 = table->nb[1];
    p.pos   = pos ? (const char *) pos->data : nullptr; p.nbp1 = pos ? pos->nb[1] : 0;
    p.dst   = (char *) dst->data;
    p.nbd1  = dst->nb[1]/td; p.nbd2 = dst->nb[2]/td; p.nbd3 = dst->nb[3]/td;
    p.n_rel    = (int) r->ne[0];
    p.n_head   = (int) r->ne[1];
    p.n_tokens = (int) r->ne[2];
    p.n_cols   = (int) table->ne[1];
    p.n_kv     = (int) dst->ne[0];
    p.n_tps    = (int) dst->ne[1];
    p.nvp      = rel_bias_nvp(r->ne[0]);
    p.n_blk    = (p.n_kv + REL_BIAS_WG - 1) / REL_BIAS_WG;
    p.threshold = ggml_get_op_params_f32(dst, 0);

    return p;
}

void ggml_sycl_rel_bias(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);

    const ggml_tensor *   r = dst->src[0];
    const rel_bias_params p = rel_bias_make_params(dst);

    GGML_ASSERT(dst->ne[2] == r->ne[1] && (int64_t) p.n_tps*dst->ne[3] == r->ne[2]);
    if (p.n_kv == 0 || p.n_tokens == 0 || p.n_head == 0) {
        return;
    }

    try {
        sycl::queue * q = ctx.stream();
        const bool f16 = dst->type == GGML_TYPE_F16;
        if (f16 && rel_bias_tiled_ok(p, dst->ne[3])) {
            rel_bias_tiled(q, p);
        } else if (p.n_rel == 16) {
            f16 ? rel_bias_launch<16, sycl::half>(q, p) : rel_bias_launch<16, float>(q, p);
        } else {
            f16 ? rel_bias_launch<0, sycl::half>(q, p) : rel_bias_launch<0, float>(q, p);
        }
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "[SYCL-REL_BIAS] ERROR: %s\n", e.what());
        throw;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the packed inputs of a flash attention that computes the bias in its kernel (see rel-bias.hpp)

#include "fattn-dpas.hpp"

#include <unordered_map>

bool ggml_sycl_rel_bias_fa_pack_layout(const ggml_tensor * rb, int masked_pos, ggml_sycl_rel_pack_layout & L) {
    const ggml_tensor * r     = rb->src[0];
    const ggml_tensor * table = rb->src[3];
    if (rb->op != GGML_OP_REL_BIAS || rb->src[4] != nullptr || rb->type != GGML_TYPE_F16 || r->ne[0] != 16 ||
        ((uintptr_t) rb->data % 16) != 0 || table->ne[1] > INT32_MAX / 2 || r->ne[1] * r->ne[2] > INT32_MAX / 4) {
        return false;
    }
    const int n_cols = (int) table->ne[1];
    const int c      = masked_pos - 1;
    if (c < 0 || c >= n_cols) {
        return false;
    }
    auto up = [](size_t x) { return (x + 255) & ~(size_t) 255; };
    L.c      = c;
    L.n_cols = n_cols;
    L.hdr    = 0;
    L.v      = 256;
    L.t2     = up(L.v  + (size_t) r->ne[1] * r->ne[2] * 48);
    L.x2     = up(L.t2 + (size_t) (c + 9) * 8 * 16);
    L.tf     = up(L.x2 + (size_t) (c + 9) * 2 * 16);
    L.tx     = up(L.tf + (size_t) n_cols * 32);
    L.end    = L.tx + (size_t) n_cols * 8;
    return L.end <= ggml_nbytes(rb);
}

static inline uint32_t rel_pack_h2(float a, float b) {
    return (uint32_t) sycl::bit_cast<uint16_t>(sycl::half(a)) | ((uint32_t) sycl::bit_cast<uint16_t>(sycl::half(b)) << 16);
}

void ggml_sycl_rel_bias_fa_pack(ggml_backend_sycl_context & ctx, const ggml_tensor * rb, const ggml_sycl_rel_pack_layout & L) {
    const ggml_tensor * r     = rb->src[0];
    const ggml_tensor * s     = rb->src[1];
    const ggml_tensor * t     = rb->src[2];
    const ggml_tensor * table = rb->src[3];
    const ggml_tensor * tau   = rb->src[5];

    constexpr int NREL = 16;
    const char * pr    = (const char *) r->data;     const int64_t nbr1 = r->nb[1], nbr2 = r->nb[2];
    const char * ps    = s ? (const char *) s->data : nullptr; const int64_t nbs1 = s ? s->nb[1] : 0;
    const char * pt    = (const char *) t->data;     const int64_t nbt1 = t->nb[1], nbt2 = t->nb[2];
    const char * pu    = tau ? (const char *) tau->data : nullptr; const int64_t nbu2 = tau ? tau->nb[2] : 0;
    const char * pc    = (const char *) table->data; const int64_t nbc1 = table->nb[1];
    const float  thr   = ggml_get_op_params_f32(rb, 0);
    const int    n_head = (int) r->ne[1];
    const int    nv     = (int) (r->ne[1] * r->ne[2]);
    const int    c      = L.c;
    const int    n_cols = L.n_cols;
    const int    nt2    = c + 9;
    char *       base   = (char *) rb->data;
    const ggml_sycl_rel_pack_layout Lc = L;

    constexpr int WG = 256;
    const int     n_items  = nv + nt2 + n_cols;
    const size_t  n_groups = (size_t) (n_items + WG - 1) / WG;
    ctx.stream()->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_groups * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int gid = (int) it.get_global_linear_id();
            auto col = [&](int j) { return (const float *) (pc + (int64_t) j * nbc1); };
            if (it.get_group(0) == 0) {
                // the header: are rows NREL, NREL + 1 the same in every column up to c
                const float x16 = col(0)[NREL];
                const float x17 = col(0)[NREL + 1];
                bool        same = true;
                for (int j = (int) it.get_local_id(0); j <= c; j += WG) {
                    same = same && col(j)[NREL] == x16 && col(j)[NREL + 1] == x17;
                }
                same = sycl::all_of_group(it.get_group(), same);
                if (it.get_local_id(0) == 0) {
                    int * h = (int *) (base + Lc.hdr);
                    h[0] = same ? 1 : 0;
                    ((float *) h)[1] = x16;
                    ((float *) h)[2] = x17;
                    h[3] = c;
                }
            }
            if (gid < nv) {
                // v of (token i, head h): the clip and the tau of rel_bias_kernel, in the same order
                const int     h  = gid % n_head;
                const int64_t i  = gid / n_head;
                const float * x  = (const float *) (pr + h * nbr1 + i * nbr2);
                const float * sc = ps ? (const float *) (ps + h * nbs1) : nullptr;
                float y[NREL];
                float sum = 0.0f;
#pragma unroll
                for (int k = 0; k < NREL; ++k) {
                    const float xk = sc ? x[k] * sc[k] : x[k];
                    y[k] = xk;
                    sum += xk * xk;
                }
                const float norm   = sycl::sqrt(sum) + 1e-6f;
                const float factor = sycl::clamp(thr / norm, 0.0f, 1.0f);
                const float tau_i  = pu ? *(const float *) (pu + i * nbu2) : 1.0f;
#pragma unroll
                for (int k = 0; k < NREL; ++k) {
                    float yk = y[k] * factor;
                    if (pu) {
                        yk *= tau_i;
                    }
                    y[k] = yk;
                }
                const float tv  = *(const float *) (pt + h * nbt1 + i * nbt2);
                const float thi = static_cast<float>(sycl::half(tv));
                sycl::uint4 * o = (sycl::uint4 *) (base + Lc.v) + (size_t) gid * 3;
                o[0] = sycl::uint4(rel_pack_h2(y[0], y[1]), rel_pack_h2(y[2], y[3]), rel_pack_h2(y[4], y[5]), rel_pack_h2(y[6], y[7]));
                o[1] = sycl::uint4(rel_pack_h2(y[8], y[9]), rel_pack_h2(y[10], y[11]), rel_pack_h2(y[12], y[13]), rel_pack_h2(y[14], y[15]));
                o[2] = sycl::uint4(rel_pack_h2(thi, tv - thi), sycl::bit_cast<uint32_t>(tv), 0u, 0u);
            } else if (gid < nv + nt2) {
                // the A operands of the distances d .. d + 7 (rows of the DPAS: d + 2 g for lanes 0..7)
                const int d = gid - nv - 8;
                sycl::uint4 * o2 = (sycl::uint4 *) (base + Lc.t2) + (size_t) (gid - nv) * 8;
                sycl::uint4 * ox = (sycl::uint4 *) (base + Lc.x2) + (size_t) (gid - nv) * 2;
                const float * cg[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    cg[g] = col(sycl::clamp(d + 2 * g, 0, c));
                }
#pragma unroll
                for (int cp = 0; cp < 8; ++cp) {
                    o2[cp] = sycl::uint4(rel_pack_h2(cg[0][2 * cp], cg[0][2 * cp + 1]), rel_pack_h2(cg[1][2 * cp], cg[1][2 * cp + 1]),
                                         rel_pack_h2(cg[2][2 * cp], cg[2][2 * cp + 1]), rel_pack_h2(cg[3][2 * cp], cg[3][2 * cp + 1]));
                }
                ox[0] = sycl::uint4(rel_pack_h2(cg[0][NREL], cg[0][NREL]), rel_pack_h2(cg[1][NREL], cg[1][NREL]),
                                    rel_pack_h2(cg[2][NREL], cg[2][NREL]), rel_pack_h2(cg[3][NREL], cg[3][NREL]));
                ox[1] = sycl::uint4(rel_pack_h2(cg[0][NREL + 1], 0.0f), rel_pack_h2(cg[1][NREL + 1], 0.0f),
                                    rel_pack_h2(cg[2][NREL + 1], 0.0f), rel_pack_h2(cg[3][NREL + 1], 0.0f));
            } else if (gid < n_items) {
                const int     j  = gid - nv - nt2;
                const float * cj = col(j);
                sycl::uint4 * of = (sycl::uint4 *) (base + Lc.tf) + (size_t) j * 2;
                of[0] = sycl::uint4(rel_pack_h2(cj[0], cj[1]), rel_pack_h2(cj[2], cj[3]), rel_pack_h2(cj[4], cj[5]), rel_pack_h2(cj[6], cj[7]));
                of[1] = sycl::uint4(rel_pack_h2(cj[8], cj[9]), rel_pack_h2(cj[10], cj[11]), rel_pack_h2(cj[12], cj[13]), rel_pack_h2(cj[14], cj[15]));
                ((sycl::float2 *) (base + Lc.tx))[j] = sycl::float2(cj[NREL], cj[NREL + 1]);
            }
        });
    });
}

// the decode form (pos set): the v of every (token, head) and the clamped pos rows, for fattn_dec_kernel
bool ggml_sycl_rel_bias_fa_dec_layout(const ggml_tensor * rb, ggml_sycl_rel_dec_layout & L) {
    const ggml_tensor * r     = rb->src[0];
    const ggml_tensor * table = rb->src[3];
    const ggml_tensor * pos   = rb->src[4];
    if (rb->op != GGML_OP_REL_BIAS || pos == nullptr || rb->type != GGML_TYPE_F16 || r->ne[0] != 16 ||
        ((uintptr_t) rb->data % 16) != 0 || !ggml_is_contiguous(rb) || rb->ne[0] > INT32_MAX / 4 ||
        r->ne[1] * r->ne[2] > INT32_MAX / 32 || table->nb[1] % sizeof(float) != 0) {
        return false;
    }
    // the table is read by the flash attention, after the node: it must outlive the graph compute
    if (table->buffer == nullptr || ggml_backend_buffer_get_usage(table->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    auto up = [](size_t x) { return (x + 255) & ~(size_t) 255; };
    L.nvp    = rel_bias_nvp(r->ne[0]);
    L.n_kv   = (int) rb->ne[0];
    L.n_head = (int) r->ne[1];
    L.v      = 0;
    L.pos    = up((size_t) r->ne[1] * r->ne[2] * L.nvp * sizeof(float));
    L.end    = L.pos + (size_t) r->ne[2] * rb->ne[0] * sizeof(int32_t);
    return L.end <= ggml_nbytes(rb);
}

static void rel_bias_fa_dec_pack(ggml_backend_sycl_context & ctx, const ggml_tensor * rb, const ggml_sycl_rel_dec_layout & L) {
    const rel_bias_params p      = rel_bias_make_params(rb);
    const int             nv     = p.n_head * p.n_tokens;
    const int             n_kv   = L.n_kv;
    const int             n_pos  = p.n_tokens * n_kv;
    float *               vo     = (float *) ((char *) rb->data + L.v);
    int32_t *             po     = (int32_t *) ((char *) rb->data + L.pos);
    constexpr int         WG     = 256;
    const size_t          n_grp  = (size_t) (nv + n_pos + WG - 1) / WG;
    GGML_ASSERT(p.n_rel == 16 && L.nvp == p.nvp);
    ctx.stream()->parallel_for(sycl::nd_range<1>(n_grp * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        const int gid = (int) it.get_global_linear_id();
        if (gid < nv) {
            // the v of rel_bias_kernel: the same task order (token, head) and the same arithmetic
            rel_bias_v(p, 16, gid / p.n_head, gid % p.n_head, vo + (size_t) gid * p.nvp);
        } else if (gid < nv + n_pos) {
            const int     q = gid - nv;
            const int64_t i = q / n_kv;
            const int     j = q % n_kv;
            po[q] = sycl::clamp(((const int32_t *) (p.pos + i * p.nbp1))[j], 0, p.n_cols - 1);
        }
    });
}

// the REL_BIAS nodes packed in this thread's graph compute, with the flash attention that reads each
static thread_local std::unordered_map<const ggml_tensor *, const ggml_tensor *> g_rel_bias_fa_skipped;

// the readers of the REL_BIAS nodes of a graph (count, last reader), built once per graph compute: the use counts of
// ggml-impl.h are not set in the graphs the scheduler hands over
struct rel_bias_readers {
    const ggml_cgraph *    g     = nullptr;
    ggml_tensor * const *  nodes = nullptr;
    int                    n     = -1;
    int                    last  = INT32_MAX;
    std::unordered_map<const ggml_tensor *, std::pair<int, const ggml_tensor *>> r;
};
static thread_local rel_bias_readers g_rel_bias_readers;

bool ggml_sycl_rel_bias_fa_pack_node(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool en_pf  = !(getenv("GGML_SYCL_FA_REL_INKERNEL") && atoi(getenv("GGML_SYCL_FA_REL_INKERNEL")) == 0);
    static const bool en_dec = !(getenv("GGML_SYCL_FA_REL_DEC") && atoi(getenv("GGML_SYCL_FA_REL_DEC")) == 0);
    const ggml_tensor * rb = cgraph->nodes[i];
    if (rb->op != GGML_OP_REL_BIAS || rb->view_src || (rb->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        !(rb->src[4] ? en_dec : en_pf)) {
        return false;
    }
    rel_bias_readers & R = g_rel_bias_readers;
    if (R.g != cgraph || R.nodes != cgraph->nodes || R.n != cgraph->n_nodes || i <= R.last) {
        R.g     = cgraph;
        R.nodes = cgraph->nodes;
        R.n     = cgraph->n_nodes;
        R.r.clear();
        for (int j = 0; j < cgraph->n_nodes; ++j) {
            if (cgraph->nodes[j]->op == GGML_OP_REL_BIAS) {
                R.r[cgraph->nodes[j]] = { 0, nullptr };
            }
        }
        for (int j = 0; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                const ggml_tensor * s = n->src[k];
                if (s == nullptr) {
                    continue;
                }
                if (s->view_src) {
                    s = s->view_src;
                }
                if (s->op == GGML_OP_REL_BIAS) {
                    auto it = R.r.find(s);
                    if (it != R.r.end()) {
                        it->second.first++;
                        it->second.second = n;
                    }
                }
            }
            // a view of a REL_BIAS node is read through the view
            if (n->view_src && n->view_src->op == GGML_OP_REL_BIAS) {
                auto it = R.r.find(n->view_src);
                if (it != R.r.end()) {
                    it->second.first += 2;
                }
            }
        }
    }
    R.last = i;

    // exactly one reader, in this graph (a split of the scheduler may hold only part of the full graph)
    const auto it = R.r.find(rb);
    if (it == R.r.end() || it->second.first != 1) {
        return false;
    }
    const ggml_tensor * fa = it->second.second;
    if (fa->op != GGML_OP_FLASH_ATTN_EXT || (fa->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    if (rb->src[4]) {
        // decode: fattn_dec_kernel reads the v and the pos rows packed here and the table
        ggml_sycl_rel_dec_layout L;
        if (!ggml_sycl_flash_attn_ext_dpas_dec_rel_ok(fa, rb) || !ggml_sycl_rel_bias_fa_dec_layout(rb, L)) {
            return false;
        }
        rel_bias_fa_dec_pack(ctx, rb, L);
        g_rel_bias_fa_skipped[rb] = fa;
        return true;
    }
    if (!ggml_sycl_flash_attn_ext_dpas_rel_inkernel_ok(fa, rb)) {
        return false;
    }
    // the pack reads the inputs of the node here: the graph allocator may reuse their memory before fa runs
    ggml_sycl_rel_pack_layout L;
    if (!ggml_sycl_rel_bias_fa_pack_layout(rb, ggml_get_op_params_i32(fa, 7), L)) {
        return false;
    }
    ggml_sycl_rel_bias_fa_pack(ctx, rb, L);
    g_rel_bias_fa_skipped[rb] = fa;
    return true;
}

bool ggml_sycl_rel_bias_fa_take(const ggml_tensor * rb, const ggml_tensor * fa) {
    auto it = g_rel_bias_fa_skipped.find(rb);
    if (it == g_rel_bias_fa_skipped.end() || it->second != fa) {
        return false;
    }
    g_rel_bias_fa_skipped.erase(it);
    return true;
}
