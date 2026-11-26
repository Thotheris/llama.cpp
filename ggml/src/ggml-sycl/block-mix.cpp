// GGML_OP_BLOCK_MIX: attention over a stack of states (ggml_block_mix), per token t and query q with the states
// s = [stack rows..., cur] of t:
//   w = softmax_k(scale*dot(s_k, query_q)), y[:, q, t] = sum_k w_k*s_k
// and ggml_block_mix_norm (op_params[1] flags): 1 the state cur is rms_norm(cur, eps_in), 2 y is replaced by
// rms_norm(y, eps_out) (times out_w = src[3] when given), 4 the raw y is kept as plane 1 of the result.
// The rounding follows ggml_compute_forward_block_mix (ggml-cpu/ops.cpp): f32 softmax, the weighted sum in k order;
// the dots and the norms are f32 sums of per-lane partials (double sums on the CPU), and the dot of the normed cur
// is the norm scale times the dot of the raw cur.
//
// One work-group per token and group of up to BM_QMAX queries (a decode step is a single work-group, so the
// kernel is latency bound there). Each work-item owns units e = tid + r*W of the rows (float4 when the rows allow
// it): it keeps its query units in registers, reads each state unit once for all the queries, and the dots are
// reduced per sub-group, then the sub-group partials are added in order by one work-item per (state, query).
// The softmax runs on one work-item per query; the weighted sum keeps the raw y in registers for the norm.

#include "block-mix.hpp"
#if GGML_SYCL_DNNL
#include "dense-dnn.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int BM_QMAX       = 4; // queries per work-group
constexpr int BM_MAX_STATES = 32;
constexpr int BM_MAX_R      = 3; // units per work-item kept in registers, more: the generic loop

struct block_mix_params {
    const char * cur;   int64_t nbc2;
    const char * stack; int64_t nbs1, nbs2;
    const char * query; int64_t nbq1;
    const char * out_w; int64_t nbw1; // nbw1 == 0: one weight for all queries
    char       * dst;   int64_t nbd1, nbd2, nbd3;
    int   n, n_stack, n_query, n_qg;
    int   flags;
    float scale, eps_in, eps_out;
    // F16: a dense f16 copy [n_tokens][n] of plane 0 of query q (nullptr: none), the B of the dense oneDNN matmuls
    sycl::half * dst_h[BM_QMAX];
    // F16, bit q: the plane-0 f32 rows of query q are not written, only dst_h[q] (GGML_SYCL_DROP_F32)
    unsigned     skip_f32;
};

template <int V> struct bm_vec;
template <> struct bm_vec<1> { using type = float; };
template <> struct bm_vec<4> { using type = sycl::float4; };

static inline float bm_hsum(float v)        { return v; }
static inline float bm_hsum(sycl::float4 v) { return (v.x() + v.y()) + (v.z() + v.w()); }

// V: floats per unit (4: n % 4 == 0 and 16-byte aligned rows), R > 0: at most R units per work-item, in
// registers; R == 0: any n, the rows are reread. R == 1 (decode, small batches) also keeps the first BM_KS states
// and the output weights in registers, so the only memory round trips are the first loads and the stores.
// part: (n_stack + 1)*BM_QMAX*n_sg dot partials, n_sg partials of the sum of squares of cur, BM_QMAX*n_sg partials
// of the output norms; dots: the dots, then the weights, (n_stack + 1)*BM_QMAX, and the norm scale of cur
constexpr int BM_KS = 4;

// F16: also the f16 copies of plane 0 (dst_h, V == 4 only), rounded to nearest even as the oneDNN f32 -> f16 reorder
template <int V, int R, bool F16 = false, int QM = BM_QMAX>
static void block_mix_kernel(const sycl::nd_item<1> & it, float * part, float * dots, const block_mix_params p) {
    static_assert(!F16 || V == 4, "the f16 copies are written per float4 unit");
    using vf = typename bm_vec<V>::type;
    constexpr int RR = R > 0 ? R : 1;
    constexpr int KS = R == 1 ? BM_KS : 0; // states kept in registers
    constexpr int KR = KS > 0 ? KS : 1;

    const int W    = (int) it.get_local_range(0);
    const int tid  = (int) it.get_local_id(0);
    const auto sg  = it.get_sub_group();
    const int sgid = (int) sg.get_group_linear_id();
    const int lane = (int) sg.get_local_linear_id();
    const int n_sg = W / WARP_SIZE;

    const size_t g  = it.get_group(0);
    const int64_t t = (int64_t) (g / p.n_qg);
    const int q0    = (int) (g % p.n_qg) * BM_QMAX;
    const int nq    = sycl::min(BM_QMAX, p.n_query - q0);
    const int ns    = p.n_stack + 1;
    const int n     = p.n;
    const int nu    = n / V; // units of V floats

    float * ssp = part + ns*BM_QMAX*n_sg;
    float * red = ssp + n_sg;

    const vf * cur = (const vf *) (p.cur + t*p.nbc2);
    auto state = [&](int k) -> const vf * {
        return k < p.n_stack ? (const vf *) (p.stack + k*p.nbs1 + t*p.nbs2) : cur;
    };

    const vf * qv[QM];
    const vf * wv[QM];
#pragma unroll
    for (int q = 0; q < QM; ++q) {
        qv[q] = (const vf *) (p.query + (int64_t) (q0 + sycl::min(q, nq - 1))*p.nbq1);
        wv[q] = p.out_w ? (const vf *) (p.out_w + (int64_t) (q0 + sycl::min(q, nq - 1))*p.nbw1) : nullptr;
    }
    const bool use_w = (p.flags & 2) && p.out_w;

    // the query units (and for R == 1 the output weights) of this work-item
    vf qr[RR][QM];
    vf wr[RR][QM];
    if constexpr (R > 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int e = tid + r*W;
#pragma unroll
            for (int q = 0; q < QM; ++q) {
                qr[r][q] = e < nu && q < nq ? qv[q][e] : vf(0.0f);
                if constexpr (R == 1) {
                    wr[r][q] = e < nu && q < nq && use_w ? wv[q][e] : vf(1.0f);
                }
            }
        }
    }

    // the dots, one state at a time: per work-item partials, then one sub-group reduction per query
    auto dot_state = [&](int k, const vf * v) {
        vf acc[QM] = {};
        vf ss = {};
#pragma unroll
        for (int r = 0; r < RR; ++r) {
            ss += v[r]*v[r];
#pragma unroll
            for (int q = 0; q < QM; ++q) {
                acc[q] += v[r]*qr[r][q];
            }
        }
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            if (q < nq) {
                const float rs = sycl::reduce_over_group(sg, bm_hsum(acc[q]), sycl::plus<float>());
                if (sg.leader()) {
                    part[(k*BM_QMAX + q)*n_sg + sgid] = rs;
                }
            }
        }
        if ((p.flags & 1) && k == p.n_stack) {
            const float rs = sycl::reduce_over_group(sg, bm_hsum(ss), sycl::plus<float>());
            if (sg.leader()) {
                ssp[sgid] = rs;
            }
        }
    };
    vf sk[KR][RR]; // the first KS states
    if constexpr (R > 0) {
        if constexpr (KS > 0) {
#pragma unroll
            for (int k = 0; k < KS; ++k) {
                if (k < ns) {
                    const vf * s = state(k);
#pragma unroll
                    for (int r = 0; r < RR; ++r) {
                        const int e = tid + r*W;
                        sk[k][r] = e < nu ? s[e] : vf(0.0f);
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < KS; ++k) {
                if (k < ns) {
                    dot_state(k, sk[k]);
                }
            }
        }
#pragma unroll 2
        for (int k = KS; k < ns; ++k) {
            const vf * s = state(k);
            vf v[RR];
#pragma unroll
            for (int r = 0; r < RR; ++r) {
                const int e = tid + r*W;
                v[r] = e < nu ? s[e] : vf(0.0f);
            }
            dot_state(k, v);
        }
    } else {
        for (int k = 0; k < ns; ++k) {
            const vf * s = state(k);
            vf acc[QM] = {};
            vf ss = {};
#pragma unroll 4
            for (int e = tid; e < nu; e += W) {
                const vf v = s[e];
                ss += v*v;
#pragma unroll
                for (int q = 0; q < QM; ++q) {
                    if (q < nq) {
                        acc[q] += v*qv[q][e];
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < QM; ++q) {
                if (q < nq) {
                    const float rs = sycl::reduce_over_group(sg, bm_hsum(acc[q]), sycl::plus<float>());
                    if (sg.leader()) {
                        part[(k*BM_QMAX + q)*n_sg + sgid] = rs;
                    }
                }
            }
            if ((p.flags & 1) && k == p.n_stack) {
                const float rs = sycl::reduce_over_group(sg, bm_hsum(ss), sycl::plus<float>());
                if (sg.leader()) {
                    ssp[sgid] = rs;
                }
            }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);

    // the sub-group partials: one sub-group per (state, query) and one for the norm of cur, the lanes read the
    // partials in a fixed order (the same sums for any timing)
    const int n_task = ns*nq + ((p.flags & 1) ? 1 : 0);
    for (int j = sgid; j < n_task; j += n_sg) {
        const float * src = j < ns*nq ? part + ((j / nq)*BM_QMAX + j % nq)*n_sg : ssp;
        float d = 0.0f;
        for (int i = lane; i < n_sg; i += WARP_SIZE) {
            d += src[i];
        }
        d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
        if (sg.leader()) {
            if (j < ns*nq) {
                dots[(j / nq)*BM_QMAX + j % nq] = d;
            } else {
                dots[ns*BM_QMAX] = 1.0f/sycl::sqrt(d/n + p.eps_in);
            }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    const float sc_in = (p.flags & 1) ? dots[ns*BM_QMAX] : 1.0f;

    // the softmax over the states, one work-item per query
    if (tid < nq) {
        const int q = tid;
        float max_val = -INFINITY;
        for (int k = 0; k < ns; ++k) {
            float d = dots[k*BM_QMAX + q];
            if (k == p.n_stack) {
                d *= sc_in;
            }
            d *= p.scale;
            dots[k*BM_QMAX + q] = d;
            max_val = sycl::fmax(max_val, d);
        }
        float sum = 0.0f;
        for (int k = 0; k < ns; ++k) {
            const float w = sycl::exp(dots[k*BM_QMAX + q] - max_val);
            dots[k*BM_QMAX + q] = w;
            sum += w;
        }
        const float inv = 1.0f/sum;
        for (int k = 0; k < ns; ++k) {
            dots[k*BM_QMAX + q] *= inv;
        }
    }
    it.barrier(sycl::access::fence_space::local_space);

    // the weighted sum in k order, each state unit read once for the queries of the group; the raw y goes to
    // plane 1 (keep_raw), to the output when it is not normed, and (R == 0) to the output until its norm is known
    vf * y[QM];
    vf * yr[QM];
#pragma unroll
    for (int q = 0; q < QM; ++q) {
        y[q]  = (vf *) (p.dst + (int64_t) (q0 + sycl::min(q, nq - 1))*p.nbd1 + t*p.nbd2);
        yr[q] = (p.flags & 4) ? (vf *) ((char *) y[q] + p.nbd3) : y[q];
    }
    const bool write_raw = R == 0 || !(p.flags & 2) || (p.flags & 4);

    float ss[QM] = {};
    vf keep[RR][QM];
    // the f16 copy of the plane 0 unit e of query q
    auto store_h = [&](int q, int e, const vf & v) {
        if constexpr (F16) {
            sycl::half * h = p.dst_h[sycl::min(q0 + q, BM_QMAX - 1)];
            if (h) {
                ((sycl::vec<sycl::half, 4> *) (h + t*n))[e] = v.template convert<sycl::half, sycl::rounding_mode::rte>();
            }
        }
    };
    // plane 0 of query q is not written in f32
    auto skip0 = [&](int q) {
        if constexpr (F16) {
            return ((p.skip_f32 >> sycl::min(q0 + q, BM_QMAX - 1)) & 1u) != 0;
        } else {
            return false;
        }
    };
    auto store_unit = [&](int e, vf * acc, vf * out) {
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            if (q < nq) {
                if (write_raw && ((p.flags & 4) || !skip0(q))) {
                    yr[q][e] = acc[q];
                }
                if (!(p.flags & 2)) {
                    store_h(q, e, acc[q]);
                }
                ss[q] += bm_hsum(acc[q]*acc[q]);
            }
            out[q] = acc[q];
        }
    };
    if constexpr (R > 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int e = tid + r*W;
            vf acc[QM] = {};
            if constexpr (KS > 0) {
#pragma unroll
                for (int k = 0; k < KS; ++k) {
                    if (k < ns) {
                        const vf v = k == p.n_stack ? sc_in*sk[k][r] : sk[k][r];
#pragma unroll
                        for (int q = 0; q < QM; ++q) {
                            acc[q] += v*dots[k*BM_QMAX + q];
                        }
                    }
                }
            }
            if (e < nu) {
#pragma unroll 4
                for (int k = KS; k < ns; ++k) {
                    const vf v = k == p.n_stack ? sc_in*cur[e] : state(k)[e];
#pragma unroll
                    for (int q = 0; q < QM; ++q) {
                        acc[q] += v*dots[k*BM_QMAX + q];
                    }
                }
                store_unit(e, acc, keep[r]);
            }
        }
    } else {
        for (int e = tid; e < nu; e += W) {
            vf acc[QM] = {};
#pragma unroll 4
            for (int k = 0; k < ns; ++k) {
                const vf v = k == p.n_stack ? sc_in*cur[e] : state(k)[e];
#pragma unroll
                for (int q = 0; q < QM; ++q) {
                    acc[q] += v*dots[k*BM_QMAX + q];
                }
            }
            store_unit(e, acc, keep[0]);
        }
    }

    if (!(p.flags & 2)) {
        return;
    }

    // rms_norm(y, eps_out) * out_w: every sub-group adds the partials of all sub-groups in the same order
#pragma unroll
    for (int q = 0; q < QM; ++q) {
        if (q < nq) {
            const float rs = sycl::reduce_over_group(sg, ss[q], sycl::plus<float>());
            if (sg.leader()) {
                red[q*n_sg + sgid] = rs;
            }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    float sc[QM];
#pragma unroll
    for (int q = 0; q < QM; ++q) {
        float d = 0.0f;
        if (q < nq) {
            for (int i = lane; i < n_sg; i += WARP_SIZE) {
                d += red[q*n_sg + i];
            }
        }
        d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
        sc[q] = 1.0f/sycl::sqrt(d/n + p.eps_out);
    }
    auto norm_unit = [&](int e, int r) {
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            if (q < nq) {
                const vf v = sc[q]*(R > 0 ? keep[r][q] : yr[q][e]);
                vf o;
                if constexpr (R == 1) {
                    o = use_w ? v*wr[r][q] : v;
                } else {
                    o = use_w ? v*wv[q][e] : v;
                }
                if (!skip0(q)) {
                    y[q][e] = o;
                }
                store_h(q, e, o);
            }
        }
    };
    if constexpr (R > 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int e = tid + r*W;
            if (e < nu) {
                norm_unit(e, r);
            }
        }
    } else {
        for (int e = tid; e < nu; e += W) {
            norm_unit(e, 0);
        }
    }
}

template <int V, int R, bool F16 = false, int QM = BM_QMAX>
static void block_mix_launch(sycl::queue * q, const block_mix_params & p, size_t n_groups, int W) {
    const int n_sg = W / WARP_SIZE;
    const int ns   = p.n_stack + 1;
    q->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) (ns*BM_QMAX + 1 + BM_QMAX)*n_sg), cgh);
        sycl::local_accessor<float, 1> dots(sycl::range<1>((size_t) ns*BM_QMAX + 1), cgh);
        cgh.parallel_for(sycl::nd_range<1>(n_groups*W, W),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                block_mix_kernel<V, R, F16, QM>(it, part.get_multi_ptr<sycl::access::decorated::no>().get(),
                                       dots.get_multi_ptr<sycl::access::decorated::no>().get(), p);
            });
    });
}

} // namespace

bool ggml_sycl_block_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * cur   = op->src[0];
    const ggml_tensor * query = op->src[1];
    const ggml_tensor * stack = op->src[2];
    const ggml_tensor * out_w = op->src[3];

    if (op->type != GGML_TYPE_F32 || cur->type != GGML_TYPE_F32 || query->type != GGML_TYPE_F32 ||
        (stack && stack->type != GGML_TYPE_F32) || (out_w && out_w->type != GGML_TYPE_F32)) {
        return false;
    }
    if (op->nb[0] != sizeof(float) || cur->nb[0] != sizeof(float) || query->nb[0] != sizeof(float) ||
        (stack && stack->nb[0] != sizeof(float)) || (out_w && out_w->nb[0] != sizeof(float))) {
        return false;
    }
    // the rows are addressed as floats
    if (cur->nb[2] % sizeof(float) || query->nb[1] % sizeof(float) || op->nb[1] % sizeof(float) ||
        op->nb[2] % sizeof(float) || op->nb[3] % sizeof(float) ||
        (stack && (stack->nb[1] % sizeof(float) || stack->nb[2] % sizeof(float))) ||
        (out_w && out_w->nb[1] % sizeof(float))) {
        return false;
    }
    const int64_t n_states = (stack ? stack->ne[1] : 0) + 1;
    return n_states <= BM_MAX_STATES && cur->ne[0] <= INT32_MAX && query->ne[1] <= INT32_MAX;
}

void ggml_sycl_block_mix(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const bool * f16_for_dense,
                         const int * pin_until) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);

    const ggml_tensor * cur   = dst->src[0];
    const ggml_tensor * query = dst->src[1];
    const ggml_tensor * stack = dst->src[2];
    const ggml_tensor * out_w = dst->src[3];

    block_mix_params p;
    p.cur   = (const char *) cur->data;   p.nbc2 = cur->nb[2];
    p.stack = stack ? (const char *) stack->data : nullptr;
    p.nbs1  = stack ? stack->nb[1] : 0;   p.nbs2 = stack ? stack->nb[2] : 0;
    p.query = (const char *) query->data; p.nbq1 = query->nb[1];
    p.out_w = out_w ? (const char *) out_w->data : nullptr;
    p.nbw1  = out_w && out_w->ne[1] > 1 ? out_w->nb[1] : 0;
    p.dst   = (char *) dst->data;
    p.nbd1  = dst->nb[1]; p.nbd2 = dst->nb[2]; p.nbd3 = dst->nb[3];
    p.n       = (int) cur->ne[0];
    p.n_stack = stack ? (int) stack->ne[1] : 0;
    p.n_query = (int) query->ne[1];
    p.n_qg    = (p.n_query + BM_QMAX - 1) / BM_QMAX;
    p.scale   = ggml_get_op_params_f32(dst, 0);
    p.flags   = ggml_get_op_params_i32(dst, 1);
    p.eps_in  = ggml_get_op_params_f32(dst, 2);
    p.eps_out = ggml_get_op_params_f32(dst, 3);
    for (int q = 0; q < BM_QMAX; ++q) {
        p.dst_h[q] = nullptr;
    }
    p.skip_f32 = 0;

    const int64_t n_tokens = cur->ne[2];
    GGML_ASSERT(p.n_stack + 1 <= BM_MAX_STATES);
    if (n_tokens == 0 || p.n_query == 0 || p.n == 0) {
        return;
    }

    const size_t n_groups = (size_t) n_tokens * p.n_qg;

    // float4 rows: n % 4 == 0 and every row 16-byte aligned (the pointers are checked, not read, on the host)
    auto al16 = [](const void * ptr) { return ((uintptr_t) ptr % 16) == 0; };
    const bool vec4 = p.n % 4 == 0 && al16(p.cur) && al16(p.query) && al16(p.dst) &&
                      (!p.stack || al16(p.stack)) && (!p.out_w || al16(p.out_w)) &&
                      p.nbc2 % 16 == 0 && p.nbs1 % 16 == 0 && p.nbs2 % 16 == 0 && p.nbq1 % 16 == 0 &&
                      p.nbw1 % 16 == 0 && p.nbd1 % 16 == 0 && p.nbd2 % 16 == 0 && p.nbd3 % 16 == 0;
    const int nu = vec4 ? p.n / 4 : p.n;

    // a few work-groups (decode, small batches): one unit per work-item, the rows read by as many work-items at
    // once as possible; many: 256, a few units each
    const int max_wg = std::min(1024, ggml_sycl_info().max_work_group_sizes[ctx.device]);
    // the units spread evenly: the fewest per work-item that keep the group at most 256 wide, then the group
    // narrowed to them (2304 floats: 192 x 3 instead of 256 x 3 with the third unit mostly idle)
    int W = std::min(256, (int) GGML_PAD((nu + (nu + 255) / 256 - 1) / ((nu + 255) / 256), WARP_SIZE));
    if (n_groups < 64) {
        W = vec4 && nu <= max_wg ? std::max(WARP_SIZE, (int) GGML_PAD(nu, WARP_SIZE)) : std::min(512, max_wg);
    }
    const int R = (nu + W - 1) / W;

    // the f16 copies of the queries whose plane-0 rows the dense oneDNN matmuls read: registered only now that the
    // kernel that writes them is sure to run (vec4, the rows in registers, one query group)
    bool f16 = false;
#if GGML_SYCL_DNNL
    if (f16_for_dense && vec4 && R >= 1 && R <= BM_MAX_R && p.n_query <= BM_QMAX) {
        for (int j = 0; j < p.n_query; ++j) {
            if (!f16_for_dense[j]) {
                continue;
            }
            ggml_tensor rows = *dst;
            rows.data     = (char *) dst->data + j*dst->nb[1];
            rows.view_src = nullptr;
            rows.ne[0] = dst->ne[0]; rows.ne[1] = n_tokens; rows.ne[2] = 1; rows.ne[3] = 1;
            rows.nb[1] = dst->nb[2]; rows.nb[2] = rows.nb[1]*n_tokens; rows.nb[3] = rows.nb[2];
            if (pin_until && pin_until[j] >= 0) {
                p.dst_h[j] = ggml_sycl_dense_dnn_act_reserve(ctx, &rows, pin_until[j]);
                p.skip_f32 |= p.dst_h[j] ? 1u << j : 0u;
            }
            if (!p.dst_h[j]) {
                p.dst_h[j] = ggml_sycl_dense_dnn_act_reserve(ctx, &rows);
            }
            f16 = f16 || p.dst_h[j];
        }
    }
#else
    GGML_UNUSED(f16_for_dense);
    GGML_UNUSED(pin_until);
#endif

    // one query (the prefill mixes): the registers and FMAs of one query instead of BM_QMAX, the same arithmetic
    // (GGML_SYCL_BLOCK_MIX_Q1=0: the BM_QMAX-query kernel)
    static const bool q1_on = ggml_sycl_get_env("GGML_SYCL_BLOCK_MIX_Q1", 1) != 0;
    const bool q1 = q1_on && p.n_query == 1;
    try {
        sycl::queue * q = ctx.stream();
        if (!vec4) {
            block_mix_launch<1, 0>(q, p, n_groups, W);
        } else if (R == 1) {
            f16 ? block_mix_launch<4, 1, true>(q, p, n_groups, W) : block_mix_launch<4, 1>(q, p, n_groups, W);
        } else if (R == 2) {
            f16 ? block_mix_launch<4, 2, true>(q, p, n_groups, W) : block_mix_launch<4, 2>(q, p, n_groups, W);
        } else if (R <= BM_MAX_R && q1) {
            f16 ? block_mix_launch<4, BM_MAX_R, true, 1>(q, p, n_groups, W) : block_mix_launch<4, BM_MAX_R, false, 1>(q, p, n_groups, W);
        } else if (R <= BM_MAX_R) {
            f16 ? block_mix_launch<4, BM_MAX_R, true>(q, p, n_groups, W) : block_mix_launch<4, BM_MAX_R>(q, p, n_groups, W);
        } else {
            block_mix_launch<4, 0>(q, p, n_groups, W);
        }
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "[SYCL-BLOCK_MIX] ERROR: %s\n", e.what());
        throw;
    }
}
