// GGML_OP_SSM_CONV_HIST: the ggml_ssm_conv sum over the window [past | x^T] read in place, the last d_conv - 1
// window columns written to the store slots (ggml_ssm_conv_hist), and with ggml_ssm_conv_hist_post the addend
// (src[4]) and a row post-op over each group of `group` channels of one token (L2 clip or RMS norm).
// The rounding follows ggml_compute_forward_ssm_conv_hist (ggml-cpu/ops.cpp): the f32 conv sum in tap order, then
// + add, then the post-op (its sum of squares an f32 tree sum here, a double sum on the CPU).
//
// One work-item per channel: a work-group is a tile of channels (one post-op group when there is a post-op, so the
// row sums are work-group reductions) and a block of up to TT consecutive tokens of one sequence, whose window
// slides through registers, so each input is read once per token block. The work-items of the last K tokens of a
// sequence write the store slots of their channel (slot k holds the window columns after token n_t - 1 - k).

#include "ssm-conv-hist.hpp"

#include <cstdio>

namespace {

constexpr int CH_TILE  = 128; // channels per work-group without a post-op
constexpr int CH_DCMAX = 8;   // the largest d_conv of the generic kernel

struct ssm_conv_hist_params {
    const char * past;  int64_t nbp1, nbp2;
    const char * x;     int64_t nbx1, nbx2;
    const char * c;     int64_t nbc1;
    char       * store; int64_t nbs1, nbs2, nbs3; int n_slots;
    const char * add;   int64_t nba1, nba2;
    char       * dst;   int64_t nbd1, nbd2;
    int   nc, d_inner, n_t, tt, n_tiles, n_tb;
    int   post;
    float param;
    int   group;
};

// DC > 0: d_conv as a compile-time constant, 0: any d_conv <= CH_DCMAX
template <int DC>
static void ssm_conv_hist_kernel(const sycl::nd_item<1> & it, const ssm_conv_hist_params p) {
    constexpr int NA = DC > 0 ? DC : CH_DCMAX;
    const int nc = DC > 0 ? DC : p.nc;

    const int    W    = (int) it.get_local_range(0);
    const int    tid  = (int) it.get_local_id(0);
    const size_t g    = it.get_group(0);
    const int    tile = (int) (g % p.n_tiles);
    const size_t rest = g / p.n_tiles;
    const int    t0   = (int) (rest % p.n_tb) * p.tt;
    const int64_t i3  = (int64_t) (rest / p.n_tb);
    const int    ntt  = sycl::min(p.tt, p.n_t - t0);

    const int  ch     = tile*W + tid;
    // without a post-op the last tile can be partial; with one the tiles are the groups (no work-group reduction
    // runs without a post-op, so the idle work-items can leave)
    const bool active = ch < p.d_inner;
    if (!active && p.post == GGML_SSM_CONV_POST_NONE) {
        return;
    }
    const int chc = active ? ch : 0;

    // column col of the window [past | x^T] of this channel
    auto win = [&](int col) -> float {
        if (col < nc - 1) {
            return *(const float *) (p.past + (int64_t) col*sizeof(float) + chc*p.nbp1 + i3*p.nbp2);
        }
        return *(const float *) (p.x + (int64_t) chc*sizeof(float) + (int64_t) (col - (nc - 1))*p.nbx1 + i3*p.nbx2);
    };

    float w[NA];
    float v[NA];
    const float * cw = (const float *) (p.c + chc*p.nbc1);
#pragma unroll
    for (int k = 0; k < NA; ++k) {
        w[k] = k < nc ? cw[k] : 0.0f;
        v[k] = k < nc - 1 ? win(t0 + k) : 0.0f;
    }

    for (int tt = 0; tt < ntt; ++tt) {
        const int i2 = t0 + tt;
        // the newest window column of token i2 is x[:, i2]
        const float xn = win(i2 + nc - 1);
#pragma unroll
        for (int k = 0; k < NA; ++k) {
            if (k == nc - 1) {
                v[k] = xn;
            }
        }

        // the same sum as ggml_compute_forward_ssm_conv_f32
        float sumf = 0.0f;
#pragma unroll
        for (int k = 0; k < NA; ++k) {
            if (k < nc) {
                sumf += v[k]*w[k];
            }
        }
        if (p.add) {
            sumf = sumf + *(const float *) (p.add + (int64_t) chc*sizeof(float) + (int64_t) i2*p.nba1 + i3*p.nba2);
        }

        if (p.post != GGML_SSM_CONV_POST_NONE) {
            const float ss = sycl::reduce_over_group(it.get_group(), active ? sumf*sumf : 0.0f, sycl::plus<float>());
            if (p.post == GGML_SSM_CONV_POST_L2_CLIP) {
                const float norm   = sycl::sqrt(ss) + 1e-6f;
                const float factor = sycl::clamp(p.param / norm, 0.0f, 1.0f);
                sumf *= factor;
            } else {
                const float mean  = ss/p.group;
                const float scale = 1.0f/sycl::sqrt(mean + p.param);
                sumf *= scale;
            }
        }

        if (active) {
            *(float *) (p.dst + (int64_t) chc*sizeof(float) + (int64_t) i2*p.nbd1 + i3*p.nbd2) = sumf;

            // slot k: the history after token n_t - 1 - k, the window columns 1 ... nc - 1 of token n_t - 1 - k
            const int k = p.n_t - 1 - i2;
            if (p.store && k < p.n_slots) {
                float * s = (float *) (p.store + chc*p.nbs1 + i3*p.nbs2 + (int64_t) k*p.nbs3);
#pragma unroll
                for (int j = 0; j < NA - 1; ++j) {
                    if (j < nc - 1) {
                        s[j] = v[j + 1];
                    }
                }
            }
        }

        // slide the window by one column
#pragma unroll
        for (int k = 0; k < NA - 1; ++k) {
            v[k] = v[k + 1];
        }
    }
}

// d_conv 4 and blocks of TT tokens, unrolled: the TT + 3 window columns (and the addends) of a work-item are loaded
// up front, TT + 3 loads in flight instead of one per token; with a post-op the TT row sums of squares are reduced
// together, per sub-group and then the sub-group partials in a fixed order (one barrier per block, not one per token)
template <int TT>
static void ssm_conv_hist_tt_kernel(const sycl::nd_item<1> & it, float * red, const ssm_conv_hist_params p) {
    constexpr int NC = 4;

    const int    W    = (int) it.get_local_range(0);
    const int    tid  = (int) it.get_local_id(0);
    const size_t g    = it.get_group(0);
    const int    tile = (int) (g % p.n_tiles);
    const size_t rest = g / p.n_tiles;
    const int    t0   = (int) (rest % p.n_tb) * TT;
    const int64_t i3  = (int64_t) (rest / p.n_tb);
    const int    ntt  = sycl::min(TT, p.n_t - t0);

    const int  ch     = tile*W + tid;
    const bool active = ch < p.d_inner;
    if (!active && p.post == GGML_SSM_CONV_POST_NONE) {
        return;
    }
    const int chc = active ? ch : 0;

    const float * xr = (const float *) (p.x + (int64_t) chc*sizeof(float) + i3*p.nbx2);
    // column j of the window block, token t0 + j - (NC - 1) of [past | x^T]
    float col[TT + NC - 1];
#pragma unroll
    for (int j = 0; j < TT + NC - 1; ++j) {
        const int c = t0 + j;
        if (c < NC - 1) {
            col[j] = *(const float *) (p.past + (int64_t) c*sizeof(float) + chc*p.nbp1 + i3*p.nbp2);
        } else {
            col[j] = j < ntt + NC - 1 ? *(const float *) ((const char *) xr + (int64_t) (c - (NC - 1))*p.nbx1) : 0.0f;
        }
    }
    float ad[TT];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
        ad[tt] = p.add && tt < ntt ?
            *(const float *) (p.add + (int64_t) chc*sizeof(float) + (int64_t) (t0 + tt)*p.nba1 + i3*p.nba2) : 0.0f;
    }
    const float * cw = (const float *) (p.c + chc*p.nbc1);
    float w[NC];
#pragma unroll
    for (int k = 0; k < NC; ++k) {
        w[k] = cw[k];
    }

    float y[TT];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
        // the same sum as ggml_compute_forward_ssm_conv_f32, then + add
        float sumf = 0.0f;
#pragma unroll
        for (int k = 0; k < NC; ++k) {
            sumf += col[tt + k]*w[k];
        }
        y[tt] = p.add ? sumf + ad[tt] : sumf;
    }

    if (p.post != GGML_SSM_CONV_POST_NONE) {
        const auto sg   = it.get_sub_group();
        const int  n_sg = W / WARP_SIZE;
        const int  sgid = (int) sg.get_group_linear_id();
        float ss[TT];
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
            ss[tt] = sycl::reduce_over_group(sg, active ? y[tt]*y[tt] : 0.0f, sycl::plus<float>());
        }
        if (sg.leader()) {
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                red[tt*n_sg + sgid] = ss[tt];
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
            float tot = 0.0f;
            for (int i = 0; i < n_sg; ++i) {
                tot += red[tt*n_sg + i];
            }
            if (p.post == GGML_SSM_CONV_POST_L2_CLIP) {
                const float norm = sycl::sqrt(tot) + 1e-6f;
                y[tt] *= sycl::clamp(p.param / norm, 0.0f, 1.0f);
            } else {
                y[tt] *= 1.0f/sycl::sqrt(tot/p.group + p.param);
            }
        }
    }

    if (!active) {
        return;
    }
    char * dr = p.dst + (int64_t) chc*sizeof(float) + i3*p.nbd2;
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
        if (tt < ntt) {
            *(float *) (dr + (int64_t) (t0 + tt)*p.nbd1) = y[tt];
        }
    }
    // slot k: the window columns 1 ... NC - 1 of token n_t - 1 - k
    if (p.store && p.n_t - t0 - ntt < p.n_slots) {
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
            const int k = p.n_t - 1 - (t0 + tt);
            if (tt < ntt && k < p.n_slots) {
                float * sl = (float *) (p.store + chc*p.nbs1 + i3*p.nbs2 + (int64_t) k*p.nbs3);
#pragma unroll
                for (int j = 0; j < NC - 1; ++j) {
                    sl[j] = col[tt + 1 + j];
                }
            }
        }
    }
}

template <int TT>
static void ssm_conv_hist_tt_launch(sycl::queue * q, const ssm_conv_hist_params & p, size_t n_groups, int W) {
    q->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) TT * (W / WARP_SIZE)), cgh);
        cgh.parallel_for(sycl::nd_range<1>(n_groups*W, W), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            ssm_conv_hist_tt_kernel<TT>(it, red.get_multi_ptr<sycl::access::decorated::no>().get(), p);
        });
    });
}

template <int DC>
static void ssm_conv_hist_launch(sycl::queue * q, const ssm_conv_hist_params & p, size_t n_groups, int W) {
    q->parallel_for(sycl::nd_range<1>(n_groups*W, W), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        ssm_conv_hist_kernel<DC>(it, p);
    });
}

} // namespace

bool ggml_sycl_ssm_conv_hist_supported(const ggml_tensor * op) {
    const ggml_tensor * past  = op->src[0];
    const ggml_tensor * x     = op->src[1];
    const ggml_tensor * c     = op->src[2];
    const ggml_tensor * store = op->src[3];
    const ggml_tensor * add   = op->src[4];

    if (op->type != GGML_TYPE_F32 || past->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || c->type != GGML_TYPE_F32 ||
        (store && store->type != GGML_TYPE_F32) || (add && add->type != GGML_TYPE_F32)) {
        return false;
    }
    if (op->nb[0] != sizeof(float) || past->nb[0] != sizeof(float) || x->nb[0] != sizeof(float) || c->nb[0] != sizeof(float) ||
        (store && store->nb[0] != sizeof(float)) || (add && add->nb[0] != sizeof(float))) {
        return false;
    }
    if (c->ne[0] < 2 || c->ne[0] > CH_DCMAX || op->ne[0] > INT32_MAX || op->ne[1] > INT32_MAX) {
        return false;
    }
    const int32_t post = ggml_get_op_params_i32(op, 0);
    if (post == GGML_SSM_CONV_POST_NONE) {
        return true;
    }
    // a post-op group is one work-group
    const int32_t group = ggml_get_op_params_i32(op, 2);
    return (post == GGML_SSM_CONV_POST_L2_CLIP || post == GGML_SSM_CONV_POST_RMS_NORM) && group > 0 && group <= 1024 &&
           op->ne[0] % group == 0;
}

void ggml_sycl_ssm_conv_hist(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    const ggml_tensor * past  = dst->src[0];
    const ggml_tensor * x     = dst->src[1];
    const ggml_tensor * c     = dst->src[2];
    const ggml_tensor * store = dst->src[3];
    const ggml_tensor * add   = dst->src[4];

    ssm_conv_hist_params p;
    p.past  = (const char *) past->data; p.nbp1 = past->nb[1]; p.nbp2 = past->nb[2];
    p.x     = (const char *) x->data;    p.nbx1 = x->nb[1];    p.nbx2 = x->nb[2];
    p.c     = (const char *) c->data;    p.nbc1 = c->nb[1];
    p.store = store ? (char *) store->data : nullptr;
    p.nbs1  = store ? store->nb[1] : 0; p.nbs2 = store ? store->nb[2] : 0; p.nbs3 = store ? store->nb[3] : 0;
    p.n_slots = store ? (int) store->ne[3] : 0;
    p.add   = add ? (const char *) add->data : nullptr;
    p.nba1  = add ? add->nb[1] : 0;     p.nba2 = add ? add->nb[2] : 0;
    p.dst   = (char *) dst->data;        p.nbd1 = dst->nb[1];  p.nbd2 = dst->nb[2];
    p.nc      = (int) c->ne[0];
    p.d_inner = (int) dst->ne[0];
    p.n_t     = (int) dst->ne[1];
    p.post    = ggml_get_op_params_i32(dst, 0);
    p.param   = ggml_get_op_params_f32(dst, 1);
    p.group   = p.post != GGML_SSM_CONV_POST_NONE ? ggml_get_op_params_i32(dst, 2) : 1;

    const int64_t n_s = dst->ne[2];
    if (p.d_inner == 0 || p.n_t == 0 || n_s == 0) {
        return;
    }

    const int W = p.post != GGML_SSM_CONV_POST_NONE ? p.group : CH_TILE;
    p.n_tiles = (p.d_inner + W - 1) / W;
    // batches: a block of tokens per work-group, the window slides through registers
    static const int tt_env = getenv("GGML_SYCL_CONV_HIST_TT") ? atoi(getenv("GGML_SYCL_CONV_HIST_TT")) : 8;
    const bool unrolled = p.nc == 4 && p.n_t >= 32 && (tt_env == 8 || tt_env == 16);
    p.tt   = unrolled ? tt_env : p.n_t >= 32 ? 8 : 1;
    p.n_tb = (p.n_t + p.tt - 1) / p.tt;
    const size_t n_groups = (size_t) p.n_tiles * p.n_tb * n_s;

    try {
        sycl::queue * q = ctx.stream();
        if (unrolled) {
            p.tt == 16 ? ssm_conv_hist_tt_launch<16>(q, p, n_groups, W) : ssm_conv_hist_tt_launch<8>(q, p, n_groups, W);
        } else if (p.nc == 4) {
            ssm_conv_hist_launch<4>(q, p, n_groups, W);
        } else {
            ssm_conv_hist_launch<0>(q, p, n_groups, W);
        }
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV_HIST] ERROR: %s\n", e.what());
        throw;
    }
}
