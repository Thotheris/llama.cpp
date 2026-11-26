//
// MIT license
// SPDX-License-Identifier: MIT
//

// Multi-column MoE GEMV for MUL_MAT_ID at a few tokens, see mmvq-id-multi.hpp.
//
// The per-column arithmetic and the sub-group reduction are the ones of the single-token kernels in mmvq.cpp
// (mul_mat_vec_q_moe and mul_mat_vec_q_moe_reorder), so a token's expert outputs do not depend on how many other
// tokens share the batch.

#include "mmvq-id-multi.hpp"

#include "common.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

#ifdef GGML_SYCL_MMVQ_HAS_XMX
#    include "esimd.hpp"
#endif

#include <algorithm>

namespace {

constexpr int MMID_MC_SCHED_WG = 256;  // work-items of the schedule kernel
constexpr int MMID_MC_NSG      = 4;    // sub-groups per work-group of the GEMV

struct mmid_mc_args {
    const char *    vx_base;
    size_t          expert_stride;
    int             ncols;
    int             nrows;
    const char *    vy;
    size_t          q8_row_bytes;
    int             ne11;
    int             n_ids;
    char *          dst;
    size_t          nb1;
    size_t          nb2;
    const int32_t * sched;
    int             n_slots;
};

// the schedule: slot_expert[n_slots] (-1 past the active experts), slot_off[n_slots + 1], routes[n_routes]
// (route = token*n_ids + slot), grouped by expert in increasing id, each group in route order
void mmid_mc_schedule(const int32_t * ids, size_t ids_nb1, int n_ids, int n_tokens, int n_as, int32_t * sched,
                      float * dst, size_t nb1, size_t nb2, int nrows, dpct::queue_ptr stream) {
    const int n_routes = n_ids * n_tokens;
    const int n_slots  = std::min(n_as, n_routes);
    // with at most 32 tokens a per-expert token mask gives each route its rank in its group in O(1); a token that
    // routes to one expert twice (or more tokens) takes the serial count instead
    const bool masks = n_tokens <= 32;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int32_t, 1>  ids_l(sycl::range<1>(n_routes), cgh);
        sycl::local_accessor<int32_t, 1>  cnt_l(sycl::range<1>(n_as), cgh);  // counts, then group offsets
        sycl::local_accessor<uint32_t, 1> msk_l(sycl::range<1>(n_as), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(MMID_MC_SCHED_WG), sycl::range<1>(MMID_MC_SCHED_WG)),
            [=](sycl::nd_item<1> it) {
                constexpr int L   = MMID_MC_SCHED_WG;
                const int     tid = (int) it.get_local_id(0);
                const auto    grp = it.get_group();
                using cnt_ref = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>;
                using msk_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
                for (int e = tid; e < n_as; e += L) {
                    cnt_l[e] = 0;
                    msk_l[e] = 0;
                }
                sycl::group_barrier(grp);
                for (int r = tid; r < n_routes; r += L) {
                    const int     t = r / n_ids;
                    const int     s = r - t * n_ids;
                    const int32_t e = *(const int32_t *) ((const char *) ids + (size_t) t * ids_nb1 + s * sizeof(int32_t));
                    const int32_t v = (e >= 0 && e < n_as) ? e : -1;
                    ids_l[r]        = v;
                    if (v >= 0) {
                        cnt_ref(cnt_l[v]).fetch_add(1);
                        if (masks) {
                            msk_ref(msk_l[v]).fetch_or(1u << t);
                        }
                    }
                }
                sycl::group_barrier(grp);

                // each work-item owns a contiguous run of experts
                const int per = (n_as + L - 1) / L;
                const int e0  = sycl::min(n_as, tid * per);
                const int e1  = sycl::min(n_as, e0 + per);
                int       act = 0;
                int       cnt = 0;
                int       dup = masks ? 0 : 1;
                for (int e = e0; e < e1; ++e) {
                    const int c = cnt_l[e];
                    act += c > 0;
                    cnt += c;
                    dup |= masks && sycl::popcount(msk_l[e]) != (uint32_t) c;
                }
                const int slot0 = sycl::exclusive_scan_over_group(grp, act, sycl::plus<int>());
                const int off0  = sycl::exclusive_scan_over_group(grp, cnt, sycl::plus<int>());
                const int n_act = sycl::reduce_over_group(grp, act, sycl::plus<int>());
                const int n_rt  = sycl::reduce_over_group(grp, cnt, sycl::plus<int>());
                const int serial = sycl::reduce_over_group(grp, dup, sycl::maximum<int>());

                int32_t * slot_expert = sched;
                int32_t * slot_off    = sched + n_slots;
                int32_t * routes      = slot_off + n_slots + 1;
                int       slot        = slot0;
                int       off         = off0;
                for (int e = e0; e < e1; ++e) {
                    const int c = cnt_l[e];
                    if (c > 0) {
                        slot_expert[slot] = e;
                        slot_off[slot]    = off;
                        ++slot;
                    }
                    cnt_l[e] = off;
                    off += c;
                }
                sycl::group_barrier(grp);

                // a route's place: its group's offset plus the routes of the same expert before it
                for (int r = tid; r < n_routes; r += L) {
                    const int e = ids_l[r];
                    if (e < 0) {
                        continue;
                    }
                    int rank = 0;
                    if (serial) {
                        for (int q = 0; q < r; ++q) {
                            rank += ids_l[q] == e;
                        }
                    } else {
                        const int t = r / n_ids;
                        rank        = sycl::popcount(msk_l[e] & ((1u << t) - 1u));
                    }
                    routes[cnt_l[e] + rank] = r;
                }
                for (int s = n_act + tid; s <= n_slots; s += L) {
                    slot_off[s] = n_rt;
                    if (s < n_slots) {
                        slot_expert[s] = -1;
                    }
                }

                // a skipped slot has no expert: its dst row is zeroed
                for (int r = 0; r < n_routes; ++r) {
                    if (ids_l[r] < 0) {
                        const int t   = r / n_ids;
                        const int s   = r - t * n_ids;
                        float *   row = (float *) ((char *) dst + (size_t) s * nb1 + (size_t) t * nb2);
                        for (int j = tid; j < nrows; j += L) {
                            row[j] = 0.0f;
                        }
                    }
                }
            });
    });
}

// the q8_1 row and the dst byte offset of route r
inline void mmid_mc_route(const mmid_mc_args & a, int r, const char *& y, size_t & doff) {
    const int t = r / a.n_ids;
    const int s = r - t * a.n_ids;
    y           = a.vy + (size_t) (t * a.ne11 + (a.ne11 == 1 ? 0 : s)) * a.q8_row_bytes;
    doff        = (size_t) s * a.nb1 + (size_t) t * a.nb2;
}

// standard block layout, NC routes of one expert per pass, one row per sub-group
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl, int NC>
void mmid_mc_plain(const mmid_mc_args & a, const sycl::nd_item<2> & it) {
    const int slot = (int) it.get_group(0);
    const int e    = a.sched[slot];
    if (e < 0) {
        return;
    }
    const auto sg  = it.get_sub_group();
    const int  row = (int) it.get_group(1) * MMID_MC_NSG + (int) sg.get_group_linear_id();
    if (row >= a.nrows) {
        return;
    }
    const int32_t * slot_off = a.sched + a.n_slots;
    const int32_t * routes   = slot_off + a.n_slots + 1;
    const int       beg      = slot_off[slot];
    const int       end      = slot_off[slot + 1];

    const block_q_t * x               = (const block_q_t *) (a.vx_base + (size_t) e * a.expert_stride);
    const int         blocks_per_row  = a.ncols / qk;
    constexpr int     blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
    const int         lane            = (int) sg.get_local_linear_id();

    for (int c0 = beg; c0 < end; c0 += NC) {
        const int          nc = sycl::min(NC, end - c0);
        const block_q8_1 * y[NC];
        size_t             doff[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const char * yj;
            mmid_mc_route(a, routes[c0 + sycl::min(j, nc - 1)], yj, doff[j]);
            y[j] = (const block_q8_1 *) yj;
        }

        float tmp[NC] = {};
        for (int i = lane / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
            const int ibx = row * blocks_per_row + i;
            const int iby = i * (qk / QK8_1);
            for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
                const int iqs = elem + vdr * (lane % (qi / vdr));
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    if (j < nc) {
                        tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j][iby], iqs);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (j < nc) {
                float sum = tmp[j];
#pragma unroll
                for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
                    sum += dpct::permute_sub_group_by_xor(sg, sum, mask);
                }
                if (lane == 0) {
                    ((float *) (a.dst + doff[j]))[row] = sum;
                }
            }
        }
    }
}

// reorder (SoA) layout, NC routes of one expert per pass, R rows per sub-group (R > 1 needs a type whose
// activations unpack once for several rows)
template <typename VD, int NC, int R>
void mmid_mc_reorder(const mmid_mc_args & a, const sycl::nd_item<2> & it) {
    using block_type   = ggml_sycl_reordered::block_q_t<VD::gtype>;
    using block_traits = typename block_type::traits;
    static_assert(R == 1 || reorder_vec_dot_shared_activations<VD::gtype>::value);
    constexpr bool is_q8_0       = VD::gtype == GGML_TYPE_Q8_0;
    constexpr bool shared_weights = reorder_vec_dot_shared_weights<VD::gtype>::value;

    const int slot = (int) it.get_group(0);
    const int e    = a.sched[slot];
    if (e < 0) {
        return;
    }
    const auto sg   = it.get_sub_group();
    const int  row0 = ((int) it.get_group(1) * MMID_MC_NSG + (int) sg.get_group_linear_id()) * R;
    if (row0 >= a.nrows) {
        return;
    }
    const int32_t * slot_off = a.sched + a.n_slots;
    const int32_t * routes   = slot_off + a.n_slots + 1;
    const int       beg      = slot_off[slot];
    const int       end      = slot_off[slot + 1];

    const char *  vx                          = a.vx_base + (size_t) e * a.expert_stride;
    constexpr int vdr_mmvq                    = is_q8_0 ? VDR_Q8_0_Q8_1_MMVQ : block_traits::vdr_mmvq;
    const int     ncols                       = a.ncols;
    const int     nrows                       = a.nrows;
    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / vdr_mmvq;
    const int     nblocks                     = nrows * blocks_per_row;
    const int     lane                        = (int) sg.get_local_linear_id();
    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    for (int c0 = beg; c0 < end; c0 += NC) {
        const int    nc = sycl::min(NC, end - c0);
        const char * y[NC];
        size_t       doff[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            mmid_mc_route(a, routes[c0 + sycl::min(j, nc - 1)], y[j], doff[j]);
        }

        float ps[NC][R] = {};
        for (int i = lane / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
            const int iby = i * block_type::block_to_q8_1_ratio();
#pragma unroll
            for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                const int iqs = elem + vdr_mmvq * (lane % block_elements_per_subgroup);
                if constexpr (R > 1) {
                    typename VD::weights wx[R];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const int  row       = sycl::min(row0 + r, nrows - 1);
                        const int  ibx       = row * blocks_per_row + i;
                        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                        wx[r]                = VD::load(vx, bx_offset, d_offset, iqs);
                    }
#pragma unroll
                    for (int j = 0; j < NC; ++j) {
                        if (j < nc) {
                            const int8_t *      q8 = (const int8_t *) y[j] + iby * QK8_1;
                            const sycl::half2 * ds = (const sycl::half2 *) (y[j] + ncols + iby * sizeof(sycl::half2));
                            const auto          av = VD::load_activations(q8, ds, iqs);
#pragma unroll
                            for (int r = 0; r < R; ++r) {
                                ps[j][r] += VD::apply(wx[r], av);
                            }
                        }
                    }
                } else {
                    const int  ibx       = row0 * blocks_per_row + i;
                    const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                    const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                    if constexpr (shared_weights) {
                        const auto wx = VD::load(vx, bx_offset, d_offset, iqs);
#pragma unroll
                        for (int j = 0; j < NC; ++j) {
                            if (j < nc) {
                                const int8_t *      q8 = (const int8_t *) y[j] + iby * QK8_1;
                                const sycl::half2 * ds =
                                    (const sycl::half2 *) (y[j] + ncols + iby * sizeof(sycl::half2));
                                ps[j][0] += VD::dot(wx, q8, ds, iqs);
                            }
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < NC; ++j) {
                            if (j < nc) {
                                const int8_t *      q8 = (const int8_t *) y[j] + iby * QK8_1;
                                const sycl::half2 * ds =
                                    (const sycl::half2 *) (y[j] + ncols + iby * sizeof(sycl::half2));
                                if constexpr (is_q8_0) {
                                    ps[j][0] += VD()(vx, bx_offset, d_offset, q8, ds, iqs, vdr_mmvq);
                                } else {
                                    ps[j][0] += VD()(vx, bx_offset, d_offset, q8, ds, iqs);
                                }
                            }
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (j < nc) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float sum = ps[j][r];
                    if constexpr (is_q8_0) {
                        // the accumulation order of the single-token kernel
                        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
                            sum += dpct::permute_sub_group_by_xor(sg, sum, mask);
                        }
                    } else {
                        sum = sycl::reduce_over_group(sg, sum, std::plus<>());
                    }
                    if (lane == 0 && row0 + r < nrows) {
                        ((float *) (a.dst + doff[j]))[row0 + r] = sum;
                    }
                }
            }
        }
    }
}

template <int R, typename F> void mmid_mc_launch(const mmid_mc_args & a, dpct::queue_ptr stream, F body) {
    const int            rows_per_wg = MMID_MC_NSG * R;
    const int            n_wg_rows   = (a.nrows + rows_per_wg - 1) / rows_per_wg;
    const sycl::range<2> local(1, MMID_MC_NSG * WARP_SIZE);
    const sycl::range<2> global((size_t) a.n_slots, (size_t) n_wg_rows * MMID_MC_NSG * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] { body(a, it); });
    });
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
void mmid_mc_plain_nc(const mmid_mc_args & a, int nc, dpct::queue_ptr stream) {
    auto go = [&](auto tag) {
        constexpr int NC = decltype(tag)::value;
        mmid_mc_launch<1>(a, stream, [](const mmid_mc_args & a, const sycl::nd_item<2> & it) {
            mmid_mc_plain<qk, qi, block_q_t, vdr, vec_dot_q_sycl, NC>(a, it);
        });
    };
    if (nc <= 2) {
        go(std::integral_constant<int, 2>{});
    } else if (nc <= 4) {
        go(std::integral_constant<int, 4>{});
    } else {
        go(std::integral_constant<int, 8>{});
    }
}

template <ggml_type T> void mmid_mc_reorder_nc(const mmid_mc_args & a, int nc, dpct::queue_ptr stream) {
    using VD          = reorder_vec_dot_q_sycl<T>;
    constexpr int R   = reorder_vec_dot_shared_activations<T>::value ? 2 : 1;
    auto          go  = [&](auto tag) {
        constexpr int NC = decltype(tag)::value;
        mmid_mc_launch<R>(a, stream, [](const mmid_mc_args & a, const sycl::nd_item<2> & it) {
            mmid_mc_reorder<VD, NC, R>(a, it);
        });
    };
    if (nc <= 2) {
        go(std::integral_constant<int, 2>{});
    } else if (nc <= 4) {
        go(std::integral_constant<int, 4>{});
    } else {
        go(std::integral_constant<int, 8>{});
    }
}


#ifdef GGML_SYCL_MMVQ_HAS_XMX
// src1 rows of the routes, quantized to q8_1 in the per-row SoA layout (quants, then the (d, s) pairs) and packed in
// schedule order, so the columns of an expert are consecutive rows q8_row_bytes apart: the 2D block loads of the XMX
// kernel read them as a matrix. The arithmetic is quantize_q8_1_impl's.
void mmid_mc_pack_q8(const char * src1, size_t nb11, size_t nb12, int ne11, int n_ids, int ne10, const int32_t * sched,
                     int n_slots, int n_routes, char * out, size_t q8_row_bytes, dpct::queue_ptr stream) {
    constexpr int EPW = QK8_1 / WARP_SIZE;
    const int     bpr = ne10 / QK8_1;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_routes * bpr * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int       g        = (int) it.get_group(0);
            const int       p        = g / bpr;
            const int       col      = g - p * bpr;
            const int32_t * slot_off = sched + n_slots;
            if (p >= slot_off[n_slots]) {
                return;
            }
            const int     r    = slot_off[n_slots + 1 + p];
            const int     t    = r / n_ids;
            const int     s    = r - t * n_ids;
            const int     lane = (int) it.get_local_id(0);
            const float * x    = (const float *) (src1 + (size_t) t * nb12 + (size_t) (ne11 == 1 ? 0 : s) * nb11) +
                              col * QK8_1 + lane * EPW;
            sycl::vec<float, EPW>  v = *reinterpret_cast<const sycl::vec<float, EPW> *>(x);
            sycl::vec<int8_t, EPW> q;
            float                  sum  = 0.0f;
            float                  amax = 0.0f;
#pragma unroll
            for (int i = 0; i < EPW; i++) {
                sum += v[i];
                amax = sycl::fmax(amax, sycl::fabs(v[i]));
            }
            sum     = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
            amax    = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
            float d = amax == 0 ? 1 : amax / 127;
#pragma unroll
            for (int i = 0; i < EPW; i++) {
                q[i] = sycl::round(v[i] / d);
            }
            d          = amax == 0 ? 0 : d;
            char * row = out + (size_t) p * q8_row_bytes;
            *reinterpret_cast<sycl::vec<int8_t, EPW> *>(row + col * QK8_1 + lane * EPW) = q;
            if (lane == 0) {
                *(sycl::half2 *) (row + ne10 + col * sizeof(sycl::half2)) = sycl::half2(sycl::half(d), sycl::half(sum));
            }
        });
}

// The XMX mul_mat_vec of esimd.hpp (xmx_mul_mat) for the routes of one active expert: a work-group is KS threads
// splitting K for one 16-row tile of one schedule slot, the A operand is the expert's packed activation rows
// (mmid_mc_pack_q8), NC columns per pass.
template <typename T, int NC, int KS>
ESIMD_INLINE void mmid_xmx(const mmid_mc_args & a, const char * vyp, const int slot, const int tile, const int lid,
                           const int ks) {
    using namespace sycl::ext::intel::esimd;
    namespace ge = ggml_sycl_esimd;

    constexpr int  NT   = ge::XMX_ROWS;
    constexpr int  RC   = NC < 8 ? NC : 8;
    constexpr int  NB   = NC / RC;
    constexpr bool pair = (RC * 32) % 64 == 0;

    if constexpr (KS > 1) {
        slm_init<KS * NC * NT * sizeof(float)>();
    }

    const int e = a.sched[slot];
    if (e < 0) {
        return;
    }
    const int32_t * slot_off = a.sched + a.n_slots;
    const int32_t * routes   = slot_off + a.n_slots + 1;
    const int       beg      = slot_off[slot];
    const int       end      = slot_off[slot + 1];

    const int  ncols = a.ncols;
    const int  nrows = a.nrows;
    const int  bpr   = ncols / QK_K;
    const int  row0  = tile * NT;
    const auto c     = T::make(a.vx_base + (size_t) e * a.expert_stride, nrows, bpr, row0);

    simd<uint32_t, NT> rows;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        rows[n] = row0 + n < nrows ? row0 + n : nrows - 1;
    }

    const unsigned yqw = ncols - 1;
    const unsigned ydw = ncols / QK8_1 * sizeof(sycl::half2) - 1;
    const unsigned yp  = (unsigned) a.q8_row_bytes - 1;

    simd<int8_t, 512> Bl = int8_t(0);
    simd<int8_t, 512> Bh = int8_t(0);
    simd<int8_t, 512> Ol = int8_t(0);
    if constexpr (T::has_min) {
        Ol = int8_t(1);
    }

    for (int c0 = beg; c0 < end; c0 += NC) {
        const int        ncv = end - c0 < NC ? end - c0 : NC;
        const char *     vy  = vyp + (size_t) c0 * a.q8_row_bytes;
        const uint32_t * yq  = (const uint32_t *) vy;
        const uint32_t * yd  = (const uint32_t *) (vy + ncols);
        const unsigned   yh  = ncv - 1;

        simd<float, NC * NT> Cf = 0.0f;
        for (int sb = lid; sb < bpr; sb += ks) {
            auto sc = T::load_scales(c, sb, rows);

            simd<float, 8 * RC> d8f[NB];
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                simd<uint32_t, 8 * RC>    ds = load_2d<uint32_t, 8, RC, 1, false, false>(yd, ydw, yh, yp, sb * 8, b * RC);
                simd<sycl::half, 16 * RC> dh = ds.template bit_cast_view<sycl::half>();
                d8f[b] = convert<float>(simd<sycl::half, 8 * RC>(dh.template select<8 * RC, 2>(0)));
            }

#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const auto              pd = T::load_part(c, sb, part);
                simd<uint32_t, 16 * RC> A2[NB];
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const int gi = 4 * part + g;
                    if constexpr (pair) {
                        if (g % 2 == 0) {
#pragma unroll
                            for (int b = 0; b < NB; ++b) {
                                A2[b] = load_2d<uint32_t, 8, RC, 2, false, false>(yq, yqw, yh, yp, sb * 64 + gi * 8,
                                                                                  b * RC);
                            }
                        }
                    } else {
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            A2[b].template select<RC * 8, 1>(0) =
                                load_2d<uint32_t, 8, RC, 1, false, false>(yq, yqw, yh, yp, sb * 64 + gi * 8, b * RC);
                        }
                    }
                    ge::xmx_group<T, NC>(Cf, pd, sc, A2, d8f, Bl, Bh, Ol, g, part);
                }
            }
        }

        if constexpr (KS > 1) {
            slm_block_store<float, NC * NT>(lid * NC * NT * sizeof(float), Cf);
            barrier();
            if (lid == 0) {
                Cf = 0.0f;
#pragma unroll 4
                for (int j = 0; j < ks; ++j) {
                    Cf += slm_block_load<float, NC * NT>(j * NC * NT * sizeof(float));
                }
            }
        }

        if (lid == 0) {
            simd<uint32_t, NT> lane(0, 1);
            simd_mask<NT>      in_rows = (lane + row0) < (uint32_t) nrows;
#pragma unroll
            for (int m = 0; m < NC; ++m) {
                if (m < ncv) {
                    const int r = routes[c0 + m];
                    const int t = r / a.n_ids;
                    const int s = r - t * a.n_ids;
                    float *   d = (float *) (a.dst + (size_t) s * a.nb1 + (size_t) t * a.nb2) + row0;
                    if (row0 + NT <= nrows && (uintptr_t) d % 16 == 0) {
                        block_store<float, NT>(d, Cf.template select<NT, 1>(m * NT));
                    } else {
                        scatter<float, NT>(d, lane * (uint32_t) sizeof(float), Cf.template select<NT, 1>(m * NT),
                                           in_rows);
                    }
                }
            }
        }
        if constexpr (KS > 1) {
            // the shared memory is reused by the next pass
            barrier();
        }
    }
}

template <typename T, int NC> void mmid_xmx_launch(const mmid_mc_args & a, const char * vyp, dpct::queue_ptr stream) {
    constexpr int KS_MAX  = ggml_sycl_esimd::xmx_n_split<NC>();
    const int     n_tiles = (a.nrows + ggml_sycl_esimd::XMX_ROWS - 1) / ggml_sycl_esimd::XMX_ROWS;
    const int     bpr     = a.ncols / QK_K;
    const int     rounds  = (bpr + KS_MAX - 1) / KS_MAX;
    const int     ks      = (bpr + rounds - 1) / rounds;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) a.n_slots, (size_t) n_tiles * ks), sycl::range<2>(1, ks)),
            sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_size<ggml_sycl_esimd::xmx_grf<T, NC, false>()> },
            [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
                mmid_xmx<T, NC, KS_MAX>(a, vyp, (int) it.get_group(0), (int) it.get_group(1), (int) it.get_local_id(1),
                                        ks);
            });
    });
}

template <template <bool> class Traits>
void mmid_xmx_nc(bool short_rows, const mmid_mc_args & a, const char * vyp, int nc, dpct::queue_ptr stream) {
    auto go = [&](auto tag) {
        constexpr int NC = decltype(tag)::value;
        if (short_rows) {
            mmid_xmx_launch<Traits<true>, NC>(a, vyp, stream);
        } else {
            mmid_xmx_launch<Traits<false>, NC>(a, vyp, stream);
        }
    };
    if (nc <= 2) {
        go(std::integral_constant<int, 2>{});
    } else if (nc <= 4) {
        go(std::integral_constant<int, 4>{});
    } else {
        go(std::integral_constant<int, 8>{});
    }
}

// Formats of groups of 32 weights with one scale each, for the part-wise XMX loop below (mmid_xmx_p): a part is 4
// groups (128 weights of K), so K needs only to be a multiple of 128. They plug into ggml_sycl_esimd::xmx_group as a
// "plain" quant type with the part's 4 groups at indices 0..3.
using sycl::ext::intel::esimd::simd;

struct mmid_xmx_plain_base {
    static constexpr bool has_min = false;
    static constexpr bool plain   = true;
    static constexpr bool min16   = false;
    struct scales_t {
        ggml_sycl_esimd::xmx_row_scales<8> d;  // the 4 groups of the part in 0..3
    };
};

// Q8_0 reorder layout: [qs: nrows*K bytes] [d: one half per 32 weights], see xmx_traits_q8_0
struct mmid_xmx_q8_0 : mmid_xmx_plain_base {
    struct ctx {
        const uint8_t *    qs;
        const sycl::half * dh;
        unsigned           pq, h;
        int                row0, gpr;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int ncols, int row0) {
        ctx c;
        c.qs   = (const uint8_t *) vx;
        c.dh   = (const sycl::half *) (c.qs + (size_t) nrows * ncols);
        c.pq   = ncols - 1;
        c.h    = nrows - 1;
        c.row0 = row0;
        c.gpr  = ncols / QK8_0;
        return c;
    }

    struct part_t {
        simd<uint32_t, 128> q[4];
    };

    static ESIMD_INLINE void load(const ctx & c, int pi, const simd<uint32_t, 16> & rows, part_t & p, scales_t & s) {
        using namespace sycl::ext::intel::esimd;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            p.q[g] = ggml_sycl_esimd::xmx_load_t<8, false>(c.qs, c.pq, c.h, pi * 32 + 8 * g, c.row0);
        }
        const simd<uint32_t, 16> blk = rows * c.gpr + pi * 4;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            s.d.v[g] = convert<float>(gather<sycl::half, 16>(c.dh, (blk + g) * (uint32_t) sizeof(sycl::half)));
        }
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) { return p.q[g]; }
};

// Q5_0 standard layout: 22-byte blocks {half d, uint32 qh, uint8 qs[16]}. Rows are 4-byte aligned (K % 64 == 0), so
// the 88 bytes of a part of 4 blocks come in as 22 dword gathers per 16 rows; blocks 0 and 2 of a part start at a
// dword, blocks 1 and 3 at a half dword. Weight k is (qs[k % 16] nibble k / 16) | bit k of qh << 4, minus 16.
// VEC (GGML_SYCL_MMID_Q5_0_VEC, default on): the rows' dwords come in as vector gathers
template <bool VEC> struct mmid_xmx_q5_0_t : mmid_xmx_plain_base {
    struct ctx {
        const uint32_t * x;
        uint32_t         row_bytes;
    };

    static ESIMD_INLINE ctx make(const void * vx, int /*nrows*/, int ncols, int /*row0*/) {
        ctx c;
        c.x         = (const uint32_t *) vx;
        c.row_bytes = (uint32_t) ((ncols / QK5_0) * sizeof(block_q5_0));
        return c;
    }

    struct part_t {
        simd<uint32_t, 16> qh[4];
        simd<uint32_t, 16> qs[4][4];  // dword j of the 16 qs bytes, per group
    };

    // bytes 2..5 of the dword pair (lo, hi)
    static ESIMD_INLINE simd<uint32_t, 16> mid(const simd<uint32_t, 16> & lo, const simd<uint32_t, 16> & hi) {
        return (lo >> 16) | (hi << 16);
    }

    static ESIMD_INLINE void set_d(scales_t & s, int g, const simd<uint32_t, 16> & bits) {
        simd<uint16_t, 16> db = convert<uint16_t>(bits);
        s.d.v[g]              = convert<float>(simd<sycl::half, 16>(db.template bit_cast_view<sycl::half>()));
    }

    static ESIMD_INLINE void load(const ctx & c, int pi, const simd<uint32_t, 16> & rows, part_t & p, scales_t & s) {
        using namespace sycl::ext::intel::esimd;
        static_assert(sizeof(block_q5_0) == 22, "");
        const simd<uint32_t, 16> off = rows * c.row_bytes + (uint32_t) (pi * 4 * sizeof(block_q5_0));
        simd<uint32_t, 16> w[22];
        if constexpr (VEC) {
            // the part's 22 dwords a row as 4 vector gathers (8 + 8 + 4 + 2 dwords a lane, [dword][lane]) rather than
            // 22 one-dword gathers: the same bits in a sixth of the messages
            simd<uint32_t, 128> v0 = gather<uint32_t, 128, 8>(c.x, off);
            simd<uint32_t, 128> v1 = gather<uint32_t, 128, 8>(c.x, off + 32);
            simd<uint32_t, 64>  v2 = gather<uint32_t, 64, 4>(c.x, off + 64);
            simd<uint32_t, 32>  v3 = gather<uint32_t, 32, 2>(c.x, off + 80);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                w[k]     = v0.template select<16, 1>(16 * k);
                w[8 + k] = v1.template select<16, 1>(16 * k);
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                w[16 + k] = v2.template select<16, 1>(16 * k);
            }
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                w[20 + k] = v3.template select<16, 1>(16 * k);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 22; ++k) {
                w[k] = gather<uint32_t, 16>(c.x, off + 4 * k);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int o = 11 * h;  // dword of block 2h, block 2h+1 starts at byte 22 past it
            set_d(s, 2 * h, w[o] & 0xFFFFu);
            p.qh[2 * h] = mid(w[o], w[o + 1]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p.qs[2 * h][j] = mid(w[o + 1 + j], w[o + 2 + j]);
            }
            set_d(s, 2 * h + 1, w[o + 5] >> 16);
            p.qh[2 * h + 1] = w[o + 6];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p.qs[2 * h + 1][j] = w[o + 7 + j];
            }
        }
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) {
        simd<uint32_t, 128> x;
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            const simd<uint32_t, 16> nib = kd < 4 ? (p.qs[g][kd] & 0x0F0F0F0Fu) : ((p.qs[g][kd - 4] >> 4) & 0x0F0F0F0Fu);
            const simd<uint32_t, 16> hb4 = (p.qh[g] >> (4 * kd)) & 0xFu;
            const simd<uint32_t, 16> hb  = ((hb4 * 0x00204081u) & 0x01010101u) << 4;
            x.template select<16, 1>(kd * 16) = nib | hb;
        }
        return ggml_sycl_esimd::xmx_bytes_sub(x, 0x10101010u);
    }
};

// mmid_xmx over parts of 128 weights for the P formats above
template <typename P, int NC, int KS>
ESIMD_INLINE void mmid_xmx_p(const mmid_mc_args & a, const char * vyp, const int slot, const int tile, const int lid,
                             const int ks) {
    using namespace sycl::ext::intel::esimd;
    namespace ge = ggml_sycl_esimd;

    constexpr int  NT   = ge::XMX_ROWS;
    constexpr int  RC   = NC < 8 ? NC : 8;
    constexpr int  NB   = NC / RC;
    constexpr bool pair = (RC * 32) % 64 == 0;

    if constexpr (KS > 1) {
        slm_init<KS * NC * NT * sizeof(float)>();
    }

    const int e = a.sched[slot];
    if (e < 0) {
        return;
    }
    const int32_t * slot_off = a.sched + a.n_slots;
    const int32_t * routes   = slot_off + a.n_slots + 1;
    const int       beg      = slot_off[slot];
    const int       end      = slot_off[slot + 1];

    const int  ncols  = a.ncols;
    const int  nrows  = a.nrows;
    const int  nparts = ncols / 128;
    const int  row0   = tile * NT;
    const auto c      = P::make(a.vx_base + (size_t) e * a.expert_stride, nrows, ncols, row0);

    simd<uint32_t, NT> rows;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        rows[n] = row0 + n < nrows ? row0 + n : nrows - 1;
    }

    const unsigned yqw = ncols - 1;
    const unsigned ydw = ncols / QK8_1 * sizeof(sycl::half2) - 1;
    const unsigned yp  = (unsigned) a.q8_row_bytes - 1;

    simd<int8_t, 512> Bl = int8_t(0);
    simd<int8_t, 512> Bh = int8_t(0);
    simd<int8_t, 512> Ol = int8_t(0);

    for (int c0 = beg; c0 < end; c0 += NC) {
        const int        ncv = end - c0 < NC ? end - c0 : NC;
        const char *     vy  = vyp + (size_t) c0 * a.q8_row_bytes;
        const uint32_t * yq  = (const uint32_t *) vy;
        const uint32_t * yd  = (const uint32_t *) (vy + ncols);
        const unsigned   yh  = ncv - 1;

        simd<float, NC * NT> Cf = 0.0f;
        for (int pi = lid; pi < nparts; pi += ks) {
            typename P::part_t   pd;
            typename P::scales_t sc;
            P::load(c, pi, rows, pd, sc);

            // d of the part's 4 groups (and the next 4, unused), [column * 8 + group]
            simd<float, 8 * RC> d8f[NB];
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                simd<uint32_t, 8 * RC>    ds = load_2d<uint32_t, 8, RC, 1, false, false>(yd, ydw, yh, yp, pi * 4, b * RC);
                simd<sycl::half, 16 * RC> dh = ds.template bit_cast_view<sycl::half>();
                d8f[b] = convert<float>(simd<sycl::half, 8 * RC>(dh.template select<8 * RC, 2>(0)));
            }

            simd<uint32_t, 16 * RC> A2[NB];
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                if constexpr (pair) {
                    if (g % 2 == 0) {
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            A2[b] = load_2d<uint32_t, 8, RC, 2, false, false>(yq, yqw, yh, yp, pi * 32 + g * 8, b * RC);
                        }
                    }
                } else {
#pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        A2[b].template select<RC * 8, 1>(0) =
                            load_2d<uint32_t, 8, RC, 1, false, false>(yq, yqw, yh, yp, pi * 32 + g * 8, b * RC);
                    }
                }
                ge::xmx_group<P, NC>(Cf, pd, sc, A2, d8f, Bl, Bh, Ol, g, 0);
            }
        }

        if constexpr (KS > 1) {
            slm_block_store<float, NC * NT>(lid * NC * NT * sizeof(float), Cf);
            barrier();
            if (lid == 0) {
                Cf = 0.0f;
#pragma unroll 4
                for (int j = 0; j < ks; ++j) {
                    Cf += slm_block_load<float, NC * NT>(j * NC * NT * sizeof(float));
                }
            }
        }

        if (lid == 0) {
            simd<uint32_t, NT> lane(0, 1);
            simd_mask<NT>      in_rows = (lane + row0) < (uint32_t) nrows;
#pragma unroll
            for (int m = 0; m < NC; ++m) {
                if (m < ncv) {
                    const int r = routes[c0 + m];
                    const int t = r / a.n_ids;
                    const int s = r - t * a.n_ids;
                    float *   d = (float *) (a.dst + (size_t) s * a.nb1 + (size_t) t * a.nb2) + row0;
                    if (row0 + NT <= nrows && (uintptr_t) d % 16 == 0) {
                        block_store<float, NT>(d, Cf.template select<NT, 1>(m * NT));
                    } else {
                        scatter<float, NT>(d, lane * (uint32_t) sizeof(float), Cf.template select<NT, 1>(m * NT),
                                           in_rows);
                    }
                }
            }
        }
        if constexpr (KS > 1) {
            barrier();
        }
    }
}

template <typename P, int NC> void mmid_xmx_p_launch(const mmid_mc_args & a, const char * vyp, dpct::queue_ptr stream) {
    constexpr int KS_MAX  = ggml_sycl_esimd::xmx_n_split<NC>();
    const int     n_tiles = (a.nrows + ggml_sycl_esimd::XMX_ROWS - 1) / ggml_sycl_esimd::XMX_ROWS;
    const int     nparts  = a.ncols / 128;
    const int     rounds  = (nparts + KS_MAX - 1) / KS_MAX;
    const int     ks      = (nparts + rounds - 1) / rounds;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) a.n_slots, (size_t) n_tiles * ks), sycl::range<2>(1, ks)),
            sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<NC >= 8 ? 256 : 128> },
            [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
                mmid_xmx_p<P, NC, KS_MAX>(a, vyp, (int) it.get_group(0), (int) it.get_group(1),
                                          (int) it.get_local_id(1), ks);
            });
    });
}

template <typename P> void mmid_xmx_p_nc(const mmid_mc_args & a, const char * vyp, int nc, dpct::queue_ptr stream) {
    if (nc <= 2) {
        mmid_xmx_p_launch<P, 2>(a, vyp, stream);
    } else if (nc <= 4) {
        mmid_xmx_p_launch<P, 4>(a, vyp, stream);
    } else {
        mmid_xmx_p_launch<P, 8>(a, vyp, stream);
    }
}
#endif  // GGML_SYCL_MMVQ_HAS_XMX

}  // namespace

bool ggml_sycl_mmid_mc_xmx_supported(enum ggml_type src0_type, bool reordered, int ncols, size_t expert_stride,
                                     const void * vx_base) {
#ifdef GGML_SYCL_MMVQ_HAS_XMX
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return reordered && ncols % QK_K == 0 && expert_stride % 64 == 0 && (uintptr_t) vx_base % 64 == 0;
        case GGML_TYPE_Q8_0:
            // 2D block loads of the int8 rows: a 64 byte aligned base, a pitch of 16 bytes
            return reordered && ncols % 128 == 0 && expert_stride % 64 == 0 && (uintptr_t) vx_base % 64 == 0;
        case GGML_TYPE_Q5_0:
            return !reordered && ncols % 128 == 0 && expert_stride % 4 == 0 && (uintptr_t) vx_base % 4 == 0;
        default:
            return false;
    }
#else
    GGML_UNUSED_VARS(src0_type, ncols, expert_stride, vx_base);
    return false;
#endif
}

void ggml_sycl_mmid_mc_xmx(enum ggml_type src0_type, const void * vx_base, size_t expert_stride, int ncols, int nrows,
                           const char * src1, size_t nb11, size_t nb12, int ne11, char * q8_packed,
                           size_t q8_row_bytes, const int32_t * ids, size_t ids_nb1, int n_ids, int n_tokens, int n_as,
                           float * dst, size_t nb1, size_t nb2, int32_t * sched, bool sched_ready,
                           dpct::queue_ptr stream) {
#ifdef GGML_SYCL_MMVQ_HAS_XMX
    const int n_routes = n_ids * n_tokens;
    const int n_slots  = std::min(n_as, n_routes);
    if (!sched_ready) {
        mmid_mc_schedule(ids, ids_nb1, n_ids, n_tokens, n_as, sched, dst, nb1, nb2, nrows, stream);
    }
    mmid_mc_pack_q8(src1, nb11, nb12, ne11, n_ids, ncols, sched, n_slots, n_routes, q8_packed, q8_row_bytes, stream);

    mmid_mc_args a;
    a.vx_base       = (const char *) vx_base;
    a.expert_stride = expert_stride;
    a.ncols         = ncols;
    a.nrows         = nrows;
    a.vy            = q8_packed;
    a.q8_row_bytes  = q8_row_bytes;
    a.ne11          = ne11;
    a.n_ids         = n_ids;
    a.dst           = (char *) dst;
    a.nb1           = nb1;
    a.nb2           = nb2;
    a.sched         = sched;
    a.n_slots       = n_slots;

    namespace ge          = ggml_sycl_esimd;
    const bool short_rows = ncols < 4 * QK_K;
    switch (src0_type) {
        case GGML_TYPE_Q4_K: mmid_xmx_nc<ge::xmx_traits_q4_k>(short_rows, a, q8_packed, n_tokens, stream); return;
        case GGML_TYPE_Q5_K: mmid_xmx_nc<ge::xmx_traits_q5_k>(short_rows, a, q8_packed, n_tokens, stream); return;
        case GGML_TYPE_Q6_K: mmid_xmx_nc<ge::xmx_traits_q6_k>(short_rows, a, q8_packed, n_tokens, stream); return;
        case GGML_TYPE_Q8_0: mmid_xmx_p_nc<mmid_xmx_q8_0>(a, q8_packed, n_tokens, stream); return;
        case GGML_TYPE_Q5_0:
            {
                static const bool vec = [] {
                    const char * e = getenv("GGML_SYCL_MMID_Q5_0_VEC");
                    return !e || atoi(e) != 0;
                }();
                if (vec) {
                    mmid_xmx_p_nc<mmid_xmx_q5_0_t<true>>(a, q8_packed, n_tokens, stream);
                } else {
                    mmid_xmx_p_nc<mmid_xmx_q5_0_t<false>>(a, q8_packed, n_tokens, stream);
                }
            }
            return;
        default: GGML_ABORT("mmid_mc_xmx: unsupported type");
    }
#else
    GGML_UNUSED_VARS(src0_type, vx_base, expert_stride, ncols, nrows, src1, nb11, nb12, ne11, q8_packed, q8_row_bytes,
                     ids, ids_nb1, n_ids, n_tokens, n_as, dst, nb1, nb2, sched, sched_ready, stream);
    GGML_ABORT("mmid_mc_xmx: built without XMX");
#endif
}

bool ggml_sycl_mmid_mc_supports_type(enum ggml_type src0_type, bool reordered) {
    switch (src0_type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
            return !reordered;
        default:
            return false;
    }
}

size_t ggml_sycl_mmid_mc_sched_ints(int n_as, int n_routes) {
    const int n_slots = std::min(n_as, n_routes);
    return (size_t) n_slots + (size_t) n_slots + 1 + (size_t) n_routes;
}

void ggml_sycl_mmid_mc(enum ggml_type src0_type, bool reordered, const void * vx_base, size_t expert_stride,
                       int ncols, int nrows, const void * vy, size_t q8_row_bytes, int ne11, const int32_t * ids,
                       size_t ids_nb1, int n_ids, int n_tokens, int n_as, float * dst, size_t nb1, size_t nb2,
                       int32_t * sched, bool sched_ready, dpct::queue_ptr stream) {
    GGML_ASSERT(ggml_sycl_mmid_mc_supports_type(src0_type, reordered));
    GGML_ASSERT(n_as <= GGML_SYCL_MMID_MC_MAX_EXPERTS && n_ids * n_tokens <= GGML_SYCL_MMID_MC_MAX_ROUTES);
    if (!sched_ready) {
        mmid_mc_schedule(ids, ids_nb1, n_ids, n_tokens, n_as, sched, dst, nb1, nb2, nrows, stream);
    }

    mmid_mc_args a;
    a.vx_base       = (const char *) vx_base;
    a.expert_stride = expert_stride;
    a.ncols         = ncols;
    a.nrows         = nrows;
    a.vy            = (const char *) vy;
    a.q8_row_bytes  = q8_row_bytes;
    a.ne11          = ne11;
    a.n_ids         = n_ids;
    a.dst           = (char *) dst;
    a.nb1           = nb1;
    a.nb2           = nb2;
    a.sched         = sched;
    a.n_slots       = std::min(n_as, n_ids * n_tokens);

    const int nc = n_tokens;
    if (reordered) {
        switch (src0_type) {
            case GGML_TYPE_Q8_0: mmid_mc_reorder_nc<GGML_TYPE_Q8_0>(a, nc, stream); return;
            case GGML_TYPE_Q4_K: mmid_mc_reorder_nc<GGML_TYPE_Q4_K>(a, nc, stream); return;
            case GGML_TYPE_Q5_K: mmid_mc_reorder_nc<GGML_TYPE_Q5_K>(a, nc, stream); return;
            case GGML_TYPE_Q6_K: mmid_mc_reorder_nc<GGML_TYPE_Q6_K>(a, nc, stream); return;
            default: GGML_ABORT("mmid_mc: unsupported reordered type");
        }
    }
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
            mmid_mc_plain_nc<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(a, nc, stream);
            return;
        case GGML_TYPE_Q4_1:
            mmid_mc_plain_nc<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(a, nc, stream);
            return;
        case GGML_TYPE_Q5_0:
            mmid_mc_plain_nc<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(a, nc, stream);
            return;
        case GGML_TYPE_Q5_1:
            mmid_mc_plain_nc<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(a, nc, stream);
            return;
        case GGML_TYPE_Q8_0:
            mmid_mc_plain_nc<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(a, nc, stream);
            return;
        case GGML_TYPE_Q4_K:
            mmid_mc_plain_nc<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(a, nc, stream);
            return;
        case GGML_TYPE_Q5_K:
            mmid_mc_plain_nc<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(a, nc, stream);
            return;
        case GGML_TYPE_Q6_K:
            mmid_mc_plain_nc<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(a, nc, stream);
            return;
        default:
            GGML_ABORT("mmid_mc: unsupported type");
    }
}
