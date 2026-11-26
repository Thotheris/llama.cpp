#include "fattn-rel-xe.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cfloat>
#include <cstdlib>

// The layout (see fattn-rel-xe.hpp for the idea). A work-group is NSG threads (ESIMD: one work-item per thread) on
// 16 NSG consecutive query rows of one head; thread t owns the 16 rows row0 .. row0 + 15, as two 8-row DPAS blocks mb,
// and all 128 output dims, as 8 accumulators of 16 dims per block. Per 32-cell tile:
//   S (16 rows x 32 cells) = Q K^T          A = Q (8 rows x 16 dims, f16, from this thread's SLM),
//                                           B = K^T (16 dims x 16 cells): a transposed 2D load of K as dim pairs
//   + bias                                  per the tile flags (below), into S's layout (row m in register m, lane
//                                           = cell)
//   P = exp(S - max), row max and sum by 2D register regions, no other thread involved
//   O (16 rows x 128) = O * rescale + P V   A = P (f16, converted in registers), B = V (16 cells x 16 dims): a VNNI
//                                           2D load of V
// Nothing goes through SLM except Q (written once). Cells past nek1 read as zeros (the surface ends there) and are
// masked; rows past neq1 compute copies of the last row and are not stored (the 2D store drops rows past the surface).
//
// The bias, per tile, from the flags of fattn_rel_tiles_aff for the thread's 64-row block:
//   uniform: one table column for every cell, the bias of each row computed once;
//   affine:  distance D = B + rr - j (rr the row in the 64-row block, j the cell in the tile): the bias of the 48
//            distances the thread's rows and cells span, three DPAS of v (A, the rows) and table columns (B, lanes =
//            distances, descending), then each row takes 16 of them by a register region at a compile-time offset;
//            cells with D < 0 or D >= W masked (checked only in tiles that reach a boundary);
//   otherwise (several sequences in the block, a wrapped cache): per row the B operand of the 16 cells' columns,
//            gathered by pos, one DPAS each, of which the row takes its own result row.
// The t term: x16 t + x17 when it is the same in every column (hdr xc), else a second DPAS of (t_hi, t_lo, 1) and
// the columns' rows 16, 17 (the affine path), and t tx.x + tx.y per cell elsewhere (as fa_rel_bias_col).

namespace syclex = sycl::ext::oneapi::experimental;

namespace {

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace esx = sycl::ext::intel::experimental::esimd;

constexpr int   XE_BC  = 32;               // cells per tile, the tile of the flags
constexpr int   XE_QB  = 8 * 128 * 2;      // the f16 Q of 8 rows in SLM, bytes
constexpr float XE_L2E = 1.4426950408889634f;
constexpr float XE_NEG = -FLT_MAX / 2;

// f(integral_constant<i>) for i = A .. B - 1: register offsets that are compile-time constants
template <int A, int B, typename F> ESIMD_INLINE void xe_for(F && f) {
    if constexpr (A < B) {
        f(std::integral_constant<int, A>{});
        xe_for<A + 1, B>(f);
    }
}

// 2D block messages on a payload built once (base, surface, shape) with the block offsets as immediates, so a load
// costs no payload moves: payload words 0-1 base, 2 width - 1 (bytes), 3 height - 1, 4 pitch - 1, 5 X, 6 Y, 7 shape
ESIMD_INLINE simd<uint32_t, 16> xe_payload(const void * base, uint32_t w1, uint32_t h1, uint32_t p1, int x, int y,
                                           int bw, int bh) {
    simd<uint32_t, 16> pl = 0;
    pl.template bit_cast_view<uint64_t>()[0] = (uint64_t) base;
    pl[2] = w1;
    pl[3] = h1;
    pl[4] = p1;
    pl[5] = (uint32_t) x;
    pl[6] = (uint32_t) y;
    pl[7] = (uint32_t) (((bh - 1) << 8) | (bw - 1));
    return pl;
}
constexpr __ESIMD_DNS::vector_type_t<uint8_t, 2> XE_CA = { (uint8_t) cache_hint::cached, (uint8_t) cache_hint::cached };
// K^T as B operands: a transposed load of 16 rows (cells) x 8 dwords (dim pairs)
template <int XO, int YO> ESIMD_INLINE simd<uint32_t, 128> xe_ld_kt(simd<uint32_t, 16> & pl) {
    simd<uint32_t, 128> old;
    return __esimd_lsc_load2d_descriptor_transpose<uint32_t, 1, 8, 16, XO, YO, 128>(1, pl.data(), old.data(), XE_CA);
}
// V as B operands: a VNNI load of 32 rows (cells) x 16 halfs (dims)
template <int XO, int YO> ESIMD_INLINE simd<uint16_t, 512> xe_ld_vn(simd<uint32_t, 16> & pl) {
    simd<uint16_t, 512> old;
    return __esimd_lsc_load2d_descriptor_transform<uint16_t, 1, 16, 32, XO, YO, 512>(1, pl.data(), old.data(), XE_CA);
}
// a 32 rows x 64 bytes block into the cache
template <int XO, int YO> ESIMD_INLINE void xe_pf(simd<uint32_t, 16> & pl) {
    simd<uint32_t, 512> old;
    __esimd_lsc_prefetch_descriptor<uint32_t, 1, 16, 32, XO, YO, 512>(1, pl.data(), old.data(), XE_CA);
}

ESIMD_INLINE int xe_min(int a, int b) { return a < b ? a : b; }
ESIMD_INLINE int xe_clamp(int x, int a, int b) { return x < a ? a : (x > b ? b : x); }

// half2(a[i], b[i]) per lane
ESIMD_INLINE simd<uint32_t, 16> xe_pack_h2(simd<float, 16> a, simd<float, 16> b) {
    simd<sycl::half, 32> h;
    h.template select<16, 2>(0) = simd<sycl::half, 16>(a);
    h.template select<16, 2>(1) = simd<sycl::half, 16>(b);
    return h.template bit_cast_view<uint32_t>();
}

// the row maxima / sums of R rows x 16 lanes (row r in elements 16 r .. 16 r + 15), in row order
template <int R, bool MAX> ESIMD_INLINE simd<float, R> xe_rows(simd<float, R * 16> t) {
    auto red = [](auto x, auto y) {
        if constexpr (MAX) {
            return max(x, y);
        } else {
            return decltype(x)(x + y);
        }
    };
    auto a = t.template bit_cast_view<float, R, 16>();
    simd<float, R * 8> t8 = red(a.template select<R, 1, 8, 1>(0, 0).read(), a.template select<R, 1, 8, 1>(0, 8).read());
    auto b = t8.template bit_cast_view<float, R, 8>();
    simd<float, R * 4> t4 = red(b.template select<R, 1, 4, 1>(0, 0).read(), b.template select<R, 1, 4, 1>(0, 4).read());
    auto c = t4.template bit_cast_view<float, R, 4>();
    simd<float, R * 2> t2 = red(c.template select<R, 1, 2, 1>(0, 0).read(), c.template select<R, 1, 2, 1>(0, 2).read());
    return red(t2.template select<R, 2>(0).read(), t2.template select<R, 2>(1).read());
}

template <int NSG, int MB, bool UNALIGNED, bool SB> struct fa_rel_xe_kernel {
    static constexpr int R = 8 * MB;  // query rows per thread
    ggml_sycl_fa_xe_params p;
    int                    n_wr;  // work-group row blocks per stream
    float                  thr;   // lazy rescale: keep a row's max until the tile max passes it by thr (0: exact)
    int                    pfd;   // prefetch distance in tiles

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL { run(it); }

    ESIMD_INLINE void run(sycl::nd_item<1> it) const {
        slm_init<NSG * MB * XE_QB>();
        constexpr auto cached = syclex::properties{ cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached> };

        const int tid    = (int) it.get_local_id(0);
        int       g      = (int) it.get_group(0);
        const int head_q = g % p.neq2;
        g /= p.neq2;
        // causal prefill: the last row blocks see the most cells; dispatch them first
        const int  out_r  = n_wr - 1 - g % n_wr;
        const int  batch  = g / n_wr;
        const int  head_k = head_q / p.gqa;
        const int  row0w  = out_r * R * NSG;
        const int  row0   = row0w + R * tid;
        const bool active = row0 < p.neq1;
        const int  blk    = active ? row0 / 64 : 0;
        const int  rr0    = row0 % 64;
        const int  nq1    = p.neq1;

        // Q * scale -> f16 A operands in this thread's SLM: [mb][dc][8 rows][16 dims]
        const uint32_t qslm = (uint32_t) tid * MB * XE_QB;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int     row = xe_min(row0 + r, nq1 - 1);
            const float * qr  = (const float *) (p.Q + batch * p.nbq3 + (int64_t) row * p.nbq1 + (int64_t) head_q * p.nbq2);
            simd<float, 128> q;
            q.template select<64, 1>(0)  = block_load<float, 64>(qr, overaligned_tag<16>{});
            q.template select<64, 1>(64) = block_load<float, 64>(qr + 64, overaligned_tag<16>{});
            q *= p.scale;
            simd<sycl::half, 128> qh = q;
#pragma unroll
            for (int dc = 0; dc < 8; ++dc) {
                slm_block_store<sycl::half, 16>(qslm + (((r / 8) * 8 + dc) * 128 + (r % 8) * 16) * 2,
                                                qh.template select<16, 1>(dc * 16));
            }
        }

        // v of the rows (the A operand of the bias DPAS), (t_hi, t_lo), (1, 0) for the t DPAS, t
        const int hv = head_q % p.pk_nhead;
        simd<sycl::half, 128> vA[MB];
        simd<uint32_t, 64>    tA[MB];
        simd<float, R>        tt;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
            tA[mb] = 0;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int        row = xe_min(row0 + r, nq1 - 1);
            const uint32_t * vp  = (const uint32_t *) (p.pk + p.pk_v) + ((int64_t) (batch * nq1 + row) * p.pk_nhead + hv) * 12;
            simd<uint32_t, 8> v8 = block_load<uint32_t, 8>(vp, overaligned_tag<16>{});
            simd<uint32_t, 4> v2 = block_load<uint32_t, 4>(vp + 8, overaligned_tag<16>{});
            vA[r / 8].template bit_cast_view<uint32_t>().template select<8, 1>((r % 8) * 8) = v8;
            tA[r / 8][(r % 8) * 8]     = (uint32_t) v2[0];
            tA[r / 8][(r % 8) * 8 + 1] = 0x3C00u; // half2(1, 0)
            tt[r]                      = sycl::bit_cast<float>((uint32_t) v2[1]);
        }
        simd<int, 4> h4  = block_load<int, 4>((const int *) (p.pk + p.pk_hdr), overaligned_tag<16>{});
        const bool   xc  = (int) h4[0] != 0;
        const float  x16 = sycl::bit_cast<float>((int) h4[1]);
        const float  x17 = sycl::bit_cast<float>((int) h4[2]);
        // the t term of the affine path when it is the same in every column (else 0 and a DPAS)
        simd<float, R> kc = 0.0f;
        if (xc) {
            kc = tt * x16 + x17;
        }

        const uint32_t * tf = (const uint32_t *) (p.pk + p.pk_tf);
        const float *    tx = (const float *) (p.pk + p.pk_tx);
        simd<int, 16> iota(0, 1);

        // the bias of a uniform tile: the column of u_pos, the pos of the block's first cell of its first row
        simd<float, R> buni;
        {
            const int u_pos = p.pos[(int64_t) (xe_min(blk * 64, nq1 - 1) + batch * nq1) * p.nbp1];
            const int ucol  = xe_clamp(u_pos, 0, p.pk_ncols - 1);
            simd<uint32_t, 8> tu = block_load<uint32_t, 8>(tf + (int64_t) ucol * 8, overaligned_tag<16>{});
            simd<uint32_t, 128> bu;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                bu.template select<16, 1>(i * 16) = (uint32_t) tu[i];
            }
            simd<sycl::half, 256> buh = bu.template bit_cast_view<sycl::half>();
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                simd<float, 128> a = xmx::dpas<8, 8, float>(buh, vA[mb]);
                buni.template select<8, 1>(mb * 8) = a.template select<8, 16>(0);
            }
            const float txx = tx[2 * ucol];
            const float txy = tx[2 * ucol + 1];
            buni += tt * txx + txy;
        }

        const char *   Kh = p.K + batch * p.nbk3 + (int64_t) head_k * p.nbk2;
        const char *   Vh = p.V + batch * p.nbv3 + (int64_t) head_k * p.nbv2;
        const uint32_t sh = (uint32_t) p.nek1 - 1;
        simd<uint32_t, 16> plk = xe_payload(Kh, 255u, sh, (uint32_t) p.nbk1 - 1, 0, 0, 8, 16);
        simd<uint32_t, 16> plv = xe_payload(Vh, 255u, sh, (uint32_t) p.nbv1 - 1, 0, 0, 16, 32);
        // the prefetch block of this thread (threads 0 .. 7: K, V x 4 column blocks of 64 bytes)
        const int  pj  = tid % 8;
        simd<uint32_t, 16> plp = xe_payload(pj < 4 ? Kh : Vh, 255u, sh, (uint32_t) (pj < 4 ? p.nbk1 : p.nbv1) - 1,
                                            (pj % 4) * 16, 0, 16, 32);

        simd<float, 128> O[MB][8];
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
#pragma unroll
            for (int db = 0; db < 8; ++db) {
                O[mb][db] = 0.0f;
            }
        }
        simd<float, R>   M = XE_NEG;
        simd<float, R>   L = 0.0f;

        const int          nw    = p.n_words;
        const int          blk0  = row0w / 64;
        const uint32_t *   tlb   = p.tl + (int64_t) (batch * p.n_rb64 + blk) * 3 * nw;
        const sycl::int2 * bwp   = p.bw + (int64_t) (batch * p.n_rb64 + blk) * p.n_tiles;
        const int          c_cl  = p.pk_c;

        for (int w = 0; w < nw; ++w) {
            // the work-group visits the tiles live in any of its blocks (every thread the same, for the barrier)
            uint32_t uw = 0;
#pragma unroll
            for (int b = 0; b < R * NSG / 64; ++b) {
                if (blk0 + b < p.n_rb64) {
                    uw |= p.tl[(int64_t) ((batch * p.n_rb64 + blk0 + b) * 3) * nw + w];
                }
            }
            const uint32_t wl = active ? tlb[w] : 0u;
            const uint32_t wn = tlb[nw + w];
            const uint32_t wa = tlb[2 * nw + w];
            for (int i = 0; i < 32; ++i) {
                if (((uw >> i) & 1u) == 0) {
                    continue;
                }
                const int tile = 32 * w + i;
                const int c0   = tile * XE_BC;
                if constexpr (SB) {
                    esx::split_barrier<esx::split_barrier_action::signal>();
                }
                if (tid < 8 && tile + pfd < p.n_tiles) {
                    plp[6] = (uint32_t) (c0 + pfd * XE_BC);
                    xe_pf<0, 0>(plp);
                }
                if (((wl >> i) & 1u) != 0) {
                    const bool uni = ((wn >> i) & 1u) == 0;
                    const bool aff = ((wa >> i) & 1u) != 0;
                    plk[6] = (uint32_t) c0;
                    plv[6] = (uint32_t) c0;

                    // K and Q of dim chunk 0 in flight during the bias
                    simd<uint32_t, 128>   kn0 = xe_ld_kt<0, 0>(plk);
                    simd<uint32_t, 128>   kn1 = xe_ld_kt<0, 16>(plk);
                    simd<sycl::half, 128> qn[MB];
#pragma unroll
                    for (int mb = 0; mb < MB; ++mb) {
                        qn[mb] = slm_block_load<sycl::half, 128>(qslm + (mb * 8) * 256);
                    }

                    // S = bias (-inf where masked), then S += Q K^T
                    simd<float, 128> S[MB][2];
                    if (uni) {
                        xe_for<0, R>([&](auto rc) {
                            constexpr int r = decltype(rc)::value;
                            const float   b = buni[r];
                            S[r / 8][0].template select<16, 1>((r % 8) * 16) = b;
                            S[r / 8][1].template select<16, 1>((r % 8) * 16) = b;
                        });
                    } else if (aff) {
                        const int * bwi = (const int *) (bwp + tile);
                        const int   B   = bwi[0];
                        const int   W   = bwi[1];
                        // a tile that reaches a boundary: the cells D < 0 or D >= W (or past nek1) are masked
                        const bool  edge = B + rr0 - 31 < 0 || B + rr0 + R - 1 >= W || (UNALIGNED && c0 + XE_BC > p.nek1);
                        xe_for<0, R>([&](auto rc) {
                            constexpr int r = decltype(rc)::value;
                            const float   k = kc[r];
                            S[r / 8][0].template select<16, 1>((r % 8) * 16) = k;
                            S[r / 8][1].template select<16, 1>((r % 8) * 16) = k;
                        });
                        // block q = 3 mb + jb, lane n: distance B + rr0 + 8 mb + 7 - 16 jb - n. Row m, cell 16 cb + l
                        // is at distance B + rr0 + 8 mb + m - 16 cb - l: lane o + l of block jb = cb (o = 7 - m) for
                        // l < 16 - o, lane l - 16 + o of block jb = cb + 1 for the rest; each block is added where it
                        // is needed. The table columns of block q + 1 are gathered during the DPAS of block q.
                        auto cols = [&](int q) {
                            simd<int, 16> d = (B + rr0 + 8 * (q / 3) + 7 - 16 * (q % 3)) - iota;
                            simd<int, 16> col = max(min(d, simd<int, 16>(c_cl)), simd<int, 16>(0));
                            return simd<uint32_t, 16>(col) * 32u;
                        };
                        simd<uint32_t, 16>  offn = cols(0);
                        simd<uint32_t, 128> tn   = gather<uint32_t, 128, 8>(tf, offn, cached);
                        xe_for<0, 3 * MB>([&](auto qc) {
                            constexpr int q  = decltype(qc)::value;
                            constexpr int mb = q / 3;
                            constexpr int jb = q % 3;
                            simd<uint32_t, 128> t32 = tn;
                            simd<uint32_t, 16>  off = offn;
                            if constexpr (q + 1 < 3 * MB) {
                                offn = cols(q + 1);
                                tn   = gather<uint32_t, 128, 8>(tf, offn, cached);
                            }
                            simd<sycl::half, 256> tb  = t32.template bit_cast_view<sycl::half>();
                            simd<float, 128>      acc = xmx::dpas<8, 8, float>(tb, vA[mb]);
                            if (!xc) {
                                simd<float, 32> x2 = gather<float, 32, 2>(tx, off / 4u, cached);
                                simd<uint32_t, 128> bx = 0;
                                bx.template select<16, 1>(0)  = xe_pack_h2(x2.template select<16, 1>(0), x2.template select<16, 1>(0));
                                bx.template select<16, 1>(16) = xe_pack_h2(x2.template select<16, 1>(16), simd<float, 16>(0.0f));
                                simd<sycl::half, 256> bxh = bx.template bit_cast_view<sycl::half>();
                                simd<sycl::half, 128> tah = tA[mb].template bit_cast_view<sycl::half>();
                                acc = xmx::dpas<8, 8, float, float>(acc, bxh, tah);
                            }
                            xe_for<0, 8>([&](auto mc) {
                                constexpr int m = decltype(mc)::value;
                                constexpr int o = 7 - m;
                                if constexpr (jb <= 1) {
                                    S[mb][jb].template select<16 - o, 1>(m * 16) += acc.template select<16 - o, 1>(m * 16 + o);
                                }
                                if constexpr (jb >= 1 && o > 0) {
                                    S[mb][jb - 1].template select<o, 1>(m * 16 + 16 - o) += acc.template select<o, 1>(m * 16);
                                }
                            });
                        });
                        if (edge) {
                            xe_for<0, R>([&](auto rc) {
                                constexpr int r = decltype(rc)::value;
                                xe_for<0, 2>([&](auto cbc) {
                                    constexpr int cb = decltype(cbc)::value;
                                    simd<int, 16> D  = (B + rr0 + r - 16 * cb) - iota;
                                    simd_mask<16> mk = (D < 0) | (D >= W);
                                    if constexpr (UNALIGNED) {
                                        mk |= (iota + (c0 + 16 * cb)) >= p.nek1;
                                    }
                                    S[r / 8][cb].template select<16, 1>((r % 8) * 16).merge(simd<float, 16>(-INFINITY), mk);
                                });
                            });
                        }
                    } else {
                        // several sequences in the block or a wrapped cache: the columns by pos, per row
                        xe_for<0, R>([&](auto rc) {
                            constexpr int r  = decltype(rc)::value;
                            constexpr int mb = r / 8;
                            constexpr int m  = r % 8;
                            const int       row  = xe_min(row0 + r, p.neq1 - 1);
                            const int32_t * prow = p.pos + (int64_t) (row + batch * p.neq1) * p.nbp1;
                            const float     t_r  = tt[r];
                            xe_for<0, 2>([&](auto cbc) {
                                constexpr int cb = decltype(cbc)::value;
                                simd<int, 16> cell = iota + (c0 + 16 * cb);
                                simd<int, 16> d;
                                if constexpr (UNALIGNED) {
                                    simd<uint32_t, 16> po = simd<uint32_t, 16>(min(cell, simd<int, 16>(p.nek1 - 1))) * 4u;
                                    d = gather<int32_t, 16>(prow, po);
                                    d.merge(simd<int, 16>(p.masked_pos), cell >= p.nek1);
                                } else {
                                    d = gather<int32_t, 16>(prow, simd<uint32_t, 16>(cell) * 4u);
                                }
                                simd<int, 16>      col = max(min(d, simd<int, 16>(p.pk_ncols - 1)), simd<int, 16>(0));
                                simd<uint32_t, 16> off = simd<uint32_t, 16>(col) * 32u;
                                simd<uint32_t, 128> t32 = gather<uint32_t, 128, 8>(tf, off, cached);
                                simd<sycl::half, 256> tb = t32.template bit_cast_view<sycl::half>();
                                simd<float, 128> acc = xmx::dpas<8, 8, float>(tb, vA[mb]);
                                simd<float, 32>  x2  = gather<float, 32, 2>(tx, off / 4u, cached);
                                simd<float, 16> x = acc.template select<16, 1>(m * 16);
                                x += t_r * x2.template select<16, 1>(0) + x2.template select<16, 1>(16);
                                x.merge(simd<float, 16>(-INFINITY), d == p.masked_pos);
                                S[mb][cb].template select<16, 1>(m * 16) = x;
                            });
                        });
                    }

                    // S += Q K^T, K and Q one dim chunk ahead (the loads of chunk dc + 1 in flight during the DPAS of dc)
                    xe_for<0, 8>([&](auto dcc) {
                        constexpr int dc = decltype(dcc)::value;
                        simd<uint32_t, 128>   k0 = kn0;
                        simd<uint32_t, 128>   k1 = kn1;
                        simd<sycl::half, 128> qa[MB];
#pragma unroll
                        for (int mb = 0; mb < MB; ++mb) {
                            qa[mb] = qn[mb];
                        }
                        if constexpr (dc < 7) {
                            kn0 = xe_ld_kt<(dc + 1) * 8, 0>(plk);
                            kn1 = xe_ld_kt<(dc + 1) * 8, 16>(plk);
#pragma unroll
                            for (int mb = 0; mb < MB; ++mb) {
                                qn[mb] = slm_block_load<sycl::half, 128>(qslm + (mb * 8 + dc + 1) * 256);
                            }
                        }
                        simd<sycl::half, 256> kb0 = k0.template bit_cast_view<sycl::half>();
                        simd<sycl::half, 256> kb1 = k1.template bit_cast_view<sycl::half>();
#pragma unroll
                        for (int mb = 0; mb < MB; ++mb) {
                            S[mb][0] = xmx::dpas<8, 8, float, float>(S[mb][0], kb0, qa[mb]);
                            S[mb][1] = xmx::dpas<8, 8, float, float>(S[mb][1], kb1, qa[mb]);
                        }
                    });

                    // the first V block in flight during the softmax
                    simd<uint16_t, 512> vn = xe_ld_vn<0, 0>(plv);

                    // online softmax: the tile's row max; a row's max moves only when the tile passes it by thr
                    simd<float, R * 16> t;
#pragma unroll
                    for (int mb = 0; mb < MB; ++mb) {
                        t.template select<128, 1>(mb * 128) = max(S[mb][0], S[mb][1]);
                    }
                    simd<float, R> Mt  = xe_rows<R, true>(t);
                    simd_mask<R>   up  = Mt > M + thr;
                    const bool      resc = up.any();
                    simd<float, R> eM  = 1.0f;
                    if (resc) {
                        simd<float, R> Mn = M;
                        Mn.merge(max(M, Mt), up);
                        eM = exp2((M - Mn) * XE_L2E);
                        M  = Mn;
                    }
                    simd<float, R> ML = M * XE_L2E;
                    simd<sycl::half, 128> P[MB][2];
                    xe_for<0, MB>([&](auto mbc) {
                        constexpr int mb = decltype(mbc)::value;
                        xe_for<0, 2>([&](auto cbc) {
                            constexpr int cb = decltype(cbc)::value;
                            xe_for<0, 8>([&](auto mc) {
                                constexpr int m = decltype(mc)::value;
                                S[mb][cb].template select<16, 1>(m * 16) =
                                    exp2(S[mb][cb].template select<16, 1>(m * 16) * XE_L2E - float(ML[mb * 8 + m]));
                            });
                            P[mb][cb] = simd<sycl::half, 128>(S[mb][cb]);
                        });
                        t.template select<128, 1>(mb * 128) = S[mb][0] + S[mb][1];
                    });
                    L = L * eM + xe_rows<R, false>(t);

                    // O = O * rescale + P V
                    xe_for<0, 8>([&](auto dbc) {
                        constexpr int db = decltype(dbc)::value;
                        simd<uint16_t, 512> v16 = vn;
                        if constexpr (db < 7) {
                            vn = xe_ld_vn<(db + 1) * 16, 0>(plv);
                        }
                        if (resc) {
                            xe_for<0, R>([&](auto rc) {
                                constexpr int r = decltype(rc)::value;
                                O[r / 8][db].template select<16, 1>((r % 8) * 16) *= float(eM[r]);
                            });
                        }
                        xe_for<0, 2>([&](auto ccc) {
                            constexpr int cc = decltype(ccc)::value;
                            simd<sycl::half, 256> vb = v16.template select<256, 1>(cc * 256).template bit_cast_view<sycl::half>();
#pragma unroll
                            for (int mb = 0; mb < MB; ++mb) {
                                O[mb][db] = xmx::dpas<8, 8, float, float>(O[mb][db], vb, P[mb][cc]);
                            }
                        });
                    });
                }
                if constexpr (SB) {
                    esx::split_barrier<esx::split_barrier_action::wait>();
                }
            }
        }

        if (!active) {
            return;
        }
        // the sink is one more cell with logit sinks[head] and no value (as the CPU adds it)
        simd<float, R> rcp;
        if (p.sinks) {
            const float     sink = p.sinks[head_q];
            simd<float, R> e    = exp2((M - sink) * XE_L2E);
            simd<float, R> a    = e / (1.0f + L * e);
            simd<float, R> b    = 1.0f / (L + exp2((sink - M) * XE_L2E));
            rcp.merge(a, b, M < sink);
        } else {
            rcp = 1.0f / L;
            rcp.merge(simd<float, R>(0.0f), L == 0.0f);
        }
        float *        dh = p.dst + ((int64_t) batch * nq1 * p.neq2 + head_q) * 128;
        const uint32_t dp = (uint32_t) p.neq2 * 512u - 1;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
#pragma unroll
            for (int db = 0; db < 8; ++db) {
                simd<float, 128> o = O[mb][db];
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    o.template select<16, 1>(m * 16) *= float(rcp[mb * 8 + m]);
                }
                store_2d<float, 16, 8>(dh, 511u, (uint32_t) nq1 - 1, dp, db * 16, row0 + 8 * mb, o);
            }
        }
    }

    auto get(syclex::properties_tag) const { return syclex::properties{ sycl::ext::intel::experimental::grf_size<MB == 2 ? 256 : 128> }; }
};

int fa_xe_env(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

template <int NSG, int MB, bool UNALIGNED>
void fa_rel_xe_launch(const ggml_sycl_fa_xe_params & p, int ne3, dpct::queue_ptr stream) {
    constexpr int WR = 8 * MB * NSG;  // rows per work-group, a multiple of the 64-row blocks of the tile flags
    static_assert(WR % 64 == 0, "work-group rows");
    const int n_wr = (p.neq1 + WR - 1) / WR;
    const sycl::nd_range<1> range((size_t) ne3 * n_wr * p.neq2 * NSG, (size_t) NSG);
    // GGML_SYCL_FA_XE_THR=t: lazy rescale (a row's max moves only when a tile passes it by t). Off by default: it
    // costs nothing measurable to rescale exactly (pp2048 at depth 0 and 32K), and with t = 5.5 a needle in a 110K
    // prompt was lost at depths 0.5 and 0.7, where the exact rescale found it
    static const float thr = getenv("GGML_SYCL_FA_XE_THR") ? (float) atof(getenv("GGML_SYCL_FA_XE_THR")) : 0.0f;
    static const int   pfd = fa_xe_env("GGML_SYCL_FA_XE_PF", 1);
    stream->parallel_for(range, fa_rel_xe_kernel<NSG, MB, UNALIGNED, true>{ p, n_wr, thr, pfd });
}

template <int NSG, int MB> void fa_rel_xe_launch_un(const ggml_sycl_fa_xe_params & p, int ne3, dpct::queue_ptr stream) {
    if (p.nek1 % XE_BC != 0) {
        fa_rel_xe_launch<NSG, MB, true>(p, ne3, stream);
    } else {
        fa_rel_xe_launch<NSG, MB, false>(p, ne3, stream);
    }
}

} // namespace

bool ggml_sycl_fa_rel_xe_ok(const ggml_sycl_fa_xe_params & p) {
    static const bool enabled = fa_xe_env("GGML_SYCL_FA_XE", 1) != 0;
    if (!enabled) {
        return false;
    }
    // 2D surfaces: 64-byte aligned bases, pitches multiples of 16 bytes and at least the 256-byte row
    auto a64 = [](const void * x) { return ((uintptr_t) x % 64) == 0; };
    return a64(p.K) && a64(p.V) && a64(p.dst) && p.nbk2 % 64 == 0 && p.nbk3 % 64 == 0 && p.nbv2 % 64 == 0 &&
           p.nbv3 % 64 == 0 && p.nbk1 % 16 == 0 && p.nbv1 % 16 == 0 && p.nbk1 >= 256 && p.nbv1 >= 256 &&
           p.nbq1 % 16 == 0 && p.nbq2 % 16 == 0 && p.nbq3 % 16 == 0 && ((uintptr_t) p.Q % 16) == 0 &&
           ((uintptr_t) p.pk % 16) == 0 && p.nek1 >= 1 && (int64_t) p.neq2 * 512 <= (int64_t) UINT32_MAX;
}

void ggml_sycl_fa_rel_xe(const ggml_sycl_fa_xe_params & p, int ne3, dpct::queue_ptr stream) {
    static const int nsg = fa_xe_env("GGML_SYCL_FA_XE_NSG", 8);
    static const int mb  = fa_xe_env("GGML_SYCL_FA_XE_MB", 2);
    if (mb == 1) {
        nsg == 16 ? fa_rel_xe_launch_un<16, 1>(p, ne3, stream) : fa_rel_xe_launch_un<8, 1>(p, ne3, stream);
    } else {
        nsg == 16 ? fa_rel_xe_launch_un<16, 2>(p, ne3, stream) : fa_rel_xe_launch_un<8, 2>(p, ne3, stream);
    }
}
