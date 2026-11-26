#include "fused-gemm.hpp"
#include "fused-gemm-esimd.hpp"
#include "quants.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <string>
#include <tuple>
#include <mutex>
#include <set>
#include <unordered_map>

namespace mx = sycl::ext::oneapi::experimental::matrix;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
static constexpr int FG_BK     = QK4_NL;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
template <typename T> struct fg_elem;

template <> struct fg_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::fp16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    static constexpr ggml_type       src   = GGML_TYPE_F16;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return (store) x; }
    static pair make(float x, float y) { return pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
static inline uint32_t fg_round_bits(float x, int drop) {
    const uint32_t u = sycl::bit_cast<uint32_t>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        return (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    return u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

struct alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

template <> struct fg_elem<sycl::ext::oneapi::bfloat16> {
    using store = sycl::ext::oneapi::bfloat16;
    using pair  = fg_bf16x2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::bf16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    static constexpr ggml_type       src   = GGML_TYPE_BF16;
    static constexpr int             mant  = 7;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return store(x); }
    static pair make(float x, float y) { return { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
template <> struct fg_elem<mx::precision::tf32> {
    using store = float;
    using pair  = sycl::float2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::tf32;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    static constexpr ggml_type       src   = GGML_TYPE_COUNT;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 1;
    static store cvt(float x) { return sycl::bit_cast<float>(fg_round_bits(x, 13) & ~0x1fffu); }
    static pair make(float x, float y) { return pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
template <typename TA, typename TB, int TM_, int TN_, int TK_, int SG_> struct fg_combo {
    using ta  = TA;
    using tb  = TB;
    using EA  = fg_elem<TA>;
    using EB  = fg_elem<TB>;
    using tsa = typename EA::store;
    using tsb = typename EB::store;
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int TK = TK_;
    static constexpr int SG = SG_;
    static constexpr int VNNI    = 4 / sizeof(tsb);  // K rows of B packed in one 32-bit word
    static constexpr int SG_ROWS = TM > 16 ? TM : 16;
    static constexpr int RPL     = SG_ROWS / SG;     // A rows one lane decodes per k step
    static constexpr int MT      = SG_ROWS / TM;
    static constexpr int BN      = TN > 32 ? TN : 32;
    static constexpr int NT      = BN / TN;
    static constexpr int WG_SIZE = FG_KSPLIT * SG;
    static constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    static constexpr bool efficient = TN <= 32;
    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

using fg_half = sycl::half;
using fg_bf16 = sycl::ext::oneapi::bfloat16;
using fg_tf32 = mx::precision::tf32;

// The combinations the ESIMD grouped GEMM (fused-gemm-esimd.cpp) stands in for: 8 x SG x 16 with f16
// or bf16 operands, the shape of one dpas, on sub-group 16 (Xe2, Xe-HPC) or 8 (Xe-HPG). Its B is
// packed and tiled as theirs is.
template <typename S> static constexpr bool fg_esimd_combo() {
    return S::TM == 8 && S::TN == S::SG && S::TK == 16 && (S::SG == 16 || S::SG == 8) &&
           std::is_same_v<typename S::ta, typename S::tb> &&
           (std::is_same_v<typename S::ta, fg_half> || std::is_same_v<typename S::ta, fg_bf16>);
}

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
template <typename F> static void fg_visit_combo(int idx, F && f);
static constexpr int FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
template <int SG> static constexpr bool fg_listed() {
#if defined(GGML_SYCL_XMX_AOT_SG)
    return SG == GGML_SYCL_XMX_AOT_SG;
#else
    return true;
#endif
}

template <typename S, typename F> static void fg_call_combo(F && f) {
    if constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

template <typename F> static void fg_visit_combo(int idx, F && f) {
    switch (idx) {
        case 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        case 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   break; // Xe2, Xe3, Xe-HPC
        case 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        default: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
// (FG_AOT_SG is defined in fused-gemm.hpp, which the ESIMD kernels share)

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
static constexpr int64_t grouped_gemm_max_tiles(int64_t total_rows, int64_t n_as, int64_t BN) {
    return total_rows <= n_as ? total_rows : n_as + (total_rows - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
template <typename S> static bool fg_device_has_combo(const std::vector<mx::combination> & combinations) {
    for (const auto & c : combinations) {
        if (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::matrix_type::fp32 &&
            c.dtype == mx::matrix_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            return true;
        }
    }
    return false;
}

template <typename T> static const char * fg_type_name() {
    return std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

static std::string fg_combo_name(int idx) {
    std::string name;
    fg_visit_combo(idx, [&](auto s) {
        using S = decltype(s);
        name = std::string(fg_type_name<typename S::ta>()) + "x" + fg_type_name<typename S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    return name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
static int fg_device_combos(const sycl::device & dev) {
    static std::mutex                            mtx;
    static std::unordered_map<sycl::device, int> known;
    std::lock_guard<std::mutex>                  lock(mtx);
    const auto                                   it = known.find(dev);
    if (it != known.end()) {
        return it->second;
    }
    int available = 0;
    try {
        const auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        const auto sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                const bool sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                if (sg_ok && fg_device_has_combo<S>(combinations)) {
                    available |= 1 << idx;
                }
            });
        }
    } catch (const sycl::exception &) {
        available = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes);
    known.emplace(dev, available);
    return available;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
template <typename S> static int64_t fg_rank(ggml_type src1_type) {
    const int64_t spills  = !S::efficient;
    const int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    const int64_t convert = S::EB::src != src1_type;
    const int64_t lossy   = S::EB::mant < 10;
    const int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    const int64_t mk      = 1024 - S::TM * S::TK;
    return ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
static bool fg_mode_meets(int mode, int32_t src1_prec) {
    if (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        return true;
    }
    if (src1_prec >= GGML_PREC_BF16) {
        return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    switch (g_ggml_sycl_dynamic_required_precision) {
        case GGML_SYCL_DYNAMIC_PRECISION_TF32: return mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        case GGML_SYCL_DYNAMIC_PRECISION_BF16: return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        default:                               return false;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
static int fg_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    const sycl::device dev     = stream->get_device();
    const int          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    const bool         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    const int          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    int     best      = -1;
    int64_t best_rank = 0;
    for (int i = 0; i < 3 && best < 0; ++i) {
        const int mode = modes[i];
        if (!fg_mode_meets(mode, src1_prec)) {
            continue;
        }
        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
            if (!(allowed & (1 << idx))) {
                continue;
            }
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                if (S::EA::mode != mode || S::EB::mode != mode) {
                    return;
                }
                const int64_t rank = fg_rank<S>(src1_type);
                if (best < 0 || rank < best_rank) {
                    best      = idx;
                    best_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    static std::mutex                                      mtx;
    static std::set<std::tuple<size_t, int, int32_t, int>> seen;
    std::lock_guard<std::mutex>                            lock(mtx);
    if (seen.emplace(std::hash<sycl::device>{}(dev), (int) src1_type, src1_prec, best).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      best >= 0 ? fg_combo_name(best).c_str() : "none (library GEMM)");
    }
    return best;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
template <typename E, typename T_src>
static void fused_gemm_pack_b(const T_src * y, typename E::store * packed, int N, int Npad, int K,
                              dpct::queue_ptr stream) {
    constexpr int V   = 4 / sizeof(typename E::store);
    const int     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const int idx = id[0];
        const int n   = idx / kqs;
        const int kq  = idx - n * kqs;
        typename E::store vals[V] = {};
        if (n < N) {
            const T_src * src = y + (size_t) n * K + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt((float) src[v]);
            }
        }
        typename E::store * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq4_nl * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq4_nl blk = xrow[kb];
    const float        d   = (float) blk.d;
#pragma unroll
    for (int j = 0; j < QK4_NL / 2; j += 2) {
        const uint8_t q0 = blk.qs[j];
        const uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq3_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    const block_iq3_s * blk = xrow + kb / (QK_K / 32);
    const int           ib8 = kb % (QK_K / 32);
    const uint8_t *     qs  = blk->qs + 8 * ib8;
    const int           qh  = blk->qh[ib8];
    const float         d   = (float) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        const int      signs = blk->signs[4 * ib8 + il];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float g1a = (float) ((grid1 >> (16 * j + 0)) & 0xff);
            const float g1b = (float) ((grid1 >> (16 * j + 8)) & 0xff);
            const float g2a = (float) ((grid2 >> (16 * j + 0)) & 0xff);
            const float g2b = (float) ((grid2 >> (16 * j + 8)) & 0xff);
            const int   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
template <typename block_q_t> struct fg_block_traits;
template <> struct fg_block_traits<block_iq4_nl> { static constexpr int qk = QK4_NL; };
template <> struct fg_block_traits<block_iq3_s>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq3_xxs> { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq4_xs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xxs> { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_s>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_s>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_m>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q8_0>    { static constexpr int qk = QK8_0; };
template <> struct fg_block_traits<block_q5_0>    { static constexpr int qk = QK5_0; };
template <> struct fg_block_traits<block_q2_K>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q3_K>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q4_K>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q5_K>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q6_K>    { static constexpr int qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#define FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    const T * blk = xrow + kb / (QK_K / 32);                                     \
    const int ib  = kb % (QK_K / 32)

template <typename E>
static __dpct_inline__ void fg_pack_quarter(const float * __restrict__ t, typename E::pair * a, int il) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

// 32 values from 16 qs bytes, as in IQ4_NL blocks and IQ4_XS sub-blocks: the low nibbles fill the
// first half of the step and the high nibbles the second, so the halves land at a[0..7] and a[8..15]
template <typename E>
static __dpct_inline__ void fg_decode_iq4(const uint8_t * __restrict__ q4, const float d, typename E::pair * a) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq4_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq4_xs);
    const float d = (float) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    fg_decode_iq4<E>(blk->qs + 16 * ib, d, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq3_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq3_xxs);
    const uint8_t *  q3    = blk->qs + 8 * ib;
    const uint16_t * gas   = (const uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    const uint32_t   aux32 = gas[0] | (gas[1] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xxs);
    const uint16_t * q2    = blk->qs + 4 * ib;
    const uint8_t *  aux8  = (const uint8_t *) q2;
    const uint32_t   aux32 = q2[2] | (q2[3] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xxs_grid + aux8[il]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xs);
    const uint16_t * q2 = blk->qs + 4 * ib;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xs_grid + (q2[il] & 511));
        const float     d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_s);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid =
            (const uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        const float   d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq1_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_s);
    const float delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d     = (float) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq1_m * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_m);
    const uint16_t * sc = (const uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const int   ib16  = 2 * ib + il / 2;
        const float d     = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        const float delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
template <typename E>
static __dpct_inline__ void fg_decode_q8_0(const int8_t * __restrict__ qs, const float d, typename E::pair * a) {
#pragma unroll
    for (int j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
static __dpct_inline__ void fg_scale_min_k4(const int j, const uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

template <typename E>
static __dpct_inline__ void fg_decode_q2_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                                           const float d, const float dmin, const int ib, typename E::pair * a) {
    const uint8_t * q = qs + 32 * (ib / 4);
    const int shift = 2 * (ib % 4);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t sc = scales[2 * ib + j / 8];
        const float dl = d * (sc & 0xF);
        const float ml = dmin * (sc >> 4);
        a[j] = E::make(dl * ((q[2 * j] >> shift) & 3) - ml,
                       dl * ((q[2 * j + 1] >> shift) & 3) - ml);
    }
}

// scales of the two 16-value halves of sub-block ib
static __dpct_inline__ void fg_q3_K_dl(const uint8_t * __restrict__ scales, const float d, const int ib, float * dl) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int s  = 2 * ib + i;
        const int sc = ((scales[s % 8] >> (4 * (s / 8))) & 0xF) | (((scales[8 + s % 4] >> (2 * (s / 4))) & 3) << 4);
        dl[i] = d * (sc - 32);
    }
}

template <typename E>
static __dpct_inline__ void fg_decode_q3_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ hmask,
                                           const uint8_t * __restrict__ scales, const float d, const int ib,
                                           typename E::pair * a) {
    // block_q3_K is 110 bytes, so qs and hmask are only 2-byte aligned
    const uint16_t * q = (const uint16_t *) (qs + 32 * (ib / 4));
    const uint16_t * h = (const uint16_t *) hmask;
    const int shift = 2 * (ib % 4);
    float dl[2];
    fg_q3_K_dl(scales, d, ib, dl);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        // per byte: 2 low bits, plus 4 if the hmask bit is set, minus 4
        const uint32_t v = ((q[j] >> shift) & 0x0303u) | (((h[j] >> ib) & 0x0101u) << 2);
        a[j] = E::make(dl[j / 8] * ((int) (v & 0xFF) - 4), dl[j / 8] * ((int) (v >> 8) - 4));
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
template <typename E>
static __dpct_inline__ void fg_decode_q4_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                                           const sycl::half2 dm, const int ib, typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    const float     d     = (float) dm[0] * sc;
    const float     m     = (float) dm[1] * mb;
    const uint8_t * q     = qs + 32 * (ib / 2);
    const int       shift = 4 * (ib % 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> shift) & 0xF) - m, d * ((q[2 * j + 1] >> shift) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
template <typename E>
static __dpct_inline__ void fg_decode_q5_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
                                           const uint8_t * __restrict__ scales, const sycl::half2 dm, const int ib,
                                           typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    const float     d     = (float) dm[0] * sc;
    const float     m     = (float) dm[1] * mb;
    const uint8_t * q     = qs + 32 * (ib / 2);
    const int       shift = 4 * (ib % 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int l = 2 * j;
        a[j] = E::make(d * (((q[l] >> shift) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> shift) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
template <typename E>
static __dpct_inline__ void fg_decode_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                                           const int8_t * __restrict__ scales, const float d, const int ib,
                                           typename E::pair * a) {
    const int       h  = ib / 4;
    const int       r  = ib % 4;
    const uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    const uint8_t * hb = qh + 32 * h;
    const int8_t *  sc = scales + 8 * h + 2 * r;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int   l  = 2 * j;
        const float dl = d * sc[j / 8];
        const int   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        const int   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q8_0 * __restrict__ xrow, const int kb, typename E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (float) xrow[kb].d, a);
}

// q5_0: value k < 16 is the low nibble of qs[k], value 16 + k its high nibble, each with bit k
// (16 + k) of qh as 16, minus 16, times d: dequantize_q5_0. The block is 22 bytes, so qh is read
// bytewise.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q5_0 * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_q5_0 * blk = xrow + kb;
    const float        d   = (float) blk->d;
    const uint32_t     qh  = blk->qh[0] | (blk->qh[1] << 8) | (blk->qh[2] << 16) | ((uint32_t) blk->qh[3] << 24);
#pragma unroll
    for (int j = 0; j < QK5_0 / 2; j += 2) {
        const uint8_t q0 = blk->qs[j];
        const uint8_t q1 = blk->qs[j + 1];
        a[j / 2]     = E::make(d * ((int) ((q0 & 0xf) | (((qh >> j) & 1) << 4)) - 16),
                               d * ((int) ((q1 & 0xf) | (((qh >> (j + 1)) & 1) << 4)) - 16));
        a[j / 2 + 8] = E::make(d * ((int) ((q0 >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16),
                               d * ((int) ((q1 >> 4) | (((qh >> (j + 17)) & 1) << 4)) - 16));
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q2_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q2_K);
    fg_decode_q2_K<E>(blk->qs, blk->scales, (float) blk->dm[0], (float) blk->dm[1], ib, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q3_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q3_K);
    fg_decode_q3_K<E>(blk->qs, blk->hmask, blk->scales, (float) blk->d, ib, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q4_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q5_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q6_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (float) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
template <typename block_q_t> struct fg_soa {
    static constexpr bool supported = false;
};

template <> struct fg_soa<block_q8_0> {
    static constexpr bool supported = true;
    // [qs][d]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int,
                                      typename E::pair * a) {
        const float d = (float) ((const sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((const int8_t *) x + ib * QK8_0, d, a);
    }
};

template <> struct fg_soa<block_q2_K> {
    static constexpr bool supported = true;

    // [qs][scales][dm]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t * scales = x + nblocks * (QK_K / 4);
        const sycl::half2 dm = ((const sycl::half2 *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q2_K<E>(x + ib * (QK_K / 4), scales + ib * (QK_K / 16), (float) dm[0], (float) dm[1],
                          kb % (QK_K / 32), a);
    }
};

template <> struct fg_soa<block_q3_K> {
    static constexpr bool supported = true;

    // [qs][hmask][scales][d]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t * hmask = x + nblocks * (QK_K / 4);
        const uint8_t * scales = hmask + nblocks * (QK_K / 8);
        const float d = (float) ((const sycl::half *) (scales + nblocks * 12))[ib];
        const int sub = kb % (QK_K / 32);
        // 16-byte aligned, see opt_for_reorder_id
        using v4 = sycl::vec<uint32_t, 4>;
        const v4 * q = (const v4 *) (x + ib * (QK_K / 4) + 32 * (sub / 4));
        const v4 * h = (const v4 *) (hmask + ib * (QK_K / 8));
        const v4 qv[2] = { q[0], q[1] };
        const v4 hv[2] = { h[0], h[1] };
        const int shift = 2 * (sub % 4);
        float dl[2];
        fg_q3_K_dl(scales + ib * 12, d, sub, dl);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t v = ((qv[w / 4][w % 4] >> shift) & 0x03030303u) | (((hv[w / 4][w % 4] >> sub) & 0x01010101u) << 2);
            const float    f = dl[w / 4];
            a[2 * w]     = E::make(f * ((int) (v & 0xFF) - 4), f * ((int) ((v >> 8) & 0xFF) - 4));
            a[2 * w + 1] = E::make(f * ((int) ((v >> 16) & 0xFF) - 4), f * ((int) (v >> 24) - 4));
        }
    }
};

template <> struct fg_soa<block_q4_K> {
    static constexpr bool supported = true;
    // [qs][scales][dm]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t *   scales = x + nblocks * (QK_K / 2);
        const sycl::half2 dm     = ((const sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_SIZE, dm, kb % (QK_K / 32), a);
    }
};

template <> struct fg_soa<block_q5_K> {
    static constexpr bool supported = true;
    // [qs][qh][scales][dm]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t *   qh     = x + nblocks * (QK_K / 2);
        const uint8_t *   scales = qh + nblocks * (QK_K / 8);
        const sycl::half2 dm     = ((const sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_SIZE, dm,
                          kb % (QK_K / 32), a);
    }
};

template <> struct fg_soa<block_q6_K> {
    static constexpr bool supported = true;
    // [ql][qh][scales][d]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t * qh     = x + nblocks * (QK_K / 2);
        const uint8_t * scales = qh + nblocks * (QK_K / 4);
        const float     d      = (float) ((const sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (const int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};

template <ggml_type type> struct fg_soa_iq4 {
    static constexpr bool supported = true;
    // [qs][per-block records]; k step kb is 32-value group kb % groups of block ib
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        constexpr int groups = ggml_sycl_reordered::block_q_t<type>::traits::qk / QK4_NL;
        const int64_t g      = (int64_t) ib * groups + kb % groups;
        fg_decode_iq4<E>(x + 16 * g, ggml_sycl_reordered::iq4_group_scale<type>(x + nblocks * groups * 16, g), a);
    }
};

template <> struct fg_soa<block_iq4_nl> : fg_soa_iq4<GGML_TYPE_IQ4_NL> {};

template <> struct fg_soa<block_iq4_xs> : fg_soa_iq4<GGML_TYPE_IQ4_XS> {};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
template <typename S, typename block_q_t, bool reordered>
static void fused_dequant_gemm_tile(
    const block_q_t * __restrict__ x,
    const typename S::tsb * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int Npad, const int K, const int ldd,
    const int b0, const int n0, const int n1,
    sycl::local_accessor<typename S::tsa, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item, ggml_sycl_grouped_gemm_rows dst_rows = {}) {
    if constexpr (fg_built<S::SG>()) {
        using EA = typename S::EA;
        using TA = typename S::ta;
        using TB = typename S::tb;
        const auto sg     = item.get_sub_group();
        const int  sg_id  = sg.get_group_id()[0];
        const int  lane   = sg.get_local_id()[0];
        const int  m0     = item.get_group(1) * S::SG_ROWS;
        const int  nstep  = K / FG_BK;
        const int  a_base = sg_id * S::SG_ROWS * FG_BK;
        const int  c_base = sg_id * S::SG_ROWS * S::BN;

        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, S::TM, S::TN> acc[S::MT][S::NT];
#pragma unroll
        for (int mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr int         KPB     = fg_block_traits<block_q_t>::qk / FG_BK; // k steps per block
        const size_t          bpr     = K / fg_block_traits<block_q_t>::qk;
        const size_t          nblocks = (size_t) M * bpr;
        size_t                row_blk[S::RPL];
        bool                  row_ok[S::RPL];
        typename EA::pair *   a[S::RPL];
#pragma unroll
        for (int r = 0; r < S::RPL; ++r) {
            const int row = m0 + r * S::SG + lane;
            row_ok[r]  = row < M;
            row_blk[r] = (size_t) (row_ok[r] ? row : 0) * bpr;
            a[r]       = (typename EA::pair *) &tile_a[a_base + (r * S::SG + lane) * FG_BK];
        }

        const auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        const int b_stride = Npad * S::VNNI;

        const int kb_begin = (sg_id * nstep) / FG_KSPLIT;
        const int kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        for (int kb = kb_begin; kb < kb_end; ++kb) {
#pragma unroll
            for (int r = 0; r < S::RPL; ++r) {
                if (row_ok[r]) {
                    if constexpr (reordered) {
                        fg_soa<block_q_t>::template stage<EA>((const uint8_t *) x, nblocks, row_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } else {
                        fg_stage_a<EA>(x + row_blk[r], kb, a[r]);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::group_barrier(sg);

#pragma unroll
            for (int kt = 0; kt < FG_BK / S::TK; ++kt) {
                const int kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_matrix<sycl::sub_group, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#pragma unroll
                for (int nt = 0; nt < S::NT; ++nt) {
                    mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#pragma unroll
                for (int mt = 0; mt < S::MT; ++mt) {
                    mx::joint_matrix<sycl::sub_group, TA, mx::use::a, S::TM, S::TK, mx::layout::row_major> sub_a;
                    mx::joint_matrix_load(sg, sub_a,
                        tile_a.template get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#pragma unroll
                    for (int nt = 0; nt < S::NT; ++nt) {
                        mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::group_barrier(sg);
        }

#pragma unroll
        for (int mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_store(sg, acc[mt][nt],
                    tile_c.template get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::row_major);
            }
        }
        sycl::group_barrier(item.get_group());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        for (int idx = item.get_local_linear_id(); idx < S::SG_ROWS * S::BN; idx += S::WG_SIZE) {
            const int r = idx % S::SG_ROWS;
            const int c = idx / S::SG_ROWS;
            const int m = m0 + r;
            const int n = n0 + c;
            if (m < M && n < n1) {
                float sum = 0.0f;
#pragma unroll
                for (int s = 0; s < FG_KSPLIT; ++s) {
                    sum += tile_c[s * S::SG_ROWS * S::BN + r * S::BN + c];
                }
                if (dst_rows.map) {
                    const mmid_row_mapping row = dst_rows.map[n];
                    float * out = (float *) (dst_rows.base + (row.i1 % dst_rows.ne1) * dst_rows.nb1 + row.i2 * dst_rows.nb2);
                    out[m] = sum;
                } else {
                    dst[(size_t) n * ldd + m] = sum;
                }
            }
        }
    }
}

template <typename S, typename block_q_t, bool reordered = false>
static void fused_dequant_gemm_launch(const void * src0, const typename S::tsb * packed, float * dst, const int M,
                                      const int N, const int Npad, const int K, const int ldd,
                                      const int64_t groups_n, const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int n0 = item.get_group(0) * S::BN;
                fused_dequant_gemm_tile<S, block_q_t, reordered>((const block_q_t *) src0, packed, dst, M, Npad, K,
                                                      ldd, n0, n0, N, tile_a, tile_c, item);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
template <typename S, typename block_q_t, bool reordered>
static bool grouped_dequant_gemm_launch(const char * src0_dd, const size_t expert_stride,
                                        const ggml_sycl_grouped_gemm_tile * tiles_ptr, const typename S::tsb * packed,
                                        float * dst, const int M, const int Npad, const int K, const int64_t n_tiles,
                                        const int64_t groups_m, dpct::queue_ptr stream,
                                        ggml_sycl_grouped_gemm_rows dst_rows = {},
                                        const uint32_t * expert_offsets = nullptr,
                                        const ggml_sycl_grouped_gemm_situ_out * epi = nullptr) {
    if constexpr (fg_esimd_combo<S>() && ggml_sycl_grouped_gemm_esimd_has<block_q_t>) {
        if (epi) {
            // the SiTU epilogue or nothing queued
            return ggml_sycl_grouped_gemm_esimd_enabled() &&
                   ggml_sycl_grouped_gemm_esimd<block_q_t, typename S::tsb, S::SG>(src0_dd, expert_stride, tiles_ptr,
                                                                                   packed, dst, M, Npad, K, dst_rows,
                                                                                   reordered, n_tiles, stream,
                                                                                   expert_offsets, epi);
        }
        // the register-operand ESIMD kernel, either weight layout: no SLM A image (GGML_SYCL_MMID_ESIMD)
        if (ggml_sycl_grouped_gemm_esimd_enabled() &&
            ggml_sycl_grouped_gemm_esimd<block_q_t, typename S::tsb, S::SG>(src0_dd, expert_stride, tiles_ptr, packed,
                                                                            dst, M, Npad, K, dst_rows, reordered,
                                                                            n_tiles, stream, expert_offsets)) {
            return true;
        }
    }
    if (epi || dst_rows.w) {
        // the epilogues are the ESIMD kernel's: nothing queued
        return false;
    }
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int                         t    = item.get_group(0);
                const ggml_sycl_grouped_gemm_tile tile = tiles_ptr[t];
                if (tile.n0 == tile.n1) {
                    return;
                }
                const block_q_t *       x    = (const block_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, block_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, item, dst_rows);
            });
    });
    return true;
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
template <typename S>
static void grouped_gemm_pack_b(const float * y, typename S::tsb * packed, const ggml_sycl_grouped_gemm_tile * tiles,
                                int Npad, int K, dpct::queue_ptr stream) {
    using E        = typename S::EB;
    constexpr int V = S::VNNI;
    const int kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const size_t                      idx  = id[0];
        const int                         kq   = idx / Npad;
        const int                         n    = idx - (size_t) kq * Npad;
        const ggml_sycl_grouped_gemm_tile tile = tiles[n / S::BN];
        const int                         row  = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        typename S::tsb vals[V] = {};
        if (row < tile.n1) {
            const float * src = y + (size_t) row * K + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        typename S::tsb * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// q5_0, q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
template <typename T> static constexpr bool fg_plain_ok() {
    return !std::is_same_v<T, block_q8_0> && !std::is_same_v<T, block_q5_0> && !std::is_same_v<T, block_q2_K> &&
           !std::is_same_v<T, block_q3_K> &&
           !std::is_same_v<T, block_q4_K> && !std::is_same_v<T, block_q5_K> &&
           !std::is_same_v<T, block_q6_K>;
}

template <typename T, bool R> struct fg_tag {
    using type                     = T;
    static constexpr bool reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
template <typename T, typename F> static bool fg_visit_layout(bool reordered, F && f) {
    if (!reordered) {
        f(fg_tag<T, false>{});
        return true;
    }
    if constexpr (fg_soa<T>::supported) {
        f(fg_tag<T, true>{});
        return true;
    }
    return false;
}

template <typename F> static bool fg_visit_type(ggml_type type, bool reordered, F && f) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:  return fg_visit_layout<block_iq4_nl>(reordered, f);
        case GGML_TYPE_IQ3_S:   return fg_visit_layout<block_iq3_s>(reordered, f);
        case GGML_TYPE_IQ4_XS:  return fg_visit_layout<block_iq4_xs>(reordered, f);
        case GGML_TYPE_IQ3_XXS: return fg_visit_layout<block_iq3_xxs>(reordered, f);
        case GGML_TYPE_IQ2_XXS: return fg_visit_layout<block_iq2_xxs>(reordered, f);
        case GGML_TYPE_IQ2_XS:  return fg_visit_layout<block_iq2_xs>(reordered, f);
        case GGML_TYPE_IQ2_S:   return fg_visit_layout<block_iq2_s>(reordered, f);
        case GGML_TYPE_IQ1_S:   return fg_visit_layout<block_iq1_s>(reordered, f);
        case GGML_TYPE_IQ1_M:   return fg_visit_layout<block_iq1_m>(reordered, f);
        case GGML_TYPE_Q8_0:    return fg_visit_layout<block_q8_0>(reordered, f);
        case GGML_TYPE_Q5_0:    return fg_visit_layout<block_q5_0>(reordered, f);
        case GGML_TYPE_Q2_K:    return fg_visit_layout<block_q2_K>(reordered, f);
        case GGML_TYPE_Q3_K:    return fg_visit_layout<block_q3_K>(reordered, f);
        case GGML_TYPE_Q4_K:    return fg_visit_layout<block_q4_K>(reordered, f);
        case GGML_TYPE_Q5_K:    return fg_visit_layout<block_q5_K>(reordered, f);
        case GGML_TYPE_Q6_K:    return fg_visit_layout<block_q6_K>(reordered, f);
        default:                return false;
    }
}

// The ESIMD grouped GEMM loads A straight into its dpas operands, so the joint_matrix tile shape
// means nothing to it; it runs on the fg_esimd_combo of the picked operand type and sub-group size. A
// grouped call whose weight it covers is moved there from any other shape of that type and sub-group
// size, if the device lists it, whatever GGML_SYCL_XMX_GATHER_SHAPES allows: the shape bits choose
// among joint_matrix tiles.
// A call the ESIMD kernel then declines (a shape it does not take) runs that combination's joint_matrix
// kernel.
// the ESIMD grouped GEMM is enabled and has a decode for this weight format and layout
static bool fg_esimd_covers(ggml_type src0_type, bool reordered) {
    if (!ggml_sycl_grouped_gemm_esimd_enabled()) {
        return false;
    }
    bool covered = false;
    fg_visit_type(src0_type, reordered, [&](auto tag) {
        using T = decltype(tag);
        covered = ggml_sycl_grouped_gemm_esimd_has<typename T::type>;
    });
    return covered;
}

static int fg_esimd_remap(dpct::queue_ptr stream, int combo, ggml_type src0_type, bool reordered) {
    if (combo < 0 || !fg_esimd_covers(src0_type, reordered)) {
        return combo;
    }
    int target = -1;
    fg_visit_combo(combo, [&](auto s) {
        using S = decltype(s);
        if constexpr (std::is_same_v<typename S::ta, typename S::tb>) {
            for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
                fg_visit_combo(idx, [&](auto t) {
                    using U = decltype(t);
                    if constexpr (fg_esimd_combo<U>() && U::SG == S::SG && std::is_same_v<typename U::ta, typename S::ta>) {
                        target = idx;
                    }
                });
            }
        }
    });
    if (target < 0 || target == combo || !(fg_device_combos(stream->get_device()) & (1 << target))) {
        return combo;
    }
    return target;
}

template <typename S>
static bool fg_fused_run(ggml_type src0_type, bool src0_reordered, const void * src0, const void * src1, ggml_type src1_type,
                         float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::queue_ptr stream) {
    const int64_t groups_n = (N + S::BN - 1) / S::BN;
    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    const int     Npad     = (int) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * Npad);
    if (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<typename S::EB>((const sycl::half *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    } else if (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<typename S::EB>((const fg_bf16 *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    } else {
        fused_gemm_pack_b<typename S::EB>((const float *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    }

    const typename S::tsb * packed = packed_b.get();
    return fg_visit_type(src0_type, src0_reordered, [&](auto tag) {
        using block_q_t = typename decltype(tag)::type;
        if constexpr (fg_plain_ok<block_q_t>()) {
            fused_dequant_gemm_launch<S, block_q_t, decltype(tag)::reordered>(src0, packed, dst, (int) M, (int) N,
                                                                              Npad, (int) K, (int) ldd, groups_n,
                                                                              groups_m, stream);
        }
    });
}

bool ggml_sycl_fused_dequant_gemm(ggml_type src0_type, bool src0_reordered, const void * src0, const void * src1, ggml_type src1_type,
                                  int32_t src1_prec, float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM.
    // src0_reordered: the persistent weight reorder may have been installed on this
    // tensor by an earlier small-batch mul_mat; the A stage then has to decode the
    // SoA layout. Types without an SoA stage (fg_soa<T>::supported == false)
    // decline via fg_visit_layout and the caller falls through to the
    // reorder-aware conversion route.
    bool plain_ok = false;
    fg_visit_type(src0_type, src0_reordered, [&](auto tag) { plain_ok = fg_plain_ok<typename decltype(tag)::type>(); });
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        return false;
    }
    if (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        return false;
    }
    if (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        return false;
    }
    const int combo = fg_pick_combo(stream, src1_type, src1_prec);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_fused_run<decltype(s)>(src0_type, src0_reordered, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    return launched;
}

template <typename S>
static bool fg_grouped_run(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                           const float * src1, float * dst, const int64_t * expert_row_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_grouped_gemm_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::queue_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    for (int64_t e = 0; e < n_as; ++e) {
        const int64_t end = expert_row_offsets[e + 1];
        for (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    const int64_t n_tiles  = tiles.size();
    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    const int     Npad     = (int) (n_tiles * S::BN);

    const int64_t max_tiles = grouped_gemm_max_tiles(expert_row_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_grouped_gemm_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(tiles_dev.get(), tiles.data(), n_tiles * sizeof(ggml_sycl_grouped_gemm_tile))));

    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (int) K, stream);

    const typename S::tsb *             packed    = packed_b.get();
    const ggml_sycl_grouped_gemm_tile * tiles_ptr = tiles_dev.get();
    const char *                        src0_dd   = (const char *) src0_base;
    return fg_visit_type(src0_type, reordered, [&](auto tag) {
        using T = decltype(tag);
        grouped_dequant_gemm_launch<S, typename T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (int) M, Npad, (int) K, n_tiles, groups_m,
                                                                       stream);
    });
}

bool ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                                    const float * src1, int32_t src1_prec, float * dst,
                                    const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t total_rows, std::vector<ggml_sycl_grouped_gemm_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    int64_t n_active = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](auto) {})) {
        return false;
    }
    if (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, total_rows, n_active)) {
        return false;
    }
    const int combo = fg_esimd_remap(stream, fg_pick_combo(stream, GGML_TYPE_F32, src1_prec), src0_type, reordered);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_grouped_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
                                               expert_row_offsets, n_as, M, K, tiles, pool, stream);
    });
    return launched;
}

// One work-group sorts routes by expert and builds the tile table.
static constexpr int MMID_SCHED_WG = 256;

size_t ggml_sycl_mmid_schedule_scratch(int64_t n_as) {
    return (size_t) MMID_SCHED_WG * n_as;
}

using mmid_sched_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                           sycl::memory_scope::work_group,
                                           sycl::access::address_space::local_space>;

// The schedule's shared steps. The exclusive scan of cnt into off (with the total in off[n_as]):
static inline void mmid_sched_scan(sycl::group<1> grp, int64_t tid, uint32_t * cnt, uint32_t * off, int64_t n_as) {
    sycl::joint_exclusive_scan(grp, cnt, cnt + n_as, off, 0u, sycl::plus<uint32_t>());
    sycl::group_barrier(grp);
    if (tid == 0) {
        off[n_as] = off[n_as - 1] + cnt[n_as - 1];
    }
    sycl::group_barrier(grp);
}

// The tile table from the expert offsets (cnt and toff are reused as the tile counts and their scan).
static inline void mmid_sched_tiles(sycl::group<1> grp, int64_t tid, uint32_t * cnt, const uint32_t * off,
                                    uint32_t * toff, int64_t n_as, int64_t n_tiles_max,
                                    ggml_sycl_grouped_gemm_tile * tiles, uint32_t * expert_offsets) {
    constexpr int WG = MMID_SCHED_WG;
    for (int64_t e = tid; e < n_as; e += WG) {
        cnt[e] = (off[e + 1] - off[e] + GGML_SYCL_FG_BN - 1) / GGML_SYCL_FG_BN;
    }
    sycl::group_barrier(grp);
    mmid_sched_scan(grp, tid, cnt, toff, n_as);

    const uint32_t n_tiles = toff[n_as];
    for (int64_t e = tid; e < n_as; e += WG) {
        const uint32_t base = off[e];
        const uint32_t end  = off[e + 1];
        for (uint32_t j = 0; j < cnt[e]; ++j) {
            const uint32_t n0 = base + j * GGML_SYCL_FG_BN;
            const uint32_t n1 = n0 + GGML_SYCL_FG_BN < end ? n0 + GGML_SYCL_FG_BN : end;
            tiles[toff[e] + j] = { (int32_t) e, (int32_t) n0, (int32_t) n1 };
        }
    }
    // the launch is bounded, so the unused tail must be tiles the GEMM skips
    for (int64_t t = n_tiles + tid; t < n_tiles_max; t += WG) {
        tiles[t] = { 0, 0, 0 };
    }
    for (int64_t e = tid; e <= n_as; e += WG) {
        expert_offsets[e] = off[e];
    }
}

// The SLM variant of the schedule: the same runs and the same output as the global-scratch kernel below, but
// each run's per-expert count, then its exclusive prefix inside the expert's slice, is a uint16 in local
// memory ([expert][run]), and the ids of a run are read 16 at a time ahead of the updates. The global variant
// spends most of its ~110 us on dependent read-modify-writes of global scratch. uint16 holds every value when
// n_routes <= 65536: a prefix is below its expert's count, which is at most n_routes.
static constexpr int MMID_SCHED_BATCH = 16;

static bool mmid_sched_slm_ok(int64_t n_as, int64_t n_routes, dpct::queue_ptr stream) {
    static const int env = [] {
        const char * e = getenv("GGML_SYCL_MMID_SCHED_SLM");
        return e ? atoi(e) : 1;
    }();
    if (!env || n_routes > 65536) {
        return false;
    }
    const size_t need = (size_t) n_as * MMID_SCHED_WG * sizeof(uint16_t) + 3 * (n_as + 1) * sizeof(uint32_t);
    return need <= stream->get_device().get_info<sycl::info::device::local_mem_size>();
}

bool ggml_sycl_build_mmid_schedule(const int32_t * ids_dev, size_t ids_token_stride, int64_t n_as, int64_t n_ids,
                                   int64_t n_tokens, int64_t n_tiles_max, mmid_row_mapping * row_mapping,
                                   ggml_sycl_grouped_gemm_tile * tiles, uint32_t * expert_offsets,
                                   uint32_t * scratch, dpct::queue_ptr stream) {
    if (n_as <= 0 || n_as > GGML_SYCL_MMID_SCHED_MAX_EXPERTS || n_ids <= 0 || n_tokens <= 0 || n_ids > INT32_MAX / n_tokens) {
        return false;
    }
    const int64_t n_routes = n_tokens * n_ids;
    if (n_tiles_max < grouped_gemm_max_tiles(n_routes, n_as, GGML_SYCL_FG_BN)) {
        return false;
    }

    if (mmid_sched_slm_ok(n_as, n_routes, stream)) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint32_t, 1> l_cnt(n_as + 1, cgh);
            sycl::local_accessor<uint32_t, 1> l_off(n_as + 1, cgh);
            sycl::local_accessor<uint32_t, 1> l_toff(n_as + 1, cgh);
            sycl::local_accessor<uint16_t, 1> l_run((size_t) n_as * MMID_SCHED_WG, cgh);
            cgh.parallel_for(sycl::nd_range<1>(MMID_SCHED_WG, MMID_SCHED_WG),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                constexpr int  WG  = MMID_SCHED_WG;
                constexpr int  B   = MMID_SCHED_BATCH;
                const auto     grp = item.get_group();
                const int64_t  tid = item.get_local_id(0);
                uint32_t *     cnt  = l_cnt.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t *     off  = l_off.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t *     toff = l_toff.get_multi_ptr<sycl::access::decorated::no>().get();
                uint16_t *     run  = l_run.get_multi_ptr<sycl::access::decorated::no>().get();

                for (int64_t i = tid; i < n_as * WG; i += WG) {
                    run[i] = 0;
                }
                for (int64_t r = tid; r < n_routes; r += WG) {
                    row_mapping[r] = { 0, 0 };
                }
                sycl::group_barrier(grp);

                const int64_t chunk   = (n_routes + WG - 1) / WG;
                const int64_t r_begin = std::min<int64_t>(tid * chunk, n_routes);
                const int64_t r_end   = std::min<int64_t>(r_begin + chunk, n_routes);
                // n_routes <= 65536, so 32-bit route math; a batch walks (token, slot) without a division a route
                auto load_batch = [&](int64_t r0, int32_t (&e)[B], uint32_t & token0, uint32_t & slot0) {
                    token0 = (uint32_t) r0 / (uint32_t) n_ids;
                    slot0  = (uint32_t) r0 - token0 * (uint32_t) n_ids;
                    uint32_t token = token0, slot = slot0;
#pragma unroll
                    for (int j = 0; j < B; ++j) {
                        int32_t v = -1;
                        if (r0 + j < r_end) {
                            v = ((const int32_t *) ((const char *) ids_dev + (size_t) token * ids_token_stride))[slot];
                        }
                        e[j] = v >= 0 && v < n_as ? v : -1;
                        if (++slot == (uint32_t) n_ids) {
                            slot = 0;
                            ++token;
                        }
                    }
                };
                for (int64_t r0 = r_begin; r0 < r_end; r0 += B) {
                    int32_t  e[B];
                    uint32_t token0, slot0;
                    load_batch(r0, e, token0, slot0);
#pragma unroll
                    for (int j = 0; j < B; ++j) {
                        if (e[j] >= 0) {
                            run[e[j] * WG + tid]++;
                        }
                    }
                }
                sycl::group_barrier(grp);

                // per expert, the runs' counts become their exclusive prefix; the total is the expert's count.
                // A sub-group takes an expert, a lane 16 consecutive runs (WG == 16 * 16).
                static_assert(WG == 256, "a lane scans WG / 16 runs");
                {
                    const auto sg   = item.get_sub_group();
                    const int  sgid = sg.get_group_linear_id();
                    const int  lane = sg.get_local_linear_id();
                    for (int64_t e = sgid; e < n_as; e += WG / 16) {
                        uint16_t * re = run + e * WG + lane * 16;
                        uint32_t   pre[16];
                        uint32_t   local = 0;
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            pre[j] = local;
                            local += re[j];
                        }
                        const uint32_t base = sycl::exclusive_scan_over_group(sg, local, sycl::plus<uint32_t>());
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            re[j] = (uint16_t) (base + pre[j]);
                        }
                        const uint32_t tot = sycl::reduce_over_group(sg, local, sycl::plus<uint32_t>());
                        if (lane == 0) {
                            cnt[e] = tot;
                        }
                    }
                }
                sycl::group_barrier(grp);
                mmid_sched_scan(grp, tid, cnt, off, n_as);

                for (int64_t r0 = r_begin; r0 < r_end; r0 += B) {
                    int32_t  e[B];
                    uint32_t token, slot;
                    load_batch(r0, e, token, slot);
#pragma unroll
                    for (int j = 0; j < B; ++j) {
                        if (e[j] >= 0) {
                            const uint32_t pos = off[e[j]] + run[e[j] * WG + tid]++;
                            row_mapping[pos]   = { (int32_t) slot, (int32_t) token };
                        }
                        if (++slot == (uint32_t) n_ids) {
                            slot = 0;
                            ++token;
                        }
                    }
                }
                sycl::group_barrier(grp);
                mmid_sched_tiles(grp, tid, cnt, off, toff, n_as, n_tiles_max, tiles, expert_offsets);
            });
        });
        return true;
    }

    stream->submit([&](sycl::handler & cgh) {
        // l_cnt is reused: route counts, then the tile count per expert
        sycl::local_accessor<uint32_t, 1> l_cnt(n_as + 1, cgh);
        sycl::local_accessor<uint32_t, 1> l_off(n_as + 1, cgh);
        sycl::local_accessor<uint32_t, 1> l_toff(n_as + 1, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(MMID_SCHED_WG, MMID_SCHED_WG),
            [=](sycl::nd_item<1> item) {
                const auto     grp = item.get_group();
                const int64_t  tid = item.get_local_id(0);
                uint32_t * cnt  = l_cnt.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t * off  = l_off.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t * toff = l_toff.get_multi_ptr<sycl::access::decorated::no>().get();

                // The routes are cut into one contiguous run per work-item. A run's per-expert counts
                // live in its own row of scratch, so each expert's rows land in route order (token,
                // then slot), the order of the host sort, and the same every run: atomic arrival
                // order would permute an expert's rows between runs, and the output with them.
                const int64_t chunk   = (n_routes + MMID_SCHED_WG - 1) / MMID_SCHED_WG;
                const int64_t r_begin = std::min<int64_t>(tid * chunk, n_routes);
                const int64_t r_end   = std::min<int64_t>(r_begin + chunk, n_routes);
                uint32_t *    mine    = scratch + tid * n_as;
                for (int64_t e = 0; e < n_as; ++e) {
                    mine[e] = 0;
                }
                // a dropped route must leave a row that names (0, 0) rather than stale memory
                for (int64_t r = tid; r < n_routes; r += MMID_SCHED_WG) {
                    row_mapping[r] = { 0, 0 };
                }
                for (int64_t r = r_begin; r < r_end; ++r) {
                    const int64_t   token  = r / n_ids;
                    const int64_t   slot   = r - token * n_ids;
                    const int32_t * id_row = (const int32_t *) ((const char *) ids_dev + token * ids_token_stride);
                    const int32_t   e      = id_row[slot];
                    if (e >= 0 && e < n_as) {
                        mine[e]++;
                    }
                }
                sycl::group_barrier(grp);

                // per expert, the runs' counts become their exclusive prefix (the run's first row
                // inside the expert's slice); the total is the expert's count
                for (int64_t e = tid; e < n_as; e += MMID_SCHED_WG) {
                    uint32_t run = 0;
                    for (int t = 0; t < MMID_SCHED_WG; ++t) {
                        const uint32_t c       = scratch[t * n_as + e];
                        scratch[t * n_as + e] = run;
                        run += c;
                    }
                    cnt[e] = run;
                }
                sycl::group_barrier(grp);

                mmid_sched_scan(grp, tid, cnt, off, n_as);

                for (int64_t r = r_begin; r < r_end; ++r) {
                    const int64_t   token  = r / n_ids;
                    const int64_t   slot   = r - token * n_ids;
                    const int32_t * id_row = (const int32_t *) ((const char *) ids_dev + token * ids_token_stride);
                    const int32_t   e      = id_row[slot];
                    if (e >= 0 && e < n_as) {
                        const uint32_t pos = off[e] + mine[e]++;
                        row_mapping[pos]   = { (int32_t) slot, (int32_t) token };
                    }
                }
                sycl::group_barrier(grp);

                mmid_sched_tiles(grp, tid, cnt, off, toff, n_as, n_tiles_max, tiles, expert_offsets);
            });
    });
    return true;
}

// The device schedule's B pack: routed src1 rows (through src1.map) -> packed [K/V][Npad][V], as
// grouped_gemm_pack_b. A work-group is one tile (BN columns) by PACK_KQ packed k rows and goes
// through SLM, so lanes read consecutive values of one src1 row and write each packed row in BN-wide
// runs: a work-item per packed value read 8 bytes of a different row per lane, at half the bandwidth.
// Tiles past the schedule's end are skipped (the GEMM skips them too); the columns of a partial
// tile past its last row are zero.
static constexpr int FG_PACK_KQ = 64;  // packed k rows (dwords of B) of one work-group

// tanh through the hardware exp: the value is rounded to half right after, far coarser than its error
// (a few f32 ulp, absolute ~1e-7 near 0), and sycl::tanh made the SiTU pack compute-bound.
static inline float fg_tanh(float y) {
    const float t = sycl::native::exp(-2.0f * sycl::fabs(y));
    return sycl::copysign((1.0f - t) / (1.0f + t), y);
}

template <typename S, bool SITU>
static void grouped_gemm_pack_b_device(const ggml_sycl_grouped_gemm_rows src1, const ggml_sycl_grouped_gemm_tile * tiles,
                                       typename S::tsb * packed, const int Npad, const int K, const int64_t n_tiles,
                                       const ggml_sycl_grouped_gemm_situ situ, dpct::queue_ptr stream) {
    using E           = typename S::EB;
    using tsb         = typename S::tsb;
    constexpr int V   = S::VNNI;
    constexpr int BN  = S::BN;
    constexpr int WG  = 256;
    constexpr int KPI = FG_PACK_KQ * BN / WG;  // packed values one work-item moves
    constexpr int L      = 16;                  // load run: lanes reading one row
    constexpr int CPR    = BN * L / WG;         // columns of one load run
    constexpr int KQ_IT  = FG_PACK_KQ / L;      // load steps down a column
    static_assert(FG_PACK_KQ * BN % WG == 0 && WG % BN == 0 && FG_PACK_KQ % KPI == 0, "pack tiling");
    static_assert(BN * L % WG == 0 && FG_PACK_KQ % L == 0 && CPR * KQ_IT == KPI, "pack load tiling");
    struct alignas(4) word {
        tsb v[V];
    };
    static_assert(sizeof(word) == 4, "one packed value is a dword");
    const int     kqs      = K / V;
    const int64_t kgroups  = (kqs + FG_PACK_KQ - 1) / FG_PACK_KQ;
    const float   beta     = situ.beta;
    const float   lb       = situ.lb;
    const float   inv_beta = 1.0f / beta;
    const float   inv_lb   = lb > 0.0f ? 1.0f / lb : 0.0f;
    const ptrdiff_t g_off  = situ.g_off;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<word, 1> tile_s(FG_PACK_KQ * (BN + 1), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_tiles, kgroups * WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> it) {
            const int                         t    = it.get_group(0);
            const ggml_sycl_grouped_gemm_tile tile = tiles[t];
            if (tile.n1 <= tile.n0) {
                return;
            }
            const int lid = it.get_local_id(1);
            const int kq0 = it.get_group(1) * FG_PACK_KQ;
            // load: a run of L work-items converts L consecutive packed values of one column, so each read
            // instruction covers L*V consecutive floats of a src1 row; run r takes columns r*CPR.. and
            // KQ_IT such steps down each
            {
                const int r    = lid / L;
                const int lane = lid % L;
#pragma unroll
                for (int cc = 0; cc < CPR; ++cc) {
                    const int c   = r * CPR + cc;
                    const int row = tile.n0 + c;
                    const float * in = nullptr;
                    if (row < tile.n1) {
                        const mmid_row_mapping mapping = src1.map[row];
                        in = (const float *) (src1.base + (mapping.i1 % src1.ne1) * src1.nb1 + mapping.i2 * src1.nb2);
                    }
#pragma unroll
                    for (int j = 0; j < KQ_IT; ++j) {
                        const int kql = j * L + lane;
                        const int kq  = kq0 + kql;
                        word      w   = {};
                        if (in && kq < kqs) {
#pragma unroll
                            for (int v = 0; v < V; ++v) {
                                float val = in[V * kq + v];
                                if constexpr (SITU) {
                                    const float * g   = (const float *) ((const char *) in + g_off);
                                    float         gv  = g[V * kq + v];
                                    const float   act = fg_tanh(val * inv_beta) * beta *
                                                      (1.0f / (1.0f + sycl::native::exp(-val)));
                                    if (lb > 0.0f) {
                                        gv = fg_tanh(gv * inv_lb) * lb;
                                    }
                                    val = act * gv;
                                }
                                w.v[v] = E::cvt(val);
                            }
                        }
                        tile_s[kql * (BN + 1) + c] = w;
                    }
                }
            }
            sycl::group_barrier(it.get_group());
            // store: BN consecutive work-items write one packed row of the tile
#pragma unroll
            for (int j = 0; j < KPI; ++j) {
                const int kql = j * (WG / BN) + lid / BN;
                const int c   = lid % BN;
                const int kq  = kq0 + kql;
                if (kq < kqs) {
                    *(word *) (packed + ((size_t) kq * Npad + (size_t) t * BN + c) * V) = tile_s[kql * (BN + 1) + c];
                }
            }
        });
    });
}

// src1 f32 rows (slot s, token t at base + s*nb1 + t*nb2) -> contiguous half rows [t][s][K], rounded as the
// pack rounds them: the B the SiTU-epilogue kernel gathers from
static void fg_src1_to_half(const ggml_sycl_grouped_gemm_rows src1, sycl::half * out, const int64_t n_tokens,
                            const int64_t K, dpct::queue_ptr stream) {
    const int64_t kh  = K / 2;
    const int64_t ne1 = src1.ne1;
    stream->parallel_for(sycl::range<1>((size_t) (n_tokens * ne1 * kh)), [=](sycl::id<1> id) {
        const int64_t i   = id[0];
        const int64_t row = i / kh;
        const int64_t k2  = i - row * kh;
        const float * in  = (const float *) (src1.base + (row % ne1) * src1.nb1 + (row / ne1) * src1.nb2) + 2 * k2;
        const sycl::float2 v = *(const sycl::float2 *) in;
        *(sycl::half2 *) (out + 2 * i) = sycl::half2((sycl::half) v.x(), (sycl::half) v.y());
    });
}

static bool fg_moe_gb_env() {
    static const bool v = [] {
        const char * e = getenv("GGML_SYCL_MOE_GB");
        return e ? atoi(e) != 0 : true;
    }();
    return v;
}

template <typename S>
static bool fg_grouped_device_run(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                                   ggml_sycl_grouped_gemm_rows src1, ggml_sycl_grouped_gemm_rows dst,
                                   const int32_t * ids_dev, size_t ids_token_stride, int64_t n_as, int64_t n_ids,
                                   int64_t n_tokens, int64_t M, int64_t K, ggml_sycl_pool & pool,
                                   dpct::queue_ptr stream, const ggml_sycl_grouped_gemm_situ * situ,
                                   ggml_sycl_mmid_sched * shared, const ggml_sycl_grouped_gemm_situ_out * epi,
                                   int combo) {
    if constexpr (S::BN != GGML_SYCL_FG_BN) {
        return false;
    } else {
        const int64_t total_rows = n_tokens * n_ids;
        const int64_t max_tiles = grouped_gemm_max_tiles(total_rows, n_as, S::BN);
        if (max_tiles > INT32_MAX / S::BN) {
            return false;
        }
        const int Npad = (int) (max_tiles * S::BN);
        ggml_sycl_mmid_sched   local;
        ggml_sycl_mmid_sched * sc = &local;
        if (shared && (shared->max_tiles == 0 ||
                       (shared->ids == ids_dev && shared->ids_stride == ids_token_stride && shared->n_as == n_as &&
                        shared->n_ids == n_ids && shared->n_tokens == n_tokens && shared->max_tiles == max_tiles))) {
            sc = shared;
        }
        // the combine's half rows (dst.w) only from the down projection whose B the gate_up epilogue packed,
        // so nothing is queued before the ESIMD kernel takes or declines the call
        const bool from_epi = situ && sc == shared && sc->max_tiles != 0 && sc->next_K == K && sc->next_combo == combo;
        if (dst.w && !from_epi) {
            return false;
        }
        if (sc->max_tiles == 0) {
            sc->rows.alloc(pool, total_rows);
            sc->tiles.alloc(pool, max_tiles);
            sc->offsets.alloc(pool, n_as + 1);
            sc->scratch.alloc(pool, ggml_sycl_mmid_schedule_scratch(n_as));
            if (!ggml_sycl_build_mmid_schedule(ids_dev, ids_token_stride, n_as, n_ids, n_tokens, max_tiles,
                                               sc->rows.get(), sc->tiles.get(), sc->offsets.get(),
                                               sc->scratch.get(), stream)) {
                return false;
            }
            sc->ids        = ids_dev;
            sc->ids_stride = ids_token_stride;
            sc->n_as       = n_as;
            sc->n_ids      = n_ids;
            sc->n_tokens   = n_tokens;
            sc->max_tiles  = max_tiles;
        }
        src1.map = sc->rows.get();
        dst.map = sc->rows.get();
        const ggml_sycl_grouped_gemm_tile * tile_ptr = sc->tiles.get();
        const bool try_epi = epi && sc == shared && sc->next_K == 0 && epi->D > 0;
        if (try_epi && sc->next_b.get() == nullptr) {
            // before packed_b: the pool frees in reverse allocation order, and this one outlives the call
            sc->next_b.alloc(pool, (size_t) epi->D * Npad * sizeof(typename S::tsb));
        }
        // the epilogue kernel gathers B from half rows of src1 instead of a packed B (GGML_SYCL_MOE_GB=0: packs)
        ggml_sycl_pool_alloc<typename S::tsb> b16_alloc;
        const typename S::tsb *               b16 = nullptr;
        if constexpr (std::is_same_v<typename S::tsb, sycl::half>) {
            if (try_epi && !situ && fg_moe_gb_env() && K % 2 == 0 && src1.nb1 % 8 == 0 && src1.nb2 % 8 == 0 &&
                (uintptr_t) src1.base % 8 == 0 && n_tokens * src1.ne1 * K * 2 <= (int64_t) UINT32_MAX) {
                // the f16 copy of the rows when a producer wrote one (same rounding), else converted here
                if (src1.h) {
                    b16 = src1.h;
                } else {
                    sycl::half * h = b16_alloc.alloc(pool, (size_t) (n_tokens * src1.ne1 * K));
                    fg_src1_to_half(src1, h, n_tokens, K, stream);
                    b16 = h;
                }
            }
        }
        ggml_sycl_pool_alloc<typename S::tsb> packed_b;
        typename S::tsb * packed = nullptr;
        auto pack = [&]() {
            packed = packed_b.alloc(pool, (size_t) K * Npad);
            if (situ) {
                grouped_gemm_pack_b_device<S, true>(src1, tile_ptr, packed, Npad, (int) K, max_tiles, *situ, stream);
            } else {
                grouped_gemm_pack_b_device<S, false>(src1, tile_ptr, packed, Npad, (int) K, max_tiles, {}, stream);
            }
        };
        if (situ && sc == shared && sc->next_K == K && sc->next_combo == combo) {
            // the previous projection's epilogue packed this B
            packed       = (typename S::tsb *) sc->next_b.get();
            sc->next_K   = 0;
        } else if (!b16) {
            pack();
        }
        const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
        if (try_epi) {
            // the SiTU epilogue writes the next projection's packed B in place of dst, when the kernel has it
            ggml_sycl_grouped_gemm_situ_out e = *epi;
            e.out                             = sc->next_b.get();
            e.b16                             = b16;
            e.b_ne1                           = (int) src1.ne1;
            bool done                         = false;
            fg_visit_type(src0_type, reordered, [&](auto tag) {
                using T = decltype(tag);
                done = grouped_dequant_gemm_launch<S, typename T::type, T::reordered>((const char *) src0_base,
                    expert_stride, tile_ptr, packed, (float *) dst.base, (int) M, Npad, (int) K, max_tiles, groups_m,
                    stream, dst, sc->offsets.get(), &e);
            });
            if (done) {
                sc->next_K     = epi->D;
                sc->next_combo = combo;
                return true;
            }
        }
        if (!packed) {
            pack();
        }
        bool launched = true;
        const bool visited = fg_visit_type(src0_type, reordered, [&](auto tag) {
            using T = decltype(tag);
            launched = grouped_dequant_gemm_launch<S, typename T::type, T::reordered>((const char *) src0_base,
                expert_stride, tile_ptr, packed, (float *) dst.base, (int) M, Npad, (int) K, max_tiles, groups_m,
                stream, dst, sc->offsets.get());
        });
        if (dst.w && !(visited && launched)) {
            // declined with nothing queued: the epilogue's B stays for the plain call
            sc->next_K = K;
            return false;
        }
        return visited;
    }
}

int ggml_sycl_grouped_dequant_gemm_device_combo(ggml_type src0_type, bool reordered, int32_t src1_prec,
                                                int64_t n_as, int64_t n_ids, int64_t n_tokens,
                                                int64_t M, int64_t K, dpct::queue_ptr stream) {
    if (n_as <= 0 || n_as > GGML_SYCL_MMID_SCHED_MAX_EXPERTS || n_ids <= 0 || n_tokens <= 0 ||
        n_ids > INT32_MAX / n_tokens) {
        return -1;
    }
    const int64_t total_rows = n_ids * n_tokens;
    // The active expert count is unknown until the GPU builds the schedule. The bound keeps wide
    // experts on the per-expert library GEMMs, which beat the joint_matrix kernel there; the ESIMD
    // kernel beats them at any width (B70, Inkling ub2048: q4_K 2304x2304 2.7 ms a call against the
    // host-synchronised loop's several), so a format it covers takes every batch.
    const bool unbounded = (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_UNBOUNDED) != 0 ||
                           fg_esimd_covers(src0_type, reordered);
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](auto) {}) ||
        !ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, total_rows, std::min(n_as, total_rows),
                                                unbounded)) {
        return -1;
    }
    const sycl::device device = stream->get_device();
    if (device.get_info<sycl::info::device::max_work_group_size>() < MMID_SCHED_WG ||
        device.get_info<sycl::info::device::local_mem_size>() < 3 * (n_as + 1) * sizeof(uint32_t)) {
        return -1;
    }
    if (grouped_gemm_max_tiles(total_rows, n_as, GGML_SYCL_FG_BN) > INT32_MAX / GGML_SYCL_FG_BN) {
        return -1;
    }
    const int combo = fg_esimd_remap(stream, fg_pick_combo(stream, GGML_TYPE_F32, src1_prec), src0_type, reordered);
    if (combo < 0) {
        return -1;
    }
    // the device schedule is built for GGML_SYCL_FG_BN-wide tiles, see fg_grouped_device_run()
    bool bn_ok = false;
    fg_visit_combo(combo, [&](auto s) { bn_ok = decltype(s)::BN == GGML_SYCL_FG_BN; });
    return bn_ok ? combo : -1;
}

bool ggml_sycl_grouped_dequant_gemm_device_supported(ggml_type src0_type, bool reordered, int32_t src1_prec,
                                                      int64_t n_as, int64_t n_ids, int64_t n_tokens,
                                                      int64_t M, int64_t K, dpct::queue_ptr stream) {
    return ggml_sycl_grouped_dequant_gemm_device_combo(src0_type, reordered, src1_prec, n_as, n_ids, n_tokens, M, K,
                                                       stream) >= 0;
}

bool ggml_sycl_grouped_dequant_gemm_device(ggml_type src0_type, bool reordered, const void * src0_base,
                                           size_t expert_stride, const ggml_sycl_grouped_gemm_rows & src1,
                                           int32_t src1_prec, const ggml_sycl_grouped_gemm_rows & dst,
                                           const int32_t * ids_dev, size_t ids_token_stride, int64_t n_as,
                                           int64_t n_ids, int64_t n_tokens, int64_t M, int64_t K, ggml_sycl_pool & pool,
                                           dpct::queue_ptr stream, const ggml_sycl_grouped_gemm_situ * situ,
                                           ggml_sycl_mmid_sched * sched, const ggml_sycl_grouped_gemm_situ_out * epi,
                                           int epi_combo) {
    const int combo = ggml_sycl_grouped_dequant_gemm_device_combo(src0_type, reordered, src1_prec, n_as, n_ids,
                                                                  n_tokens, M, K, stream);
    if (src1.ne1 <= 0 || dst.ne1 <= 0 || combo < 0) {
        return false;
    }
    // the epilogue packs for the next projection's combination, which must be this one's
    if (epi_combo != combo) {
        epi = nullptr;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_grouped_device_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
            ids_dev, ids_token_stride, n_as, n_ids, n_tokens, M, K, pool, stream, situ, sched, epi, combo);
    });
    return launched;
}
