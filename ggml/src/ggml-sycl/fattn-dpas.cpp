#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-dpas.hpp"
#include "fattn-rel-xe.hpp"
#include <vector>
#include "rel-bias.hpp"

// GGML_SYCL_SUPPORT_XMX=OFF builds skip the DPAS below, as do spir64_gen builds for sub-group 8
// targets (Xe-HPG) or mixed target lists, which cannot compile it
#if !defined(GGML_SYCL_SUPPORT_XMX) || (defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 16)
#define GGML_SYCL_FA_DPAS_BUILD 0
#else
#define GGML_SYCL_FA_DPAS_BUILD 1
#endif

#if GGML_SYCL_FA_DPAS_BUILD
static constexpr int   FA_DPAS_WG    = 256;      // work-group size
static constexpr int   FA_DPAS_NSG   = 16;       // sub-groups of 16 lanes
static constexpr int   FA_DPAS_MAX_Q = 31;       // larger batches go to the prefill kernels
static constexpr int   FA_DPAS_MAX_G = 32;       // query heads per KV head
static constexpr float FA_DPAS_MINF  = -1.0e30f; // finite start value of the running max

static bool fattn_dpas_support_arch(gpu_arch arch) {
    return arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31 ||
           arch == gpu_arch::intel_gpu_lnl_m   || arch == gpu_arch::intel_gpu_ptl_h   ||
           arch == gpu_arch::intel_gpu_ptl_u   || arch == gpu_arch::intel_gpu_wcl;
}

// head sizes with a compiled kernel (multiples of the 32-dim q4_0 / q8_0 block)
static bool fattn_dpas_head_ok(int64_t d) {
    return d == 64 || d == 96 || d == 128 || d == 256 || d == 512;
}

static bool fattn_dpas_aligned16(const ggml_tensor * t) {
    return t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0 && (uintptr_t) t->data % 16 == 0;
}

// F16 K/V with a mask per query head or the rel-bias table (ggml_flash_attn_ext_set_rel_bias), head size 128:
// the small-batch kernel below (f16 DPAS, split keys) for decode, small batches and row-exact ops, the rel-bias
// prefill kernel for larger rel-bias batches
static bool fattn_dpas_f16_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * pos   = dst->src[7];
    const ggml_tensor * bias  = dst->src[6];

    if (dst->src[5] != nullptr) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        (sinks && (sinks->type != GGML_TYPE_F32 || sinks->nb[0] != sizeof(float) || sinks->ne[0] < Q->ne[2]))) {
        return false;
    }
    if (Q->ne[0] != 128 || K->ne[0] != 128 || V->ne[0] != 128) {
        return false;
    }
    if (Q->ne[1] < 1 || K->ne[1] < 1 || K->ne[1] > INT32_MAX / 2 || K->ne[2] < 1 || V->ne[2] != K->ne[2] ||
        V->ne[1] != K->ne[1] || Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] > FA_DPAS_MAX_G) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(sycl::half) || V->nb[0] != sizeof(sycl::half) ||
        !fattn_dpas_aligned16(Q) || !fattn_dpas_aligned16(K) || !fattn_dpas_aligned16(V)) {
        return false;
    }
    if (pos) {
        // the per-cell pos rows are read as 16-byte vectors
        return mask == nullptr && bias && pos->type == GGML_TYPE_I32 && bias->type == GGML_TYPE_F16 &&
               pos->nb[0] == sizeof(int32_t) && bias->nb[0] == sizeof(sycl::half) && pos->ne[0] == K->ne[1] &&
               pos->ne[1] == Q->ne[1] * Q->ne[3] && pos->nb[1] % 16 == 0 && (uintptr_t) pos->data % 16 == 0 &&
               bias->ne[1] == Q->ne[1] && bias->ne[3] == Q->ne[3] && Q->ne[2] % bias->ne[2] == 0;
    }
    return mask && mask->type == GGML_TYPE_F16 && mask->nb[0] == sizeof(sycl::half) && mask->ne[0] >= K->ne[1] &&
           mask->ne[1] >= Q->ne[1] && Q->ne[2] % mask->ne[2] == 0 && Q->ne[3] % mask->ne[3] == 0;
}
#endif // GGML_SYCL_FA_DPAS_BUILD

bool ggml_sycl_flash_attn_ext_dpas_supported(int device, const ggml_tensor * dst) {
#if !GGML_SYCL_FA_DPAS_BUILD
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!g_ggml_sycl_fa_dpas || !fattn_dpas_support_arch(ggml_sycl_info().devices[device].hw_info.arch)) {
        return false;
    }

    // a per-head mask or the rel-bias table: only the F16 kernels read them
    if (dst->src[6] != nullptr || (dst->src[3] != nullptr && dst->src[3]->ne[2] > 1)) {
        return fattn_dpas_f16_supported(dst);
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q->type != GGML_TYPE_F32 || (K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0) || V->type != K->type ||
        (sinks && sinks->type != GGML_TYPE_F32)) {
        return false;
    }
    if (!fattn_dpas_head_ok(K->ne[0]) || V->ne[0] != K->ne[0] || Q->ne[0] != K->ne[0]) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > FA_DPAS_MAX_Q || K->ne[1] < 1) {
        return false;
    }
    if (K->ne[2] < 1 || V->ne[2] != K->ne[2] || Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] > FA_DPAS_MAX_G) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[0] != sizeof(sycl::half) ||
                 mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1])) {
        return false;
    }
    if (sinks && (sinks->nb[0] != sizeof(float) || sinks->ne[0] < Q->ne[2])) {
        return false;
    }
    if (Q->nb[0] != sizeof(float)) {
        return false;
    }
    // q4_0 / q8_0 rows are read as dwords from the dword at or before the row start: rows must start on 2-byte bounds
    // (head sizes with an odd block count, e.g. 96, put every other head 2 bytes into a dword)
    // so a row's first/last dword may hold 2 bytes outside it; aligned dwords never cross the 128-byte buffer alignment
    for (const ggml_tensor * t : { K, V }) {
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 2 != 0 || t->nb[2] % 2 != 0 || t->nb[3] % 2 != 0 ||
            (uintptr_t) t->data % 2 != 0) {
            return false;
        }
    }
    return true;
#endif
}

#if GGML_SYCL_FA_DPAS_BUILD

typedef short fa_dpas_short8 __attribute__((ext_vector_type(8)));
typedef int   fa_dpas_int4   __attribute__((ext_vector_type(4)));
typedef int   fa_dpas_int8   __attribute__((ext_vector_type(8)));
typedef float fa_dpas_float8 __attribute__((ext_vector_type(8)));

// Xe2 DPAS (SIMD16, systolic depth 8, repeat count 8) as inline vISA: joint_matrix has no s4 type.
// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s4, 16 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s4(const fa_dpas_short8 & a, const fa_dpas_int4 & b, fa_dpas_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s4.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s8, 32 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s8(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s8.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// acc += a x b. a: 8 rows x 16 f16, one per lane per row. b: 16 x 16 f16 as pairs, 8 dwords per lane, one column per lane.
static inline void fa_dpas_f16(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(acc);
#endif
}

// Block b of a row = f16 scale + NQ dwords of quants (q4_0: 18 bytes, NQ 4; q8_0: 34 bytes, NQ 8), at byte (2 + 4 NQ) b.
// w = the NQ + 1 dwords from that byte / 4. An odd block starts 2 bytes into w[0].
template <int NQ> static inline void fa_dpas_split_block(const uint32_t * w, bool odd, uint32_t & sc, uint32_t * q) {
    if (!odd) {
        sc = w[0] & 0xFFFFu;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            q[i] = (w[i] >> 16) | (w[i + 1] << 16);
        }
    } else {
        sc = w[0] >> 16;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            q[i] = w[i + 1];
        }
    }
}

struct fa_dpas_params {
    const char *   Q;
    const char *   K;
    const char *   V;
    const char *   mask;
    float *        dst;
    float *        parts;  // per-split partial results, nsplit > 1 only
    sycl::float2 * meta;   // per-split (max, sum), nsplit > 1 only
    float          scale;
    const float *  sinks;  // per query head softmax sink logit, or null
    int            G;      // query heads per KV head
    int            nq;     // query tokens per work-group
    int            nqc;    // token chunks
    int            ne01;
    int            ne02;
    int            ne11;
    int            ne33;
    int            nsplit;
    int            chunk;  // keys per split
    int64_t        nb01, nb02, nb03;
    int64_t        nb11, nb12, nb13;
    int64_t        nb21, nb22, nb23;
    int64_t        nb31, nb33;
    // FA_DPAS_MASK_HEAD: the mask head stride (halfs) and head count
    int64_t        nb32;
    int            ne32;
    // FA_DPAS_MASK_REL: the mask of (query c, head h, cell j) of stream s is
    //   bias[c nbb1 + (h % neb2) nbb2 + s nbb3 + pos[(c + s ne01) nbp1 + j]]  (strides in elements)
    const int32_t *    pos;
    const sycl::half * bias;
    int64_t            nbp1;
    int64_t            nbb1, nbb2, nbb3;
    int                neb2;
    int                masked_pos; // the pos of the cells that are masked in every row, or -1
};

// the mask of the small-batch kernel: none or one per query token (q4_0 / q8_0 / f16), one per query token and head
// (f16), or the rel-bias table gathered by the cell pos (f16)
enum fa_dpas_mask_mode { FA_DPAS_MASK_TOKEN = 0, FA_DPAS_MASK_HEAD = 1, FA_DPAS_MASK_REL = 2 };
// the K/V type of the small-batch kernel
enum fa_dpas_kv_type { FA_DPAS_KV_Q4_0 = 0, FA_DPAS_KV_Q8_0 = 1, FA_DPAS_KV_F16 = 2 };

// One work-group = one KV head x nq query tokens x one slice of the keys: S = Q K^T by s8 x s4 / s8 x s8 DPAS on the
// raw K blocks, online softmax, O^T += V^T P^T by f16 DPAS. ST: stage K/V tiles in SLM (1: 16-byte, 2: dword loads).
// LG: 256-register file (no spills at 32 rows, same occupancy). KT: the K/V type (fa_dpas_kv_type); f16 K/V take
// S = Q K^T by f16 DPAS from Q in SLM. MM: the mask (fa_dpas_mask_mode); with a per-head or rel-bias mask a key tile
// whose mask is -inf in every row of the work-group is skipped (it would add exact zeros).
template <int D, int NR16, int KG, int TC, int ST, bool LG, int KT, int MM>
static void fattn_dpas_kernel(const fa_dpas_params p, const int nkvh, const int ne03, dpct::queue_ptr stream) {
    constexpr bool F16  = KT == FA_DPAS_KV_F16;
    constexpr bool Q8   = KT == FA_DPAS_KV_Q8_0;
    constexpr int RMAX  = 16 * NR16;
    constexpr int RTMAX = RMAX / 8;                         // 8-row tiles in Q.K
    constexpr int TK    = 16 * KG;                          // keys per tile
    static_assert(QK4_0 == QK8_0, "q4_0 and q8_0 blocks hold the same number of dims");
    constexpr int BB    = Q8 ? (int) sizeof(block_q8_0) : (int) sizeof(block_q4_0); // bytes per block
    constexpr int NQ    = (BB - 2) / 4;                     // quant dwords per block
    constexpr int NB    = D / QK4_0;                        // blocks per row
    constexpr int DC    = D / 16;                           // f16: 16-dim chunks of Q.K
    constexpr int NQA   = F16 ? DC : NB;                    // Q A-operands per 8-row tile
    constexpr int DPS   = D <= 128 ? 8 : D <= 256 ? 16 : 32; // head dims per sub-group in P.V, inside one 32-dim block
    constexpr int NPV   = D / DPS;                          // sub-groups doing P.V, the rest only reach its barrier
    constexpr bool ALLPV = NPV == FA_DPAS_NSG;              // every sub-group does P.V: no runtime check in the loops
    constexpr int XT    = DPS / 8;                          // 8-dim tiles per sub-group in P.V
    constexpr int PS    = TK + 16;                          // f16 row stride of P
    constexpr int RW    = F16 ? D / 2 : NB * BB / 4;        // dwords per row
    constexpr bool ODD  = !F16 && (NB * BB) % 4 != 0;       // odd block count: rows may start 2 bytes into a dword
    constexpr int SW    = RW + 1;                           // odd SLM row stride: no bank conflicts for lane = key
    static_assert(D % QK4_0 == 0 && D % DPS == 0 && NPV <= FA_DPAS_NSG && QK4_0 % DPS == 0, "unsupported head size");
    static_assert(ST != 1 || RW % 4 == 0, "16-byte staging needs rows of 16-byte multiples");
    static_assert(!F16 || ST == 0, "f16 K/V are read from global memory");
    static_assert(F16 || MM == FA_DPAS_MASK_TOKEN, "per-head and rel-bias masks need f16 K/V");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<fa_dpas_short8, 1> sQ8(sycl::range<1>(RTMAX * NQA * 16), cgh); // Q s8 / f16 [tile][block][lane]
        sycl::local_accessor<fa_dpas_float8, 1> sDQ(sycl::range<1>(F16 ? 1 : RTMAX * NB), cgh); // Q scales [tile][block]
        sycl::local_accessor<float, 1>          sS(sycl::range<1>(RMAX * TK), cgh);        // S [row][key]
        sycl::local_accessor<fa_dpas_int8, 1>   sP(sycl::range<1>(RMAX * PS / 16), cgh);   // P f16 [row][key]
        sycl::local_accessor<float, 1>          sM(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sL(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sA(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<uint32_t, 1>       sKV(sycl::range<1>(ST ? TK * SW : 1), cgh);
        const sycl::nd_range<3> ndr(sycl::range<3>((size_t) ne03 * p.nqc, nkvh, (size_t) p.nsplit * FA_DPAS_WG),
                                    sycl::range<3>(1, 1, FA_DPAS_WG));
        auto kern = [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0) / p.nqc;
                const int c0    = (it.get_group(0) % p.nqc) * p.nq;
                const int G     = p.G;
                const int R     = p.nq * G;
                const int RT    = (R + 7) / 8;
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                short *   sQs   = (short *) &sQ8[0];
                short *   sPs   = (short *) &sP[0];

                // f16: Q * scale -> f16 A-operands, element m of lane l = row 8 t + m, dim 16 dc + l
                if constexpr (F16) {
                    for (int e = tid; e < RT * 8 * DC; e += FA_DPAS_WG) {
                        const int  r    = e / DC;
                        const int  dc   = e % DC;
                        const int  c    = c0 + r / G;
                        const bool real = r < R && c < p.ne01;
                        const float * qr = (const float *) (p.Q + seq * p.nb03 + (int64_t) (real ? c : 0) * p.nb01 +
                                                            (int64_t) (kvh * G + (real ? r % G : 0)) * p.nb02) + dc * 16;
                        const int t = r / 8;
                        const int m = r % 8;
#pragma unroll
                        for (int l = 0; l < 16; ++l) {
                            const sycl::half h = sycl::half(real ? qr[l] * p.scale : 0.0f);
                            sQs[((t * DC + dc) * 16 + l) * 8 + m] = (short) sycl::bit_cast<uint16_t>(h);
                        }
                    }
                }
                // Q -> s8, one scale per row and block. Lane l holds the 2 dims of K column elements 2 l, 2 l + 1:
                // q4_0 packs dims l and l + 16 into byte l, q8_0 stores dims in order
                for (int e = tid; e < (F16 ? 0 : RT * 8 * NB); e += FA_DPAS_WG) {
                    const int  r    = e / NB;
                    const int  b    = e % NB;
                    const int  c    = c0 + r / G;
                    const bool real = r < R && c < p.ne01;
                    const float * qr = (const float *) (p.Q + seq * p.nb03 + (int64_t) (real ? c : 0) * p.nb01 +
                                                        (int64_t) (kvh * G + (real ? r % G : 0)) * p.nb02) + b * QK4_0;
                    float x[QK4_0];
                    float amax = 0.0f;
#pragma unroll
                    for (int i = 0; i < QK4_0; ++i) {
                        x[i] = real ? qr[i] * p.scale : 0.0f;
                        amax = sycl::fmax(amax, sycl::fabs(x[i]));
                    }
                    const float dq = amax / 127.0f;
                    const float iq = amax > 0.0f ? 127.0f / amax : 0.0f;
                    const int   t  = r / 8;
                    const int   m  = r % 8;
                    ((float *) &sDQ[0])[(t * NB + b) * 8 + m] = dq;
#pragma unroll
                    for (int l = 0; l < 16; ++l) {
                        const int lo = (int) sycl::rint(x[Q8 ? 2 * l : l] * iq);
                        const int hi = (int) sycl::rint(x[Q8 ? 2 * l + 1 : l + 16] * iq);
                        sQs[((t * NB + b) * 16 + l) * 8 + m] = (short) ((lo & 0xFF) | ((hi & 0xFF) << 8));
                    }
                }
                // rows without a query keep P = 0
                for (int e = tid; e < RMAX * PS; e += FA_DPAS_WG) {
                    sPs[e] = 0;
                }
                if (tid < RMAX) {
                    // the sink is one more key with logit sinks[head] and no value; the first key slice owns it
                    const bool sink = p.sinks && split == 0 && tid < R;
                    sM[tid] = sink ? p.sinks[kvh * G + tid % G] : FA_DPAS_MINF;
                    sL[tid] = sink ? 1.0f : 0.0f;
                    sA[tid] = 1.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int    k_begin = split * p.chunk;
                const int    k_end   = sycl::min(k_begin + p.chunk, p.ne11);
                const char * Kh      = p.K + seq * p.nb13 + kvh * p.nb12;
                const char * Vh      = p.V + seq * p.nb23 + kvh * p.nb22;
                const sycl::half * mrow = p.mask ? (const sycl::half *) (p.mask + (seq % p.ne33) * p.nb33) : nullptr;

                // the mask of row r (query c = c0 + r / G, head kvh G + r % G) at key j, for the per-head and rel-bias
                // modes: the bias table entry of the pos of the cell, as the CPU reads it (bp[pp[ic]])
                auto mask_at = [&](int r, int c, int j) -> float {
                    const int h = kvh * G + r % G;
                    if constexpr (MM == FA_DPAS_MASK_REL) {
                        const int32_t pj = p.pos[(int64_t) (c + seq * p.ne01) * p.nbp1 + j];
                        return static_cast<float>(p.bias[(int64_t) c * p.nbb1 + (int64_t) (h % p.neb2) * p.nbb2 +
                                                         (int64_t) seq * p.nbb3 + pj]);
                    } else {
                        return static_cast<float>(mrow[((int64_t) c * p.nb31 + (int64_t) (h % p.ne32) * p.nb32) /
                                                           (int64_t) sizeof(sycl::half) + j]);
                    }
                };

                // O^T: lane = query row, element m = head dim
                fa_dpas_float8 o[NR16][XT];
#pragma unroll
                for (int t = 0; t < NR16; ++t) {
#pragma unroll
                    for (int x = 0; x < XT; ++x) {
                        o[t][x] = 0.0f;
                    }
                }
                // the dims of this sub-group lie in one V block
                const int vb  = (DPS * w) / QK4_0;
                const int vd0 = (DPS * w) % QK4_0;

                // keys past k_end repeat key k_begin, their scores are -inf
                auto stage = [&](const char * base, int64_t nb, int t0) {
                    if constexpr (ST == 2) {
                        for (int e = tid; e < TK * RW; e += FA_DPAS_WG) {
                            const int key = e / RW;
                            const int q   = e - key * RW;
                            const int j   = t0 + key < k_end ? t0 + key : k_begin;
                            sKV[key * SW + q] = ((const uint32_t *) (base + (int64_t) j * nb))[q];
                        }
                        return;
                    }
                    constexpr int RV  = ST == 1 ? RW / 4 : 1;
                    constexpr int NV  = TK * RV;
                    constexpr int PER = (NV + FA_DPAS_WG - 1) / FA_DPAS_WG;
                    sycl::uint4 v[PER];
#pragma unroll
                    for (int i = 0; i < PER; ++i) {
                        const int e = tid + i * FA_DPAS_WG;
                        if (e < NV) {
                            const int key = e / RV;
                            const int q   = e - key * RV;
                            const int j   = t0 + key < k_end ? t0 + key : k_begin;
                            v[i] = ((const sycl::uint4 *) (base + (int64_t) j * nb))[q];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < PER; ++i) {
                        const int e = tid + i * FA_DPAS_WG;
                        if (e < NV) {
                            const int key = e / RV;
                            const int q   = e - key * RV;
                            const int o   = key * SW + 4 * q;
                            sKV[o + 0] = v[i][0];
                            sKV[o + 1] = v[i][1];
                            sKV[o + 2] = v[i][2];
                            sKV[o + 3] = v[i][3];
                        }
                    }
                };

                // the NQ + 1 dwords of block b: of tile key kk in SLM, or of a row that starts 2 ph bytes into dword row[0]
                auto load_block = [&](const uint32_t * row, int ph, int kk, int b, uint32_t * wv) {
                    const int wd = (2 * ph + BB * b) / 4;
                    if constexpr (ST) {
                        GGML_UNUSED(row);
#pragma unroll
                        for (int i = 0; i < NQ + 1; ++i) {
                            wv[i] = sKV[kk * SW + wd + i];
                        }
                    } else {
                        GGML_UNUSED(kk);
#pragma unroll
                        for (int i = 0; i < NQ + 1; ++i) {
                            wv[i] = row[wd + i];
                        }
                    }
                };

                // start loading the rows of keys t0 .. t0 + TK - 1 into L1 while the current phase computes
                auto prefetch_rows = [&](const char * base, int64_t nb, int t0) {
                    constexpr int NL = RW * 4 / 64 + 1; // cache lines per row
                    for (int e = tid; e < TK * NL; e += FA_DPAS_WG) {
                        const int j = t0 + e / NL;
                        if (j < k_end) {
                            char * pa = const_cast<char *>(base) + (int64_t) j * nb + (e % NL) * 64;
                            sycl::ext::oneapi::experimental::prefetch(pa, sycl::ext::oneapi::experimental::properties{ sycl::ext::oneapi::experimental::prefetch_hint_L1 });
                        }
                    }
                };

                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    if constexpr (MM != FA_DPAS_MASK_TOKEN) {
                        // a tile whose mask is -inf in every row adds nothing: skip it (uniform over the work-group)
                        bool live = false;
                        if constexpr (MM == FA_DPAS_MASK_REL) {
                            // the rel-bias rows of a query token share its pos row
                            live = p.masked_pos < 0;
                            for (int e = tid; !live && e < p.nq * TK; e += FA_DPAS_WG) {
                                const int c  = c0 + e / TK;
                                const int jj = t0 + e % TK;
                                live = c < p.ne01 && jj < k_end &&
                                       p.pos[(int64_t) (c + seq * p.ne01) * p.nbp1 + jj] != p.masked_pos;
                            }
                        } else {
                            for (int e = tid; !live && e < R * TK; e += FA_DPAS_WG) {
                                const int r  = e / TK;
                                const int c  = c0 + r / G;
                                const int jj = t0 + e % TK;
                                live = c < p.ne01 && jj < k_end && mask_at(r, c, jj) != -INFINITY;
                            }
                        }
                        if (!sycl::any_of_group(it.get_group(), live)) {
                            continue;
                        }
                    }
                    if constexpr (ST) {
                        stage(Kh, p.nb11, t0);
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    prefetch_rows(Vh, p.nb21, t0);

                    // S = Q K^T, job = (key group, TC row tiles)
                    const int nch = (RT + TC - 1) / TC;
                    for (int job = w; job < KG * nch; job += FA_DPAS_NSG) {
                        const int  kg = job % KG;
                        const int  ch = job / KG;
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const char *     kb = Kh + (int64_t) (jv ? j : k_begin) * p.nb11;
                        const int        kp = (ST || !ODD) ? 0 : (int) (((uintptr_t) kb >> 1) & 1); // row starts 2 bytes into a dword
                        const uint32_t * kr = (const uint32_t *) (kb - 2 * kp);
                        fa_dpas_float8 s[TC];
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            s[tt] = 0.0f;
                        }
                        if constexpr (F16) {
                            // B = K^T (16 dims x 16 keys): lane = key, the 8 dwords of dims 16 dc .. 16 dc + 15
                            const sycl::uint4 * kr4 = (const sycl::uint4 *) kb;
#pragma unroll
                            for (int dc = 0; dc < DC; ++dc) {
                                const sycl::uint4  k0 = kr4[2 * dc];
                                const sycl::uint4  k1 = kr4[2 * dc + 1];
                                const fa_dpas_int8 bv = { (int) k0[0], (int) k0[1], (int) k0[2], (int) k0[3],
                                                          (int) k1[0], (int) k1[1], (int) k1[2], (int) k1[3] };
#pragma unroll
                                for (int tt = 0; tt < TC; ++tt) {
                                    const int t = ch * TC + tt;
                                    if (t < RT) {
                                        fa_dpas_f16(sQ8[(t * DC + dc) * 16 + lane], bv, s[tt]);
                                    }
                                }
                            }
                        }
#pragma unroll
                        for (int b = 0; b < (F16 ? 0 : NB); ++b) {
                            uint32_t wv[NQ + 1];
                            uint32_t sc;
                            uint32_t q[NQ];
                            load_block(kr, kp, kg * 16 + lane, b, wv);
                            fa_dpas_split_block<NQ>(wv, (b + kp) % 2 != 0, sc, q);
                            const float dk = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc));
#pragma unroll
                            for (int tt = 0; tt < TC; ++tt) {
                                const int t = ch * TC + tt;
                                if (t < RT) {
                                    fa_dpas_int8 ia;
                                    if constexpr (Q8) {
                                        const fa_dpas_int8 bv = { (int) q[0], (int) q[1], (int) q[2], (int) q[3],
                                                                  (int) q[4], (int) q[5], (int) q[6], (int) q[7] };
                                        fa_dpas_s8_s8(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    } else {
                                        // u4 (n + 8) -> s4 n
                                        const fa_dpas_int4 bv = { (int) (q[0] ^ 0x88888888u), (int) (q[1] ^ 0x88888888u),
                                                                  (int) (q[2] ^ 0x88888888u), (int) (q[3] ^ 0x88888888u) };
                                        fa_dpas_s8_s4(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    }
                                    const fa_dpas_float8 ds = sDQ[t * NB + b] * dk;
                                    s[tt] += __builtin_convertvector(ia, fa_dpas_float8) * ds;
                                }
                            }
                        }
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            const int t = ch * TC + tt;
                            if (t < RT) {
#pragma unroll
                                for (int m = 0; m < 8; ++m) {
                                    const int r = t * 8 + m;
                                    const int c = c0 + r / G;
                                    if (r < R) {
                                        const bool  ok = jv && c < p.ne01;
                                        float mv;
                                        if constexpr (MM == FA_DPAS_MASK_TOKEN) {
                                            mv = (ok && mrow) ? static_cast<float>(mrow[((int64_t) c * p.nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
                                        } else {
                                            mv = ok ? mask_at(r, c, j) : 0.0f;
                                        }
                                        sS[r * TK + kg * 16 + lane] = ok ? s[tt][m] + mv : -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (ST) {
                        stage(Vh, p.nb21, t0); // the softmax does not use sKV
                    }

                    // online softmax per row -> P
                    for (int r = w; r < R; r += FA_DPAS_NSG) {
                        float sv[KG];
                        float mt = -INFINITY;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            sv[i] = sS[r * TK + i * 16 + lane];
                            mt    = sycl::fmax(mt, sv[i]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);
                        float       lt    = 0.0f;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            const float pv = sycl::native::exp(sv[i] - m_new);
                            sPs[r * PS + i * 16 + lane] = (short) sycl::bit_cast<uint16_t>(sycl::half(pv));
                            lt += pv;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    prefetch_rows(Kh, p.nb11, t0 + TK);

                    // O^T = diag(alpha) O^T + V^T P^T. A = V^T (8 dims x 16 keys, lane = key), B = P^T (16 keys x 16 rows)
#pragma unroll
                    for (int t = 0; t < NR16; ++t) {
                        const float a = sA[t * 16 + lane];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
                            o[t][x] *= a;
                        }
                    }
#pragma unroll 1
                    for (int kg = 0; kg < ((ALLPV || w < NPV) ? KG : 0); ++kg) {
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const char *     vbp = Vh + (int64_t) (jv ? j : k_begin) * p.nb21;
                        const int        vp  = (ST || !ODD) ? 0 : (int) (((uintptr_t) vbp >> 1) & 1);
                        const uint32_t * vr  = (const uint32_t *) (vbp - 2 * vp);
                        fa_dpas_short8 va[XT];
                        if constexpr (F16) {
                            // A = V^T (8 dims x 16 keys): lane = key, its 8 halfs of dims DPS w + 8 x ..
                            GGML_UNUSED(vr);
#pragma unroll
                            for (int x = 0; x < XT; ++x) {
                                va[x] = 0;
                                if (jv) {
                                    va[x] = *(const fa_dpas_short8 *) (vbp + (DPS * w + 8 * x) * sizeof(sycl::half));
                                }
                            }
                        }
                        uint32_t wv[NQ + 1];
                        uint32_t sc = 0;
                        uint32_t q[NQ];
                        if constexpr (!F16) {
                            load_block(vr, vp, kg * 16 + lane, vb, wv);
                            fa_dpas_split_block<NQ>(wv, (vb + vp) % 2 != 0, sc, q);
                        }
                        const float dv = jv ? static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc)) : 0.0f;
#pragma unroll
                        for (int x = 0; x < (F16 ? 0 : XT); ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = vd0 + x * 8 + m;
                                int       v;
                                if constexpr (Q8) {
                                    v = (int8_t) ((q[d / 4] >> (8 * (d % 4))) & 0xFF);
                                } else {
                                    const int byte = d % 16;
                                    v = (int) ((q[byte / 4] >> (8 * (byte % 4) + 4 * (d / 16))) & 0xF) - 8;
                                }
                                va[x][m] = (short) sycl::bit_cast<uint16_t>(sycl::half((float) v * dv));
                            }
                        }
#pragma unroll
                        for (int t = 0; t < NR16; ++t) {
                            const fa_dpas_int8 pb = sP[((t * 16 + lane) * PS + kg * 16) / 16];
#pragma unroll
                            for (int x = 0; x < XT; ++x) {
                                fa_dpas_f16(va[x], pb, o[t][x]);
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int t = 0; t < NR16; ++t) {
                    const int r = t * 16 + lane;
                    const int c = c0 + r / G;
                    if ((ALLPV || w < NPV) && r < R && c < p.ne01) {
                        const int64_t jdu = ((int64_t) seq * p.ne01 + c) * p.ne02 + kvh * G + r % G;
                        const float   l   = sL[r];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = w * DPS + x * 8 + m;
                                if (p.nsplit == 1) {
                                    p.dst[jdu * D + d] = l > 0.0f ? o[t][x][m] / l : 0.0f;
                                } else {
                                    p.parts[(jdu * p.nsplit + split) * D + d] = o[t][x][m];
                                }
                            }
                        }
                    }
                }
                if (p.nsplit > 1 && tid < R && c0 + tid / G < p.ne01) {
                    const int64_t jdu = ((int64_t) seq * p.ne01 + c0 + tid / G) * p.ne02 + kvh * G + tid % G;
                    p.meta[jdu * p.nsplit + split] = sycl::float2(sM[tid], sL[tid]);
                }
            };
        if constexpr (LG) {
            cgh.parallel_for(ndr, sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> }, kern);
        } else {
            cgh.parallel_for(ndr, kern);
        }
    });
}

// merge the per-split partial results (the max and sum of each split in meta); a row without any unmasked key or sink
// gets 0, as on the CPU
template <int D>
static void fattn_dpas_combine(const float * parts, const sycl::float2 * meta, float * dst, int nsplit, int ne01,
                               int ne02, int ne03, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne03, ne02, (size_t) ne01 * D), sycl::range<3>(1, 1, D)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int     tid = it.get_local_id(2);
                const int64_t jdu = ((int64_t) it.get_group(0) * ne01 + it.get_group(2)) * ne02 + it.get_group(1);
                const sycl::float2 * m = meta + jdu * nsplit;
                float mx = m[0].x();
                for (int l = 1; l < nsplit; ++l) {
                    mx = sycl::fmax(mx, m[l].x());
                }
                float num = 0.0f;
                float den = 0.0f;
                for (int l = 0; l < nsplit; ++l) {
                    const float a = sycl::native::exp(m[l].x() - mx);
                    num += a * parts[(jdu * nsplit + l) * D + tid];
                    den += a * m[l].y();
                }
                dst[jdu * D + tid] = den > 0.0f ? num / den : 0.0f;
            });
    });
}

// row_exact (ggml_flash_attn_ext_set_row_exact): one query token per work-group and the key splits of a one-token
// batch, so that every row is computed exactly as it is alone
template <int D, int NR16, int KG, int TC, int ST, int KT, int MM>
static void fattn_dpas_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool row_exact = false) {
    constexpr int TK = 16 * KG;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * pos  = dst->src[7];
    const ggml_tensor * bias = dst->src[6];

    const int ne01 = Q->ne[1];
    const int ne11 = K->ne[1];
    const int nkvh = K->ne[2];
    const int ne03 = Q->ne[3];

    fa_dpas_params p;
    p.Q    = (const char *) Q->data;
    p.K    = (const char *) K->data;
    p.V    = (const char *) V->data;
    p.mask = mask ? (const char *) mask->data : nullptr;
    p.sinks = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    p.dst  = (float *) dst->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.G    = Q->ne[2] / K->ne[2];
    p.nq   = row_exact ? 1 : std::max(1, std::min(ne01, 16 * NR16 / p.G));
    p.nqc  = (ne01 + p.nq - 1) / p.nq;
    p.ne01 = ne01;
    p.ne02 = Q->ne[2];
    p.ne11 = ne11;
    p.ne33 = mask ? mask->ne[3] : 1;
    p.nb01 = Q->nb[1];
    p.nb02 = Q->nb[2];
    p.nb03 = Q->nb[3];
    p.nb11 = K->nb[1];
    p.nb12 = K->nb[2];
    p.nb13 = K->nb[3];
    p.nb21 = V->nb[1];
    p.nb22 = V->nb[2];
    p.nb23 = V->nb[3];
    p.nb31 = mask ? mask->nb[1] : 0;
    p.nb33 = mask ? mask->nb[3] : 0;
    p.nb32 = mask ? mask->nb[2] : 0;
    p.ne32 = mask ? mask->ne[2] : 1;
    p.pos  = pos ? (const int32_t *) pos->data : nullptr;
    p.bias = bias ? (const sycl::half *) bias->data : nullptr;
    p.nbp1 = pos ? pos->nb[1] / (int64_t) sizeof(int32_t) : 0;
    p.nbb1 = bias ? bias->nb[1] / (int64_t) sizeof(sycl::half) : 0;
    p.nbb2 = bias ? bias->nb[2] / (int64_t) sizeof(sycl::half) : 0;
    p.nbb3 = bias ? bias->nb[3] / (int64_t) sizeof(sycl::half) : 0;
    p.neb2 = bias ? bias->ne[2] : 1;
    p.masked_pos = pos ? ggml_get_op_params_i32(dst, 7) : -1;

    // split the keys so that all work-groups run in one wave
    // SLM fits about 4 work-groups per Xe2 core with 16 rows, 2 with 32. Xe2 core = 8 EUs, nsm = EUs / 16.
    const int xe_cores = std::max(1, ggml_sycl_info().devices[ctx.device].nsm * 2);
    const int wave     = xe_cores * (NR16 == 1 ? 4 : 2);
    const int ntiles   = (ne11 + TK - 1) / TK;
    int nsplit = std::max(1, std::min(wave / (nkvh * ne03 * (row_exact ? 1 : p.nqc)), ntiles));
    p.chunk    = ((ntiles + nsplit - 1) / nsplit) * TK;
    nsplit     = (ne11 + p.chunk - 1) / p.chunk;
    p.nsplit   = nsplit;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit * ggml_nelements(dst));
        meta.alloc((size_t) nsplit * ggml_nrows(dst));
    }
    p.parts = parts.get();
    p.meta  = meta.get();

    dpct::queue_ptr stream = ctx.stream();
    // the large register file pays off for 32-row launches from about 4K keys
    if (NR16 == 2 && ne11 >= 4096) {
        fattn_dpas_kernel<D, NR16, KG, TC, ST, true, KT, MM>(p, nkvh, ne03, stream);
    } else {
        fattn_dpas_kernel<D, NR16, KG, TC, ST, false, KT, MM>(p, nkvh, ne03, stream);
    }

    if (nsplit > 1) {
        float *        parts_p = parts.get();
        sycl::float2 * meta_p  = meta.get();
        float *        dst_p   = (float *) dst->data;
        if constexpr (KT == FA_DPAS_KV_F16) {
            fattn_dpas_combine<D>(parts_p, meta_p, dst_p, nsplit, ne01, Q->ne[2], Q->ne[3], stream);
            return;
        }
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(nsplit * sizeof(sycl::float2)), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(Q->ne[3], Q->ne[2], (size_t) ne01 * D), sycl::range<3>(1, 1, D)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_combine_results<D>(parts_p, meta_p, dst_p, nsplit,
                                                  lm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

template <bool Q8> static void fattn_dpas_dispatch(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    constexpr int KTQ = Q8 ? FA_DPAS_KV_Q8_0 : FA_DPAS_KV_Q4_0;

    const int D    = K->ne[0];
    const int G    = Q->ne[2] / K->ne[2];
    const int ne01 = Q->ne[1];

    bool aligned16 = true;
    for (const ggml_tensor * t : { K, V }) {
        aligned16 = aligned16 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0 &&
                    (uintptr_t) t->data % 16 == 0;
    }

    // single token decode: staged K/V tiles are faster at long context
    if (D == 256 && ne01 == 1 && G <= 16 && aligned16) {
        fattn_dpas_case<256, 1, 8, 1, 1, KTQ, FA_DPAS_MASK_TOKEN>(ctx, dst);
        return;
    }
    const bool rows16 = ne01 * G <= 16;
    // 256-key tiles for one row tile; 128 at head size 512 so S and the Q copy stay in SLM for the planned waves
#define FATTN_DPAS_D(D_, KG16_)                                                                        \
    case D_:                                                                                           \
        if (rows16) {                                                                                  \
            fattn_dpas_case<D_, 1, KG16_, 1, 0, KTQ, FA_DPAS_MASK_TOKEN>(ctx, dst);                                     \
        } else {                                                                                       \
            fattn_dpas_case<D_, 2, 8, 2, 0, KTQ, FA_DPAS_MASK_TOKEN>(ctx, dst);                                         \
        }                                                                                              \
        break;
    // q8_0 at head size 128: its 136-byte rows gather poorly per lane, stage them in SLM by coalesced dword loads
    bool aligned4 = true;
    for (const ggml_tensor * t : { K, V }) {
        aligned4 = aligned4 && t->nb[1] % 4 == 0 && t->nb[2] % 4 == 0 && t->nb[3] % 4 == 0 && (uintptr_t) t->data % 4 == 0;
    }
    if (Q8 && D == 128 && aligned4) {
        if (rows16) {
            fattn_dpas_case<128, 1, 16, 1, 2, KTQ, FA_DPAS_MASK_TOKEN>(ctx, dst);
        } else {
            fattn_dpas_case<128, 2, 8, 2, 2, KTQ, FA_DPAS_MASK_TOKEN>(ctx, dst);
        }
        return;
    }
    switch (D) {
        FATTN_DPAS_D( 64, 16)
        FATTN_DPAS_D( 96, 16)
        FATTN_DPAS_D(128, 16)
        FATTN_DPAS_D(256, 16)
        FATTN_DPAS_D(512, 8)
        default:
            GGML_ABORT("fattn-dpas: unsupported head size %d", D);
    }
#undef FATTN_DPAS_D
}

// ---------------------------------------------------------------------------------------------------------------------
// rel-bias prefill: ggml_flash_attn_ext_set_rel_bias with F16 K/V and head size 128, larger batches. The algorithm of
// the Vulkan backend's flash_attn_prefill_rel.comp on f16 DPAS:
//
// A work-group owns FA_REL_WR = 64 query rows of one head: 16 sub-groups, sub-group sg holds the rows rb = sg % 4
// (16 rows, one per lane) and dh = sg / 4, which picks both the 8 cells of the FA_REL_BC = 32 cell tile whose scores
// the sub-group computes and the 32 output dims of the 128 it accumulates:
//   S^T (8 cells x 16 rows) = K (8 x 128) Q^T (128 x 16)       accumulator lane = query row, element = cell
// the four sub-groups of a row block exchange their partial row max through SLM, write their P (f16) to SLM, and each
// reads P of all the tile's cells back as B matrices:
//   O^T (32 dims x 16 rows) += V^T (32 x 32 cells) P^T (32 x 16)
// The row sum is kept per sub-group (over its own cells, with the common max) and summed at the end. Q (scaled, f16)
// stays in SLM. K and V^T tiles are staged in SLM as A matrices in register order (lane l, dword g = row 2 g + l / 8,
// columns 2 (l % 8), 2 (l % 8) + 1), so each lane reads its operand with one 16-byte load.
// Each lane gathers the bias of its row and its sub-group's 8 cells (bias[pos], -inf where masked) into registers.
// The tile flags come from fattn_rel_tiles (once per row block for all heads): a dead tile (no row of the block sees
// any of its cells, pos == masked_pos everywhere) is skipped; a uniform tile (every pos of its cells in every row of
// the block equals u_pos, the pos of the first cell of the first row: the far cells of a full attention at depth)
// takes the bias of u_pos, read once per lane, instead of the gather.

static constexpr int FA_REL_WR = 64; // query rows per work-group
static constexpr int FA_REL_BC = 32; // cells per tile

struct fa_rel_params {
    const char *       Q;
    const char *       K;
    const char *       V;
    const int32_t *    pos;
    const sycl::half * bias;
    const float *      sinks;
    float *            dst;
    float              scale;
    int                neq1, neq2, nek1, gqa, n_rb, n_tiles, n_words;
    int64_t            nbq1, nbq2, nbq3;       // bytes
    int64_t            nbk1, nbk2, nbk3;       // bytes
    int64_t            nbv1, nbv2, nbv3;       // bytes
    int64_t            nbp1;                   // int32
    int64_t            nbb1, nbb2, nbb3;       // halfs
    int                neb2;
    int                masked_pos;
    // in-kernel bias (FUSED): the packed inputs of the REL_BIAS node (rel-bias.hpp) and the affine tile data
    const char *       pk;
    int64_t            pk_hdr, pk_v, pk_t2, pk_x2, pk_tf, pk_tx;
    int                pk_c, pk_ncols, pk_nhead;
    const sycl::int2 * bw;                     // (B, W) per (stream, row block, tile)
};

// acc += a x b, a = the 8 x 16 f16 A matrix as 4 dwords per lane in register order (lane l, dword g = row 2 g + l / 8,
// columns 2 (l % 8) + {0, 1}), b as in fa_dpas_f16
static inline void fa_dpas_f16_a4(const fa_dpas_int4 & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(acc);
#endif
}

// the tile flags of fattn_rel_prefill_kernel: for the FA_REL_WR-row block y of stream z and the tiles 32 x .. 32 x + 31,
// two words at tl[((z n_rb + y) 2 + {0: live, 1: non-uniform}) n_words + x]:
//   live: bit i set when some row of the block sees some cell of tile 32 x + i (a pos != masked_pos)
//   non-uniform: bit i set when some pos of a cell of the tile in some row of the block differs from u_pos
template <bool UNALIGNED>
static void fattn_rel_tiles(const fa_rel_params p, uint32_t * tl, int ne3, dpct::queue_ptr stream) {
    constexpr int WG = 256;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> bits(sycl::range<1>(2), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne3, p.n_rb, (size_t) p.n_words * WG), sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                auto      sg    = it.get_sub_group();
                const int batch = it.get_group(0);
                const int y     = it.get_group(1);
                const int x     = it.get_group(2);
                if (tid < 2) {
                    bits[tid] = 0;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int row_base = y * FA_REL_WR;
                const int u_pos    = p.pos[(int64_t) (sycl::min(row_base, p.neq1 - 1) + batch * p.neq1) * p.nbp1];
                // FA_REL_WR is a multiple of the sub-group size: the lanes of a sub-group visit the same tiles
                const int  row_u  = row_base + tid % FA_REL_WR;
                const bool row_ok = row_u < p.neq1;
                const int32_t * prow = p.pos + (int64_t) (sycl::min(row_u, p.neq1 - 1) + batch * p.neq1) * p.nbp1;
                const int t0    = x * 32;
                const int t_end = sycl::min(t0 + 32, p.n_tiles);
                for (int t = t0 + tid / FA_REL_WR; t < t_end; t += WG / FA_REL_WR) {
                    bool live = false;
                    bool nonu = false;
                    if (row_ok) {
                        const int c0 = t * FA_REL_BC;
                        if constexpr (UNALIGNED) {
                            for (int i = 0; i < FA_REL_BC; ++i) {
                                const int c = c0 + i;
                                if (c < p.nek1) {
                                    const int d = prow[c];
                                    live = live || d != p.masked_pos;
                                    nonu = nonu || d != u_pos;
                                } else {
                                    nonu = true;
                                }
                            }
                        } else {
#pragma unroll
                            for (int i = 0; i < FA_REL_BC / 4; ++i) {
                                const sycl::int4 d = ((const sycl::int4 *) (prow + c0))[i];
#pragma unroll
                                for (int k = 0; k < 4; ++k) {
                                    live = live || d[k] != p.masked_pos;
                                    nonu = nonu || d[k] != u_pos;
                                }
                            }
                        }
                    }
                    live = sycl::any_of_group(sg, live);
                    nonu = sycl::any_of_group(sg, nonu);
                    if (sg.get_local_linear_id() == 0) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            b0(bits[0]), b1(bits[1]);
                        if (live) {
                            b0.fetch_or(1u << (t - t0));
                        }
                        if (nonu) {
                            b1.fetch_or(1u << (t - t0));
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid < 2) {
                    tl[((int64_t) (batch * p.n_rb + y) * 2 + tid) * p.n_words + x] = bits[tid];
                }
            });
    });
}

// the tile flags of the in-kernel bias: the words of fattn_rel_tiles, a third word per 32 tiles,
//   affine: bit i set when every cell j of tile 32 x + i and row r of the block (in the block / tile: rr = r - row0,
//   jj = j - c0) has pos == masked_pos where D = B + rr - jj < 0 or D >= W, and pos == min(D, c) elsewhere (c the clamp
//   column, cells past nek1 not checked), with B and W at bw[(z n_rb + y) n_tiles + tile]
// Pass 1 takes B from any cell with pos < c (all of them must agree), pass 2 checks the cells with that B and takes W as
// the smallest D >= 0 of a masked cell, which must be past the largest D of a live cell.
template <bool UNALIGNED>
static void fattn_rel_tiles_aff(const fa_rel_params p, uint32_t * tl, sycl::int2 * bw, int ne3, dpct::queue_ptr stream) {
    constexpr int WG   = 256;
    constexpr int TPW  = WG / FA_REL_WR; // tiles in flight per work-group pass
    constexpr int IMAX = INT32_MAX;
    constexpr int IMIN = INT32_MIN;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> bits(sycl::range<1>(3), cgh);
        sycl::local_accessor<int, 1>      sb(sycl::range<1>(5 * 32), cgh); // bmin, bmax, wmin, dmax, bad per tile
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne3, p.n_rb, (size_t) p.n_words * WG), sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                auto      sg    = it.get_sub_group();
                const int batch = it.get_group(0);
                const int y     = it.get_group(1);
                const int x     = it.get_group(2);
                if (tid < 3) {
                    bits[tid] = 0;
                }
                if (tid < 32) {
                    sb[tid] = IMAX; sb[32 + tid] = IMIN; sb[64 + tid] = IMAX; sb[96 + tid] = IMIN; sb[128 + tid] = 0;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int row_base = y * FA_REL_WR;
                const int u_pos    = p.pos[(int64_t) (sycl::min(row_base, p.neq1 - 1) + batch * p.neq1) * p.nbp1];
                const int rr       = tid % FA_REL_WR;
                const int row_u    = row_base + rr;
                const bool row_ok  = row_u < p.neq1;
                const int32_t * prow = p.pos + (int64_t) (sycl::min(row_u, p.neq1 - 1) + batch * p.neq1) * p.nbp1;
                const int t0    = x * 32;
                const int t_end = sycl::min(t0 + 32, p.n_tiles);
                const int c     = p.pk_c;
                // 4 pos of the row from cell cc (a multiple of 4); UNALIGNED: IMIN past nek1
                auto ld4 = [&](int cc) {
                    if constexpr (UNALIGNED) {
                        sycl::int4 r;
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            r[k] = cc + k < p.nek1 ? prow[cc + k] : IMIN;
                        }
                        return r;
                    } else {
                        return *(const sycl::int4 *) (prow + cc);
                    }
                };
                using atom = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                              sycl::access::address_space::local_space>;

                // pass 1: the live / non-uniform words and the candidates of B
                for (int t = t0 + tid / FA_REL_WR; t < t_end; t += TPW) {
                    bool live = false;
                    bool nonu = false;
                    int  bmin = IMAX;
                    int  bmax = IMIN;
                    if (row_ok) {
                        const int c0 = t * FA_REL_BC;
#pragma unroll
                        for (int i4 = 0; i4 < FA_REL_BC / 4; ++i4) {
                            const sycl::int4 d4 = ld4(c0 + 4 * i4);
#pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const int i = 4 * i4 + k;
                                const int d = d4[k];
                                if (UNALIGNED && d == IMIN) {
                                    nonu = true;
                                    continue;
                                }
                                live = live || d != p.masked_pos;
                                nonu = nonu || d != u_pos;
                                if (d >= 0 && d < c) {
                                    bmin = sycl::min(bmin, d - rr + i);
                                    bmax = sycl::max(bmax, d - rr + i);
                                }
                            }
                        }
                    }
                    live = sycl::any_of_group(sg, live);
                    nonu = sycl::any_of_group(sg, nonu);
                    bmin = sycl::reduce_over_group(sg, bmin, sycl::minimum<int>());
                    bmax = sycl::reduce_over_group(sg, bmax, sycl::maximum<int>());
                    if (sg.get_local_linear_id() == 0) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            b0(bits[0]), b1(bits[1]);
                        if (live) {
                            b0.fetch_or(1u << (t - t0));
                        }
                        if (nonu) {
                            b1.fetch_or(1u << (t - t0));
                        }
                        atom(sb[t - t0]).fetch_min(bmin);
                        atom(sb[32 + t - t0]).fetch_max(bmax);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // pass 2: check every cell against B (no candidate: every live cell is clamped, B far enough)
                for (int t = t0 + tid / FA_REL_WR; t < t_end; t += TPW) {
                    const int bmin = sb[t - t0];
                    const int B    = bmin == IMAX ? c + 2 * FA_REL_BC : bmin;
                    int  wmin = IMAX;
                    int  dmax = IMIN;
                    bool bad  = false;
                    if (row_ok) {
                        const int c0 = t * FA_REL_BC;
#pragma unroll
                        for (int i4 = 0; i4 < FA_REL_BC / 4; ++i4) {
                            const sycl::int4 d4 = ld4(c0 + 4 * i4);
#pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const int i = 4 * i4 + k;
                                const int d = d4[k];
                                if (UNALIGNED && d == IMIN) {
                                    continue;
                                }
                                const int D = B + rr - i;
                                if (d == p.masked_pos) {
                                    if (D >= 0) {
                                        wmin = sycl::min(wmin, D);
                                    }
                                } else {
                                    bad  = bad || D < 0 || d != sycl::min(D, c);
                                    dmax = sycl::max(dmax, D);
                                }
                            }
                        }
                    }
                    bad  = sycl::any_of_group(sg, bad);
                    wmin = sycl::reduce_over_group(sg, wmin, sycl::minimum<int>());
                    dmax = sycl::reduce_over_group(sg, dmax, sycl::maximum<int>());
                    if (sg.get_local_linear_id() == 0) {
                        atom(sb[64 + t - t0]).fetch_min(wmin);
                        atom(sb[96 + t - t0]).fetch_max(dmax);
                        if (bad) {
                            atom(sb[128 + t - t0]).fetch_or(1);
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                if (tid < 32 && t0 + tid < t_end) {
                    const int  bmin = sb[tid];
                    const int  bmax = sb[32 + tid];
                    const int  wmin = sb[64 + tid];
                    const int  dmax = sb[96 + tid];
                    const bool aff  = (bmin == IMAX || bmin == bmax) && sb[128 + tid] == 0 && dmax < wmin;
                    bw[(int64_t) (batch * p.n_rb + y) * p.n_tiles + t0 + tid] =
                        sycl::int2(bmin == IMAX ? c + 2 * FA_REL_BC : bmin, wmin);
                    if (aff) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>(bits[2]).fetch_or(1u << tid);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (tid < 3) {
                    tl[((int64_t) (batch * p.n_rb + y) * 3 + tid) * p.n_words + x] = bits[tid];
                }
            });
    });
}

// rows-of-the-lane bias of the in-kernel path for an arbitrary table column d (the f32 dot of the f16 table and r)
static inline float fa_rel_bias_col(const fa_rel_params & p, int d, const fa_dpas_int8 & vb, float t) {
    const sycl::uint4 *  tf = (const sycl::uint4 *) (p.pk + p.pk_tf) + (int64_t) d * 2;
    const sycl::uint4    a0 = tf[0];
    const sycl::uint4    a1 = tf[1];
    const sycl::float2   tx = ((const sycl::float2 *) (p.pk + p.pk_tx))[d];
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const sycl::half2 a = sycl::bit_cast<sycl::half2>(i < 4 ? a0[i] : a1[i - 4]);
        const sycl::half2 r = sycl::bit_cast<sycl::half2>((uint32_t) vb[i]);
        acc += static_cast<float>(a[0]) * static_cast<float>(r[0]);
        acc += static_cast<float>(a[1]) * static_cast<float>(r[1]);
    }
    return acc + (tx.x() * t + tx.y());
}

// k ? b : a for a mask k of all zeros or all ones
static inline uint32_t fa_blend(uint32_t k, uint32_t a, uint32_t b) {
    return (a & ~k) | (b & k);
}

// the v of a row of the in-kernel path as the B operand of the bias DPAS (lane = row)
static inline fa_dpas_int8 fa_rel_load_vb(const sycl::uint4 * vp) {
    const sycl::uint4 v0 = vp[0];
    const sycl::uint4 v1 = vp[1];
    return { (int) v0[0], (int) v0[1], (int) v0[2], (int) v0[3], (int) v1[0], (int) v1[1], (int) v1[2], (int) v1[3] };
}

template <bool UNALIGNED, bool FUSED>
static void fattn_rel_prefill_kernel(const fa_rel_params p, const uint32_t * tl, int ne3, dpct::queue_ptr stream) {
    constexpr int SG     = 16;
    constexpr int WG     = 256;
    constexpr int WR     = FA_REL_WR;
    constexpr int BC     = FA_REL_BC;
    constexpr int D      = 128;
    constexpr int RB     = WR / SG;     // row blocks of 16 rows
    constexpr int DSPLIT = WG / SG / RB; // sub-groups per row block
    constexpr int DS     = D / DSPLIT;  // output dims per sub-group
    constexpr int OA_N   = DS / 8;      // O^T accumulators per sub-group
    constexpr int DC     = D / 16;      // 16-dim chunks of Q.K
    constexpr int SC     = BC / DSPLIT; // cells of the scores of one sub-group (one 8-row accumulator)
    constexpr int PSTR   = BC + 8;      // f16 row stride of P
    constexpr int NK     = BC * D / 8 / WG;
    static_assert(SC == 8 && NK * WG * 8 == BC * D && (BC / 2) * (D / 8) == WG && DSPLIT == 4, "tile shape");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::uint4, 1>  sQ(sycl::range<1>(DC * WR * 16 / 8), cgh);  // Q [dc][row][16]
        sycl::local_accessor<sycl::uint4, 1>  sKV(sycl::range<1>(DC * BC * 16 / 8), cgh); // the K, then V^T tile
        sycl::local_accessor<sycl::uint4, 1>  sP(sycl::range<1>(WR * PSTR / 8), cgh);     // P [row][PSTR]
        sycl::local_accessor<sycl::float4, 1> sX(sycl::range<1>(WR), cgh);                // [row][DSPLIT]
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne3, p.n_rb, (size_t) p.neq2 * WG), sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid    = it.get_local_id(2);
                auto      sg     = it.get_sub_group();
                const int lane   = tid % SG;
                const int sgi    = tid / SG;
                const int rb     = sgi % RB;
                const int dh     = sgi / RB;
                const int head_q = it.get_group(2);
                const int head_k = head_q / p.gqa;
                const int batch  = it.get_group(0);
                // causal prefill: the last row blocks see the most cells; dispatch them first
                const int out_r  = p.n_rb - 1 - (int) it.get_group(1);
                const int row0   = out_r * WR;
                sycl::half * sQh = (sycl::half *) &sQ[0];
                float *      sXf = (float *) &sX[0];

                // Q * scale -> f16 [d / 16][row][d % 16]
#pragma unroll
                for (int i = 0; i < WR * D / 4 / WG; ++i) {
                    const int f   = i * WG + tid;
                    const int q   = f / (D / 4);
                    const int d   = (f % (D / 4)) * 4;
                    const int row = sycl::min(row0 + q, p.neq1 - 1);
                    const sycl::float4 v = *(const sycl::float4 *) (p.Q + batch * p.nbq3 + (int64_t) row * p.nbq1 +
                                                                    (int64_t) head_q * p.nbq2 + d * sizeof(float)) * p.scale;
                    sycl::half * qd = sQh + ((d / 16) * WR + q) * 16 + d % 16;
                    qd[0] = sycl::half(v.x());
                    qd[1] = sycl::half(v.y());
                    qd[2] = sycl::half(v.z());
                    qd[3] = sycl::half(v.w());
                }
                // (the barrier before the first K tile store publishes Q)

                const int u_pos = p.pos[(int64_t) (sycl::min(row0, p.neq1 - 1) + batch * p.neq1) * p.nbp1];
                constexpr int NWT = FUSED ? 3 : 2;
                const uint32_t * tl_live = tl + (int64_t) (batch * p.n_rb + out_r) * NWT * p.n_words;
                const uint32_t * tl_nonu = tl_live + p.n_words;
                const uint32_t * tl_aff  = tl_live + 2 * p.n_words;

                // the row of the lane (rows past neq1 compute a copy of the last one and are not stored)
                const int row_u = row0 + rb * 16 + lane;
                const int row_c = sycl::min(row_u, p.neq1 - 1);
                const int32_t *    prow = p.pos + (int64_t) (row_c + batch * p.neq1) * p.nbp1;
                const sycl::half * brow = FUSED ? nullptr : p.bias + (int64_t) row_c * p.nbb1 +
                                          (int64_t) (head_q % p.neb2) * p.nbb2 + (int64_t) batch * p.nbb3;

                // FUSED: v of the row (the B operand of the bias DPAS) is reloaded where it is used (keeping it in
                // registers through the tile loop spills); the t term is x16 t + x17 when it is the same for every
                // column, else a second DPAS of the t_hi, t_lo, 1 column
                const sycl::uint4 * vp = nullptr;
                float t_f = 0.0f;
                float kc  = 0.0f;
                bool  xc  = true;
                float b_uni;
                if constexpr (FUSED) {
                    vp = (const sycl::uint4 *) (p.pk + p.pk_v) +
                         ((int64_t) (batch * p.neq1 + row_c) * p.pk_nhead + head_q % p.pk_nhead) * 3;
                    const sycl::uint4 v2 = vp[2];
                    t_f   = sycl::bit_cast<float>(v2[1]);
                    const int *   hdr = (const int *) (p.pk + p.pk_hdr);
                    const float * hdf = (const float *) hdr;
                    xc = hdr[0] != 0;
                    kc = xc ? hdf[1] * t_f + hdf[2] : 0.0f;
                    b_uni = fa_rel_bias_col(p, sycl::clamp(u_pos, 0, p.pk_ncols - 1), fa_rel_load_vb(vp), t_f);
                } else {
                    b_uni = static_cast<float>(brow[u_pos]);
                }

                const char * Kh = p.K + batch * p.nbk3 + (int64_t) head_k * p.nbk2;
                const char * Vh = p.V + batch * p.nbv3 + (int64_t) head_k * p.nbv2;

                fa_dpas_float8 oa[OA_N];
#pragma unroll
                for (int j = 0; j < OA_N; ++j) {
                    oa[j] = 0.0f;
                }
                constexpr float NEG = -FLT_MAX / 2;
                float Mf = NEG;
                float Lf = 0.0f;

                int      wi = -1;
                uint32_t wl = 0;
                uint32_t wn = 0;
                uint32_t wa = 0;
                const sycl::int2 * bwp = FUSED ? p.bw + (int64_t) (batch * p.n_rb + out_r) * p.n_tiles : nullptr;
                for (int tile = 0; tile < p.n_tiles; ++tile) {
                    if (tile / 32 != wi) {
                        wi = tile / 32;
                        wl = tl_live[wi];
                        wn = tl_nonu[wi];
                        if constexpr (FUSED) {
                            wa = tl_aff[wi];
                        }
                    }
                    // a tile whose cells are all masked adds nothing: skip it (uniform over the work-group)
                    if (((wl >> (tile % 32)) & 1u) == 0) {
                        continue;
                    }
                    const bool uni = ((wn >> (tile % 32)) & 1u) == 0;
                    const int  c0  = tile * BC;

                    // K: thread f gathers the 4 dwords of lane f % 16 of the A matrix f / 16 = (8-cell block pc, chunk dc)
                    sycl::uint4 kr[NK];
#pragma unroll
                    for (int i = 0; i < NK; ++i) {
                        const int f  = i * WG + tid;
                        const int l  = f % 16;
                        const int pc = (f / 16) / DC;
                        const int dc = (f / 16) % DC;
#pragma unroll
                        for (int g = 0; g < 4; ++g) {
                            const int kv = c0 + pc * 8 + 2 * g + l / 8;
                            kr[i][g] = (UNALIGNED && kv >= p.nek1) ? 0u :
                                *(const uint32_t *) (Kh + (int64_t) kv * p.nbk1 + (dc * 8 + l % 8) * sizeof(uint32_t));
                        }
                    }
                    // V: thread f loads the cells c0 + 2 (f % 16) + {0, 1} of the 8 dims 8 (f / 16) ..
                    const int   vkv = c0 + 2 * (tid % (BC / 2));
                    const int   vd8 = tid / (BC / 2);
                    sycl::uint4 va(0u);
                    sycl::uint4 vb(0u);
                    if (!UNALIGNED || vkv < p.nek1) {
                        va = *(const sycl::uint4 *) (Vh + (int64_t) vkv * p.nbv1 + vd8 * 16);
                    }
                    if (!UNALIGNED || vkv + 1 < p.nek1) {
                        vb = *(const sycl::uint4 *) (Vh + (int64_t) (vkv + 1) * p.nbv1 + vd8 * 16);
                    }

                    // the bias of the SC cells of the sub-group in the row of the lane (masked cells: -inf)
                    const int cs0 = c0 + dh * SC;
                    float     bl[SC];
                    if (uni) {
#pragma unroll
                        for (int e = 0; e < SC; ++e) {
                            bl[e] = b_uni;
                        }
                    } else if (FUSED && ((wa >> (tile % 32)) & 1u) != 0) {
                        // the cell e of the sub-group in the row of the lane is at distance D = Bs + lane - e: the
                        // bias of the 24 distances w0 .. w0 + 23 for all 16 rows by three DPAS (lane = row), then
                        // lane l takes the distances w0 + l .. w0 + l + 7 by a shift of l
                        const sycl::int2 bwv = bwp[tile];
                        const int Bs = bwv.x() + rb * 16 - dh * SC;
                        const int w0 = Bs - 7;
                        const fa_dpas_int8 vB = fa_rel_load_vb(vp);
                        float win[24];
#pragma unroll
                        for (int j = 0; j < 3; ++j) {
                            const int d = sycl::clamp(w0 + 8 * j + lane / 8, -8, p.pk_c) + 8;
                            const sycl::uint4 a = ((const sycl::uint4 *) (p.pk + p.pk_t2))[(int64_t) d * 8 + lane % 8];
                            fa_dpas_float8     acc = 0.0f;
                            const fa_dpas_int4 aa  = { (int) a[0], (int) a[1], (int) a[2], (int) a[3] };
                            fa_dpas_f16_a4(aa, vB, acc);
                            if (!xc) {
                                const sycl::uint4 ax = lane % 8 < 2 ?
                                    ((const sycl::uint4 *) (p.pk + p.pk_x2))[(int64_t) d * 2 + lane % 8] : sycl::uint4(0u);
                                const fa_dpas_int4 axa = { (int) ax[0], (int) ax[1], (int) ax[2], (int) ax[3] };
                                fa_dpas_int8 tB = 0;
                                tB[0] = (int) vp[2][0];
                                tB[1] = 0x3C00; // half2(1, 0)
                                fa_dpas_f16_a4(axa, tB, acc);
                            }
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                win[8 * j + m] = acc[m];
                            }
                        }
                        // the shift by the lane as bit blends of values: a select of two array elements is
                        // folded into a load at a select of two addresses, which puts the window in memory
                        const uint32_t k8 = 0u - (uint32_t) ((lane >> 3) & 1);
                        const uint32_t k4 = 0u - (uint32_t) ((lane >> 2) & 1);
                        const uint32_t k2 = 0u - (uint32_t) ((lane >> 1) & 1);
                        const uint32_t k1 = 0u - (uint32_t) (lane & 1);
                        uint32_t s16[16];
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            s16[i] = fa_blend(k8, sycl::bit_cast<uint32_t>(win[i]), sycl::bit_cast<uint32_t>(win[i + 8]));
                        }
                        uint32_t s12[12];
#pragma unroll
                        for (int i = 0; i < 12; ++i) {
                            s12[i] = fa_blend(k4, s16[i], s16[i + 4]);
                        }
                        uint32_t s10[10];
#pragma unroll
                        for (int i = 0; i < 10; ++i) {
                            s10[i] = fa_blend(k2, s12[i], s12[i + 2]);
                        }
#pragma unroll
                        for (int e = 0; e < SC; ++e) {
                            const float v = sycl::bit_cast<float>(fa_blend(k1, s10[SC - 1 - e], s10[SC - e]));
                            const int   D = Bs + lane - e;
                            const bool  m = D < 0 || D >= bwv.y() || (UNALIGNED && cs0 + e >= p.nek1);
                            bl[e] = m ? -INFINITY : v + kc;
                        }
                    } else if constexpr (FUSED) {
                        // an irregular tile (several sequences in the block, a wrapped cache): per cell
                        const fa_dpas_int8 vB = fa_rel_load_vb(vp);
#pragma unroll
                        for (int e = 0; e < SC; ++e) {
                            const int c = cs0 + e;
                            const int d = (!UNALIGNED || c < p.nek1) ? prow[c] : p.masked_pos;
                            bl[e] = d == p.masked_pos ? -INFINITY : fa_rel_bias_col(p, sycl::clamp(d, 0, p.pk_ncols - 1), vB, t_f);
                        }
                    } else if constexpr (UNALIGNED) {
#pragma unroll
                        for (int e = 0; e < SC; ++e) {
                            const int c = cs0 + e;
                            bl[e] = c < p.nek1 ? static_cast<float>(brow[prow[c]]) : -INFINITY;
                        }
                    } else {
#pragma unroll
                        for (int e4 = 0; e4 < SC / 4; ++e4) {
                            const sycl::int4 d = *(const sycl::int4 *) (prow + cs0 + 4 * e4);
#pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                bl[4 * e4 + k] = static_cast<float>(brow[d[k]]);
                            }
                        }
                    }

                    // the V^T and P of the previous tile are no longer read
                    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                    for (int i = 0; i < NK; ++i) {
                        sKV[i * WG + tid] = kr[i];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // S^T of the SC cells dh SC .. of the tile and the 16 rows of the row block: K Q^T + bias
                    fa_dpas_float8 sa = 0.0f;
#pragma unroll
                    for (int dc = 0; dc < DC; ++dc) {
                        const sycl::uint4  q0 = sQ[(dc * WR + rb * 16 + lane) * 2];
                        const sycl::uint4  q1 = sQ[(dc * WR + rb * 16 + lane) * 2 + 1];
                        const fa_dpas_int8 qb = { (int) q0[0], (int) q0[1], (int) q0[2], (int) q0[3],
                                                  (int) q1[0], (int) q1[1], (int) q1[2], (int) q1[3] };
                        const sycl::uint4  kw = sKV[(dh * DC + dc) * 16 + lane];
                        const fa_dpas_int4 ka = { (int) kw[0], (int) kw[1], (int) kw[2], (int) kw[3] };
                        fa_dpas_f16_a4(ka, qb, sa);
                    }
                    float mloc = NEG;
#pragma unroll
                    for (int e = 0; e < SC; ++e) {
                        sa[e] += bl[e];
                        mloc = sycl::fmax(mloc, sa[e]);
                    }
                    sXf[(rb * 16 + lane) * DSPLIT + dh] = mloc;
                    it.barrier(sycl::access::fence_space::local_space);

                    // the K tile is no longer read: V^T into its space. The A matrix (vc, vd8) of V^T: lane l, dword
                    // g = (V[16 vc + 2 (l % 8) + {0, 1}][8 vd8 + 2 g + l / 8]); the cell pair 2 pr, 2 pr + 1 is lane
                    // pr % 8 (the even dims) and lane 8 + pr % 8 (the odd dims)
                    {
                        const int pr  = tid % (BC / 2);
                        const int vc  = pr / 8;
                        const int blk = (vc * (D / 8) + vd8) * 16;
                        sKV[blk + pr % 8]     = (va & 0xFFFFu) | (vb << 16);
                        sKV[blk + 8 + pr % 8] = (va >> 16) | (vb & 0xFFFF0000u);
                    }

                    // online softmax of the row of the lane: the max over the whole tile (the same in the DSPLIT
                    // sub-groups of the row block), the partial sum over the cells of the sub-group
                    const sycl::float4 mx   = sX[rb * 16 + lane];
                    const float        Mnew = sycl::fmax(Mf, sycl::fmax(sycl::fmax(mx.x(), mx.y()), sycl::fmax(mx.z(), mx.w())));
                    const float        eM   = sycl::native::exp(Mf - Mnew);
                    Lf *= eM;
                    uint32_t pk[SC / 2];
#pragma unroll
                    for (int e = 0; e < SC; e += 2) {
                        const float p0 = sycl::native::exp(sa[e] - Mnew);
                        const float p1 = sycl::native::exp(sa[e + 1] - Mnew);
                        Lf += p0;
                        Lf += p1;
                        pk[e / 2] = (uint32_t) sycl::bit_cast<uint16_t>(sycl::half(p0)) |
                                    ((uint32_t) sycl::bit_cast<uint16_t>(sycl::half(p1)) << 16);
                    }
                    sP[(rb * 16 + lane) * (PSTR / 8) + dh] = sycl::uint4(pk[0], pk[1], pk[2], pk[3]);

                    // rescale O (eM is exactly 1 where the max did not move)
                    if (sycl::any_of_group(sg, Mnew > Mf)) {
#pragma unroll
                        for (int j = 0; j < OA_N; ++j) {
                            oa[j] *= eM;
                        }
                    }
                    Mf = Mnew;
                    it.barrier(sycl::access::fence_space::local_space);

                    // O^T += V^T P^T: P^T as B matrices, dword i of lane l = P[row l][16 c + 2 i + {0, 1}]
#pragma unroll
                    for (int c = 0; c < BC / 16; ++c) {
                        const sycl::uint4  p0 = sP[(rb * 16 + lane) * (PSTR / 8) + 2 * c];
                        const sycl::uint4  p1 = sP[(rb * 16 + lane) * (PSTR / 8) + 2 * c + 1];
                        const fa_dpas_int8 pb = { (int) p0[0], (int) p0[1], (int) p0[2], (int) p0[3],
                                                  (int) p1[0], (int) p1[1], (int) p1[2], (int) p1[3] };
#pragma unroll
                        for (int j = 0; j < OA_N; ++j) {
                            const sycl::uint4  vw = sKV[(c * (D / 8) + dh * OA_N + j) * 16 + lane];
                            const fa_dpas_int4 vA = { (int) vw[0], (int) vw[1], (int) vw[2], (int) vw[3] };
                            fa_dpas_f16_a4(vA, pb, oa[j]);
                        }
                    }
                }

                // the row sum: the partial sums of the DSPLIT sub-groups of the row block, in a fixed order
                it.barrier(sycl::access::fence_space::local_space);
                sXf[(rb * 16 + lane) * DSPLIT + dh] = Lf;
                it.barrier(sycl::access::fence_space::local_space);
                {
                    const sycl::float4 ls = sX[rb * 16 + lane];
                    Lf = ((ls.x() + ls.y()) + ls.z()) + ls.w();
                }

                // the sink is one more cell with logit sinks[head] and no value (as the CPU adds it)
                float rcp;
                if (p.sinks) {
                    const float sink = p.sinks[head_q];
                    if (Mf < sink) {
                        const float e = sycl::exp(Mf - sink);
                        rcp = e / (1.0f + Lf * e);
                    } else {
                        rcp = 1.0f / (Lf + sycl::exp(sink - Mf));
                    }
                } else {
                    rcp = Lf == 0.0f ? 0.0f : 1.0f / Lf;
                }

                if (row_u < p.neq1) {
                    float * o = p.dst + ((int64_t) (batch * p.neq1 + row_u) * p.neq2 + head_q) * D + dh * DS;
#pragma unroll
                    for (int j = 0; j < OA_N; ++j) {
                        *(sycl::float4 *) (o + 8 * j)     = sycl::float4(oa[j][0], oa[j][1], oa[j][2], oa[j][3]) * rcp;
                        *(sycl::float4 *) (o + 8 * j + 4) = sycl::float4(oa[j][4], oa[j][5], oa[j][6], oa[j][7]) * rcp;
                    }
                }
            });
    });
}

static void fattn_rel_prefill(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool fused) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * pos  = dst->src[7];
    const ggml_tensor * bias = dst->src[6];

    fa_rel_params p;
    p.Q     = (const char *) Q->data;
    p.K     = (const char *) K->data;
    p.V     = (const char *) V->data;
    p.pos   = (const int32_t *) pos->data;
    p.bias  = (const sycl::half *) bias->data;
    p.sinks = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    p.dst   = (float *) dst->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.neq1    = Q->ne[1];
    p.neq2    = Q->ne[2];
    p.nek1    = K->ne[1];
    p.gqa     = Q->ne[2] / K->ne[2];
    p.n_rb    = (p.neq1 + FA_REL_WR - 1) / FA_REL_WR;
    p.n_tiles = (p.nek1 + FA_REL_BC - 1) / FA_REL_BC;
    p.n_words = (p.n_tiles + 31) / 32;
    p.nbq1 = Q->nb[1];
    p.nbq2 = Q->nb[2];
    p.nbq3 = Q->nb[3];
    p.nbk1 = K->nb[1];
    p.nbk2 = K->nb[2];
    p.nbk3 = K->nb[3];
    p.nbv1 = V->nb[1];
    p.nbv2 = V->nb[2];
    p.nbv3 = V->nb[3];
    p.nbp1 = pos->nb[1] / (int64_t) sizeof(int32_t);
    p.nbb1 = bias->nb[1] / (int64_t) sizeof(sycl::half);
    p.nbb2 = bias->nb[2] / (int64_t) sizeof(sycl::half);
    p.nbb3 = bias->nb[3] / (int64_t) sizeof(sycl::half);
    p.neb2 = bias->ne[2];
    p.masked_pos = ggml_get_op_params_i32(dst, 7);
    p.pk = nullptr;
    p.bw = nullptr;

    const int ne3 = Q->ne[3];
    dpct::queue_ptr stream = ctx.stream();
    if (fused) {
        ggml_sycl_rel_pack_layout L;
        const bool ok = ggml_sycl_rel_bias_fa_pack_layout(bias, p.masked_pos, L); // packed at the REL_BIAS node
        GGML_ASSERT(ok);
        p.pk       = (const char *) bias->data;
        p.pk_hdr   = L.hdr;
        p.pk_v     = L.v;
        p.pk_t2    = L.t2;
        p.pk_x2    = L.x2;
        p.pk_tf    = L.tf;
        p.pk_tx    = L.tx;
        p.pk_c     = L.c;
        p.pk_ncols = L.n_cols;
        p.pk_nhead = bias->src[0]->ne[1];
        ggml_sycl_pool_alloc<uint32_t>   tl(ctx.pool(), (size_t) ne3 * p.n_rb * 3 * p.n_words);
        ggml_sycl_pool_alloc<sycl::int2> bw(ctx.pool(), (size_t) ne3 * p.n_rb * p.n_tiles);
        p.bw = bw.get();
        if (p.nek1 % FA_REL_BC != 0) {
            fattn_rel_tiles_aff<true>(p, tl.get(), bw.get(), ne3, stream);
        } else {
            fattn_rel_tiles_aff<false>(p, tl.get(), bw.get(), ne3, stream);
        }
        auto old_kernel = [&](const fa_rel_params & q) {
            if (q.nek1 % FA_REL_BC != 0) {
                fattn_rel_prefill_kernel<true, true>(q, tl.get(), ne3, stream);
            } else {
                fattn_rel_prefill_kernel<false, true>(q, tl.get(), ne3, stream);
            }
        };
        // the Xe2-style ESIMD kernel (fattn-rel-xe.cpp) on the same tile flags; GGML_SYCL_FA_XE=0 keeps the SIMT one
        ggml_sycl_fa_xe_params xp;
        xp.Q = p.Q; xp.K = p.K; xp.V = p.V; xp.pos = p.pos; xp.sinks = p.sinks; xp.dst = p.dst; xp.scale = p.scale;
        xp.neq1 = p.neq1; xp.neq2 = p.neq2; xp.nek1 = p.nek1; xp.gqa = p.gqa;
        xp.n_rb64 = p.n_rb; xp.n_tiles = p.n_tiles; xp.n_words = p.n_words;
        xp.nbq1 = p.nbq1; xp.nbq2 = p.nbq2; xp.nbq3 = p.nbq3;
        xp.nbk1 = p.nbk1; xp.nbk2 = p.nbk2; xp.nbk3 = p.nbk3;
        xp.nbv1 = p.nbv1; xp.nbv2 = p.nbv2; xp.nbv3 = p.nbv3;
        xp.nbp1 = p.nbp1; xp.masked_pos = p.masked_pos;
        xp.pk = p.pk; xp.pk_hdr = p.pk_hdr; xp.pk_v = p.pk_v; xp.pk_tf = p.pk_tf; xp.pk_tx = p.pk_tx;
        xp.pk_c = p.pk_c; xp.pk_ncols = p.pk_ncols; xp.pk_nhead = p.pk_nhead;
        xp.tl = tl.get(); xp.bw = bw.get();
        static_assert(FA_REL_WR == 64 && FA_REL_BC == 32, "fattn-rel-xe takes 64-row blocks of 32-cell tiles");
        if (!ggml_sycl_fa_rel_xe_ok(xp)) {
            old_kernel(p);
            return;
        }
        // GGML_SYCL_FA_XE_CHECK=1 (debug, graphs off): run both kernels and print the largest difference per call
        static const bool check = getenv("GGML_SYCL_FA_XE_CHECK") && atoi(getenv("GGML_SYCL_FA_XE_CHECK")) != 0;
        if (check) {
            const size_t n = (size_t) ggml_nelements(dst);
            ggml_sycl_pool_alloc<float> ref(ctx.pool(), n);
            fa_rel_params q = p;
            q.dst = ref.get();
            old_kernel(q);
            ggml_sycl_fa_rel_xe(xp, ne3, stream);
            std::vector<float> a(n), b(n);
            stream->memcpy(a.data(), ref.get(), n * sizeof(float));
            stream->memcpy(b.data(), p.dst, n * sizeof(float)).wait();
            double md = 0.0, mr = 0.0;
            size_t bad = 0, imax = 0;
            for (size_t i = 0; i < n; ++i) {
                const double d = fabs((double) a[i] - (double) b[i]);
                if (!(d <= 1e30)) {
                    bad++;
                    continue;
                }
                if (d > md) {
                    md = d;
                    imax = i;
                }
                mr = std::max(mr, fabs((double) a[i]));
            }
            static int calls = 0;
            fprintf(stderr, "FA_XE_CHECK %d: q %d k %d s %d max|diff| %.3g at %zu (row %zu head %zu) max|ref| %.3g nonfinite %zu\n",
                    calls++, p.neq1, p.nek1, ne3, md, imax, imax / 128 / p.neq2, (imax / 128) % p.neq2, mr, bad);
            return;
        }
        ggml_sycl_fa_rel_xe(xp, ne3, stream);
        return;
    }
    ggml_sycl_pool_alloc<uint32_t> tl(ctx.pool(), (size_t) ne3 * p.n_rb * 2 * p.n_words);
    if (p.nek1 % FA_REL_BC != 0) {
        fattn_rel_tiles<true>(p, tl.get(), ne3, stream);
        fattn_rel_prefill_kernel<true, false>(p, tl.get(), ne3, stream);
    } else {
        fattn_rel_tiles<false>(p, tl.get(), ne3, stream);
        fattn_rel_prefill_kernel<false, false>(p, tl.get(), ne3, stream);
    }
}

// the prefill kernel can compute the bias of the REL_BIAS node rb itself (rel-bias.hpp): fattn_dpas_f16_dispatch
// takes fattn_rel_prefill for fa, and rb is the table form with n_rel 16
bool ggml_sycl_flash_attn_ext_dpas_rel_inkernel_ok(const ggml_tensor * fa, const ggml_tensor * rb) {
    if (fa->op != GGML_OP_FLASH_ATTN_EXT || fa->src[7] == nullptr || fa->src[6] != rb || fa->src[1]->type != GGML_TYPE_F16 ||
        ggml_get_op_params_i32(fa, 6) != 0 || fa->src[0]->ne[1] <= FA_DPAS_MAX_Q || !fattn_dpas_f16_supported(fa)) {
        return false;
    }
    // every row of the op reads its own v: the bias rows are the query rows of fa
    const ggml_tensor * r = rb->src[0];
    if (rb->ne[1] != fa->src[0]->ne[1] || rb->ne[3] != fa->src[0]->ne[3] || r->ne[2] != fa->src[0]->ne[1] * fa->src[0]->ne[3] ||
        fa->src[0]->ne[2] % r->ne[1] != 0) {
        return false;
    }
    ggml_sycl_rel_pack_layout L;
    return ggml_sycl_rel_bias_fa_pack_layout(rb, ggml_get_op_params_i32(fa, 7), L);
}

// ---------------------------------------------------------------------------------------------------------------------
// decode: up to FA_DEC_MAX_Q query tokens, f16 K/V, head size 128, a per-head f16 mask, G query heads per KV head.
// One work-group = one query token x one KV head x one slice of the keys, FA_DEC_NSG sub-groups of 16 lanes. A lane
// holds 8 of the 128 dims (2 (l + 16 i) and 2 (l + 16 i) + 1) of Q * scale, of the K and V rows and of O, so one row
// is one sub-group block read. A sub-group takes FA_DEC_KP keys per pass: it reads their mask entries, K rows and V
// rows together (one wait for memory per pass), sums the lane parts of the Q.K products with a transposing butterfly
// so that lane l gets key l % FA_DEC_KP, then runs the online softmax and P.V with P broadcast from the key lanes.
// Scores, P and O are f32. Every sub-group keeps its own running max, sum and O; the work-group merges them at the end
// and writes the output, or the (max, sum, O) of its slice for fattn_dpas_combine. The slicing depends only on the key
// count, so a row is computed the same in every batch.
static constexpr int FA_DEC_NSG       = 8;
static constexpr int FA_DEC_WG        = 16 * FA_DEC_NSG;
static constexpr int FA_DEC_KP        = 8; // keys per sub-group pass; 16 needs more registers than the 256 there are
static constexpr int FA_DEC_MAX_Q     = 4;
static constexpr int FA_DEC_MAX_SPLIT = 16;
static constexpr int FA_DEC_CHUNK     = 0;   // fixed keys per slice (GGML_SYCL_FA_DEC_CHUNK); 0: by the key count

struct fa_dec_params {
    const char *   Q;
    const char *   K;
    const char *   V;
    const char *   mask;
    const float *  sinks;
    float *        dst;
    float *        parts;
    sycl::float2 * meta;
    float          scale;
    int            ne01, ne02, ne11, ne32, ne33, nsplit, chunk;
    int64_t        nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb32, nb33;
    // REL: the bias of the packed REL_BIAS node (rel-bias.hpp) in place of the mask
    const float *   rv;
    const int32_t * rpos;
    const float *   rtab;
    int64_t         rnbc1;
    int             rnvp, rnkv, rnhead;
};

// two f16 of a K or V row as floats
static inline sycl::float2 fa_dec_h2(uint32_t u) {
    const sycl::half2 h = sycl::bit_cast<sycl::half2>(u);
    return sycl::float2(static_cast<float>(h[0]), static_cast<float>(h[1]));
}

// v[t] holds lane l's part of the dot product with key t of KP; afterwards v[0] of lane l is the whole dot product
// with key l % KP: sums across the lane groups of KP lanes, then a transposing butterfly (15 shuffles for 16 keys)
template <int KP> static inline void fa_dec_transpose_sum(const sycl::sub_group & sg, int lane, float (&v)[KP]) {
#pragma unroll
    for (int h = 8; h >= KP; h /= 2) {
#pragma unroll
        for (int t = 0; t < KP; ++t) {
            v[t] += sycl::permute_group_by_xor(sg, v[t], h);
        }
    }
#pragma unroll
    for (int h = KP / 2; h >= 1; h /= 2) {
        const bool hi = (lane & h) != 0;
#pragma unroll
        for (int t = 0; t < h; ++t) {
            const float send = hi ? v[t] : v[t + h];
            const float keep = hi ? v[t + h] : v[t];
            v[t] = keep + sycl::permute_group_by_xor(sg, send, h);
        }
    }
}

template <int G, bool REL>
static void fattn_dec_kernel(const fa_dec_params p, const int nkvh, const int ne03, dpct::queue_ptr stream) {
    constexpr int D = 128;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sm(sycl::range<1>(FA_DEC_NSG * G), cgh);
        sycl::local_accessor<float, 1> sl(sycl::range<1>(FA_DEC_NSG * G), cgh);
        sycl::local_accessor<float, 1> so(sycl::range<1>(FA_DEC_NSG * G * D), cgh);
        // the KV heads of one key slice are neighbours in the dispatch order: they read the same K/V cells
        const sycl::nd_range<3> ndr(sycl::range<3>((size_t) ne03 * p.ne01, p.nsplit, (size_t) nkvh * FA_DEC_WG),
                                    sycl::range<3>(1, 1, FA_DEC_WG));
        auto kern = [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int tid   = it.get_local_id(2);
            const int split = it.get_group(1);
            const int kvh   = it.get_group(2);
            const int seq   = it.get_group(0) / p.ne01;
            const int c     = it.get_group(0) % p.ne01;
            auto      sg    = it.get_sub_group();
            const int w     = tid / 16;
            const int lane  = tid % 16;

            float q[G][8];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float * qr = (const float *) (p.Q + seq * p.nb03 + (int64_t) c * p.nb01 + (int64_t) (kvh * G + g) * p.nb02);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const sycl::float2 qq = *(const sycl::float2 *) (qr + 2 * (lane + 16 * i));
                    q[g][2 * i]     = qq[0] * p.scale;
                    q[g][2 * i + 1] = qq[1] * p.scale;
                }
            }

            const int    k_begin = split * p.chunk;
            const int    k_end   = sycl::min(k_begin + p.chunk, p.ne11);
            const char * Kh      = p.K + seq * p.nb13 + kvh * p.nb12;
            const char * Vh      = p.V + seq * p.nb23 + kvh * p.nb22;
            const sycl::half * mrow[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                mrow[g] = p.mask ? (const sycl::half *) (p.mask + (seq % p.ne33) * p.nb33 + (int64_t) c * p.nb31 +
                                                         (int64_t) ((kvh * G + g) % p.ne32) * p.nb32) :
                                   nullptr;
            }

            // REL: the v of the G heads of the token (uniform over the work-group) and its pos row
            constexpr int RNV = 18;
            [[maybe_unused]] float           rvv[G][RNV];
            [[maybe_unused]] const int32_t * rprow = nullptr;
            if constexpr (REL) {
                const int64_t ti = (int64_t) seq * p.ne01 + c;
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float * vv = p.rv + (ti * p.rnhead + kvh * G + g) * p.rnvp;
#pragma unroll
                    for (int k = 0; k < RNV; ++k) {
                        rvv[g][k] = vv[k];
                    }
                }
                rprow = p.rpos + ti * p.rnkv;
            }

            float m[G];
            float l[G];
            float o[G][8];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                m[g] = FA_DPAS_MINF;
                l[g] = 0.0f;
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    o[g][d] = 0.0f;
                }
            }

            for (int j0 = k_begin + w * FA_DEC_KP; j0 < k_end; j0 += FA_DEC_NSG * FA_DEC_KP) {
                const int  j  = j0 + lane % FA_DEC_KP;
                const bool jv = j < k_end;
                float      s[G];
                bool       live = false;
                if constexpr (REL) {
                    // the f16 bias of rel_bias_kernel: the same dot of the table column and v, rounded to f16
                    float         cc[RNV];
                    const float * col = p.rtab + (int64_t) rprow[jv ? j : j0] * p.rnbc1;
#pragma unroll
                    for (int k = 0; k < RNV; ++k) {
                        cc[k] = col[k];
                    }
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        s[g] = jv ? static_cast<float>(sycl::half(ggml_sycl_rel_bias_dot<RNV>(cc, rvv[g]))) : -INFINITY;
                        live = live || s[g] != -INFINITY;
                    }
                } else {
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        s[g] = jv ? (mrow[g] ? static_cast<float>(mrow[g][j]) : 0.0f) : -INFINITY;
                        live = live || s[g] != -INFINITY;
                    }
                }
                // FA_DEC_KP keys masked in every row add exact zeros
                if (!sycl::any_of_group(sg, live)) {
                    continue;
                }

                // one sub-group block read per row; mask, K and V of the pass are in flight together
                namespace syclex = sycl::ext::oneapi::experimental;
                constexpr auto props = syclex::properties{ syclex::data_placement_striped, syclex::full_group };
                sycl::vec<uint32_t, 4> kw[FA_DEC_KP];
                sycl::vec<uint32_t, 4> vw[FA_DEC_KP];
#pragma unroll
                for (int t = 0; t < FA_DEC_KP; ++t) {
                    const int jj = sycl::min(j0 + t, k_end - 1);
                    auto kr = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        (const uint32_t *) (Kh + (int64_t) jj * p.nb11));
                    auto vr = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        (const uint32_t *) (Vh + (int64_t) jj * p.nb21));
                    syclex::group_load(sg, kr, kw[t], props);
                    syclex::group_load(sg, vr, vw[t], props);
                }

                float pr[G];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    float dot[FA_DEC_KP];
#pragma unroll
                    for (int t = 0; t < FA_DEC_KP; ++t) {
                        float a = 0.0f;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const sycl::float2 kf = fa_dec_h2(kw[t][i]);
                            a += q[g][2 * i] * kf[0] + q[g][2 * i + 1] * kf[1];
                        }
                        dot[t] = a;
                    }
                    fa_dec_transpose_sum<FA_DEC_KP>(sg, lane, dot);

                    const float sv = s[g] == -INFINITY ? -INFINITY : dot[0] + s[g];
                    const float mt = sycl::reduce_over_group(sg, sv, sycl::maximum<float>());
                    const float mn = sycl::fmax(m[g], mt);
                    const float a  = sycl::native::exp(m[g] - mn);
                    pr[g]          = sv == -INFINITY ? 0.0f : sycl::native::exp(sv - mn);
                    l[g]           = l[g] * a + (lane < FA_DEC_KP ? pr[g] : 0.0f); // lanes past FA_DEC_KP repeat keys
                    m[g]           = mn;
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        o[g][d] *= a;
                    }
                }

#pragma unroll
                for (int t = 0; t < FA_DEC_KP; ++t) {
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const float pk = sycl::select_from_group(sg, pr[g], t);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const sycl::float2 vf = fa_dec_h2(vw[t][i]);
                            o[g][2 * i] += pk * vf[0];
                            o[g][2 * i + 1] += pk * vf[1];
                        }
                    }
                }
            }

#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float lt = sycl::reduce_over_group(sg, l[g], sycl::plus<float>());
                if (lane == 0) {
                    sm[w * G + g] = m[g];
                    sl[w * G + g] = lt;
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *(sycl::float2 *) &so[(w * G + g) * D + 2 * (lane + 16 * i)] = sycl::float2(o[g][2 * i], o[g][2 * i + 1]);
                }
            }
            it.barrier(sycl::access::fence_space::local_space);

            // work-item tid = head dim tid; the sink is one more key with logit sinks[head] and no value, the first
            // key slice owns it
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int  h    = kvh * G + g;
                const bool sink = p.sinks && split == 0;
                float      M    = sink ? p.sinks[h] : FA_DPAS_MINF;
#pragma unroll
                for (int ws = 0; ws < FA_DEC_NSG; ++ws) {
                    M = sycl::fmax(M, sm[ws * G + g]);
                }
                float L = sink ? sycl::native::exp(p.sinks[h] - M) : 0.0f;
                float O = 0.0f;
#pragma unroll
                for (int ws = 0; ws < FA_DEC_NSG; ++ws) {
                    const float a = sycl::native::exp(sm[ws * G + g] - M);
                    L += a * sl[ws * G + g];
                    O += a * so[(ws * G + g) * D + tid];
                }
                const int64_t jdu = ((int64_t) seq * p.ne01 + c) * p.ne02 + h;
                if (p.nsplit == 1) {
                    p.dst[jdu * D + tid] = L > 0.0f ? O / L : 0.0f;
                } else {
                    p.parts[(jdu * p.nsplit + split) * D + tid] = O;
                    if (tid == 0) {
                        p.meta[jdu * p.nsplit + split] = sycl::float2(M, L);
                    }
                }
            }
        };
        // the 128-register file spills the K and V rows of a pass
        cgh.parallel_for(ndr, sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> }, kern);
    });
}

template <int G> static void fattn_dec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool rel) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    fa_dec_params p;
    p.Q     = (const char *) Q->data;
    p.K     = (const char *) K->data;
    p.V     = (const char *) V->data;
    p.mask  = mask ? (const char *) mask->data : nullptr;
    p.sinks = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    p.dst   = (float *) dst->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.ne01 = Q->ne[1];
    p.ne02 = Q->ne[2];
    p.ne11 = K->ne[1];
    p.ne32 = mask ? mask->ne[2] : 1;
    p.ne33 = mask ? mask->ne[3] : 1;
    p.nb01 = Q->nb[1];
    p.nb02 = Q->nb[2];
    p.nb03 = Q->nb[3];
    p.nb11 = K->nb[1];
    p.nb12 = K->nb[2];
    p.nb13 = K->nb[3];
    p.nb21 = V->nb[1];
    p.nb22 = V->nb[2];
    p.nb23 = V->nb[3];
    p.nb31 = mask ? mask->nb[1] : 0;
    p.nb32 = mask ? mask->nb[2] : 0;
    p.nb33 = mask ? mask->nb[3] : 0;
    p.rv    = nullptr;
    p.rpos  = nullptr;
    p.rtab  = nullptr;
    p.rnbc1 = 0;
    p.rnvp = p.rnkv = p.rnhead = 0;
    if (rel) {
        // the mask is the packed REL_BIAS node (ggml_sycl_rel_bias_fa_pack_node)
        ggml_sycl_rel_dec_layout L;
        const bool ok = ggml_sycl_rel_bias_fa_dec_layout(mask, L);
        GGML_ASSERT(ok);
        const ggml_tensor * table = mask->src[3];
        p.mask   = nullptr;
        p.rv     = (const float *) ((const char *) mask->data + L.v);
        p.rpos   = (const int32_t *) ((const char *) mask->data + L.pos);
        p.rtab   = (const float *) table->data;
        p.rnbc1  = table->nb[1] / (int64_t) sizeof(float);
        p.rnvp   = L.nvp;
        p.rnkv   = L.n_kv;
        p.rnhead = L.n_head;
    }

    // up to 256 keys in one work-group (no combine pass), else at most FA_DEC_MAX_SPLIT slices of whole work-group
    // passes, at least 128 keys each
    constexpr int pass  = FA_DEC_NSG * FA_DEC_KP;
    const int     ne11  = K->ne[1];
    int           chunk = ne11;
    // GGML_SYCL_FA_DEC_CHUNK=C (keys, rounded to a multiple of pass): slices of C keys at fixed positions, so that a
    // row does not depend on the key count of the view (a speculative verify and the decode of the same token can see
    // different padded n_kv; masked keys and empty slices add exact zeros). Off by default: C = 256 costs 0.05 ms per
    // token at 8K context and 0.4 ms at 32K (more slices, longer combine), larger C starves the SWA layers
    static const int fixed = [] {
        const char * e = getenv("GGML_SYCL_FA_DEC_CHUNK");
        const int    c = e ? atoi(e) : FA_DEC_CHUNK;
        return c <= 0 ? 0 : (c + pass - 1) / pass * pass;
    }();
    if (fixed > 0) {
        chunk = std::min(ne11, fixed);
    } else if (ne11 > 256) {
        chunk = (ne11 + FA_DEC_MAX_SPLIT - 1) / FA_DEC_MAX_SPLIT;
        chunk = std::max(128, (chunk + pass - 1) / pass * pass);
    }
    p.chunk  = chunk;
    p.nsplit = (ne11 + chunk - 1) / chunk;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (p.nsplit > 1) {
        parts.alloc((size_t) p.nsplit * ggml_nelements(dst));
        meta.alloc((size_t) p.nsplit * ggml_nrows(dst));
    }
    p.parts = parts.get();
    p.meta  = meta.get();

    dpct::queue_ptr stream = ctx.stream();
    if (rel) {
        fattn_dec_kernel<G, true>(p, K->ne[2], Q->ne[3], stream);
    } else {
        fattn_dec_kernel<G, false>(p, K->ne[2], Q->ne[3], stream);
    }
    if (p.nsplit > 1) {
        fattn_dpas_combine<128>(p.parts, p.meta, p.dst, p.nsplit, Q->ne[1], Q->ne[2], Q->ne[3], stream);
    }
}

// the decode kernel takes the f16 per-head mask path for up to FA_DEC_MAX_Q tokens; returns false when it does not apply
static bool fattn_dec_takes(const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_SYCL_FA_DEC") ? atoi(getenv("GGML_SYCL_FA_DEC")) != 0 : true;
    // a row-exact op (speculative verify) of up to FA_DPAS_MAX_Q rows runs the kernel of the one-token decode, which
    // computes every row on its own (one work-group per token): its rows are the rows of the decode without
    // speculation. GGML_SYCL_FA_DEC_ROWEXACT=0: past FA_DEC_MAX_Q rows the one-row DPAS variant
    static const bool rowexact = getenv("GGML_SYCL_FA_DEC_ROWEXACT") ? atoi(getenv("GGML_SYCL_FA_DEC_ROWEXACT")) != 0 : true;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int           G = Q->ne[2] / K->ne[2];
    const int nq_max = rowexact && ggml_get_op_params_i32(dst, 6) != 0 ? FA_DPAS_MAX_Q : FA_DEC_MAX_Q;
    return enabled && dst->src[6] == nullptr && Q->ne[1] <= nq_max && K->ne[0] == 128 && V->ne[0] == 128 &&
           K->nb[1] % 16 == 0 && V->nb[1] % 16 == 0 && G >= 1 && G <= 4;
}

// rel: the mask is a REL_BIAS node packed for the kernel (ggml_sycl_rel_bias_fa_take)
static bool fattn_dec_dispatch(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool rel) {
    if (!fattn_dec_takes(dst)) {
        return false;
    }
    switch (dst->src[0]->ne[2] / dst->src[1]->ne[2]) {
        case 1: fattn_dec_case<1>(ctx, dst, rel); return true;
        case 2: fattn_dec_case<2>(ctx, dst, rel); return true;
        case 3: fattn_dec_case<3>(ctx, dst, rel); return true;
        case 4: fattn_dec_case<4>(ctx, dst, rel); return true;
        default: return false;
    }
}

bool ggml_sycl_flash_attn_ext_dpas_dec_rel_ok(const ggml_tensor * fa, const ggml_tensor * rb) {
    if (fa->op != GGML_OP_FLASH_ATTN_EXT || fa->src[3] != rb || fa->src[6] != nullptr || fa->src[1]->type != GGML_TYPE_F16 ||
        !fattn_dpas_f16_supported(fa) || !fattn_dec_takes(fa)) {
        return false;
    }
    // the kernel's (token, head) is (seq * ne01 + c, h): rb is [n_kv, ne01, n_head, ne03] with n_kv >= the keys
    const ggml_tensor * Q = fa->src[0];
    const ggml_tensor * r = rb->src[0];
    return rb->ne[1] == Q->ne[1] && rb->ne[2] == Q->ne[2] && rb->ne[3] == Q->ne[3] && r->ne[1] == Q->ne[2] &&
           r->ne[2] == Q->ne[1] * Q->ne[3] && rb->ne[0] >= fa->src[1]->ne[1] && rb->src[3]->ne[0] == 18;
}

// f16 K/V with a per-head mask or the rel-bias table: rel-bias batches past FA_DPAS_MAX_Q rows run the prefill kernel,
// the rest (decode, small batches, row-exact ops) the small-batch kernel
static void fattn_dpas_f16_dispatch(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const bool rel       = dst->src[6] != nullptr;
    const bool row_exact = ggml_get_op_params_i32(dst, 6) != 0;
    const int  G         = Q->ne[2] / K->ne[2];
    const int  ne01      = Q->ne[1];

    GGML_ASSERT(K->ne[0] == 128);

    if (rel && !row_exact && ne01 > FA_DPAS_MAX_Q) {
        fattn_rel_prefill(ctx, dst, ggml_sycl_rel_bias_fa_take(dst->src[6], dst));
        return;
    }
    // a REL_BIAS mask packed for the decode kernel is read only by it
    const bool rel_dec = !rel && ggml_sycl_rel_bias_fa_take(dst->src[3], dst);
    if (!rel && fattn_dec_dispatch(ctx, dst, rel_dec)) {
        return;
    }
    GGML_ASSERT(!rel_dec);
    // a row-exact op takes the variant of a one-token batch
    const bool rows16 = (row_exact ? 1 : ne01) * G <= 16;
    if (rel) {
        if (rows16) {
            fattn_dpas_case<128, 1, 16, 1, 0, FA_DPAS_KV_F16, FA_DPAS_MASK_REL>(ctx, dst, row_exact);
        } else {
            fattn_dpas_case<128, 2, 8, 2, 0, FA_DPAS_KV_F16, FA_DPAS_MASK_REL>(ctx, dst, row_exact);
        }
    } else {
        if (rows16) {
            fattn_dpas_case<128, 1, 16, 1, 0, FA_DPAS_KV_F16, FA_DPAS_MASK_HEAD>(ctx, dst, row_exact);
        } else {
            fattn_dpas_case<128, 2, 8, 2, 0, FA_DPAS_KV_F16, FA_DPAS_MASK_HEAD>(ctx, dst, row_exact);
        }
    }
}

void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (dst->src[1]->type == GGML_TYPE_F16) {
        fattn_dpas_f16_dispatch(ctx, dst);
    } else if (dst->src[1]->type == GGML_TYPE_Q8_0) {
        fattn_dpas_dispatch<true>(ctx, dst);
    } else {
        fattn_dpas_dispatch<false>(ctx, dst);
    }
}

#else

void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("fattn-dpas: not built for this target");
}

bool ggml_sycl_flash_attn_ext_dpas_rel_inkernel_ok(const ggml_tensor * fa, const ggml_tensor * rb) {
    GGML_UNUSED(fa);
    GGML_UNUSED(rb);
    return false;
}

bool ggml_sycl_flash_attn_ext_dpas_dec_rel_ok(const ggml_tensor * fa, const ggml_tensor * rb) {
    GGML_UNUSED(fa);
    GGML_UNUSED(rb);
    return false;
}

#endif // GGML_SYCL_FA_DPAS_BUILD
