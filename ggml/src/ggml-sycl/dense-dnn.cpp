//
// MIT license
// SPDX-License-Identifier: MIT
//

#include "dense-dnn.hpp"

#if GGML_SYCL_DNNL

#include "convert.hpp"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <map>
#include <mutex>
#include <tuple>

extern int g_ggml_sycl_enable_dnn; // ggml-sycl.cpp, GGML_SYCL_ENABLE_DNN

int g_ggml_sycl_dense_dnn            = 1;
int g_ggml_sycl_dense_dnn_min_cols   = 48;
int g_ggml_sycl_dense_dnn_bf16       = 0;
int g_ggml_sycl_dense_dnn_max_mb     = 2048;
int g_ggml_sycl_dense_dnn_max_weight_mb = 256;
int g_ggml_sycl_dense_dnn_act_cache  = 1;
int g_ggml_sycl_dense_dnn_plain      = 1;
static int g_ggml_sycl_dense_dnn_debug = 0;
static int g_ggml_sycl_dense_dnn_f32_direct = 0;
static int g_ggml_sycl_dense_dnn_concat_f16 = 0;
static int g_ggml_sycl_dense_dnn_drop_f32 = 1;
static int g_ggml_sycl_ctrl_f16 = 1;
static int g_ggml_sycl_dense_dnn_n_act = 0;

// M of the problem that picks the blocked weight layout; every M then runs against that layout
static constexpr int64_t DENSE_DNN_M_REF = 2048;
// converted src1 copies kept per context: one projection input is read by up to eight weights in a row; with
// GGML_SYCL_DROP_F32 some are pinned until their last reader, so 8 (else 4) are used
static constexpr int DENSE_DNN_N_ACT = 8;
static constexpr int DENSE_DNN_N_ACT_MAX = 16;
static int dense_dnn_n_act() {
    if (g_ggml_sycl_dense_dnn_n_act > 0) {
        return std::min(g_ggml_sycl_dense_dnn_n_act, DENSE_DNN_N_ACT_MAX);
    }
    return g_ggml_sycl_dense_dnn_drop_f32 || g_ggml_sycl_ctrl_f16 ? DENSE_DNN_N_ACT : 4;
}

void ggml_sycl_dense_dnn_init_env() {
    g_ggml_sycl_dense_dnn            = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN", 1);
    g_ggml_sycl_dense_dnn_min_cols   = std::max(1, ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_MIN_COLS", 48));
    g_ggml_sycl_dense_dnn_bf16       = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_BF16", 0);
    g_ggml_sycl_dense_dnn_max_mb     = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_MAX_MB", 2048);
    g_ggml_sycl_dense_dnn_max_weight_mb = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_MAX_WEIGHT_MB", 256);
    g_ggml_sycl_dense_dnn_act_cache  = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_ACT_CACHE", 1);
    g_ggml_sycl_dense_dnn_plain      = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_PLAIN", 1);
    g_ggml_sycl_dense_dnn_debug      = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_DEBUG", 0);
    g_ggml_sycl_dense_dnn_f32_direct = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_F32_DIRECT", 0);
    g_ggml_sycl_dense_dnn_concat_f16 = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_CONCAT_F16", 0);
    g_ggml_sycl_dense_dnn_drop_f32   = ggml_sycl_get_env("GGML_SYCL_DROP_F32", 1);
    g_ggml_sycl_ctrl_f16             = ggml_sycl_get_env("GGML_SYCL_CTRL_F16", 1);
    g_ggml_sycl_dense_dnn_n_act      = ggml_sycl_get_env("GGML_SYCL_DENSE_DNN_N_ACT", 0);
}

bool ggml_sycl_dense_dnn_drop_f32() {
    return g_ggml_sycl_dense_dnn && g_ggml_sycl_dense_dnn_act_cache && !g_ggml_sycl_dense_dnn_bf16 &&
           g_ggml_sycl_dense_dnn_drop_f32;
}

using dnnl_dt  = dnnl::memory::data_type;
using dnnl_tag = dnnl::memory::format_tag;

static dnnl_dt dense_dnn_dt() {
    return g_ggml_sycl_dense_dnn_bf16 ? dnnl_dt::bf16 : dnnl_dt::f16;
}

static bool dense_dnn_type_ok(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
#endif
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

static const ggml_tensor * dense_dnn_base(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

bool ggml_sycl_dense_dnn_eligible(int device, const ggml_tensor * dst) {
    if (!g_ggml_sycl_dense_dnn || !g_ggml_sycl_enable_dnn || dst->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!src0 || !src1 || src1->ne[1] < g_ggml_sycl_dense_dnn_min_cols) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !dense_dnn_type_ok(src0->type)) {
        return false;
    }
    // 2D only: rows of src0 contiguous, src1 rows contiguous with any row stride, dst contiguous
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src0) ||
        src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 || !ggml_is_contiguous(dst)) {
        return false;
    }
    // only model weights: their bytes never change, so a copy made once stays right
    const ggml_tensor * base = dense_dnn_base(src0);
    if (!base->buffer || ggml_backend_buffer_get_usage(base->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        ggml_backend_buffer_is_host(base->buffer) || !src1->buffer || ggml_backend_buffer_is_host(src1->buffer)) {
        return false;
    }
    // [TAG_GGML_PREC] src1 runs in f16 or bf16 here
    const int32_t src1_prec = ggml_sycl_src1_prec(dst);
    const int32_t need      = g_ggml_sycl_dense_dnn_bf16 ? GGML_PREC_BF16 : GGML_PREC_F16;
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        (src1_prec != GGML_PREC_UNDEFINED && src1_prec < need)) {
        return false;
    }
    return ggml_sycl_dnnl_has_optimized_gemm(device);
}

// ---------------------------------------------------------------------------------------------------------------
// persistent weight copies, shared by every context on the device

struct dense_dnn_weight {
    ggml_backend_buffer_t buffer = nullptr; // buffer of the base weight
    void *                data   = nullptr; // device bytes in the layout wdesc
    dnnl::memory::desc    wdesc;
    size_t                bytes  = 0;
    int                   device = 0;
    bool                  failed = false;
};

// device, weight bytes, ne0, ne1, ggml type, oneDNN type
using dense_dnn_wkey = std::tuple<int, const void *, int64_t, int64_t, int, int>;

static std::mutex                                g_dense_dnn_mutex;
static std::map<dense_dnn_wkey, dense_dnn_weight> g_dense_dnn_weights;
static size_t                                    g_dense_dnn_bytes[GGML_SYCL_MAX_DEVICES] = {};

void ggml_sycl_dense_dnn_drop_buffer(ggml_backend_buffer_t buffer) {
    std::lock_guard<std::mutex> lock(g_dense_dnn_mutex);
    for (auto it = g_dense_dnn_weights.begin(); it != g_dense_dnn_weights.end();) {
        if (it->second.buffer == buffer) {
            if (it->second.data) {
                sycl::queue & q = dpct::get_device(it->second.device).default_queue();
                q.wait();
                ggml_sycl_free_device(it->second.data, q);
                g_dense_dnn_bytes[it->second.device] -= it->second.bytes;
            }
            it = g_dense_dnn_weights.erase(it);
        } else {
            ++it;
        }
    }
}

// the weight as K x N in the blocked layout of an M_REF x K x N f16/bf16 matmul (or plain [N][K])
static dnnl::memory::desc dense_dnn_weight_desc(const dnnl::engine & eng, int64_t K, int64_t N, dnnl_dt dt) {
    if (g_ggml_sycl_dense_dnn_plain) {
        return dnnl::memory::desc({ K, N }, dt, dnnl_tag::ba);
    }
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    const dnnl::memory::desc src_md({ DENSE_DNN_M_REF, K }, dt, dnnl_tag::ab);
    const dnnl::memory::desc w_md({ K, N }, dt, dnnl_tag::any);
    const dnnl::memory::desc dst_md({ DENSE_DNN_M_REF, N }, dnnl_dt::f32, dnnl_tag::ab);
    const auto pd = dnnl::matmul::primitive_desc(eng, src_md, w_md, dst_md, attr);
    return pd.weights_desc();
}

static const dense_dnn_weight * dense_dnn_get_weight(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                                     dnnl_dt dt, queue_ptr q) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const dense_dnn_wkey key{ ctx.device, src0->data, K, N, (int) src0->type, (int) dt };

    std::lock_guard<std::mutex> lock(g_dense_dnn_mutex);
    auto it = g_dense_dnn_weights.find(key);
    if (it != g_dense_dnn_weights.end()) {
        return it->second.failed ? nullptr : &it->second;
    }
    dense_dnn_weight & w = g_dense_dnn_weights[key];
    w.failed = true;
    w.device = ctx.device;

    const ggml_tensor * base = dense_dnn_base(src0);
    w.buffer = base->buffer;
    const size_t offs = (const char *) src0->data - (const char *) base->data;
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(base->extra);
    const bool reordered = extra && extra->optimized_feature.reorder;
    // a row view: whole rows of a 2D base, the reorder layout spans the whole base
    if (base != src0 && (offs % base->nb[1] != 0 || base->ne[0] != K || ggml_nrows(base) * base->nb[1] != ggml_nbytes(base) ||
                         (reordered && (base->ne[2] != 1 || base->ne[3] != 1)))) {
        GGML_LOG_DEBUG("%s: %s: unsupported view, using the generic path\n", __func__, src0->name);
        return nullptr;
    }

    const int64_t t_start = ggml_time_us();
    const dnnl::engine eng = ctx.engine_dnnl(q);
    w.wdesc = dense_dnn_weight_desc(eng, K, N, dt);
    const int64_t t_desc = ggml_time_us();
    w.bytes = w.wdesc.get_size();
    if (w.bytes > (size_t) g_ggml_sycl_dense_dnn_max_weight_mb * 1024 * 1024 ||
        g_dense_dnn_bytes[ctx.device] + w.bytes > (size_t) g_ggml_sycl_dense_dnn_max_mb * 1024 * 1024) {
        GGML_LOG_DEBUG("%s: %s: %.1f MB over the budget, using the generic path\n", __func__, src0->name, w.bytes / 1048576.0);
        return nullptr;
    }

    // f32 rows of the weight
    float * rows   = nullptr;
    float * tmp    = nullptr;
    if (src0->type == GGML_TYPE_F32) {
        rows = (float *) src0->data;
    } else {
        // a reordered weight decodes as a whole; a plain one row by row
        const int64_t n_dq  = reordered ? ggml_nelements(base) : K * N;
        const void *  dq_in = reordered ? base->data : src0->data;
        tmp = (float *) ggml_sycl_malloc_device(n_dq * sizeof(float), *q);
        if (!tmp) {
            return nullptr;
        }
        // ggml_get_to_fp32_sycl() reads the reorder flag through dst->src[0]->extra
        ggml_tensor fake_src0 = *(reordered ? base : src0);
        fake_src0.extra       = reordered ? base->extra : nullptr;
        ggml_tensor fake_dst  = {};
        fake_dst.src[0]       = &fake_src0;
        const to_fp32_sycl_t to_fp32 = ggml_get_to_fp32_sycl(src0->type, &fake_dst);
        to_fp32(dq_in, tmp, n_dq, q);
        rows = tmp + (reordered ? (int64_t) (offs / base->nb[1]) * K : 0);
    }

    const int64_t t_dq = ggml_time_us();
    w.data = ggml_sycl_malloc_device(w.bytes, *q, GGML_SYCL_MEM_BUFFER);
    if (!w.data) {
        q->wait();
        ggml_sycl_free_device(tmp, *q);
        return nullptr;
    }
    {
        dnnl::stream       st = ctx.stream_dnnl(q);
        // an F32 weight may be a column range of its rows (the halves of a concat matmul, see below)
        const int64_t      ld_rows = src0->type == GGML_TYPE_F32 ? (int64_t) (src0->nb[1] / sizeof(float)) : K;
        dnnl::memory::desc plain_md({ K, N }, dnnl_dt::f32, dnnl::memory::dims{ 1, ld_rows });
        dnnl::memory       plain(plain_md, eng, rows);
        dnnl::memory       blocked(w.wdesc, eng, w.data);
        dnnl::reorder(plain, blocked).execute(st, plain, blocked);
    }
    q->wait();
    const int64_t t_ro = ggml_time_us();
    if (tmp) {
        ggml_sycl_free_device(tmp, *q);
    }
    if (g_ggml_sycl_dense_dnn_debug) {
        fprintf(stderr, "[dense-dnn] %s [%s %lldx%lld] %.1f MB: desc %.1f ms, dequant %.1f ms, reorder %.1f ms, free %.1f ms\n",
                src0->name, ggml_type_name(src0->type), (long long) N, (long long) K, w.bytes / 1048576.0,
                (t_desc - t_start) / 1e3, (t_dq - t_desc) / 1e3, (t_ro - t_dq) / 1e3, (ggml_time_us() - t_ro) / 1e3);
    }

    w.failed = false;
    const size_t before = g_dense_dnn_bytes[ctx.device];
    g_dense_dnn_bytes[ctx.device] += w.bytes;
    GGML_LOG_DEBUG("%s: %s [%s %lldx%lld] -> %.1f MB %s\n", __func__, src0->name, ggml_type_name(src0->type),
                   (long long) N, (long long) K, w.bytes / 1048576.0, g_ggml_sycl_dense_dnn_bf16 ? "bf16" : "f16");
    if (before / (256u << 20) != g_dense_dnn_bytes[ctx.device] / (256u << 20)) {
        GGML_LOG_INFO("%s: dense weight copies on device %d: %.0f MB\n", __func__, ctx.device,
                      g_dense_dnn_bytes[ctx.device] / 1048576.0);
    }
    return &w;
}

// ---------------------------------------------------------------------------------------------------------------
// per-context state

struct dense_dnn_act {
    const void * src   = nullptr; // f32 source
    size_t       bytes = 0;       // extent of the source
    int64_t      K = 0, M = 0, ld = 0;
    int          dt    = 0;
    void *       buf   = nullptr; // M x K converted, dense
    size_t       cap   = 0;
    bool         valid = false;
    uint64_t     used  = 0;
    int          pin_until = -1; // GGML_SYCL_DROP_F32: the only copy of its source until graph node pin_until
};

struct ggml_sycl_dense_dnn_ctx {
    queue_ptr                                                        q = nullptr;
    std::array<dense_dnn_act, DENSE_DNN_N_ACT_MAX>                   acts;
    bool                                                             any_valid = false;
    uint64_t                                                         tick      = 0;
    int                                                              cur_node  = -1; // for the pins
    // M, N, K, dt, dst dt, tanh post-op, accumulate
    std::map<std::tuple<int64_t, int64_t, int64_t, int, int, bool, bool>, std::pair<dnnl::matmul, dnnl::memory::desc>> matmuls;
    // M, K, ld of the f32 source, ld of the result, dt
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int>, dnnl::reorder>  reorders;
    // f32 matmuls on strided operands: M, N, K, lda, ldw, accumulate
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool>, std::pair<dnnl::matmul, dnnl::memory::desc>> f32_matmuls;

    ~ggml_sycl_dense_dnn_ctx() {
        if (q) {
            q->wait();
            for (auto & a : acts) {
                ggml_sycl_free_device(a.buf, *q);
            }
        }
    }
};

void ggml_sycl_dense_dnn_ctx_free(ggml_sycl_dense_dnn_ctx * state) {
    delete state;
}

static ggml_sycl_dense_dnn_ctx & dense_dnn_state(ggml_backend_sycl_context & ctx) {
    if (!ctx.dense_dnn) {
        ctx.dense_dnn    = new ggml_sycl_dense_dnn_ctx();
        ctx.dense_dnn->q = ctx.stream();
    }
    return *ctx.dense_dnn;
}

void ggml_sycl_dense_dnn_graph_begin(ggml_backend_sycl_context & ctx) {
    if (ctx.dense_dnn) {
        for (auto & a : ctx.dense_dnn->acts) {
            a.valid     = false;
            a.pin_until = -1;
        }
        ctx.dense_dnn->any_valid = false;
        ctx.dense_dnn->cur_node  = -1;
    }
}

void ggml_sycl_dense_dnn_set_node(ggml_backend_sycl_context & ctx, int i) {
    if (ctx.dense_dnn) {
        ctx.dense_dnn->cur_node = i;
    }
}

// a pinned slot holds the only copy of its source (the f32 bytes were never written): dropping it before its last
// reader means the reader would convert garbage, so the whole-graph proof of the producer was wrong
static void dense_dnn_drop_slot(const ggml_sycl_dense_dnn_ctx & st, dense_dnn_act & a, const char * why) {
    if (a.pin_until >= 0 && a.pin_until >= st.cur_node) {
        GGML_ABORT("%s: f16-only activation %p (%lld x %lld) dropped at node %d, before its last reader %d\n", why,
                   a.src, (long long) a.M, (long long) a.K, st.cur_node, a.pin_until);
    }
    a.valid     = false;
    a.pin_until = -1;
}

void ggml_sycl_dense_dnn_node_writes(ggml_backend_sycl_context & ctx, const ggml_tensor * node) {
    if (!ctx.dense_dnn || !ctx.dense_dnn->any_valid || !node->data) {
        return;
    }
    const char * d0 = (const char *) node->data;
    const char * d1 = d0 + ggml_nbytes(node);
    bool any = false;
    for (auto & a : ctx.dense_dnn->acts) {
        if (a.valid) {
            const char * s0 = (const char *) a.src;
            if (d0 < s0 + a.bytes && s0 < d1) {
                dense_dnn_drop_slot(*ctx.dense_dnn, a, "node_writes");
            }
            any = any || a.valid;
        }
    }
    ctx.dense_dnn->any_valid = any;
}

// dst[M][ld_dst] (dt) = src[M][ld_src] (f32), K columns
static void dense_dnn_convert(ggml_backend_sycl_context & ctx, ggml_sycl_dense_dnn_ctx & st, queue_ptr q, const void * src,
                              int64_t M, int64_t K, int64_t ld_src, void * dst, int64_t ld_dst, dnnl_dt dt) {
    const dnnl::engine eng = ctx.engine_dnnl(q);
    const dnnl::memory::desc src_md({ M, K }, dnnl_dt::f32, dnnl::memory::dims{ ld_src, 1 });
    const dnnl::memory::desc dst_md({ M, K }, dt, dnnl::memory::dims{ ld_dst, 1 });
    auto key = std::make_tuple(M, K, ld_src, ld_dst, (int) dt);
    auto it  = st.reorders.find(key);
    if (it == st.reorders.end()) {
        it = st.reorders.emplace(key, dnnl::reorder(dnnl::reorder::primitive_desc(eng, src_md, eng, dst_md))).first;
    }
    dnnl::memory src_mem(src_md, eng, const_cast<void *>(src));
    dnnl::memory dst_mem(dst_md, eng, dst);
    it->second.execute(ctx.stream_dnnl(q), src_mem, dst_mem);
}

// the least recently used slot (an invalid one first) with room for need bytes; nullptr when that cannot be allocated
// (a slot pinned past the current node is never taken)
static dense_dnn_act * dense_dnn_take_slot(ggml_sycl_dense_dnn_ctx & st, size_t need, queue_ptr q) {
    dense_dnn_act * slot = nullptr;
    for (int i = 0; i < dense_dnn_n_act(); ++i) {
        dense_dnn_act & a = st.acts[i];
        if (a.valid && a.pin_until >= 0 && a.pin_until >= st.cur_node) {
            continue;
        }
        if (!slot || (!a.valid && slot->valid) || (a.valid == slot->valid && a.used < slot->used)) {
            slot = &a;
        }
    }
    if (!slot) {
        return nullptr;
    }
    slot->valid     = false;
    slot->pin_until = -1;
    if (slot->cap < need) {
        if (slot->buf) {
            q->wait();
            ggml_sycl_free_device(slot->buf, *q);
        }
        // a little past the rows: the MoE gather of these rows (ggml_sycl_dense_dnn_act_lookup) prefetches ahead of
        // the K loop, which must not touch an unmapped page past the last row
        slot->buf = ggml_sycl_malloc_device(need + 4096, *q);
        slot->cap = slot->buf ? need : 0;
        if (!slot->buf) {
            slot->valid = false;
            return nullptr;
        }
    }
    return slot;
}

static void dense_dnn_set_slot(ggml_sycl_dense_dnn_ctx & st, dense_dnn_act * slot, const ggml_tensor * src1, dnnl_dt dt) {
    slot->src   = src1->data;
    slot->bytes = ggml_nbytes(src1);
    slot->K     = src1->ne[0];
    slot->M     = src1->ne[1];
    slot->ld    = src1->nb[1] / sizeof(float);
    slot->dt    = (int) dt;
    slot->used  = st.tick;
    slot->valid = g_ggml_sycl_dense_dnn_act_cache != 0;
    slot->pin_until = -1;
    st.any_valid = st.any_valid || slot->valid;
}

// src1 as a dense M x K f16/bf16 matrix, converted once while its bytes are unchanged
static const void * dense_dnn_get_act(ggml_backend_sycl_context & ctx, ggml_sycl_dense_dnn_ctx & st,
                                      const ggml_tensor * src1, dnnl_dt dt, queue_ptr q) {
    const int64_t K  = src1->ne[0];
    const int64_t M  = src1->ne[1];
    const int64_t ld = src1->nb[1] / sizeof(float);
    ++st.tick;

    for (auto & a : st.acts) {
        if (a.valid && a.src == src1->data && a.K == K && a.M == M && a.ld == ld && a.dt == (int) dt) {
            a.used = st.tick;
            return a.buf;
        }
    }
    dense_dnn_act * slot = dense_dnn_take_slot(st, (size_t) M * K * 2, q);
    if (!slot) {
        return nullptr;
    }

    dense_dnn_convert(ctx, st, q, src1->data, M, K, ld, slot->buf, K, dt);

    dense_dnn_set_slot(st, slot, src1, dt);
    return slot->buf;
}

sycl::half * ggml_sycl_dense_dnn_act_reserve(ggml_backend_sycl_context & ctx, const ggml_tensor * t, int pin_until) {
    if (!g_ggml_sycl_dense_dnn || !g_ggml_sycl_dense_dnn_act_cache || g_ggml_sycl_dense_dnn_bf16 || !t->data ||
        t->type != GGML_TYPE_F32 || t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != sizeof(float) ||
        t->nb[1] % sizeof(float) != 0) {
        return nullptr;
    }
    ggml_sycl_dense_dnn_ctx & st = dense_dnn_state(ctx);
    ++st.tick;
    // a stale copy of the same bytes must not stay valid next to the new one
    for (auto & a : st.acts) {
        if (a.valid && a.src == t->data) {
            dense_dnn_drop_slot(st, a, "act_reserve");
        }
    }
    dense_dnn_act * slot = dense_dnn_take_slot(st, (size_t) t->ne[0] * t->ne[1] * 2, ctx.stream());
    if (!slot) {
        return nullptr;
    }
    dense_dnn_set_slot(st, slot, t, dnnl_dt::f16);
    if (pin_until >= 0) {
        if (!slot->valid) {
            return nullptr; // ACT_CACHE=0: a reader would not find the copy
        }
        slot->pin_until = pin_until;
    }
    return (sycl::half *) slot->buf;
}

const sycl::half * ggml_sycl_dense_dnn_act_lookup(ggml_backend_sycl_context & ctx, const void * src, int64_t K, int64_t M,
                                                  int64_t ld) {
    if (!ctx.dense_dnn || !ctx.dense_dnn->any_valid) {
        return nullptr;
    }
    for (const auto & a : ctx.dense_dnn->acts) {
        if (a.valid && a.src == src && a.K == K && a.M == M && a.ld == ld && a.dt == (int) dnnl_dt::f16) {
            return (const sycl::half *) a.buf;
        }
    }
    return nullptr;
}

bool ggml_sycl_dense_dnn_weight_ready(int device, const ggml_tensor * src0) {
    const dense_dnn_wkey key{ device, src0->data, src0->ne[0], src0->ne[1], (int) src0->type, (int) dense_dnn_dt() };
    std::lock_guard<std::mutex> lock(g_dense_dnn_mutex);
    auto it = g_dense_dnn_weights.find(key);
    return it != g_dense_dnn_weights.end() && !it->second.failed;
}

bool ggml_sycl_dense_dnn_converts_src1(int device, const ggml_tensor * dst) {
    if (!ggml_sycl_dense_dnn_eligible(device, dst)) {
        return false;
    }
    return !(dst->src[0]->type == GGML_TYPE_F32 && g_ggml_sycl_dense_dnn_f32_direct);
}

// c[M][N] (+)= a[M][K] * w[N][K]^T with f32 operands, a and w with row strides lda and ldw (elements), c dense; the
// f32 inputs are rounded to f16 inside the GEMM (fpmath f16) as the generic F32 path of ggml_sycl_op_mul_mat_sycl does
static void dense_dnn_f32_mm(ggml_backend_sycl_context & ctx, ggml_sycl_dense_dnn_ctx & st, queue_ptr q, int64_t M,
                             int64_t N, int64_t K, const float * a, int64_t lda, const float * w, int64_t ldw, float * c,
                             bool accumulate) {
    const dnnl::engine eng = ctx.engine_dnnl(q);
    const dnnl::memory::desc a_md({ M, K }, dnnl_dt::f32, dnnl::memory::dims{ lda, 1 });
    const dnnl::memory::desc w_md({ K, N }, dnnl_dt::f32, dnnl::memory::dims{ 1, ldw });
    const dnnl::memory::desc c_md({ M, N }, dnnl_dt::f32, dnnl_tag::ab);

    auto key = std::make_tuple(M, N, K, lda, ldw, accumulate);
    auto it  = st.f32_matmuls.find(key);
    if (it == st.f32_matmuls.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if DNNL_VERSION_MAJOR > 3 || (DNNL_VERSION_MAJOR == 3 && DNNL_VERSION_MINOR >= 4)
        // split-K with atomics otherwise makes short wide problems differ run to run (see gemm.hpp)
        attr.set_deterministic(true);
#endif
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
        if (accumulate) {
            dnnl::post_ops po;
            po.append_sum(1.0f);
            attr.set_post_ops(po);
        }
        const auto pd = dnnl::matmul::primitive_desc(eng, a_md, w_md, c_md, attr);
        it = st.f32_matmuls.emplace(key, std::make_pair(dnnl::matmul(pd), pd.scratchpad_desc())).first;
        if (g_ggml_sycl_dense_dnn_debug) {
            fprintf(stderr, "[dense-dnn] f32 matmul %lldx%lldx%lld lda %lld ldw %lld%s: %s\n", (long long) M, (long long) N,
                    (long long) K, (long long) lda, (long long) ldw, accumulate ? " +=" : "", pd.impl_info_str());
        }
    }
    ggml_sycl_pool_alloc<uint8_t> scratch(ctx.pool());
    const dnnl::memory::desc & scratch_md = it->second.second;
    void * scratch_ptr = scratch_md.get_size() > 0 ? scratch.alloc(scratch_md.get_size()) : nullptr;

    dnnl::memory a_mem(a_md, eng, const_cast<float *>(a));
    dnnl::memory w_mem(w_md, eng, const_cast<float *>(w));
    dnnl::memory c_mem(c_md, eng, c);
    dnnl::memory s_mem(scratch_md, eng, scratch_ptr);
    it->second.first.execute(ctx.stream_dnnl(q), { { DNNL_ARG_SRC, a_mem },
                                                   { DNNL_ARG_WEIGHTS, w_mem },
                                                   { DNNL_ARG_DST, c_mem },
                                                   { DNNL_ARG_SCRATCHPAD, s_mem } });
}

// dst[M][N] (ddt) = [tanh](act[M][K] * weight), the tanh on the f32 accumulator before the conversion
static void dense_dnn_run(ggml_backend_sycl_context & ctx, ggml_sycl_dense_dnn_ctx & st, queue_ptr q,
                          const dense_dnn_weight * w, const void * act, int64_t M, int64_t N, int64_t K, void * dst,
                          dnnl_dt dt, dnnl_dt ddt = dnnl_dt::f32, bool tanh = false, bool accumulate = false) {
    const dnnl::engine eng = ctx.engine_dnnl(q);
    const dnnl::memory::desc src_md({ M, K }, dt, dnnl_tag::ab);
    const dnnl::memory::desc dst_md({ M, N }, ddt, dnnl_tag::ab);

    auto key = std::make_tuple(M, N, K, (int) dt, (int) ddt, tanh, accumulate);
    auto it  = st.matmuls.find(key);
    if (it == st.matmuls.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if DNNL_VERSION_MAJOR > 3 || (DNNL_VERSION_MAJOR == 3 && DNNL_VERSION_MINOR >= 4)
        // split-K with atomics otherwise makes short wide problems differ run to run (see gemm.hpp)
        attr.set_deterministic(true);
#endif
        if (tanh || accumulate) {
            dnnl::post_ops po;
            if (accumulate) {
                po.append_sum(1.0f);
            }
            if (tanh) {
                po.append_eltwise(dnnl::algorithm::eltwise_tanh, 0.0f, 0.0f);
            }
            attr.set_post_ops(po);
        }
        const int64_t t0 = ggml_time_us();
        const auto pd = dnnl::matmul::primitive_desc(eng, src_md, w->wdesc, dst_md, attr);
        it = st.matmuls.emplace(key, std::make_pair(dnnl::matmul(pd), pd.scratchpad_desc())).first;
        if (g_ggml_sycl_dense_dnn_debug) {
            fprintf(stderr, "[dense-dnn] matmul %lldx%lldx%lld: %.1f ms, %s\n", (long long) M, (long long) N, (long long) K,
                    (ggml_time_us() - t0) / 1e3, pd.impl_info_str());
        }
    }

    ggml_sycl_pool_alloc<uint8_t> scratch(ctx.pool());
    const dnnl::memory::desc & scratch_md = it->second.second;
    void * scratch_ptr = scratch_md.get_size() > 0 ? scratch.alloc(scratch_md.get_size()) : nullptr;

    dnnl::memory a_mem(src_md, eng, const_cast<void *>(act));
    dnnl::memory w_mem(w->wdesc, eng, w->data);
    dnnl::memory c_mem(dst_md, eng, dst);
    dnnl::memory s_mem(scratch_md, eng, scratch_ptr);
    it->second.first.execute(ctx.stream_dnnl(q), { { DNNL_ARG_SRC, a_mem },
                                                   { DNNL_ARG_WEIGHTS, w_mem },
                                                   { DNNL_ARG_DST, c_mem },
                                                   { DNNL_ARG_SCRATCHPAD, s_mem } });
}

// rows of a 2D-able f32 tensor: M rows of ne[0] floats at a constant stride, or false
static bool dense_dnn_rows(const ggml_tensor * t, int64_t & M, int64_t & ld) {
    if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->ne[3] != 1) {
        return false;
    }
    if (t->ne[1] == 1) {
        M  = t->ne[2];
        ld = t->nb[2] / sizeof(float);
        return t->nb[2] % sizeof(float) == 0;
    }
    if (t->ne[2] == 1) {
        M  = t->ne[1];
        ld = t->nb[1] / sizeof(float);
        return t->nb[1] % sizeof(float) == 0;
    }
    return false;
}

// CONCAT(a, b, dim 0) -> RESHAPE -> MUL_MAT(F32 weight) at node i, the concat used only there (see the header)
static bool dense_dnn_concat_match(int device, const ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_dense_dnn || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * cat  = cgraph->nodes[i];
    const ggml_tensor * view = cgraph->nodes[i + 1];
    const ggml_tensor * mm   = cgraph->nodes[i + 2];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || (cat->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        ggml_node_get_use_count(cgraph, i) != 1 || cat->type != GGML_TYPE_F32 || !ggml_is_contiguous(cat)) {
        return false;
    }
    if ((view->op != GGML_OP_RESHAPE && view->op != GGML_OP_VIEW) || view->view_src != cat || view->view_offs != 0 ||
        (view->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, i + 1) != 1 ||
        !ggml_is_contiguous(view)) {
        return false;
    }
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != view || mm->src[0]->type != GGML_TYPE_F32 ||
        !ggml_sycl_dense_dnn_eligible(device, mm)) {
        return false;
    }
    const ggml_tensor * a = cat->src[0];
    const ggml_tensor * b = cat->src[1];
    const ggml_tensor * w = mm->src[0];
    int64_t Ma, lda, Mb, ldb;
    return dense_dnn_rows(a, Ma, lda) && dense_dnn_rows(b, Mb, ldb) && Ma == view->ne[1] && Mb == view->ne[1] &&
           a->ne[0] + b->ne[0] == w->ne[0] && view->ne[0] == w->ne[0] && w->nb[1] % sizeof(float) == 0;
}

bool ggml_sycl_dense_dnn_concat_reads_f16(int device, const ggml_cgraph * cgraph, int i) {
    return g_ggml_sycl_ctrl_f16 && !g_ggml_sycl_dense_dnn_concat_f16 && !g_ggml_sycl_dense_dnn_bf16 &&
           g_ggml_sycl_dense_dnn_act_cache && dense_dnn_concat_match(device, cgraph, i);
}

// the f16 copies of the column halves of the concat matmul weight (made by an earlier call)
bool ggml_sycl_dense_dnn_concat_ready(int device, const ggml_cgraph * cgraph, int i) {
    if (!ggml_sycl_dense_dnn_concat_reads_f16(device, cgraph, i)) {
        return false;
    }
    const ggml_tensor * a = cgraph->nodes[i]->src[0];
    const ggml_tensor * b = cgraph->nodes[i]->src[1];
    const ggml_tensor * w = cgraph->nodes[i + 2]->src[0];
    const dense_dnn_wkey ka{ device, w->data, a->ne[0], w->ne[1], (int) w->type, (int) dnnl_dt::f16 };
    const dense_dnn_wkey kb{ device, (const char *) w->data + a->ne[0] * sizeof(float), b->ne[0], w->ne[1], (int) w->type,
                             (int) dnnl_dt::f16 };
    std::lock_guard<std::mutex> lock(g_dense_dnn_mutex);
    auto ia = g_dense_dnn_weights.find(ka);
    auto ib = g_dense_dnn_weights.find(kb);
    return ia != g_dense_dnn_weights.end() && !ia->second.failed && ib != g_dense_dnn_weights.end() && !ib->second.failed;
}

int ggml_sycl_dense_dnn_concat_mul_mat(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) try {
    if (!dense_dnn_concat_match(ctx.device, cgraph, i)) {
        return 0;
    }
    const ggml_tensor * cat  = cgraph->nodes[i];
    const ggml_tensor * view = cgraph->nodes[i + 1];
    ggml_tensor *       mm   = cgraph->nodes[i + 2];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || (cat->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        ggml_node_get_use_count(cgraph, i) != 1 || cat->type != GGML_TYPE_F32 || !ggml_is_contiguous(cat)) {
        return 0;
    }
    if ((view->op != GGML_OP_RESHAPE && view->op != GGML_OP_VIEW) || view->view_src != cat || view->view_offs != 0 ||
        (view->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, i + 1) != 1 ||
        !ggml_is_contiguous(view)) {
        return 0;
    }
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != view || mm->src[0]->type != GGML_TYPE_F32 ||
        !ggml_sycl_dense_dnn_eligible(ctx.device, mm)) {
        return 0;
    }
    const ggml_tensor * a = cat->src[0];
    const ggml_tensor * b = cat->src[1];
    int64_t Ma, lda, Mb, ldb;
    if (!dense_dnn_rows(a, Ma, lda) || !dense_dnn_rows(b, Mb, ldb)) {
        return 0;
    }
    const ggml_tensor * w = mm->src[0];
    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    const int64_t M = view->ne[1];
    if (Ma != M || Mb != M || a->ne[0] + b->ne[0] != K || view->ne[0] != K || w->nb[1] % sizeof(float) != 0) {
        return 0;
    }

    queue_ptr q = ctx.stream();
    ggml_sycl_dense_dnn_ctx & st = dense_dnn_state(ctx);
    if (g_ggml_sycl_ctrl_f16 && !g_ggml_sycl_dense_dnn_concat_f16 && !g_ggml_sycl_dense_dnn_bf16 &&
        g_ggml_sycl_dense_dnn_act_cache) {
        // GGML_SYCL_CTRL_F16: one f16 matmul per half against an f16 copy of its column range of the weight,
        // the second accumulating; each half's f16 copy comes from the activation cache (written by its producer,
        // or converted once and shared by the matmuls that read the same rows)
        ggml_tensor wh[2] = { *w, *w };
        const ggml_tensor * half[2] = { a, b };
        const int64_t       ldh[2]  = { lda, ldb };
        const dense_dnn_weight * wc[2];
        const void *             act[2];
        for (int h = 0; h < 2; ++h) {
            wh[h].view_src  = nullptr;
            wh[h].view_offs = 0;
            wh[h].ne[0]     = half[h]->ne[0];
            wh[h].data      = (char *) w->data + (h ? a->ne[0] * sizeof(float) : 0);
            wc[h] = dense_dnn_get_weight(ctx, &wh[h], dnnl_dt::f16, q);
        }
        if (wc[0] && wc[1]) {
            // a half without a cache slot (every slot pinned) is converted into a scratch copy; a half whose f32
            // bytes were dropped (GGML_SYCL_DROP_F32) is always in its pinned slot
            ggml_sycl_pool_alloc<sycl::half> tmp[2] = { ggml_sycl_pool_alloc<sycl::half>(ctx.pool()),
                                                        ggml_sycl_pool_alloc<sycl::half>(ctx.pool()) };
            for (int h = 0; h < 2; ++h) {
                ggml_tensor rows = *half[h];
                rows.view_src = nullptr;
                rows.ne[1] = M; rows.ne[2] = 1; rows.ne[3] = 1;
                rows.nb[1] = ldh[h] * sizeof(float); rows.nb[2] = rows.nb[1] * M; rows.nb[3] = rows.nb[2];
                act[h] = dense_dnn_get_act(ctx, st, &rows, dnnl_dt::f16, q);
                if (!act[h]) {
                    act[h] = tmp[h].alloc(M * half[h]->ne[0]);
                    dense_dnn_convert(ctx, st, q, half[h]->data, M, half[h]->ne[0], ldh[h], tmp[h].get(), half[h]->ne[0],
                                      dnnl_dt::f16);
                }
            }
            float * c = (float *) mm->data;
            dense_dnn_run(ctx, st, q, wc[0], act[0], M, N, a->ne[0], c, dnnl_dt::f16);
            dense_dnn_run(ctx, st, q, wc[1], act[1], M, N, b->ne[0], c, dnnl_dt::f16, dnnl_dt::f32, false, true);
            return 2;
        }
    }
    if (g_ggml_sycl_dense_dnn_concat_f16) {
        // both halves converted into one f16 matrix, then the matmul against the weight copy
        const dnnl_dt dt = dense_dnn_dt();
        const dense_dnn_weight * wc = dense_dnn_get_weight(ctx, w, dt, q);
        if (!wc) {
            return 0;
        }
        ggml_sycl_pool_alloc<uint16_t> act(ctx.pool(), M * K);
        dense_dnn_convert(ctx, st, q, a->data, M, a->ne[0], lda, act.get(), K, dt);
        dense_dnn_convert(ctx, st, q, b->data, M, b->ne[0], ldb, act.get() + a->ne[0], K, dt);
        dense_dnn_run(ctx, st, q, wc, act.get(), M, N, K, (float *) mm->data, dt);
        return 2;
    }
    const int64_t ldw = w->nb[1] / sizeof(float);
    float * c = (float *) mm->data;
    dense_dnn_f32_mm(ctx, st, q, M, N, a->ne[0], (const float *) a->data, lda, (const float *) w->data, ldw, c, false);
    dense_dnn_f32_mm(ctx, st, q, M, N, b->ne[0], (const float *) b->data, ldb, (const float *) w->data + a->ne[0], ldw, c,
                     true);
    return 2;
} catch (const dnnl::error & e) {
    GGML_LOG_WARN("%s: oneDNN error (%s), using the generic path\n", __func__, e.what());
    return 0;
}

bool ggml_sycl_dense_dnn_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                 ggml_tensor * dst) try {
    queue_ptr     q  = ctx.stream();
    const dnnl_dt dt = dense_dnn_dt();

    if (src0->type == GGML_TYPE_F32 && g_ggml_sycl_dense_dnn_f32_direct) {
        int64_t M, lda;
        if (dense_dnn_rows(src1, M, lda) && src0->nb[1] % sizeof(float) == 0) {
            dense_dnn_f32_mm(ctx, dense_dnn_state(ctx), q, M, src0->ne[1], src0->ne[0], (const float *) src1->data, lda,
                             (const float *) src0->data, src0->nb[1] / sizeof(float), (float *) dst->data, false);
            return true;
        }
    }

    const dense_dnn_weight * w = dense_dnn_get_weight(ctx, src0, dt, q);
    if (!w) {
        return false;
    }
    ggml_sycl_dense_dnn_ctx & st = dense_dnn_state(ctx);
    const void * act = dense_dnn_get_act(ctx, st, src1, dt, q);
    if (!act) {
        return false;
    }

    dense_dnn_run(ctx, st, q, w, act, src1->ne[1], src0->ne[1], src0->ne[0], (float *) dst->data, dt);
    return true;
} catch (const dnnl::error & e) {
    GGML_LOG_WARN("%s: oneDNN error for %s (%s), using the generic path\n", __func__, dst->name, e.what());
    return false;
}

bool ggml_sycl_dense_dnn_mul_mat_f16(ggml_backend_sycl_context & ctx, const ggml_tensor * dst, sycl::half * out,
                                     bool tanh) try {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!ggml_sycl_dense_dnn_converts_src1(ctx.device, dst) || src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 ||
        src1->ne[3] != 1 || src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0) {
        return false;
    }
    queue_ptr     q  = ctx.stream();
    const dnnl_dt dt = dense_dnn_dt();
    const dense_dnn_weight * w = dense_dnn_get_weight(ctx, src0, dt, q);
    if (!w) {
        return false;
    }
    ggml_sycl_dense_dnn_ctx & st = dense_dnn_state(ctx);
    const void * act = dense_dnn_get_act(ctx, st, src1, dt, q);
    if (!act) {
        return false;
    }
    dense_dnn_run(ctx, st, q, w, act, src1->ne[1], src0->ne[1], src0->ne[0], out, dt, dnnl_dt::f16, tanh);
    return true;
} catch (const dnnl::error & e) {
    GGML_LOG_WARN("%s: oneDNN error for %s (%s), using the generic path\n", __func__, dst->name, e.what());
    return false;
}

#endif // GGML_SYCL_DNNL
