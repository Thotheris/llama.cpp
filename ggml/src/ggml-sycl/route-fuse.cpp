// Decode MoE routing with a selection bias in one launch (GGML_SYCL_ROUTE_FUSE, default on).
//
// llm_graph_context::build_moe_ffn with sigmoid gating, exp_probs_b and normalised, scaled weights emits, per layer
//   probs = sigmoid(logits) -> RESHAPE -> ADD(probs, bias) -> ARGSORT -> VIEW (top k) -> GET_ROWS(probs, top k)
//   -> RESHAPE -> SUM_ROWS -> SCALE (x + eps) -> DIV -> RESHAPE -> SCALE (w_scale)
// i.e. twelve nodes and eight kernels of one sub-group or less. The topk-moe fusion (topk-moe.cpp) declines the bias.
// Here one work-group per token does the lot: the sigmoid as op_sigmoid, the top k of probs + bias by rank (ties: the
// lower expert, as in topk-moe.cpp), the unbiased probs of the chosen experts, their sum with the tree of the
// short-row sum_rows kernel, the two SCALE ops as scale * x + bias and the DIV. Only the top-k view of the argsort
// output is written (its only reader). Gated to rows (tokens) <= 4.

#include "route-fuse.hpp"

#include <cfloat>

#include "ggml-impl.h"

namespace {

constexpr int ROUTE_MAX_USED = 16; // the choices of a token

// one work-group of NE work-items per token, one per expert: the rank of an expert is the number of experts with a
// larger biased prob, or an equal one and a lower index, so the top k are the ranks below k (in the order of a
// descending sort, ties to the lower expert as in topk-moe.cpp)
template <int NE>
void route_kernel(const float * __restrict__ logits, const float * __restrict__ bias, float * __restrict__ weights,
                  int32_t * __restrict__ ids, int ids_s, int n_used, float s1, float b1, float s2, float b2,
                  float * __restrict__ sel_l, float * __restrict__ w_l, const sycl::nd_item<1> & it) {
    const int row = (int) it.get_group(0);
    const int e   = (int) it.get_local_id(0);

    logits += (size_t) row * NE;
    weights += (size_t) row * n_used;
    ids += (size_t) row * ids_s;

    const float x  = logits[e];
    const float p  = 1.0f / (1.0f + sycl::exp(-x));
    const float sb = p + bias[e];
    const float sv = sycl::isnan(sb) ? -FLT_MAX : sb;
    sel_l[e]       = sv;
    it.barrier(sycl::access::fence_space::local_space);

    int                   rank = 0;
    const sycl::float4 *  s4   = (const sycl::float4 *) sel_l;
#pragma unroll 8
    for (int j = 0; j < NE / 4; ++j) {
        const sycl::float4 o = s4[j];
        rank += (o.x() > sv) || (o.x() == sv && 4 * j + 0 < e);
        rank += (o.y() > sv) || (o.y() == sv && 4 * j + 1 < e);
        rank += (o.z() > sv) || (o.z() == sv && 4 * j + 2 < e);
        rank += (o.w() > sv) || (o.w() == sv && 4 * j + 3 < e);
    }
    if (rank < n_used) {
        ids[rank] = e;
        w_l[rank] = p;
    }
    it.barrier(sycl::access::fence_space::local_space);

    if (e < n_used) {
        // the sum_rows tree: lane i holds w[i], halves added from WARP_SIZE/2 down
        float v[WARP_SIZE];
#pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            v[i] = i < n_used ? w_l[i] : 0.0f;
        }
#pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s >>= 1) {
#pragma unroll
            for (int i = 0; i < s; ++i) {
                v[i] = v[i] + v[i + s];
            }
        }
        const float sum = s1 * v[0] + b1;
        const float wn  = w_l[e] / sum;
        weights[e]      = s2 * wn + b2;
    }
}

template <int NE>
void route_launch(sycl::queue * q, const float * logits, const float * bias, float * weights, int32_t * ids, int ids_s,
                  int n_rows, int n_used, float s1, float b1, float s2, float b2) {
    q->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::float4, 1> sel_l(sycl::range<1>(NE / 4), cgh);
        sycl::local_accessor<float, 1> w_l(sycl::range<1>(ROUTE_MAX_USED), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) n_rows * NE, NE),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             route_kernel<NE>(logits, bias, weights, ids, ids_s, n_used, s1, b1, s2, b2,
                                              (float *) sel_l.get_multi_ptr<sycl::access::decorated::no>().get(),
                                              w_l.get_multi_ptr<sycl::access::decorated::no>().get(), it);
                         });
    });
}

} // namespace

int ggml_sycl_route_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = ggml_sycl_get_env("GGML_SYCL_ROUTE_FUSE", 1) != 0;
    if (!on) {
        return 0;
    }
    ggml_tensor * probs = cgraph->nodes[i];
    if (probs->op != GGML_OP_UNARY || ggml_get_unary_op(probs) != GGML_UNARY_OP_SIGMOID) {
        return 0;
    }
    static const ggml_op ops[] = { GGML_OP_UNARY,   GGML_OP_RESHAPE,  GGML_OP_ADD,   GGML_OP_ARGSORT,
                                   GGML_OP_VIEW,    GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SUM_ROWS,
                                   GGML_OP_SCALE,   GGML_OP_DIV,      GGML_OP_RESHAPE, GGML_OP_SCALE };
    constexpr int        n_ops = (int) (sizeof(ops) / sizeof(ops[0]));
    if (i + n_ops > cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    for (int k = 0; k < n_ops; ++k) {
        if (nd[k]->op != ops[k]) {
            return 0;
        }
    }
    const ggml_tensor * logits = probs->src[0];
    ggml_tensor *       pr     = nd[1];
    ggml_tensor *       add    = nd[2];
    ggml_tensor *       srt    = nd[3];
    ggml_tensor *       topk   = nd[4];
    ggml_tensor *       gr     = nd[5];
    ggml_tensor *       grr    = nd[6];
    ggml_tensor *       sum    = nd[7];
    ggml_tensor *       sc1    = nd[8];
    ggml_tensor *       dv     = nd[9];
    ggml_tensor *       dvr    = nd[10];
    ggml_tensor *       sc2    = nd[11];
    const ggml_tensor * bias   = add->src[0] == probs ? add->src[1] : nullptr;

    const int64_t ne = logits->ne[0];
    const int64_t nr = logits->ne[1];
    if (!bias || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) || ggml_nrows(logits) != nr ||
        nr < 1 || nr > 4 || probs->type != GGML_TYPE_F32 || !ggml_is_contiguous(probs) ||
        bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) || ggml_nelements(bias) != ne ||
        pr->src[0] != probs || srt->src[0] != add || ggml_get_op_params_i32(srt, 0) != GGML_SORT_ORDER_DESC ||
        topk->src[0] != srt || topk->view_offs != 0 || topk->type != GGML_TYPE_I32 || topk->nb[0] != sizeof(int32_t) ||
        topk->ne[1] != nr || gr->src[0] != pr || gr->src[1] != topk || grr->src[0] != gr || sum->src[0] != grr ||
        sc1->src[0] != sum || dv->src[0] != grr || dv->src[1] != sc1 || dvr->src[0] != dv || sc2->src[0] != dvr ||
        sc2->type != GGML_TYPE_F32 || !ggml_is_contiguous(sc2) || grr->ne[0] != topk->ne[0] || grr->ne[1] != nr ||
        sum->ne[0] != 1 || sum->ne[1] != nr) {
        return 0;
    }
    const int n_used = (int) topk->ne[0];
    if (n_used < 1 || n_used > ROUTE_MAX_USED || n_used > ne || ggml_nelements(sc2) != (int64_t) n_used * nr) {
        return 0;
    }
    const int outs[2] = { i + 4, i + 11 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, outs, 2)) {
        return 0;
    }
    // outputs must not alias the inputs read later in the kernel
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    if (overlap(sc2, logits) || overlap(sc2, bias) || overlap(topk, logits) || overlap(topk, bias) ||
        overlap(topk, sc2)) {
        return 0;
    }

    float s1, b1, s2, b2;
    memcpy(&s1, (const float *) sc1->op_params + 0, sizeof(float));
    memcpy(&b1, (const float *) sc1->op_params + 1, sizeof(float));
    memcpy(&s2, (const float *) sc2->op_params + 0, sizeof(float));
    memcpy(&b2, (const float *) sc2->op_params + 1, sizeof(float));

    const float * l  = (const float *) logits->data;
    const float * b  = (const float *) bias->data;
    float *       w  = (float *) sc2->data;
    int32_t *     id = (int32_t *) topk->data;
    const int     is = (int) (topk->nb[1] / sizeof(int32_t));
    sycl::queue * q  = ctx.stream();
    switch (ne) {
        case 64:  route_launch<64>(q, l, b, w, id, is, (int) nr, n_used, s1, b1, s2, b2); break;
        case 128: route_launch<128>(q, l, b, w, id, is, (int) nr, n_used, s1, b1, s2, b2); break;
        case 256: route_launch<256>(q, l, b, w, id, is, (int) nr, n_used, s1, b1, s2, b2); break;
        default:  return 0;
    }
    return n_ops - 1;
}
