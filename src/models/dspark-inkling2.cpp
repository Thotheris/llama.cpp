#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <vector>

// Inkling2 DSpark draft (the mtp.safetensors of Inkling-2-Mini), see vLLM inkling2/nvidia/dspark.py:
//   * the target features are the raw block partials of target layer visits [5, 23, 41] and the final normed
//     hidden state, concatenated per token
//   * draft layer i mixes them with softmax(fuse[i]) and normalizes (dflare_norm), then projects the mix to its
//     context K/V (attn_k_ctx with its own k norm and rope, attn_v_ctx)
//   * a block is [anchor, mask x (block_size - 1)] at the anchor position onward: the anchor is the normalized
//     target embedding of the last sampled token, every mask slot the same mask_embd vector
//   * attention is bidirectional over the context and the block, layers 0 and 2 see sliding_window positions
//     before the block
//   * a recurrent markov head chains the slots greedily: context [hidden, f_rnn] (f_rnn: the softmax(fuse[-1]) mix of
//     the last committed token's features), [state, markov_w1[prev]], gated state update, latent tanh(out), whose
//     hidden_out projection is added to the hidden state before the shared lm head (the reference adds the folded
//     lm_head @ hidden_out to the base logits, the same sum), then the target's final softcap

void llama_model_dflash::load_tensors_inkling2() {
    LLAMA_LOAD_LOCALS;

    const ggml_tensor * w1_meta = ml->get_tensor_meta("markov_w1.weight");
    if (w1_meta == nullptr) {
        throw std::runtime_error("Inkling2 DSpark draft requires markov_w1");
    }
    const int64_t rank        = w1_meta->ne[0];
    const int64_t n_vocab_pad = w1_meta->ne[1]; // the padded vocab of the shared lm head
    const int64_t n_tap       = (int64_t) target_layer_ids.size();

    dspark_markov_w1    = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1,  "weight"), { rank, n_vocab_pad }, 0);
    dspark_joint_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "weight"), { 2*rank + 2*n_embd, 3*rank }, 0);
    dspark_joint_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "bias"),   { 3*rank }, 0);

    ink_hidden_out   = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_OUT, "weight"), { rank, n_embd }, 0);
    ink_hidden_out_b = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_OUT, "bias"),   { n_embd }, 0);
    ink_norm         = create_tensor(tn(LLM_TENSOR_DSPARK_NORM,       "weight"), { n_embd }, 0);
    ink_fuse         = create_tensor(tn(LLM_TENSOR_DSPARK_FUSE,       "weight"), { n_tap, n_layer + 1 }, 0);
    ink_mask         = create_tensor(tn(LLM_TENSOR_DSPARK_MASK_VEC,   "weight"), { n_embd }, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,       "weight"), { n_embd }, 0);

    ink_layers.resize(n_layer);
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        auto & ink   = ink_layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "weight", i), { n_embd }, 0);
        layer.wq          = create_tensor(tn(LLM_TENSOR_ATTN_Q,      "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk          = create_tensor(tn(LLM_TENSOR_ATTN_K,      "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv          = create_tensor(tn(LLM_TENSOR_ATTN_V,      "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", i), { n_embd_head_k * n_head, n_embd }, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);

        ink.wk_ctx      = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_K_CTX,      "weight", i), { n_embd, n_embd_k_gqa }, 0);
        ink.wv_ctx      = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_V_CTX,      "weight", i), { n_embd, n_embd_v_gqa }, 0);
        ink.k_ctx_norm  = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_K_CTX_NORM, "weight", i), { n_embd_head_k }, 0);
        ink.dflare_norm = create_tensor(tn(LLM_TENSOR_DSPARK_DFLARE_NORM,     "weight", i), { n_embd }, 0);
    }

    LLAMA_LOG_INFO("%s: Inkling2 DSpark draft (markov rank = %lld, %lld taps, padded vocab = %lld)\n", __func__,
            (long long) rank, (long long) n_tap, (long long) n_vocab_pad);
}

llama_model_dflash::graph_inkling2::graph_inkling2(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    GGML_ASSERT(use_iswa && "Inkling2 DSpark draft expects sliding-window layers");

    if (ubatch.embd || ubatch.embd_row) {
        build_inject_inkling2();
    } else {
        build_block_inkling2();
    }
}

// the markov head's lm head pass of every draft step, lm_head*(h + hidden_out*latent + hidden_out_b), is split into
// one batched lm_head*h of all steps and a per-step [n_vocab, rank] product with the folded weight (vLLM folds it
// the same way, in bf16); this makes the folded weight once, with the target's lm head on its own device
void llama_model_dflash::init_ctx_other(const llama_model & model_other, const std::vector<ggml_backend_t> & backends) const {
    ggml_tensor * head = model_other.output;
    if (backbone != BACKBONE_INKLING2 || output != nullptr || head == nullptr || model_other.output_s != nullptr ||
        head->buffer == nullptr || ink_fold_head == head || getenv("LLAMA_DSPARK_NO_FOLD")) {
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(head->buffer));
    ggml_backend_t backend = nullptr;
    for (ggml_backend_t b : backends) {
        if (dev != nullptr && ggml_backend_get_device(b) == dev) {
            backend = b;
            break;
        }
    }
    if (backend == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        LLAMA_LOG_INFO("%s: the lm head is not on a GPU, the markov head runs it once per draft step\n", __func__);
        return;
    }

    const int64_t n_vocab = (int64_t) vocab.n_tokens();
    const int64_t rank    = ink_hidden_out->ne[0];
    const int64_t n_embd  = head->ne[0];
    GGML_ASSERT(ink_hidden_out->ne[1] == n_embd && n_vocab <= head->ne[1]);

    {
        ggml_init_params params = { 2*ggml_tensor_overhead(), nullptr, true };
        ink_fold_ctx.reset(ggml_init(params));
        ink_fold_w = ggml_new_tensor_2d(ink_fold_ctx.get(), GGML_TYPE_F16, rank, n_vocab);
        ink_fold_b = ggml_new_tensor_1d(ink_fold_ctx.get(), GGML_TYPE_F32, n_vocab);
        ggml_set_name(ink_fold_w, "dspark_fold.weight");
        ggml_set_name(ink_fold_b, "dspark_fold.bias");
        ink_fold_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ink_fold_ctx.get(), ggml_backend_get_default_buffer_type(backend)));
        if (!ink_fold_buf) {
            throw std::runtime_error("dspark: unable to allocate the folded markov head");
        }
        ggml_backend_buffer_set_usage(ink_fold_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // hidden_out^T [n_embd, rank] in F32, made on the host
    std::vector<float> ho_t((size_t) n_embd*rank);
    {
        GGML_ASSERT(ggml_is_contiguous(ink_hidden_out));
        std::vector<uint8_t> raw(ggml_nbytes(ink_hidden_out));
        ggml_backend_tensor_get(ink_hidden_out, raw.data(), 0, raw.size());
        std::vector<float> ho((size_t) n_embd*rank);
        const auto * traits = ggml_get_type_traits(ink_hidden_out->type);
        if (ink_hidden_out->type == GGML_TYPE_F32) {
            memcpy(ho.data(), raw.data(), raw.size());
        } else {
            GGML_ASSERT(traits->to_float != nullptr);
            traits->to_float(raw.data(), ho.data(), (int64_t) ho.size());
        }
        for (int64_t d = 0; d < n_embd; ++d) {
            for (int64_t j = 0; j < rank; ++j) {
                ho_t[(size_t) j*n_embd + d] = ho[(size_t) d*rank + j];
            }
        }
    }

    ggml_init_params params = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(params) };
    ggml_cgraph * gf = ggml_new_graph(ctx.get());

    ggml_tensor * ho = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, rank);
    ggml_set_input(ho);

    ggml_tensor * lm = ggml_view_2d(ctx.get(), head, n_embd, n_vocab, head->nb[1], 0);

    ggml_tensor * w = ggml_mul_mat(ctx.get(), lm, ho);                   // [n_vocab, rank]
    ggml_tensor * b = ggml_mul_mat(ctx.get(), lm, ink_hidden_out_b);    // [n_vocab, 1]
    ggml_build_forward_expand(gf, ggml_cpy(ctx.get(), ggml_transpose(ctx.get(), w), ink_fold_w));
    ggml_build_forward_expand(gf, ggml_cpy(ctx.get(), ggml_reshape_1d(ctx.get(), b, n_vocab), ink_fold_b));

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(galloc, gf);
    if (ok) {
        ggml_backend_tensor_set(ho, ho_t.data(), 0, ggml_nbytes(ho));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    ggml_gallocr_free(galloc);

    if (!ok) {
        LLAMA_LOG_WARN("%s: failed to fold the markov head, it runs the lm head once per draft step\n", __func__);
        ink_fold_w = ink_fold_b = nullptr;
        ink_fold_buf.reset();
        ink_fold_ctx.reset();
        return;
    }
    ink_fold_head = head;

    LLAMA_LOG_INFO("%s: folded the markov head into the lm head on %s (%.1f MiB)\n", __func__,
            ggml_backend_name(backend), ggml_backend_buffer_get_size(ink_fold_buf.get())/1024.0/1024.0);
}

ggml_tensor * llama_model_dflash::graph_inkling2::mix_taps(ggml_tensor * feat, ggml_tensor * alpha, int r) {
    const int64_t n_tap = feat->ne[1];

    // [1, n_tap, 1] row r of the weights, broadcast over the hidden size and the tokens
    ggml_tensor * w = ggml_view_3d(ctx0, alpha, 1, n_tap, 1, alpha->nb[0], alpha->nb[1], r*alpha->nb[1]);
    ggml_tensor * x = ggml_mul(ctx0, feat, w);

    ggml_tensor * sum = nullptr;
    for (int64_t t = 0; t < n_tap; ++t) {
        ggml_tensor * xt = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], t*x->nb[1]);
        sum = sum ? ggml_add(ctx0, sum, xt) : ggml_cont(ctx0, xt);
    }
    return sum; // [n_embd, n]
}

// embd batch: the target features of committed positions -> context K/V of every draft layer, and the markov
// head's feature mix of the output rows (h_nextn, read back by the runtime as f_rnn)
void llama_model_dflash::graph_inkling2::build_inject_inkling2() {
    const auto & m = static_cast<const llama_model_dflash &>(model);

    const int64_t n_tap      = (int64_t) m.target_layer_ids.size();
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    GGML_ASSERT(n_embd_inp == n_tap*n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);
    ggml_tensor * feat = build_inp_target_features(*this, *inp, n_embd_inp);
    res->add_input(std::move(inp));
    cb(feat, "inp_target_features", -1);

    // one softmax over the taps per row, the mix weights of each draft layer and of the markov head
    ggml_tensor * alpha = ggml_soft_max(ctx0, m.ink_fuse); // [n_tap, n_layer + 1]
    cb(alpha, "dflare_alpha", -1);

    ggml_tensor * feat3 = ggml_reshape_3d(ctx0, feat, n_embd, n_tap, n_tokens);

    for (int il = 0; il < n_layer; ++il) {
        const auto & ink = m.ink_layers[il];

        ggml_tensor * ctx = build_norm(mix_taps(feat3, alpha, il), ink.dflare_norm, nullptr, LLM_NORM_RMS, il);
        cb(ctx, "dflare_ctx", il);

        ggml_tensor * Kcur = build_lora_mm(ink.wk_ctx, ctx);
        ggml_tensor * Vcur = build_lora_mm(ink.wv_ctx, ctx);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = build_norm(Kcur, ink.k_ctx_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Kcur, "Kcur_injected", il);
        cb(Vcur, "Vcur_injected", il);

        const bool    is_swa = hparams.is_swa(il);
        const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
        ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
        ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
        ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa[il] : inp_attn_iswa->self_k_rot[il];
        ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa[il] : inp_attn_iswa->self_v_rot[il];
        if (k_rot) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
        }
        if (v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
        }
        ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
        ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
    }

    // f_rnn of the output rows only (the runtime marks the tail of each sequence)
    if (n_outputs > 0) {
        ggml_tensor * out_ids = build_inp_out_ids();

        ggml_tensor * feat_out = ggml_get_rows(ctx0, feat, out_ids);
        feat_out = ggml_reshape_3d(ctx0, feat_out, n_embd, n_tap, n_outputs);

        ggml_tensor * f_rnn = build_norm(mix_taps(feat_out, alpha, n_layer), m.ink_norm, nullptr, LLM_NORM_RMS, -1);
        cb(f_rnn, "f_rnn", -1);

        res->t_h_nextn = f_rnn;
        ggml_build_forward_expand(gf, f_rnn);
    }
}

// token batch: [anchor, mask...] blocks, one per sequence, all of the same size; the first n_outputs/n_blocks slots
// of each block are its outputs: the anchor slot (unread) and the drafts
void llama_model_dflash::graph_inkling2::build_block_inkling2() {
    const auto & m = static_cast<const llama_model_dflash &>(model);

    GGML_ASSERT(cparams.ctx_other != nullptr && "Inkling2 DSpark draft shares the target's embeddings and lm head");
    const llama_model * model_tgt = llama_get_model(cparams.ctx_other);
    GGML_ASSERT(model_tgt->tok_norm != nullptr && "Inkling2 DSpark draft requires the target's embedding norm");

    const int64_t n_blocks = ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0 && "Inkling2 DSpark draft requires equal-size blocks");
    const int64_t n_block = n_tokens / n_blocks;

    // tokens and the per-token f_rnn state (the same for every token of a block)
    auto inp = std::make_unique<llm_graph_input_embd_h>(n_embd, n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;
    inp->h = build_inp_embd_state(inp.get(), n_embd);
    ggml_tensor * tokens = inp->tokens;
    ggml_tensor * state  = inp->h;
    res->add_input(std::move(inp));
    // the state is read only by the markov chain; keep it in the graph so that it is always allocated
    ggml_build_forward_expand(gf, state);

    // the anchor of every block: a strided view of the tokens
    ggml_tensor * anchors = ggml_cont_1d(ctx0, ggml_view_2d(ctx0, tokens, 1, n_blocks, n_block*tokens->nb[0], 0), n_blocks);

    ggml_tensor * x = ggml_get_rows(ctx0, get_tok_embd(*this, model), anchors);
    x = build_norm(x, model_tgt->tok_norm, nullptr, LLM_NORM_RMS, -1);
    cb(x, "anchor_embd", -1);
    if (n_block > 1) {
        ggml_tensor * masks = ggml_repeat_4d(ctx0, m.ink_mask, n_embd, n_block - 1, n_blocks, 1);
        x = ggml_concat(ctx0, ggml_reshape_3d(ctx0, x, n_embd, 1, n_blocks), masks, 1);
    }
    x = ggml_reshape_2d(ctx0, x, n_embd, n_tokens);
    cb(x, "block_embd", -1);

    // the whole block shares the mask row of its anchor (bidirectional, window of the anchor)
    inp_attn_iswa->self_attn_mask_swa.block_uniform = true;

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = build_norm(x, layer.attn_norm, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * Qcur = build_lora_mm(layer.wq, cur);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        Qcur = build_rope(Qcur, inp_pos, il);
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(inp_attn_iswa, layer.wo, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        x = ggml_add(ctx0, cur, x);
        cb(x, "ffn_inp", il);

        cur = build_norm(x, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        x = ggml_add(ctx0, cur, x);
        cb(x, "l_out", il);
    }

    ggml_tensor * hidden = build_norm(x, model.output_norm, nullptr, LLM_NORM_RMS, -1); // [n_embd, n_tokens]
    cb(hidden, "result_norm", -1);

    const auto [lm_head, lm_head_s] = get_lm_head(*this, model);
    const int64_t n_vocab = (int64_t) model.vocab.n_tokens(); // the unpadded vocab of the drafts
    GGML_ASSERT(n_vocab <= lm_head->ne[1]);
    const float cap = hparams.f_final_logit_softcapping;

    const auto logits_of = [&](ggml_tensor * h) {
        ggml_tensor * l = build_lora_mm(lm_head, h, lm_head_s);
        // drop the padded vocabulary
        l = ggml_cont(ctx0, ggml_view_2d(ctx0, l, n_vocab, l->ne[1], l->nb[1], 0));
        if (cap > 0.0f) {
            l = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, l, 1.0f/cap)), cap);
        }
        return l; // [n_vocab, n_blocks]
    };

    if (n_outputs == 0) {
        return;
    }

    // the outputs: the first n_out slots of every block; anything else (the reserve graphs, a block longer than
    // trained) gets the plain lm head logits of its output rows
    const int64_t n_out = n_outputs % n_blocks == 0 ? n_outputs / n_blocks : 0;
    bool chain = n_out > 0 && n_block <= (int64_t) hparams.dflash_block_size && ubatch.output != nullptr;
    for (int64_t b = 0; chain && b < n_blocks; ++b) {
        for (int64_t i = 0; i < n_block; ++i) {
            chain = chain && (bool) ubatch.output[b*n_block + i] == (i < n_out);
        }
    }
    if (!chain) {
        ggml_tensor * out = logits_of(ggml_get_rows(ctx0, hidden, build_inp_out_ids()));
        cb(out, "result_output", -1);
        res->t_logits = out;
        ggml_build_forward_expand(gf, out);
        return;
    }

    // slot i of every block: a strided view [n_embd, n_blocks]
    const auto slot = [&](ggml_tensor * t, int64_t i) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, t, t->ne[0], n_blocks, n_block*t->nb[1], i*t->nb[1]));
    };

    ggml_tensor * f_rnn = slot(state, 0);
    cb(f_rnn, "f_rnn", -1);

    const int64_t rank = m.dspark_markov_w1->ne[0];

    ggml_tensor * prev = anchors;
    ggml_tensor * hs   = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, n_blocks), 0.0f);

    // with the folded head: lm_head*h of every draft slot in one pass, [n_vocab_padded, n_blocks*(n_out - 1)]
    // slot-major, and lm_head*(hidden_out*latent + hidden_out_b) per step from the fold
    const bool folded = m.ink_fold_w != nullptr && m.ink_fold_head == lm_head && lm_head_s == nullptr;
    ggml_tensor * base = nullptr;
    if (folded) {
        ggml_tensor * hd = ggml_view_3d(ctx0, hidden, n_embd, n_blocks, n_out - 1, n_block*hidden->nb[1], hidden->nb[1], hidden->nb[1]);
        hd = ggml_reshape_2d(ctx0, ggml_cont(ctx0, hd), n_embd, n_blocks*(n_out - 1));
        base = build_lora_mm(lm_head, hd);
        cb(base, "markov_base_logits", -1);
    }

    std::vector<ggml_tensor *> cols;
    std::vector<ggml_tensor *> confs; // [1, n_blocks] per step: the draft's top-1 probability
    for (int64_t i = 1; i < n_out; ++i) {
        ggml_tensor * h_i = slot(hidden, i);

        ggml_tensor * joint = ggml_concat(ctx0, hs, ggml_get_rows(ctx0, m.dspark_markov_w1, prev), 0);
        joint = ggml_concat(ctx0, joint, h_i,   0);
        joint = ggml_concat(ctx0, joint, f_rnn, 0); // [2r + 2 n_embd, n_blocks]

        ggml_tensor * raw = ggml_add(ctx0, ggml_mul_mat(ctx0, m.dspark_joint_proj, joint), m.dspark_joint_proj_b); // [3r, n_blocks]

        const auto chunk = [&](int64_t k) {
            return ggml_view_2d(ctx0, raw, rank, n_blocks, raw->nb[1], k*rank*ggml_element_size(raw));
        };
        ggml_tensor * gate = ggml_sigmoid(ctx0, chunk(0));
        ggml_tensor * cand = ggml_tanh(ctx0, chunk(1));
        ggml_tensor * lat  = ggml_tanh(ctx0, chunk(2));

        // state = gate * state + (1 - gate) * cand
        hs = ggml_add(ctx0, cand, ggml_mul(ctx0, gate, ggml_sub(ctx0, hs, cand)));
        cb(hs, "markov_state", (int) i);

        ggml_tensor * col;
        if (folded) {
            ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base->nb[1], (i - 1)*n_blocks*base->nb[1]);
            col = ggml_add(ctx0, ggml_add(ctx0, ggml_mul_mat(ctx0, m.ink_fold_w, lat), m.ink_fold_b), base_i);
            if (cap > 0.0f) {
                col = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, col, 1.0f/cap)), cap);
            }
        } else {
            // lm_head(h) + lm_head(hidden_out(latent)) as one lm head pass
            ggml_tensor * corr = ggml_add(ctx0, ggml_mul_mat(ctx0, m.ink_hidden_out, lat), m.ink_hidden_out_b);
            col = logits_of(ggml_add(ctx0, h_i, corr));
        }
        cb(col, "markov_logits", (int) i);

        cols.push_back(col);

        // the vulkan argmax is one workgroup per row (0.7 ms over this vocab); top-1 is ~20x faster
        ggml_tensor * top = ggml_top_k(ctx0, col, 1); // [1, n_blocks]

        // its probability, the confidence that truncates an adaptive draft: softmax(col)[top] of every block
        ggml_tensor * sm = ggml_reshape_3d(ctx0, ggml_soft_max(ctx0, col), 1, n_vocab, n_blocks);
        confs.push_back(ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, sm, top), 1, n_blocks));

        if (i + 1 < n_out) {
            prev = ggml_reshape_1d(ctx0, top, n_blocks);
        }
    }

    // the anchor slot is not a draft: repeat the first draft's logits (no extra lm head pass) unless it is alone
    ggml_tensor * cat = cols.empty() ? logits_of(slot(hidden, 0)) : cols[0];
    for (ggml_tensor * col : cols) {
        cat = ggml_concat(ctx0, cat, col, 1);
    }

    // cat is slot-major; restore the block-major output order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, n_out);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, n_out, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_outputs);
    cb(out, "result_output", -1);

    res->t_logits = out;
    ggml_build_forward_expand(gf, out);

    // h_nextn: the confidence of each output row in its first element (the anchor slot reads 1), as the dspark
    // confidence head
    if (!confs.empty()) {
        ggml_tensor * conf = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_blocks), 1.0f);
        for (ggml_tensor * c : confs) {
            conf = ggml_concat(ctx0, conf, c, 1);
        }
        conf = ggml_reshape_3d(ctx0, conf, 1, n_blocks, n_out);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3)); // [1, n_out, n_blocks]
        conf = ggml_pad(ctx0, ggml_reshape_2d(ctx0, conf, 1, n_outputs), n_embd - 1, 0, 0, 0);
        cb(conf, "dspark_conf", -1);
        res->t_h_nextn = conf;
        ggml_build_forward_expand(gf, conf);
    }
}
