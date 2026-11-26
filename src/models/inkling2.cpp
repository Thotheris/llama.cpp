#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    prec_policy.set_acc(output, GGML_PREC_F32);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    attn_in.assign(n_layer_phys(), nullptr);
    attn_qkv_inp_norm.assign(n_layer_phys(), nullptr);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // q, gate, rel and sink project the same input: stacked into attn_in when they can be, else loaded separately
        const bool in_stacked = plan_concat_rows(ml, {
                    { tn(LLM_TENSOR_ATTN_Q,         "weight", i).str(), n_embd_head*n_head },
                    { tn(LLM_TENSOR_ATTN_GATE,      "weight", i).str(), n_embd_head*n_head },
                    { tn(LLM_TENSOR_ATTN_REL,       "weight", i).str(), n_rel*n_head },
                    { tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i).str(), n_head },
                }, n_embd, format("blk.%d.attn_in.weight", i), &attn_in[i], &layer.wk);
        const int in_flags = in_stacked ? TENSOR_SKIP : 0;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, in_flags);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, in_flags);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, in_flags);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, in_flags);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            // and the three as rows of one weight: one norm over the three inputs of a token (the separate ones stay
            // for batches)
            plan_concat_rows(ml, {
                    { tn(LLM_TENSOR_ATTN_Q_INP_NORM, "weight", i).str(), 1 },
                    { tn(LLM_TENSOR_ATTN_K_INP_NORM, "weight", i).str(), 1 },
                    { tn(LLM_TENSOR_ATTN_V_INP_NORM, "weight", i).str(), 1 },
                }, n_embd, format("blk.%d.attn_qkv_inp_norm.weight", i), &attn_qkv_inp_norm[i], &layer.attn_q_inp_norm);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

// ggml does not fold constants: make the weight-only expressions of the graph once, from the loaded weights, with the
// same float ops as the graph computed them, and put each next to its weight (same buffer type)
void llama_model_inkling2::load_arch_derived() {
    const int64_t n_conv = hparams.n_shortconv_l_cache;
    const int64_t n_rel  = hparams.n_relative;

    // the buffer type of a weight, never a CPU extra type (they only accept tensors they can repack)
    auto buft_of = [](const ggml_tensor * w) -> ggml_backend_buffer_type_t {
        if (!w || !w->buffer || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w)) {
            return nullptr;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(w->buffer);
        if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft)) {
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                buft = ggml_backend_dev_buffer_type(dev);
            }
        }
        return buft;
    };

    enum kind_t { CONV, TABLE, SINKS, FLIP };
    struct item {
        kind_t        kind;
        const ggml_tensor * src; // the weight it derives from (the key)
        int           il;
        float         scale;
        ggml_tensor * dst;
    };
    std::map<ggml_backend_buffer_type_t, std::vector<item>> plan;

    for (int il = 0; il < n_layer_phys(); ++il) {
        const auto & layer = layers[il];
        const bool dynamic = !hparams.is_swa(il);

        // the scales of convolve(): K and V of the full-attention layers have input-dependent taps
        const std::pair<const ggml_tensor *, float> convs[] = {
            { layer.attn_k_conv,    1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1) },
            { layer.attn_v_conv,    1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1) },
            { layer.attn_post_conv, 1.0f/sqrtf(n_conv + 1) },
            { layer.ffn_post_conv,  1.0f/sqrtf(n_conv + 1) },
        };
        for (const auto & [w, scale] : convs) {
            if (ggml_backend_buffer_type_t buft = buft_of(w)) {
                plan[buft].push_back({ CONV, w, il, scale, nullptr });
            }
        }
        if (ggml_backend_buffer_type_t buft = buft_of(layer.attn_rel_bias)) {
            plan[buft].push_back({ TABLE, layer.attn_rel_bias, il, 0.0f, nullptr });
            plan[buft].push_back({ SINKS, layer.attn_rel_bias, il, 0.0f, nullptr });
        }
    }
    // one flip per buffer type, shared by its convolutions
    for (auto & [buft, items] : plan) {
        items.push_back({ FLIP, nullptr, -1, 0.0f, nullptr });
    }

    size_t n_bytes = 0;
    for (auto & [buft, items] : plan) {
        ggml_init_params params = {
            /*.mem_size   =*/ items.size()*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("inkling2: failed to create the context of the derived tensors");
        }

        for (auto & it : items) {
            const int64_t n_head = it.il >= 0 ? hparams.n_head(it.il) : 0;
            switch (it.kind) {
                case CONV:  it.dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, it.src->ne[0], it.src->ne[1]);         break;
                case TABLE: it.dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rel + 2,     it.src->ne[1] + 2);     break;
                case SINKS: it.dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);                               break;
                case FLIP:  it.dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_conv);                               break;
            }
            switch (it.kind) {
                case CONV:  ggml_format_name(it.dst, "%s.folded", it.src->name); break;
                case TABLE: ggml_format_name(it.dst, "blk.%d.attn_rel_table", it.il); break;
                case SINKS: ggml_format_name(it.dst, "blk.%d.attn_sinks_zero", it.il); break;
                case FLIP:  ggml_format_name(it.dst, "conv_flip"); break;
            }
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            ggml_free(ctx);
            throw std::runtime_error(format("inkling2: unable to allocate the %s buffer of the derived tensors", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        n_bytes += ggml_backend_buffer_get_size(buf);

        ggml_tensor * flip = nullptr;
        std::vector<float> src;
        std::vector<float> dst;
        for (auto & it : items) {
            if (it.src) {
                src.resize(ggml_nelements(it.src));
                ggml_backend_tensor_get(it.src, src.data(), 0, ggml_nbytes(it.src));
            }
            dst.assign(ggml_nelements(it.dst), 0.0f);
            switch (it.kind) {
                case CONV: {
                    // conv_weight(): (w + e_last)*scale, an ADD of the one-hot last tap then a SCALE
                    for (int64_t c = 0; c < it.src->ne[1]; ++c) {
                        for (int64_t k = 0; k < n_conv; ++k) {
                            const float e = k == n_conv - 1 ? 1.0f : 0.0f;
                            const float s = src[c*n_conv + k] + e;
                            dst[c*n_conv + k] = s*it.scale;
                        }
                    }
                } break;
                case TABLE: {
                    // concat(concat(attn_rel_bias, rel_rows, 0), rel_cols, 1): each distance column is [bias, -1, 0],
                    // then the column past the extent [0, ..., -1, 0] and the masked column [0, ..., 0, -inf]
                    const int64_t n_extent = it.src->ne[1];
                    const int64_t ld       = n_rel + 2;
                    for (int64_t j = 0; j < n_extent; ++j) {
                        for (int64_t i = 0; i < n_rel; ++i) {
                            dst[j*ld + i] = src[j*n_rel + i];
                        }
                        dst[j*ld + n_rel]     = -1.0f;
                        dst[j*ld + n_rel + 1] =  0.0f;
                    }
                    // rel_cols(): the sum of the two padded fills
                    dst[n_extent*ld + n_rel]           = -1.0f + 0.0f;
                    dst[n_extent*ld + n_rel + 1]       =  0.0f + 0.0f;
                    dst[(n_extent + 1)*ld + n_rel]     =  0.0f + 0.0f;
                    dst[(n_extent + 1)*ld + n_rel + 1] =  0.0f + -INFINITY;
                } break;
                case SINKS: break;
                case FLIP: {
                    std::vector<int32_t> idx(n_conv);
                    for (int64_t k = 0; k < n_conv; ++k) {
                        idx[k] = (int32_t) (n_conv - 1 - k);
                    }
                    ggml_backend_tensor_set(it.dst, idx.data(), 0, ggml_nbytes(it.dst));
                    flip = it.dst;
                    continue;
                }
            }
            ggml_backend_tensor_set(it.dst, dst.data(), 0, ggml_nbytes(it.dst));
        }

        for (auto & it : items) {
            switch (it.kind) {
                case CONV:  derived_convs[it.src] = { it.dst, it.scale, flip }; break;
                case TABLE: derived_rels[it.src].table = it.dst; break;
                case SINKS: derived_rels[it.src].sinks = it.dst; break;
                case FLIP:  break;
            }
        }

        add_derived_buffer(ctx, buf);
    }

    LLAMA_LOG_INFO("%s: %zu derived convolution taps and %zu relative-bias tables, %.2f MiB\n", __func__,
            derived_convs.size(), derived_rels.size(), n_bytes/1024.0/1024.0);
}

bool llama_model_inkling2::plan_concat_rows(llama_model_loader & ml, const std::vector<std::pair<std::string, int64_t>> & parts,
        int64_t ne0, const std::string & name, ggml_tensor ** dst, ggml_tensor * const * buft_ref) {
    concat_rows_job job = { {}, name, GGML_TYPE_COUNT, ne0, 0, dst, buft_ref };
    for (const auto & [part, rows] : parts) {
        const ggml_tensor * meta = ml.get_tensor_meta(part.c_str());
        if (!meta || meta->ne[0] != ne0 || meta->ne[1] != rows || meta->ne[2] != 1 || meta->ne[3] != 1 ||
            (job.type != GGML_TYPE_COUNT && meta->type != job.type)) {
            LLAMA_LOG_INFO("%s: %s: parts differ in type or shape, loading them separately\n", __func__, name.c_str());
            return false;
        }
        job.type = meta->type;
        job.ne1 += rows;
        job.parts.push_back(part);
    }
    concat_jobs.push_back(std::move(job));
    return true;
}

// the quantized rows of a 2D tensor are contiguous and independent, so a row concatenation is a byte concatenation
void llama_model_inkling2::load_arch_tensors_derived(llama_model_loader & ml) {
    if (concat_jobs.empty()) {
        return;
    }

    // one context and buffer per buffer type, the one of a tensor of the same layer (never a CPU extra type)
    std::map<ggml_backend_buffer_type_t, std::vector<concat_rows_job *>> groups;
    for (auto & job : concat_jobs) {
        ggml_backend_buffer_type_t buft = *job.buft_ref ? weight_buft(*job.buft_ref) : nullptr;
        if (!buft) {
            throw std::runtime_error(format("inkling2: no buffer type for %s", job.name.c_str()));
        }
        if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft)) {
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                buft = ggml_backend_dev_buffer_type(dev);
            }
        }
        groups[buft].push_back(&job);
    }

    size_t n_bytes = 0;
    std::vector<uint8_t> data;
    for (auto & [buft, jobs] : groups) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*jobs.size(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("inkling2: failed to create the context of the stacked weights");
        }
        for (auto * job : jobs) {
            *job->dst = ggml_new_tensor_2d(ctx, job->type, job->ne0, job->ne1);
            ggml_set_name(*job->dst, job->name.c_str());
        }
        alloc_derived_tensors(ctx, buft, ml.no_alloc);

        for (auto * job : jobs) {
            ggml_tensor * t = *job->dst;
            n_bytes += ggml_nbytes(t);
            if (ml.no_alloc) {
                continue;
            }
            data.resize(ggml_nbytes(t));
            size_t offs = 0;
            for (const auto & part : job->parts) {
                const auto * w = ml.get_weight(part.c_str());
                GGML_ASSERT(w);
                const size_t n = ggml_nbytes(w->tensor);
                GGML_ASSERT(offs + n <= data.size());
                ml.files.at(w->idx)->seek(w->offs, SEEK_SET);
                ml.files.at(w->idx)->read_raw(data.data() + offs, n);
                offs += n;
            }
            GGML_ASSERT(offs == data.size());
            ggml_backend_tensor_set(t, data.data(), 0, data.size());
        }
    }

    LLAMA_LOG_INFO("%s: %zu row-stacked projection weight(s), %.2f MiB, replace their separately stored parts\n",
            __func__, concat_jobs.size(), n_bytes/1024.0/1024.0);
    concat_jobs.clear();
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale, as one op:
// cur*scale * clamp(threshold / (||cur*scale||_2 + 1e-6), 0, 1)
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    return ggml_l2_clip(ctx0, cur, scale, threshold);
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
// the RMS norm is per row (one state of one token), so each state is normalised once per graph
ggml_tensor * llama_model_inkling2::graph::mix_norm(ggml_tensor * x) {
    auto it = mix_norms.find(x);
    if (it != mix_norms.end()) {
        return it->second;
    }
    ggml_tensor * normed = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens), 1e-6f);
    mix_norms.emplace(x, normed);
    return normed;
}

// the normalised blocks, latest first, [n_embd, n_blocks, n_tokens]; a list extends a cached shorter one by one concat
ggml_tensor * llama_model_inkling2::graph::mix_stack(const std::vector<ggml_tensor *> & blocks) {
    if (blocks.empty()) {
        return nullptr;
    }
    auto it = mix_stacks.find(blocks);
    if (it != mix_stacks.end()) {
        return it->second;
    }
    ggml_tensor * earlier = mix_stack(std::vector<ggml_tensor *>(blocks.begin(), blocks.end() - 1));
    ggml_tensor * stacked = mix_norm(blocks.back());
    if (earlier) {
        stacked = ggml_concat(ctx0, stacked, earlier, 1);
    }
    mix_stacks.emplace(blocks, stacked);
    return stacked;
}

ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    // states [block_{n-1}, ..., block_0, cur], each row RMS-normalised; the stack and cur are separate sources, no concat
    // out[e, q, t] = sum_s softmax_s(states[:, s, t].query[:, q]/sqrt(n_embd))*states[e, s, t], one op
    return ggml_block_mix(ctx0, mix_norm(cur), mix_stack(blocks), query, 1.0f/sqrtf(n_embd));
}

// up to a matrix-vector batch (LLAMA_INKLING2_MIX_NORM=0 disables it), and past one (LLAMA_INKLING2_MIX_NORM_BATCH=0
// disables that): the norm of cur and the norm after the mix are then no passes of their own over the states (a batch
// rounds the normed cur as the decode step does, norm scale times the raw dot)
// A speculative verify of 2..GGML_SYCL_ROW_EXACT_MAX (8; above 8 the gate_mix_mv form below is not taken, so the rows
// are not row-exact: see build_gates) tokens: every row an output in a context that can roll back,
// the batch whose attention build_attn_mha makes row-exact. It builds the one-token form of the graph where the batched
// one rounds differently, for a backend that runs each column of its weight products as a one-token decode
// (GGML_SYCL_ROW_EXACT in ggml-sycl.cpp), so that every row is the decode of its token alone.
// GGML_SYCL_ROW_EXACT=0: the batched form.
bool llama_model_inkling2::graph::row_exact_verify() const {
    static const bool enabled = !getenv("GGML_SYCL_ROW_EXACT") || atoi(getenv("GGML_SYCL_ROW_EXACT")) != 0;
    static const int  max_n   = getenv("GGML_SYCL_ROW_EXACT_MAX") ? atoi(getenv("GGML_SYCL_ROW_EXACT_MAX")) : 8;
    // (a batch of one token per sequence is no verify: the attention of llama-graph.cpp is not row-exact then)
    return enabled && cparams.n_rs_seq > 0 && n_tokens > 1 && n_tokens <= max_n && n_outputs == n_tokens &&
           ubatch.n_tokens > ubatch.n_seqs_unq;
}

bool llama_model_inkling2::graph::mix_fused() const {
    static const bool disabled = getenv("LLAMA_INKLING2_MIX_NORM") && atoi(getenv("LLAMA_INKLING2_MIX_NORM")) == 0;
    static const bool batch_off = getenv("LLAMA_INKLING2_MIX_NORM_BATCH") && atoi(getenv("LLAMA_INKLING2_MIX_NORM_BATCH")) == 0;
    return !disabled && (n_tokens <= 8 || !batch_off);
}

ggml_tensor * llama_model_inkling2::graph::mix_normed(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query,
        ggml_tensor * out_w, float eps_out, ggml_tensor ** raw) {
    // the cached norm of cur when there is one, else the op normalises the raw cur (eps of mix_norm)
    ggml_tensor * stack = mix_stack(blocks);
    auto it = mix_norms.find(cur);
    const bool cached = it != mix_norms.end();
    ggml_tensor * c = cached ? it->second : ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    ggml_tensor * out = ggml_block_mix_norm(ctx0, c, stack, query, 1.0f/sqrtf(n_embd), cached ? -1.0f : 1e-6f, out_w, eps_out, raw != nullptr);
    if (raw) {
        *raw = ggml_view_3d(ctx0, out, n_embd, out->ne[1], out->ne[2], out->nb[1], out->nb[2], out->nb[3]);
        out  = ggml_view_3d(ctx0, out, n_embd, out->ne[1], out->ne[2], out->nb[1], out->nb[2], 0);
    }
    return out;
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
// post: an L2 clip (threshold post_param) or RMS norm (eps post_param) of each row of `group` channels of the result
// residual: residual + result (no post-op), the addend of the convolution op where it can be
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset,
        ggml_ssm_conv_post post,
        float post_param,
        int64_t group,
        ggml_tensor * residual) {
    GGML_ASSERT(!residual || (!dynamic && post == GGML_SSM_CONV_POST_NONE));
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * xs = ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs);

    // rollback mode keeps the states of the last Kw tokens (rollback slots 0 ... Kw - 1); a single-token step has
    // no earlier state to keep. The convolution op writes them all (its store has Kw slots), unless
    // LLAMA_INKLING2_ROLLBACK_SET_ROWS=1: then it writes slot 0 and the others are windows of x stored by SET_ROWS
    const bool rollback = cparams.n_rs_seq > 0 && n_seq_tokens > 1;
    static const bool rollback_set_rows = getenv("LLAMA_INKLING2_ROLLBACK_SET_ROWS") && atoi(getenv("LLAMA_INKLING2_ROLLBACK_SET_ROWS")) != 0;
    const int64_t n_store_slots = rollback && !rollback_set_rows ? std::min<int64_t>(n_seq_tokens, (int64_t) cparams.n_rs_seq + 1) : 1;

    // the window [past | cur^T]: only the input-dependent taps and the SET_ROWS rollback store need it as a tensor
    ggml_tensor * x = nullptr;
    if (dynamic || (rollback && rollback_set_rows)) {
        x = ggml_concat(ctx0, past, ggml_transpose(ctx0, xs), 0);
    }

    // the residual and the output scale are folded into the taps: s*(cur + conv(x, w)) = conv(x, s*(w + e_last))
    const float scale = 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1);

    // the input-dependent taps as one window product: one sequence, or one token per sequence (a multi-sequence
    // decode step), where the tokens are evenly strided in x; other multi-sequence batches keep the loop below
    const bool dyn_window = dynamic && (n_seqs == 1 || n_seq_tokens == 1);

    // the input-dependent taps, the post-op and the convolution are one op (with or without rollback)
    const bool fuse = !dynamic || dyn_window;

    // the term of the input-dependent taps, [n_channels, n_tokens]
    ggml_tensor * dyn_term = nullptr;
    if (dyn_window) {
        const int64_t n_heads     = dynamic->ne[0]/n_conv;
        const int64_t n_embd_head = n_channels/n_heads;

        // the tap of lag multiplies x[t + n_conv - 1 - lag]: flip the taps to the order of the window, u = n_conv - 1 - lag
        // [1, n_conv, n_tokens, n_heads] -> [n_conv, 1, n_heads, n_tokens]
        ggml_tensor * dyn = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        ggml_tensor * taps = ggml_view_4d(ctx0, dyn, 1, n_conv, n_tokens, n_heads, dyn->nb[0], dyn->nb[2], dyn->nb[1], 0);
        ggml_tensor * flip = ggml_view_3d(ctx0, conv_flip(weight), n_conv, n_tokens, n_heads, 0, 0, 0);

        taps = ggml_scale(ctx0, ggml_get_rows_ext(ctx0, taps, flip, GGML_TYPE_F32), scale);
        taps = ggml_view_4d(ctx0, taps, n_conv, 1, n_heads, n_tokens, taps->nb[2], taps->nb[3], taps->nb[2], 0);

        // the sliding windows of x, [n_conv, n_embd_head, n_heads, n_tokens]: one multiply and a sum over the window;
        // the next token's window starts one column later in its sequence, or at the next sequence
        ggml_tensor * window = ggml_view_4d(ctx0, x, n_conv, n_embd_head, n_heads, n_tokens,
                x->nb[1], n_embd_head*x->nb[1], n_seqs == 1 ? x->nb[0] : x->nb[2], 0);

        dyn_term = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, window, taps)), n_channels, n_tokens);
    }

    if (rollback && rollback_set_rows) {
        store_conv_rollback(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));
    }

    ggml_tensor * out;
    {
        // one op: the convolution of the window, and its last n_conv - 1 columns into the cells of the ubatch,
        // the same values and cells as build_conv_state_store; past reads the gathered copy, not the cells
        // slot k of the rollback slots (the layout of store_conv_rollback) is k*get_size() cells further
        const int64_t n_cols   = n_conv - 1;
        const size_t  row_size = state->nb[1];
        ggml_tensor * store = ggml_view_4d(ctx0, state, n_cols, n_channels, n_seqs, n_store_slots,
                n_cols*ggml_element_size(state), row_size, (size_t) inp->mctx->get_size()*row_size,
                inp->mctx->get_head()*row_size + offset*n_cols*ggml_element_size(state));
        // the residual sum is commutative: residual + conv rounds as the conv op's addend conv + residual
        if (fuse && (dyn_term || residual || post != GGML_SSM_CONV_POST_NONE)) {
            ggml_tensor * add = dyn_term ? ggml_reshape_3d(ctx0, dyn_term, n_channels, n_seq_tokens, n_seqs) :
                                residual ? ggml_reshape_3d(ctx0, residual, n_channels, n_seq_tokens, n_seqs) : nullptr;
            out = ggml_ssm_conv_hist_post(ctx0, past, xs, conv_weight(weight, scale), store, add, post, post_param, (int) group);
            return ggml_reshape_2d(ctx0, out, n_channels, n_tokens);
        }
        out = ggml_ssm_conv_hist(ctx0, past, xs, conv_weight(weight, scale), store);
    }
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dyn_term) {
        out = ggml_add(ctx0, out, dyn_term);
    } else if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        // scaled once while contiguous: a strided SCALE of each tap is not supported on every backend (a CPU split each)
        dynamic = ggml_scale(ctx0, ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens), scale);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    if (post != GGML_SSM_CONV_POST_NONE) {
        out = ggml_reshape_3d(ctx0, out, group, n_channels/group, n_tokens);
        out = post == GGML_SSM_CONV_POST_L2_CLIP ? clip(out, nullptr, post_param) : ggml_rms_norm(ctx0, out, post_param);
        out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);
    }

    if (residual) {
        out = ggml_add(ctx0, residual, out);
    }

    return out;
}

// [TAG_RECURRENT_ROLLBACK_SPLITS] the rollback slots of build_conv_state_store, without slot 0: the fused conv writes
// the final state (the cells of the ubatch, which are slot 0) itself. A rollback of k tokens reads slot k: window w
// of x starts at column n - (Kw - 1) + w and is slot Kw - 1 - w, so the windows 0 ... Kw - 2 go to the slots
// Kw - 1 ... 1. The slot order is built once per graph.
void llama_model_inkling2::graph::store_conv_rollback(llm_graph_input_rs * inp, ggml_tensor * state, ggml_tensor * x, int64_t n_cols, size_t offs) {
    const int64_t n        = x->ne[0] - n_cols; // n_seq_tokens
    const int64_t channels = x->ne[1];
    const int64_t n_seqs   = x->ne[2];

    const int64_t K  = (int64_t) cparams.n_rs_seq + 1;
    const int64_t Kw = std::min(n, K);
    if (Kw < 2) {
        return;
    }

    if (!rs_slot_rev_t) {
        // {K-1, ..., 0}; its size never depends on the ubatch, so the reserved graph covers every tail
        rs_slot_rev_t = ggml_arange(ctx0, 0.0f, (float) K, 1.0f);
        rs_slot_rev_t = ggml_scale_bias(ctx0, rs_slot_rev_t, -1.0f, (float) (K - 1));
        rs_slot_rev_t = ggml_cast(ctx0, rs_slot_rev_t, GGML_TYPE_I32);
        cb(rs_slot_rev_t, "s_slot_rev", -1);
    }

    // {Kw-1, ..., 1}
    ggml_tensor * slot_order = ggml_view_1d(ctx0, rs_slot_rev_t, Kw - 1, ggml_row_size(GGML_TYPE_I32, K - Kw));

    ggml_tensor * windows = ggml_view_4d(ctx0, x,
            n_cols, Kw - 1, channels, n_seqs,
            ggml_element_size(x), x->nb[1], x->nb[2],
            ggml_row_size(x->type, n - (Kw - 1)));

    const size_t row_size = state->nb[1];
    ggml_tensor * slots = ggml_view_4d(ctx0, state,
            n_cols, K, channels, n_seqs,
            (size_t) inp->mctx->get_size()*row_size, ggml_row_size(state->type, n_cols), row_size,
            inp->mctx->get_head()*row_size + offs);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, slots, windows, slot_order));
}

// s*(w + e_last) for the taps w [n_conv, n_channels], once per graph and weight
ggml_tensor * llama_model_inkling2::graph::conv_weight(ggml_tensor * weight, float scale) {
    const auto & derived = static_cast<const llama_model_inkling2 &>(model).derived_convs;
    if (auto it = derived.find(weight); it != derived.end() && it->second.scale == scale) {
        return it->second.taps;
    }
    ggml_tensor *& res = conv_weights[{weight, scale}];
    if (!res) {
        const int n_conv = hparams.n_shortconv_l_cache;
        if (!conv_last_t) {
            conv_last_t = ggml_pad_ext(ctx0, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 1), 1.0f), n_conv - 1, 0, 0, 0, 0, 0, 0, 0);
        }
        res = ggml_scale(ctx0, ggml_add(ctx0, weight, conv_last_t), scale);
    }
    return res;
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
// cur_norm and message_norm are the row RMS norms (eps 1e-6) of cur and message, [n_embd, 1, n_tokens], possibly strided views
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * cur_norm, ggml_tensor * message, ggml_tensor * message_norm,
        const loop_layer & loop, int ctrl, ggml_tensor * gate) {
    if (!gate) {
        ggml_tensor * down = loop.ctrl_down[ctrl];
        if (!down) {
            // a row view of the stacked q, k, v weight: the same matrix
            const int64_t rank = loop.ctrl_down_qkv->ne[1]/3;
            down = ggml_view_2d(ctx0, loop.ctrl_down_qkv, 2*n_embd, rank, loop.ctrl_down_qkv->nb[1], (ctrl - LOOP_CTRL_Q)*rank*loop.ctrl_down_qkv->nb[1]);
        }
        gate = ggml_reshape_2d(ctx0, ggml_concat(ctx0, cur_norm, message_norm, 0), 2*n_embd, n_tokens);
        gate = build_lora_mm(down, gate);
    }

    ggml_tensor * gain_forget = loop.ctrl_gain_forget[ctrl];
    if (!gain_forget && loop.ctrl_gain_forget_qkv && ctrl != LOOP_CTRL_FFN) {
        // a row view of the stacked q, k, v weight: the same matrix
        const ggml_tensor * w = loop.ctrl_gain_forget_qkv;
        gain_forget = ggml_view_2d(ctx0, loop.ctrl_gain_forget_qkv, w->ne[0], 2*n_embd, w->nb[1], (ctrl - LOOP_CTRL_Q)*2*n_embd*w->nb[1]);
    }

    if (gain_forget && n_tokens <= 8 && loras->empty()) {
        // a matrix-vector batch: the stacked gain|forget projection and the blend as one op, with the rounding of the
        // mul_mat_vec and GATE_MIX below
        return ggml_gate_mix_mv(ctx0, cur, message, gain_forget, gate);
    }

    if (gain_forget) {
        // one matmul over the stacked rows [gain | forget], then the tanh and the blend as one op:
        // (x - tanh(forget)*x) + tanh(gain)*m with the rounding of the unfused chain below
        return ggml_gate_mix(ctx0, cur, message, build_lora_mm(gain_forget, gate), true);
    }

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

// [2, n_extent]: the rows of the bias table that take -t and the 1 of [r, t, 1]
ggml_tensor * llama_model_inkling2::graph::rel_rows(int64_t n_extent) {
    ggml_tensor *& rows = n_extent == hparams.n_swa ? rel_rows_swa : rel_rows_full;
    if (!rows || rows->ne[1] != n_extent) {
        rows = ggml_pad(ctx0, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_extent), -1.0f), 1, 0, 0, 0);
    }
    return rows;
}

// [n_relative + 2, 2]: the columns of the bias table for the distances past the extent (-t) and the masked cells (-inf)
ggml_tensor * llama_model_inkling2::graph::rel_cols() {
    if (!rel_cols_t) {
        const int n_rel = hparams.n_relative;

        ggml_tensor * far  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 1), -1.0f);
        ggml_tensor * drop = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 1), -INFINITY);

        rel_cols_t = ggml_add(ctx0,
                ggml_pad_ext(ctx0, far,  n_rel,     1, 0, 1, 0, 0, 0, 0),
                ggml_pad_ext(ctx0, drop, n_rel + 1, 0, 1, 0, 0, 0, 0, 0));
    }
    return rel_cols_t;
}

// I32 [n_shortconv_l_cache]: n - 1, ..., 1, 0
ggml_tensor * llama_model_inkling2::graph::conv_flip(const ggml_tensor * weight) {
    const auto & derived = static_cast<const llama_model_inkling2 &>(model).derived_convs;
    if (auto it = derived.find(weight); it != derived.end() && it->second.flip) {
        return it->second.flip;
    }
    if (!conv_flip_t) {
        const int n_conv = hparams.n_shortconv_l_cache;
        conv_flip_t = ggml_argsort(ctx0, ggml_arange(ctx0, 0.0f, (float) n_conv, 1.0f), GGML_SORT_ORDER_DESC);
    }
    return conv_flip_t;
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop, ggml_tensor * messages_norm) {
    GGML_ASSERT(!messages == !loop);

    tap_layer_inp(n_visit++, cur);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // row j of x [n_embd, n, n_tokens] as a strided view, [n_embd, n_tokens] or [n_embd, 1, n_tokens]
    auto row_2d = [&](ggml_tensor * x, int j) {
        return ggml_view_2d(ctx0, x, n_embd, n_tokens, x->nb[2], j*x->nb[1]);
    };
    auto row_3d = [&](ggml_tensor * x, int j) {
        return ggml_view_3d(ctx0, x, n_embd, 1, n_tokens, x->nb[1], x->nb[2], j*x->nb[1]);
    };

    // the messages and their row norms, shared by the controllers of this layer
    if (messages && !messages_norm) {
        messages_norm = ggml_rms_norm(ctx0, messages, 1e-6f);
    }

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    ggml_tensor * mixed     = nullptr;
    // one token of a full-attention layer: the [n_embd, 3] tensor whose rows are the q, k and v inputs
    ggml_tensor * rows3     = nullptr;
    // the attention input norms done by the mix op: the input norm (no loop), or the plain row norm of the
    // controllers (loop), inputs_norm
    bool          normed_in   = false;
    ggml_tensor * inputs_norm = nullptr;
    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    const bool    join        = ip % hparams.attn_res_block_size != 0;
    if (layer.attn_res_query && mix_fused()) {
        // one token only: with more, the q/k/v inputs would be strided row views of the [n_embd, 3, n_tokens]
        // result, and a non-contiguous B makes the mat-vec skip its q8_1 (MMVQ) path, unlike the contiguous
        // RMS_NORM*MUL output of the unfused chain (a different rounding)
        // a batch past the matrix-vector sizes reads strided rows either way (the matmuls convert their B)
        // a speculative verify (row_exact_verify) takes it too: its products then run each column as the decode does
        const bool qkv_w = !is_swa && static_cast<const llama_model_inkling2 &>(model).attn_qkv_inp_norm[ip] && loras->empty() &&
                (n_tokens == 1 || n_tokens > 8 || row_exact_verify());
        if (messages) {
            inputs_norm = mix_normed(blocks, cur, layer.attn_res_query, nullptr, 1e-6f, &mixed);
        } else if (is_swa || qkv_w) {
            mixed = mix_normed(blocks, cur, layer.attn_res_query,
                    is_swa ? layer.attn_norm : static_cast<const llama_model_inkling2 &>(model).attn_qkv_inp_norm[ip], hparams.f_norm_rms_eps);
            normed_in = true;
        } else {
            mixed = mix(blocks, cur, layer.attn_res_query);
        }
        if (!join) {
            // the stack of the later mixes holds the norm of this input: beside the mix op, not after it
            ggml_build_forward_expand(gf, mix_norm(cur));
        }
        for (int j = 0; j < 3; ++j) {
            inputs[j] = row_2d(mixed, is_swa ? 0 : j);
        }
    } else if (layer.attn_res_query) {
        mixed = mix(blocks, cur, layer.attn_res_query);
        // strided row views: the norms and the controllers below read strides, no copy is needed
        for (int j = 0; j < 3; ++j) {
            inputs[j] = row_2d(mixed, is_swa ? 0 : j);
        }
        if (!is_swa && n_tokens == 1) {
            rows3 = mixed;
        }
    }

    if (messages) {
        // one norm over all the mixed inputs (a SWA layer has a single one)
        if (!inputs_norm) {
            inputs_norm = ggml_rms_norm(ctx0, mixed ? mixed : ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), 1e-6f);
        }

        // one token: the gates of the three controllers as one batched matrix-vector product over the stacked weight,
        // the gate inputs [input norm | message norm] of the three as one concat (a SWA layer repeats its input norm
        // by a zero stride); each gate column is the one of the separate matmul
        ggml_tensor * gates = nullptr;
        if (loop->ctrl_down_qkv && n_tokens == 1 && loras->empty()) {
            const int64_t rank = loop->ctrl_down_qkv->ne[1]/3;
            ggml_tensor * in_n  = ggml_view_2d(ctx0, inputs_norm, n_embd, 3, mixed && !is_swa ? inputs_norm->nb[1] : 0, 0);
            ggml_tensor * msg_n = ggml_view_2d(ctx0, messages_norm, n_embd, 3, messages_norm->nb[1], LOOP_CTRL_Q*messages_norm->nb[1]);
            ggml_tensor * g_in  = ggml_reshape_3d(ctx0, ggml_concat(ctx0, in_n, msg_n, 0), 2*n_embd, 1, 3);
            gates = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, loop->ctrl_down_qkv, 2*n_embd, rank, 3), g_in);
        }

        if (gates && loop->ctrl_gain_forget_qkv) {
            // and the three blends as one GATE_MIX_MV, each row with its controller's weight: [n_embd, 3]
            const ggml_tensor * w = loop->ctrl_gain_forget_qkv;
            ggml_tensor * x   = ggml_view_2d(ctx0, mixed ? mixed : cur, n_embd, 3, mixed && !is_swa ? mixed->nb[1] : 0, 0);
            ggml_tensor * msg = ggml_view_2d(ctx0, messages, n_embd, 3, messages->nb[1], LOOP_CTRL_Q*messages->nb[1]);
            ggml_tensor * gf  = ggml_view_3d(ctx0, loop->ctrl_gain_forget_qkv, w->ne[0], 2*n_embd, 3, w->nb[1], 2*n_embd*w->nb[1], 0);
            ggml_tensor * mixd = ggml_gate_mix_mv(ctx0, x, msg, gf, ggml_view_2d(ctx0, gates, gates->ne[0], 3, gates->nb[2], 0));
            rows3 = is_swa ? nullptr : mixd;
            if (is_swa) {
                // the three SWA inputs share the norm weight: one row norm over the three
                mixd = build_norm(mixd, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            }
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(ctx0, mixd, n_embd, 1, mixd->nb[1], j*mixd->nb[1]);
            }
        } else {
            rows3 = nullptr;
            // the messages make the Q, K and V inputs of a SWA layer differ
            for (int j = 0; j < 3; ++j) {
                const int ctrl = LOOP_CTRL_Q + j;
                inputs[j] = control(inputs[j], row_3d(inputs_norm, mixed && !is_swa ? j : 0),
                        row_2d(messages, ctrl), row_3d(messages_norm, ctrl), *loop, ctrl,
                        gates ? ggml_view_2d(ctx0, gates, gates->ne[0], 1, gates->nb[1], j*gates->nb[2]) : nullptr);
            }
            if (is_swa) {
                for (int j = 0; j < 3; ++j) {
                    inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
                }
            }
        }
    }

    if (is_swa && messages) {
        // normed above
    } else if (normed_in) {
        // normed by the mix op: the rows of its result, contiguous rows of one tensor as the norm's would be
        if (is_swa) {
            inputs[0] = inputs[1] = inputs[2] = ggml_reshape_2d(ctx0, mixed, n_embd, n_tokens);
        } else {
            for (int j = 0; j < 3; ++j) {
                // one token: row views of the contiguous [n_embd, 3] result (the mat-vecs read one q8_1 quantization of it)
                inputs[j] = n_tokens == 1 ? ggml_view_2d(ctx0, mixed, n_embd, 1, mixed->nb[1], j*mixed->nb[1]) : row_2d(mixed, j);
            }
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else if (rows3 && static_cast<const llama_model_inkling2 &>(model).attn_qkv_inp_norm[ip] && loras->empty()) {
        // one row norm over the three inputs with the stacked weights (each row as its separate norm), and the
        // inputs as row views of the result: the mat-vecs read one q8_1 quantization of it
        ggml_tensor * normed = build_norm(ggml_reshape_2d(ctx0, rows3, n_embd, 3),
                static_cast<const llama_model_inkling2 &>(model).attn_qkv_inp_norm[ip], nullptr, LLM_NORM_RMS, il);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_view_2d(ctx0, normed, n_embd, 1, normed->nb[1], j*normed->nb[1]);
        }
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        // q, the output gate, r and the sink logit project inputs[0]: one matmul over the stacked weight when there is one
        const int64_t n_rel = hparams.n_relative;
        const int64_t nq    = n_embd_head*n_head;
        ggml_tensor * Qcur  = nullptr;
        ggml_tensor * Rcur  = nullptr;
        ggml_tensor * sinks = nullptr;
        ggml_tensor * gate  = nullptr;
        // up to a matrix-vector batch the rows of the stacked matmul are computed as the separate ones. Up to 128
        // tokens the stacked matmul is also used: one wide matmul fills the GPU better than four narrow ones (B70:
        // +4% at 24 tokens, never slower), it only changes which tile/split-k each row gets, i.e. rounding. A large
        // batch multiplies row views of the stacked weight as before (same speed, the rounding of the separate
        // weights). LLAMA_INKLING2_ATTN_IN_MAX=<n> moves the limit.
        ggml_tensor * w_in = static_cast<const llama_model_inkling2 &>(model).attn_in[ip];
        static const int attn_in_mv_max = getenv("LLAMA_INKLING2_ATTN_IN_MAX") ? atoi(getenv("LLAMA_INKLING2_ATTN_IN_MAX")) : 128;
        if (w_in && n_tokens <= attn_in_mv_max) {
            ggml_tensor * y  = build_lora_mm(w_in, inputs[0]); // [q | gate | r | sink, n_tokens]
            const size_t  es = ggml_element_size(y);

            Qcur  = ggml_view_3d(ctx0, y, n_embd_head, n_head, n_tokens, n_embd_head*es, y->nb[1], 0);
            gate  = ggml_view_2d(ctx0, y, nq,                  n_tokens, y->nb[1], nq*es);
            Rcur  = ggml_view_3d(ctx0, y, n_rel,       n_head, n_tokens, n_rel*es, y->nb[1], 2*nq*es);
            sinks = ggml_view_3d(ctx0, y, 1,           n_head, n_tokens, es,       y->nb[1], (2*nq + n_rel*n_head)*es);
        } else {
            // the separate weights, or row views of the stacked one
            auto rows = [&](ggml_tensor * w, int64_t r0, int64_t nr) {
                return w_in ? ggml_view_2d(ctx0, w_in, w_in->ne[0], nr, w_in->nb[1], r0*w_in->nb[1]) : w;
            };
            Qcur  = ggml_reshape_3d(ctx0, build_lora_mm(rows(layer.wq,             0,                   nq),           inputs[0]), n_embd_head, n_head, n_tokens);
            gate  =                       build_lora_mm(rows(layer.wqkv_gate,      nq,                  nq),           inputs[0]);
            Rcur  = ggml_reshape_3d(ctx0, build_lora_mm(rows(layer.attn_rel,       2*nq,                n_rel*n_head), inputs[0]), n_rel, n_head, n_tokens);
            sinks = ggml_reshape_3d(ctx0, build_lora_mm(rows(layer.attn_sink_proj, 2*nq + n_rel*n_head, n_head),       inputs[0]), 1, n_head, n_tokens);
        }
        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

        const int64_t n_stream = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];

        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        // the bias of a cell is the dot of its distance column of the table with [r, t, 1]; the table has a column for the distances past the extent (-t) and one for the masked cells (-inf)
        const int64_t n_extent = layer.attn_rel_bias->ne[1];

        const auto & derived = static_cast<const llama_model_inkling2 &>(model).derived_rels;
        const auto   rel_it  = derived.find(layer.attn_rel_bias);
        const llama_model_inkling2::derived_rel rel = rel_it != derived.end() ? rel_it->second : llama_model_inkling2::derived_rel{};

        ggml_tensor * bias = rel.table ? rel.table :
            ggml_concat(ctx0, ggml_concat(ctx0, layer.attn_rel_bias, rel_rows(n_extent), 0), rel_cols(), 1);

        // one op: clip r, scale it by tau, project [r, t, 1] on the table and take along the distances,
        // [n_kv, n_tokens/n_stream, n_head, n_stream] in F16 for the mask of flash attention
        const ggml_type type_b = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

        // a batch (past the matrix-vector sizes): the op writes the bias of every table column per token and head,
        // [n_extent + 2, n_tokens/n_stream, n_head, n_stream], and flash attention gathers it by the distances
        // itself, instead of reading a dense mask that grows with n_kv (the distance n_extent + 1 is the masked cell)
        static const bool fa_rel_disabled = getenv("LLAMA_INKLING2_FA_DENSE_MASK") != nullptr;
        const bool fa_rel = cparams.flash_attn && !fa_rel_disabled && n_tokens/n_stream > 8;

        ggml_tensor * kq_b = ggml_rel_bias(ctx0, Rcur, layer.attn_rel_clip, hparams.f_relative_clip, is_swa ? nullptr : tau,
                sinks, bias, fa_rel ? nullptr : rel_pos, (int) n_stream, type_b);
        cb(kq_b, "kq_b", il);
        // queued right after the projection: it runs beside the k/v work instead of on the path into the attention
        ggml_build_forward_expand(gf, kq_b);

        // K is L2-clipped and V RMS-normalised per head, in the op of the convolution where it can be
        ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0,
                GGML_SSM_CONV_POST_L2_CLIP, hparams.f_qk_clip, n_embd_head);
        ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa,
                GGML_SSM_CONV_POST_RMS_NORM, hparams.f_norm_rms_eps, n_embd_head);

        Qcur = clip(Qcur, layer.attn_q_clip, hparams.f_qk_clip);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        // r is clipped (and scaled by tau on the full-attention layers) inside the bias op above
        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        sinks = rel.sinks ? rel.sinks : ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);

        cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il,
                fa_rel ? rel_pos : nullptr, (int32_t) n_extent + 1);

        // a layer that does not start a block: the attention output joins the residual stream inside the
        // convolution op

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa,
                GGML_SSM_CONV_POST_NONE, 0.0f, 0, join ? inpSA : nullptr);
        cb(cur, "attn_out", il);

        if (!join) {
            blocks.push_back(inpSA);
        }
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    if (mix_fused() && messages) {
        // the controller reads the mix and its row norm, both from the mix op
        ggml_tensor * raw = nullptr;
        ggml_tensor * nrm = mix_normed(blocks, cur, layer.ffn_res_query, nullptr, 1e-6f, &raw);
        cur = control(ggml_reshape_2d(ctx0, raw, n_embd, n_tokens), nrm,
                row_2d(messages, LOOP_CTRL_FFN), row_3d(messages_norm, LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    } else if (mix_fused()) {
        cur = ggml_reshape_2d(ctx0, mix_normed(blocks, cur, layer.ffn_res_query, layer.ffn_norm, hparams.f_norm_rms_eps), n_embd, n_tokens);
    } else {
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
        if (messages) {
            cur = control(cur, ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), 1e-6f),
                    row_2d(messages, LOOP_CTRL_FFN), row_3d(messages_norm, LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
        }
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    }

    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps);

    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd,
            GGML_SSM_CONV_POST_NONE, 0.0f, 0, ffn_inp);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::tap_layer_inp(int visit, ggml_tensor * cur) {
    // only the extracted visits are touched: with no drafter attached the graph is unchanged
    const auto & taps = cparams.embeddings_layer_inp;
    if (visit < 0 || (size_t) visit >= taps.size() || !taps[visit]) {
        return;
    }
    if (!ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx0, cur);
    }
    res->t_layer_inp[visit] = cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    if (mix_fused()) {
        cur = ggml_reshape_2d(ctx0, mix_normed(blocks, cur, model.output_res_query, model.output_norm, hparams.f_norm_rms_eps), n_embd, n_tokens);
    } else {
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    }

    tap_layer_inp(n_visit, cur);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
