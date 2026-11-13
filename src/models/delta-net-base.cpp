#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);
    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head  = mctx_cur->get_head();
    const auto mem_size = mctx_cur->get_size();

    const int64_t n_seqs = ubatch.n_seqs;

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    const int64_t row_count = (conv_kernel_size - 1) * conv_channels;

    const size_t row_size  = ggml_row_size(conv_states_all->type, row_count);

    if (cparams.n_rs_seq == 0) {
        const int64_t s_idx  = conv_input->ne[0] - conv_states->ne[0];
        const int64_t s_slot = 0;

        ggml_tensor * conv_state_last =
            ggml_view_3d(ctx0, conv_input,
                    conv_kernel_size - 1, conv_channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, s_idx));
        cb(conv_state_last, "conv_state_last", il);

        ggml_tensor * conv_state_update =
            ggml_view_2d(ctx0, conv_states_all,
                    row_count, n_seqs, conv_states_all->nb[1],
                    (s_slot * mem_size + kv_head) * row_size);
        cb(conv_state_update, "conv_state_update", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
    } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // this logic assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are inside
        //   the same ubatch, which `split_equal()` guarantees via its n_keep_tail argument

        const int64_t K = (int64_t) cparams.n_rs_seq + 1;
        std::vector<ggml_tensor *> staged_copies;
        for (int64_t t = 1; t <= K; ++t) {
            const int64_t s_idx  = std::max<int64_t>(0, conv_input->ne[0] - conv_states->ne[0] - K + t);
            const int64_t s_slot = K - t;

            ggml_tensor * conv_state_last =
                ggml_view_3d(ctx0, conv_input,
                        conv_kernel_size - 1, conv_channels, n_seqs,
                        conv_input->nb[1], conv_input->nb[2],
                        ggml_row_size(conv_input->type, s_idx));

            if (mctx_cur->staged_publication_enabled() && s_slot > 0) {
                auto * stage=mctx_cur->staging_tensor(il,false,uint32_t(s_slot));
                auto * publication=mctx_cur->publication_tensor(il,false,uint32_t(s_slot));
                GGML_ASSERT(stage && publication);
                auto * staged=ggml_cpy(ctx0,conv_state_last,stage);
                staged_copies.push_back(staged);
                res->add_recurrent_snapshot({publication,uint32_t(il),uint32_t(s_slot),false,0});
            } else {
                ggml_tensor * conv_state_update = mctx_cur->spill_enabled() && s_slot > 0
                    ? ggml_new_tensor_2d(ctx0, conv_states_all->type, row_count, n_seqs)
                    : ggml_view_2d(ctx0,
                        conv_states_all, row_count, n_seqs,
                        conv_states_all->nb[1],
                        (s_slot * mem_size + kv_head) * row_size);
                ggml_build_forward_expand(gf,ggml_cpy(ctx0,conv_state_last,conv_state_update));
                if (mctx_cur->spill_enabled() && s_slot > 0)
                    res->add_recurrent_snapshot({conv_state_update,uint32_t(il),uint32_t(s_slot),false,0});
            }
        }
        if (!staged_copies.empty()) GGML_ASSERT(mctx_cur->publish_staged(ctx0,gf,il,false,staged_copies));
    }

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);
    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    if (mctx_cur->spill_enabled()) {
        std::vector<ggml_tensor *> staged_copies;
        for (int64_t slot=0;slot<n_written;++slot) {
            auto * source = ggml_view_2d(ctx0, src, D, n_seqs, src->nb[1], size_t(slot)*src->nb[2]);
            if (slot > 0 && mctx_cur->staged_publication_enabled()) {
                auto * stage=mctx_cur->staging_tensor(il,true,uint32_t(slot));
                auto * publication=mctx_cur->publication_tensor(il,true,uint32_t(slot));
                GGML_ASSERT(stage && publication);
                auto * staged=ggml_cpy(ctx0,source,stage);
                staged_copies.push_back(staged);
                res->add_recurrent_snapshot({publication,uint32_t(il),uint32_t(slot),true,0});
            } else {
                auto * destination = slot == 0
                    ? ggml_view_2d(ctx0, ssm_states_all, D, n_seqs, ssm_states_all->nb[1], size_t(kv_head)*row_size)
                    : ggml_new_tensor_2d(ctx0, ssm_states_all->type, D, n_seqs);
                ggml_build_forward_expand(gf,ggml_cpy(ctx0,source,destination));
                if (slot > 0)
                    res->add_recurrent_snapshot({destination,uint32_t(il),uint32_t(slot),true,0});
            }
        }
        if (!staged_copies.empty()) GGML_ASSERT(mctx_cur->publish_staged(ctx0,gf,il,true,staged_copies));
    } else {
        ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
            D, n_seqs, n_written,
            ssm_states_all->nb[1],
            (size_t) mem_size * row_size,
            (size_t) kv_head * row_size);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
    }

    return output;
}
