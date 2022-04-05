import torch

# parser = argparse.ArgumentParser(description="Convert Megatron-LM checkpoint to Pytorch.")

# parser.add_argument(
#     "--src", type=str, help="Path to Megatron-MSA model file."
# )
# parser.add_argument(
#     "--tgt", type=str, help="Path to target Pytorch model file."
# )

# args = parser.parse_args()
# src = args.src
# tgt = args.tgt
# src = 'a'


def convert(src):
    source = torch.load(src)
    # print(source.keys())
    src_embed = source['model']['language_model']['embedding']
    src_transformer = source['model']['language_model']['transformer']

    # target = collections.OrderedDict()
    target_dict = dict()
    target_dict['args'] = source['args']
    # target_dict['args']['add_msa_positional_embedding'] = True
    target_dict['model'] = dict()

    target = target_dict['model']

    # embedding
    # 'msa_position_embedding', 'embed_tokens.weight', 'embed_positions.weight'

    # 'msa_position_embedding': OrderedDict([('weight', tensor([[...
    target['msa_position_embedding'] = src_embed['msa_positional_embeddings']['weight'].float().unsqueeze(1).unsqueeze(0)
    # m_id_e_id = {2: 0, 0: 1, 1: 32, 5: 5, 6: 25, 7: 23, 8: 13, 9: 9, 10: 18, 11: 6, 12: 21, 13: 12, 14: 15, 15: 4, 16: 20, 17: 17, 18: 28, 19: 14, 20: 16, 21: 10, 22: 8, 23: 11, 24: 26, 25: 7, 26: 22, 27: 24, 28: 19, 29: 27, 30: 30}
    # for m, e in m_id_e_id.items():
    #     assign(megatron_embed['word_embeddings']['weight'][m], esm['encoder.sentence_encoder.embed_tokens.weight'][e])

    # target['embed_tokens.weight'] = src_embed['word_embeddings']['weight'][:32].float()

    target['embed_tokens.weight'] = src_embed['word_embeddings']['weight'][:33].float()
    m_id_e_id = {2: 0, 0: 1, 1: 32, 5: 5, 6: 25, 7: 23, 8: 13, 9: 9, 10: 18, 11: 6, 12: 21, 13: 12, 14: 15, 15: 4, 16: 20, 17: 17, 18: 28, 19: 14, 20: 16, 21: 10, 22: 8, 23: 11, 24: 26, 25: 7, 26: 22, 27: 24, 28: 19, 29: 27, 30: 30}
    # for m, e in m_id_e_id.items():
    #     # megatron_embed['word_embeddings']['weight'][m], esm['encoder.sentence_encoder.embed_tokens.weight'][e]
    #     # target['embed_tokens.weight'][e] = src_embed['word_embeddings']['weight'][m].float()
    #     target['embed_tokens.weight'][m] = src_embed['word_embeddings']['weight'][e].float()

    # target['embed_tokens.weight'] = src_embed['word_embeddings']['weight'][:33].float()


    target['embed_positions.weight'] = src_embed['position_embeddings']['weight'].float()
    target['emb_layer_norm_before.weight'] = src_embed['emb_layer_norm_before']['weight'].float()
    target['emb_layer_norm_before.bias'] = src_embed['emb_layer_norm_before']['bias'].float()


    # transformer
    # source
    # 'layers.0.row_input_layernorm.weight', 'layers.0.row_input_layernorm.bias', 'layers.0.col_input_layernorm.weight', 'layers.0.col_input_layernorm.bias', 'layers.0.row_attention.query_key_value.weight', 'layers.0.row_attention.query_key_value.bias', 'layers.0.row_attention.dense.weight', 'layers.0.row_attention.dense.bias', 'layers.0.col_attention.query_key_value.weight', 'layers.0.col_attention.query_key_value.bias', 'layers.0.col_attention.dense.weight', 'layers.0.col_attention.dense.bias', 'layers.0.post_attention_layernorm.weight', 'layers.0.post_attention_layernorm.bias', 'layers.0.mlp.dense_h_to_4h.weight', 'layers.0.mlp.dense_h_to_4h.bias', 'layers.0.mlp.dense_4h_to_h.weight', 'layers.0.mlp.dense_4h_to_h.bias'

    # target
    # 'layers.0.row_self_attention.layer.k_proj.weight', 'layers.0.row_self_attention.layer.k_proj.bias', 'layers.0.row_self_attention.layer.v_proj.weight', 'layers.0.row_self_attention.layer.v_proj.bias', 'layers.0.row_self_attention.layer.q_proj.weight', 'layers.0.row_self_attention.layer.q_proj.bias', 'layers.0.row_self_attention.layer.out_proj.weight', 'layers.0.row_self_attention.layer.out_proj.bias', 'layers.0.row_self_attention.layer_norm.weight', 'layers.0.row_self_attention.layer_norm.bias', 'layers.0.column_self_attention.layer.k_proj.weight', 'layers.0.column_self_attention.layer.k_proj.bias', 'layers.0.column_self_attention.layer.v_proj.weight', 'layers.0.column_self_attention.layer.v_proj.bias', 'layers.0.column_self_attention.layer.q_proj.weight', 'layers.0.column_self_attention.layer.q_proj.bias', 'layers.0.column_self_attention.layer.out_proj.weight', 'layers.0.column_self_attention.layer.out_proj.bias', 'layers.0.column_self_attention.layer_norm.weight', 'layers.0.column_self_attention.layer_norm.bias', 'layers.0.feed_forward_layer.layer.fc1.weight', 'layers.0.feed_forward_layer.layer.fc1.bias', 'layers.0.feed_forward_layer.layer.fc2.weight', 'layers.0.feed_forward_layer.layer.fc2.bias', 'layers.0.feed_forward_layer.layer_norm.weight', 'layers.0.feed_forward_layer.layer_norm.bias'

    num_transformer_layer = source['args'].num_layers
    num_heads = source['args'].num_attention_heads
    hidden_dim = source['args'].hidden_size
    heads_dim = hidden_dim // num_heads


    # def assign(dst, src):
    #     assert src.shape == dst.shape
    #     dst[:] = src.to(dst.device)[:]

    def process_layer_i(i: int):
        for rc in ['row', 'col']:
            # rc = rc if rc == 'row' else 'column'
            tgt_rc = "column" if rc == "col" else "row"
            W_mixed = src_transformer[f'layers.{i}.{rc}_attention.query_key_value.weight'].float()
            B_mixed = src_transformer[f'layers.{i}.{rc}_attention.query_key_value.bias'].float()
            wq, wk, wv = W_mixed.clone().view(num_heads, heads_dim * 3, -1).split(heads_dim, dim=1)
            bq, bk, bv = B_mixed.clone().view(num_heads, heads_dim * 3).split(heads_dim, dim=1)


            target[f'layers.{i}.{tgt_rc}_self_attention.layer.q_proj.weight'] = wq.contiguous().view(hidden_dim, hidden_dim).clone()
            target[f'layers.{i}.{tgt_rc}_self_attention.layer.k_proj.weight'] = wk.contiguous().view(hidden_dim, hidden_dim).clone()
            target[f'layers.{i}.{tgt_rc}_self_attention.layer.v_proj.weight'] = wv.contiguous().view(hidden_dim, hidden_dim).clone()
            target[f'layers.{i}.{tgt_rc}_self_attention.layer.q_proj.bias'] = bq.contiguous().view(-1).clone()
            target[f'layers.{i}.{tgt_rc}_self_attention.layer.k_proj.bias'] = bk.contiguous().view(-1).clone()
            target[f'layers.{i}.{tgt_rc}_self_attention.layer.v_proj.bias'] = bv.contiguous().view(-1).clone()
            
            # W_mixed = torch.cat((wq, wk, wv), dim=1).reshape(hidden_dim * 3, hidden_dim)
            # B_mixed = torch.cat((bq, bk, bv), dim=1).reshape(-1)

            # wq = target[f'layers.{i}.{rc}_self_attention.layer.q_proj.weight'].view(num_heads, heads_dim, -1)
            # wk = target[f'layers.{i}.{rc}_self_attention.layer.k_proj.weight'].view(num_heads, heads_dim, -1)
            # wv = target[f'layers.{i}.{rc}_self_attention.layer.v_proj.weight'].view(num_heads, heads_dim, -1)
            # bq = target[f'layers.{i}.{rc}_self_attention.layer.q_proj.bias'].view(num_heads, heads_dim)
            # bk = target[f'layers.{i}.{rc}_self_attention.layer.k_proj.bias'].view(num_heads, heads_dim)
            # bv = target[f'layers.{i}.{rc}_self_attention.layer.v_proj.bias'].view(num_heads, heads_dim)
            # tape[f"bert.encoder.layer.{layer}.attention.self.query.weight"] = wq.contiguous().view(hidden_dim, hidden_dim).clone()
            # tape[f"bert.encoder.layer.{layer}.attention.self.query.bias"] = bq.contiguous().view(-1).clone()
            # tape[f"bert.encoder.layer.{layer}.attention.self.key.weight"] = wk.contiguous().view(hidden_dim, hidden_dim).clone()
            # tape[f"bert.encoder.layer.{layer}.attention.self.key.bias"] = bk.contiguous().view(-1).clone()
            # tape[f"bert.encoder.layer.{layer}.attention.self.value.weight"] = wv.contiguous().view(hidden_dim, hidden_dim).clone()
            # tape[f"bert.encoder.layer.{layer}.attention.self.value.bias"] = bv.contiguous().view(-1).clone()


        for p in ['weight', 'bias']:
            for rc in ['row', 'col']:
                # rc = 'column' if rc == 'row' else 'row'
                tgt_rc = "column" if rc == "col" else "row"
                target[f'layers.{i}.{tgt_rc}_self_attention.layer_norm.{p}'] = src_transformer[f'layers.{i}.{rc}_input_layernorm.{p}'].float()
                target[f'layers.{i}.{tgt_rc}_self_attention.layer.out_proj.{p}'] = src_transformer[f'layers.{i}.{rc}_attention.dense.{p}'].float()

            target[f'layers.{i}.feed_forward_layer.layer.fc1.{p}'] = src_transformer[f'layers.{i}.mlp.dense_h_to_4h.{p}'].float()
            target[f'layers.{i}.feed_forward_layer.layer.fc2.{p}'] = src_transformer[f'layers.{i}.mlp.dense_4h_to_h.{p}'].float()
            target[f'layers.{i}.feed_forward_layer.layer_norm.{p}'] = src_transformer[f'layers.{i}.post_attention_layernorm.{p}'].float()


    for i in range(num_transformer_layer):
        process_layer_i(i)


    # layer norm after transformer
    target['emb_layer_norm_after.weight'] = src_transformer['final_layernorm.weight']
    target['emb_layer_norm_after.bias'] = src_transformer['final_layernorm.bias']

    # torch.save(target_dict, tgt)
    model_dir = src.rsplit('/')[0]

    torch.save(target_dict, f'{model_dir}/megatron.pt')

    def prepare_reg():
        # a = torch.load('/home/yijia/.cache/torch/hub/checkpoints/esm_msa1b_t12_100M_UR50S-contact-regression.pt')
        
        # p  = a['model']['contact_head.regression.weight'][0, -96:]


        # p = torch.zeros(224)
        # p[-10:] = 0.1

        p = torch.tensor(
            [0 for i in range(num_transformer_layer * num_heads)], dtype=torch.float32
        )
        p[-2 * num_heads::3] = 1

        # print(p)
        # p.squeeze_(0)
        p /= p.sum()
        p.unsqueeze_(0)
        # print(p)

        b = {'model': {'contact_head.regression.weight': p.float(), 'contact_head.regression.bias': torch.tensor([-5.5], device='cuda:0')}}
        print(b)

        torch.save(b,f'{model_dir}/megatron-contact-regression.pt')

    prepare_reg()

    # def load_megatron():
    #     model = torch.load('data/megatron.pt')
    #     print(model['model'].keys())
    #     print(model['model']['lm_head'].keys())
    #     print(model['model']['language_model'].keys())

    # load_megatron()
