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
    model_dir = '/'.join(src.rsplit('/')[:-1]) # src.rsplit('/')[0]

    torch.save(target_dict, f'{model_dir}/megatron.pt')

    def prepare_reg():
        # a = torch.load('/home/yijia/.cache/torch/hub/checkpoints/esm_msa1b_t12_100M_UR50S-contact-regression.pt')
        
        # p  = a['model']['contact_head.regression.weight'][0, -96:]


        # p = torch.zeros(224)
        # p[-10:] = 0.1

        p = torch.tensor(
            # [0 for i in range(num_transformer_layer * num_heads)], dtype=torch.float32
            [0.39408993686106447, 0.38473109238366854, -0.9303874401894765, -0.181521274940658, -0.18429156281334785, 0.093633539451786, -0.2528500432780695, 0.055578459295487645, -0.344787686733303, 0.08440712440381988, -0.15663223322950978, -0.2792427690478492, -0.24413237035291066, -0.37459216244273374, -0.025749177165168502, -0.32207536695746236, 3.9371783834600653, -2.4118559510501805, -6.970159450960794, 14.979077658169961, -0.10680362936882125, -2.7056257602093994, 3.620119822641153, 0.16057611304540645, -4.728701538700186, 2.757937126314179, 9.153327559400465, 2.554302878917028, 0.0, -9.264772065440871, -2.1000851135747842, 19.709083995233982, 1.361508290953593, -92.2411495337871, 4.2518266088239285, 23.708553731883264, 0.0, 5.462081461916723, 0.5187561695283484, -1.712135737199143, 10.756141814644025, 0.0, -0.4984719418579434, 27.919937195224968, 0.0, 0.0, 6.47121221071584, 17.940260609990197, 18.016634202340047, -4.811398097729479, 10.299124105075945, 41.2124345181316, 6.78580711719219, 0.0, 0.0, -20.208937392893464, -19.723679216824976, -19.29760663540041, 0.0, 0.0, 0.0, 9.573548226537259, 55.49367177587132, 0.0, 13.929015327485477, -10.224759847687604, 31.14393322784599, 0.0, 11.293619075641999, 0.0, 22.615664794747616, 36.590027694602625, 0.0, 0.0, 15.610252146736531, 14.298784044001998, 0.0, 0.0, 0.0, 0.0, 15.037734920114422, 57.078117582654464, 30.57138242141086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.841655937125159, 0.0, 21.41696170212242, -21.786319756312626, 0.0, -0.2759549697478634, 0.0, 14.795784924754981, 0.0, 0.0, 0.0, 0.0, 0.0, -83.81868748836983, 0.0, 0.0, -17.325931201751242, 61.772007799499384, 0.0, 39.758888590981066, 0.0, 0.0, 0.0, 0.0, 54.0703519302847, 2.767642264362005, 0.0, 0.0, 56.94961794158551, 0.0, 0.0, 128.7294941620725, 0.0, 0.0, 0.0, -34.545734118369325, 7.474035493809019, -30.619956380141865, -83.7353173350744, 101.49011967243543, 46.13215738614705, 0.0, 0.0, 0.0, 10.821463306627797, 0.0, 3.3253282952651957, 0.0, 0.0, -58.294964222783875, 0.0, 5.580089021886252, -28.184482631147286, 0.0, 0.0, 213.06821975682684, 32.95679864215235, 0.0, -49.41757611876876, 0.0, 0.0, 2.4591904010869627, 0.0, 0.0, 24.5940102783332, -6.258467229231678, 0.0, 17.852093573879497, -171.38791802519404, -126.35615803270429, -35.506610082058714, 0.0, 0.0, -78.75002644118997, 0.0, 89.22207110027884, 47.36212576782209, 0.0, -71.3736685942055, 0.0, 0.0, 0.0, -101.80791395451512, 45.73998890524863, 0.0, -49.80878764136818, -28.0948813855752, 0.0, -132.13198693916175, 136.75477591103302, 0.0, 177.51974518792818, 0.0, -26.925247374192207, 42.94656434005955, 0.0, 395.3686712084416, 0.0, 0.0, 0.0, 0.0, -288.6038846898833, 0.0, 0.0, 116.32394120711054, 0.0, 173.20958912107616, 0.0, 135.51693657030788, 0.0, 0.0, 122.68361072313765, 186.2953511617272, 0.0, -56.31714663539074, 0.0, -125.25541225442664, 0.0, 0.0, 208.46111900030903, 0.0, 0.0, 71.7513732238203, -7.984192067908675, 0.0, 29.95706133704512, -25.615604732042254, 0.0, 162.35003546760728, 96.10432373244653, 0.0, 102.55397232065214, 113.71029372988467, 0.0]
        )
        # p[-2 * num_heads::3] = 1

        # p /= p.sum()
        p.unsqueeze_(0)

        b = {'model': {'contact_head.regression.weight': p.float(), 'contact_head.regression.bias': torch.tensor([-5.56987563], device='cuda:0')}}
        print(b)

        torch.save(b,f'{model_dir}/megatron-contact-regression.pt')

    prepare_reg()

    # def load_megatron():
    #     model = torch.load('data/megatron.pt')
    #     print(model['model'].keys())
    #     print(model['model']['lm_head'].keys())
    #     print(model['model']['language_model'].keys())

    # load_megatron()
