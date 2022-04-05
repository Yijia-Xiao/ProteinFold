import torch

m = torch.load('data/megatron.pt')
e = torch.load('/home/yijia/.cache/torch/hub/checkpoints/esm_msa1b_t12_100M_UR50S.pt')
print(f"{m['args']}=")
print(f"{e['args']}=")

keys = []
def recursive(dic):
    if isinstance(dic, dict):
        keys.append(dic.keys)


# keys = ['embed_dim', 'embed_positions_msa', 'layers', 'dropout', 'ffn_embed_dim', 'attention_heads', 'attention_dropout', 'activation_dropout', 'max_tokens', 'max_positions']

