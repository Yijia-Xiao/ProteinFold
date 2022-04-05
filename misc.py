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


# LAYERNUM=2
# HIDDENSIZE=72
# HEAD=3
#
# MAX_TOKENS=2048
#
# MAX_ALIGNS=1024
# MAX_LENGTH=512
# POS_EMBED=16
#
# --num-layers $LAYERNUM \
# --hidden-size $HIDDENSIZE \
# --num-attention-heads $HEAD \
# --seq-length $MAX_LENGTH \
# --max-position-embeddings $MAX_LENGTH \
# --max-msa-position-embeddings $MAX_ALIGNS \
# --vocab-file $MYPATH/msa_tools/msa_vocab.txt \
# --add-msa-positional-embedding \
# --add-post-embedding-layernorm \
