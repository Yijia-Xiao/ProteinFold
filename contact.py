import matplotlib.pyplot as plt
import fold
import torch
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
from fold.converter import convert
import argparse

parser = argparse.ArgumentParser(description="Convert Megatron-LM checkpoint to Pytorch.")

parser.add_argument(
    "--model", type=str, help="Path to Megatron-MSA model file."
)
args = parser.parse_args()

model_path = args.model
convert(model_path)

torch.set_grad_enabled(False)

#  [markdown]
# ## Data Loading
# 
# This sets up some sequence loading utilities for ESM-1b (`read_sequence`) and the MSA Transformer (`read_msa`).

# 
# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

#  [markdown]
# ## Run MSA Transformer Contact Prediction
# 
# The MSAs used here are samples from the [`trRosetta` (v1) dataset](https://yanglab.nankai.edu.cn/trRosetta/benchmark/), also used in the MSA Transformer paper.

# msa_transformer, msa_alphabet = fold.pretrained.esm_msa1b_t12_100M_UR50S()
# msa_transformer, msa_alphabet = fold.pretrained.megatron_msa_1B("./data/megatron.pt")
msa_transformer, msa_alphabet = fold.pretrained.megatron_msa_1B(model_path.split('/')[0] + "/megatron.pt")
print(model_path.split('/')[0] + "/megatron.pt")
msa_transformer = msa_transformer.eval().cuda()
msa_batch_converter = msa_alphabet.get_batch_converter()

msa_data = [
    read_msa("./data/sample.a2m", 96),
]
msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
msa_batch_tokens = msa_batch_tokens.cuda()
print(msa_batch_tokens.size(), msa_batch_tokens.dtype)  # Should be a 3D tensor with dtype torch.int64.

# msa_contacts = msa_transformer.predict_contacts(msa_batch_tokens).cpu()
msa_ret = msa_transformer.predict_tots(msa_batch_tokens)
msa_contacts = msa_ret['contacts'].cpu()
msa_row = msa_ret['row_attentions'].cpu()

fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
for ax, contact, msa in zip(axes, msa_contacts, msa_batch_strs):
    seqlen = len(msa[0])
    ax.imshow(contact[:seqlen, :seqlen], cmap="Blues")

# plt.show()
plt.savefig('sample.png')



def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

# 12, 8
def plot(i, j):
    plt.figure()
    # plt.imshow(msa_row[0][i][j][1:, 1:].float())
    plt.imshow(apc(symmetrize(msa_row[0][i][j][1:, 1:].float().softmax(dim=-1))), cmap='Blues')
    plt.savefig('row.png')


layers, heads = msa_transformer.args.num_layers, msa_transformer.args.num_attention_heads

fig, axs = plt.subplots(layers, heads)

# for i in range(6, layers):
for i in range(layers - 2, layers):
    for j in range(heads):
        axs[i, j].imshow(apc(symmetrize(msa_row[0][i][j][1:, 1:].float().softmax(dim=-1))), cmap='Blues')

plt.savefig('maps.png')
