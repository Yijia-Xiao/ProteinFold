import matplotlib.pyplot as plt
import fold
import torch
import os
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string


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
msa_transformer, msa_alphabet = fold.pretrained.megatron_msa_1B()
msa_transformer = msa_transformer.eval().cuda()
msa_batch_converter = msa_alphabet.get_batch_converter()

msa_data = [
    read_msa("./examples/1a3a_1_A.a3m", 64),
    read_msa("./examples/5ahw_1_A.a3m", 64),
    read_msa("./examples/1xcr_1_A.a3m", 64),
]
msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
msa_batch_tokens = msa_batch_tokens.cuda()
print(msa_batch_tokens.size(), msa_batch_tokens.dtype)  # Should be a 3D tensor with dtype torch.int64.

msa_contacts = msa_transformer.predict_contacts(msa_batch_tokens).cpu()

fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
for ax, contact, msa in zip(axes, msa_contacts, msa_batch_strs):
    seqlen = len(msa[0])
    ax.imshow(contact[:seqlen, :seqlen], cmap="Blues")

# plt.show()
plt.savefig('map.png')
