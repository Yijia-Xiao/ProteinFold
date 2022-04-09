import matplotlib.pyplot as plt
import fold
import torch
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
from fold.converter import convert
import argparse
from fold.inference import Inference
from fold.loader import sample_reader


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


model = Inference('./data/megatron-1b.pt')

def a2m_reader(file):
    ret = []
    with open(file, 'r') as f:
        for l in f.readlines():
            ret.append(('', l.strip()))
    return ret


msa_data = [
    # read_msa("./data/sample.a2m", 96),
    a2m_reader("data/CAMEO-trRosettaA2M/5OD1_A.a2m")[:96],
    a2m_reader("data/CAMEO-trRosettaA2M/5Y08_A.a2m")[:96],
    a2m_reader("data/CAMEO-trRosettaA2M/5Z3F_A.a2m")[:96],
]


msa_data, msa_row, msa_ret = model(msa_data, require_contacts=True)

msa_contacts = msa_ret['contacts'].cpu()
msa_row = msa_ret['row_attentions'].cpu()

fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
for ax, contact, msa in zip(axes, msa_contacts, msa_data):
    seqlen = len(msa[0][1])
    ax.imshow(contact[:seqlen, :seqlen], cmap="Blues")

# plt.show()
plt.savefig('sample.png')

