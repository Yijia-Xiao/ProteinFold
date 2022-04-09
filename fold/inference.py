# import matplotlib.pyplot as plt
import fold
import torch
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
# from fold.converter import convert
# import argparse

torch.set_grad_enabled(False)
class Inference(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.deletekeys = dict.fromkeys(string.ascii_lowercase)
        self.deletekeys["."] = None
        self.deletekeys["*"] = None
        self.translation = str.maketrans(self.deletekeys)

    def read_sequence(self, filename: str) -> Tuple[str, str]:
        """ Reads the first (reference) sequences from a fasta or MSA file."""
        record = next(SeqIO.parse(filename, "fasta"))
        return record.description, str(record.seq)

    def remove_insertions(self, sequence: str) -> str:
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(self.translation)

    def read_msa(self, filename: str, nseq: int) -> List[Tuple[str, str]]:
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        return [(record.description, self.remove_insertions(str(record.seq)))
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


    def load_model(self):
        m_path = '/'.join(self.model_path.rsplit('/')[:-1]) + "/megatron.pt"
        msa_transformer, msa_alphabet = fold.pretrained.megatron_msa_1B(m_path)

        self.msa_transformer = msa_transformer.eval().cuda() # .half()
        self.msa_batch_converter = msa_alphabet.get_batch_converter()

    # def inference(model_path, msa_data):
    def __call__(self, msa_data, require_contacts=True):
        # msa_data is a list of msa samples

        # msa_data = [
        #     read_msa("./data/sample.a2m", 32),
        # ]
        self.load_model()

        msa_batch_labels, msa_batch_strs, msa_batch_tokens = self.msa_batch_converter(msa_data)
        msa_batch_tokens = msa_batch_tokens.cuda()

        if require_contacts:
            msa_ret = self.msa_transformer.predict_tots(msa_batch_tokens)
            msa_contacts = msa_ret['contacts'].cpu()
            msa_row = msa_ret['row_attentions'].cpu()

            return msa_data, msa_row, msa_ret
        else:
            msa_ret = self.msa_transformer.predict_heads(msa_batch_tokens)
            msa_row = msa_ret['row_attentions'].cpu()

            return msa_data, msa_row, None