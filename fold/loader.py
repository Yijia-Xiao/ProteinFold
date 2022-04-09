import os
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform


alphabet_str = 'ARNDCQEGHILKMFPSTWYV-'
id_to_char = dict()
for i in range(len(alphabet_str)):
    id_to_char[i] = alphabet_str[i]


def map_id_to_token(token_id):
    return id_to_char[token_id]


def sample_reader(sample_path):
    with np.load(sample_path) as data:
        msa_ids = data['msa']
        xyzCa = data['xyzca']
        msa_ids_2D_list = msa_ids.tolist()

        msa = []
        for seq in msa_ids_2D_list:
            msa_str = ''.join(list(map(map_id_to_token, seq)))
            msa.append(msa_str)
        contact = np.less(squareform(pdist(xyzCa)), 8.0).astype(np.int64)
        sample = {'name': sample_path.split('/')[-1], 'msa': msa, 'contact': contact}
    return sample

# sample = sample_reader('data/train/1np6_1_A.npz')

def train_reader(train_path):
    files = os.listdir(train_path)
    dataset = [sample_reader(os.path.join(train_path, s)) for s in files]
    return dataset


# train_data = train_reader('./data/train/')

def process_cameo_sample(cameo_path, file_name):
    msa_path = f'{cameo_path}/CAMEO-trRosettaA2M'
    label_path = f'{cameo_path}/CAMEO-GroundTruth'
    msa = open(os.path.join(msa_path, file_name + '.a2m'), 'r').read().splitlines()
    with open(os.path.join(label_path, file_name + '.native.pkl'), 'rb') as f:
        labels = pickle.load(f, encoding="bytes")

    assert msa[0].strip() == labels[b'sequence'].decode()

    # an entry of < 0 indicates an invalid distance.
    dist_mat = labels[b'atomDistMatrix'][b'CbCb']
    seq_len = len(dist_mat)
    binary_labels = np.zeros((seq_len, seq_len), dtype=int).tolist()
    for row in range(seq_len):
        for col in range(seq_len):
            if dist_mat[row][col] >= 0:
                if dist_mat[row][col] < 8:
                    binary_labels[row][col] = 1
            else:
                binary_labels[row][col] = -1
    return {
        'name': file_name,
        'msa': msa,
        'binary_labels': binary_labels,
    }


# sample = process_cameo_sample('./data', '6GCJ_A')

def cameo_reader(cameo_path):
    # CAMEO-GroundTruth
    # CAMEO-trRosettaA2M

    trRosetta_data = []
    label_path = f'{cameo_path}/CAMEO-GroundTruth'
    names = [i.split('.')[0] for i in os.listdir(label_path)]
    for name in names:
        data = process_cameo_sample(cameo_path, name)
        trRosetta_data.append(data)

    return trRosetta_data



# cameo_data = cameo_reader('./data')
