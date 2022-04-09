import json
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from fold.inference import Inference
from fold.loader import train_reader, cameo_reader


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


def train_classification_net(data, label):
    sklearn_solver = 'liblinear'
    if sklearn_solver == 'liblinear':
        net = LogisticRegression(penalty='l1', C=1 / 0.15, solver='liblinear')
    elif sklearn_solver == 'saga':
        net = LogisticRegression(penalty='l1', C=1 / 0.15, solver='saga', n_jobs=32)
    net.fit(data, label)
    ret = {}
    ret['net.intercept_'] = net.intercept_
    ret['net.coef_'] = net.coef_
    ret['net.score(X, Y)'] = net.score(data, label)
    ret['net'] = net
    return ret


range_dic = {'short': [6, 12], 'mid': [12, 24], 'long': [24, 2048], 'midlong': [12, 2048], 'all': [-1, 2048]} # , 'midlong': [12, 2048]}
frac_list = [1, 2, 5]


MAX_DEPTH = 32
MAX_LENGT = 768

MAX_DEPTH = 128
MAX_LENGT = 512

def train():
    train_samples = train_reader('./data/train/')
    train_data = []
    model = Inference('./data/megatron-1b.pt')

    esm_train = []
    bin_train = []

    # train
    def construct_train(heads, bin_label):
        num_layer, num_head, seqlen, _ = heads.size()

        attentions = heads.view(num_layer * num_head, seqlen, seqlen)
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(1, 2, 0)

        for i in range(seqlen):
            for j in range(i + 6, seqlen):
                # based on ESM paper
                esm_train.append(attentions[i][j].tolist())
                bin_train.append(bin_label[i][j])

    for idx, s_ in enumerate(train_samples):
        # batch size = 1 [[]]
        sample = [[('', seq) for seq in s_['msa'][: MAX_DEPTH]]]

        train_data.append(sample)
        p, q, r = model(sample, require_contacts=False)
        # print(str(idx) * 20, r['contacts'].shape)

        label = s_['contact']
        heads = q[0][:, :, 1:, 1:]
        print(heads.shape)
        construct_train(heads, label)

    # return train_data

    logging.info('start train')

    net = train_classification_net(esm_train, bin_train)
    logging.info('stop train')
    logging.info(net)

    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f)
    # with open('net.pickle', 'rb') as f:
    #     net = pickle.load(f)
    return net

# train()

def calculate_contact_precision(name, pred, label, local_range, local_frac=5, ignore_index=-1):
    """
        local_range: eg. local_range=[12, 24], calculate midium range contacts
        local_frac: eg. local_frac=5, calculate P@L/5, local_frac=2, calculate P@L/2
    """
    for i in range(len(label)):
        for j in range(len(label)):
            if (abs(i - j) < local_range[0] or abs(i - j) >= local_range[1]):
                label[i][j] = ignore_index

    correct = 0
    total = 0

    predictions = pred
    labels = label.reshape(-1)

    valid_masks = (labels != ignore_index)
    confidence = predictions[:, 1]
    valid_masks = valid_masks.type_as(confidence)
    masked_prob = (confidence * valid_masks).view(-1)
    seq_len = int(len(labels) ** 0.5)
    most_likely = masked_prob.topk(seq_len // local_frac, sorted=False)
    selected = labels.view(-1).gather(0, most_likely.indices)
    selected[selected < 0] = 0
    correct += selected.sum().long()
    total += selected.numel()
    return correct, total


def predict_sample(net, sample_tuple):
    sample, heads = sample_tuple
    logging.info(sample['msa'][0])

    label = torch.from_numpy(np.array(sample['binary_labels']))
    num_layer, num_head, seqlen, _ = heads.size()
    attentions = heads.view(num_layer * num_head, seqlen, seqlen)
    attentions = apc(symmetrize(attentions))
    attentions = attentions.permute(1, 2, 0)

    proba = net['net'].predict_proba(attentions.reshape(-1, num_layer * num_head).cpu())
    net_pred = net['net'].predict(attentions.reshape(-1, num_layer * num_head).cpu())
    # proba = net['net'].predict_proba(attentions.reshape(-1, 144).cpu())
    # proba = parallel(delayed(net['net'].predict_proba)(attentions.reshape(-1, 144)) for job_id in range(job_num))
    # cor, tot = calculate_contact_precision(sample['name'], torch.from_numpy(proba).to('cuda'), label.to('cuda'), local_range=range_, frac=frac)
    proba = torch.from_numpy(proba).float()
    label = label.float()
    eval_dic = dict()
    for r in range_dic:
        eval_dic[r] = dict()
        for f in frac_list:
            eval_dic[r][f] = dict()
            for c in ['cor', 'tot']:
                eval_dic[r][f][c] = 0

    for range_name in range_dic:
        for fra in frac_list:
            cor, tot = calculate_contact_precision(sample['name'], proba.clone(), label.clone(), local_range=range_dic[range_name], local_frac=fra)
            # logging.info(cor.item(), tot)
            eval_dic[range_name][fra]['cor'] += cor.item()
            eval_dic[range_name][fra]['tot'] += tot
    logging.info(eval_dic)
    # return eval_dic
    return (eval_dic, (net_pred, label.clone()))


def evaluate(net):
    cameo_data = cameo_reader('./data')
    model = Inference('./data/megatron-1b.pt')

    test_samples = []
    for sample in tqdm(cameo_data):
        if len(sample['msa'][0]) > MAX_LENGT:
            continue

        msa = [[('', seq) for seq in sample['msa'][: MAX_DEPTH]]]
        p, q, r = model(msa, require_contacts=False)
        print(model.msa_transformer.training)
        # label = s_['contact']
        heads = q[0][:, :, 1:, 1:]
        # print(heads.shape)
        test_samples.append((sample, heads))

    parallel = Parallel(n_jobs=32, batch_size=1)

    eval_dict_pred_tuple_list = parallel(delayed(predict_sample)(net, sample) for sample in test_samples)

    eval_dict_list = [t[0] for t in eval_dict_pred_tuple_list]
    pred_list = [t[1] for t in eval_dict_pred_tuple_list]
    json.dump(eval_dict_list, open('eval_dict_list.json', 'w'))

    # logging.info(eval_dict_list)
    merge_dict = dict()
    for r in range_dic:
        merge_dict[r] = dict()
        for f in frac_list:
            merge_dict[r][f] = dict()
            for c in ['cor', 'tot']:
                merge_dict[r][f][c] = 0

    for eval_di in eval_dict_list:
        for r in range_dic:
            for f in frac_list:
                for c in ['cor', 'tot']:
                    merge_dict[r][f][c] += eval_di[r][f][c]
    # logging.info(merge_dict)
    return merge_dict, pred_list

def main():
    net = train()
    # load the trained net regression to model
    eval_dic, ret_contact = evaluate(net)
    for r in range_dic:
        for f in frac_list:
            eval_dic[r][f]['acc'] = eval_dic[r][f]['cor'] / eval_dic[r][f]['tot']


    logging.info(eval_dic)
    logging.info(eval_dic['long'])
    torch.save(ret_contact, f'./data/cameo.pt')

# main()
with open('net.pickle', 'rb') as f:
    net = pickle.load(f)

eval_dic, ret_contact = evaluate(net)

for r in range_dic:
    for f in frac_list:
        eval_dic[r][f]['acc'] = eval_dic[r][f]['cor'] / eval_dic[r][f]['tot']

logging.info(eval_dic)
logging.info(eval_dic['long'])
torch.save(ret_contact, f'./data/cameo.pt')


# {'short': {1: {'cor': 3267, 'tot': 34296, 'acc': 0.095258922323303},
#   2: {'cor': 1933, 'tot': 17113, 'acc': 0.11295506340209198},
#   5: {'cor': 818, 'tot': 6813, 'acc': 0.12006458241596947}},
#  'mid': {1: {'cor': 3226, 'tot': 34296, 'acc': 0.09406344763237695},
#   2: {'cor': 1822, 'tot': 17113, 'acc': 0.10646876643487407},
#   5: {'cor': 842, 'tot': 6813, 'acc': 0.12358725965066784}},
#  'long': {1: {'cor': 2903, 'tot': 34296, 'acc': 0.08464543970142291},
#   2: {'cor': 1602, 'tot': 17113, 'acc': 0.09361304271606381},
#   5: {'cor': 722, 'tot': 6813, 'acc': 0.10597387347717599}},
#  'midlong': {1: {'cor': 3444, 'tot': 34296, 'acc': 0.10041987403778867},
#   2: {'cor': 1918, 'tot': 17113, 'acc': 0.11207853678490037},
#   5: {'cor': 847, 'tot': 6813, 'acc': 0.12432115074123}},
#  'all': {1: {'cor': 28467, 'tot': 34296, 'acc': 0.830038488453464},
#   2: {'cor': 16413, 'tot': 17113, 'acc': 0.9590954245310582},
#   5: {'cor': 6714, 'tot': 6813, 'acc': 0.9854689564068693}}}

