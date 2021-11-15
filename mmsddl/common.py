'''
Author:
    Sherry Yu
Initial Version:
    Nov-2020
Function:
   Define some common functions for train and evaluate
'''

import numpy as np
import torch
from nutsflow import *
from mmsddl.get_cfg import DEVICE
from mmsdcommon.metrics import roc_auc_score
from mmsdcommon.preprocess import normalise_acc, normalise_eda, normalise_bvp, \
    normalise_hr
from collections import namedtuple

def probabilities(pred):
    return torch.softmax(pred, 1)


def to_numpy(x):
    return x.detach().cpu().numpy()


def sample_negatives(label, data):
    return label[label<1], data[label<1]


def mae(arr1, arr2):
    np.mean(np.abs(arr1 - arr2), axis=1)


@nut_processor
def MakeBatch(samples, cfg, batchsize, test = False):
    for batch in samples >> Chunk(batchsize):
        if test:
            meta, szrids, targets, data = batch >> Unzip()
        else:
            meta, targets, data = batch >> Unzip()
        data_batch = torch.tensor(data).to(DEVICE)
        if cfg.network == 'cnn':
            # change channel location for pytorch compatibility
            data_batch = data_batch.permute(0, 2, 1)
        elif cfg.network == 'lstm':
            # change the input to 2 second segments
            b = data_batch.shape[0]
            data_batch = data_batch.reshape(b, cfg.win_len // cfg.subsequence, -1)
        elif cfg.network == 'cnnlstm':
            b, l, c = data_batch.shape
            data_batch = data_batch.reshape(b, cfg.win_len // cfg.subsequence,
                                            -1, c)
        elif cfg.network == 'convae':
            # change channel location for pytorch compatibility
            data_batch = data_batch.permute(0, 2, 1)

        tar_batch = torch.tensor(targets).to(DEVICE)
        if test:
            yield szrids, tar_batch, data_batch
        else:
            yield tar_batch, data_batch


@nut_function
def PredBatch(batch, net):
    szrids, targets, data = batch
    preds = net(data)
    probs = probabilities(preds)
    preds = torch.max(probs, 1)[1].view_as(targets)

    return szrids, to_numpy(targets), to_numpy(preds), to_numpy(probs)


@nut_function
def PredBatchAE(batch, net, criterion):
    szrids, labels, data = batch
    preds = net(data)
    losses = criterion(preds, data.float())
    return szrids, to_numpy(labels), to_numpy(preds), to_numpy(losses)


@nut_function
def TrainBatch(batch, net, optimizer, criterion, ae=False):
    targets, data = batch
    preds = net(data)
    if ae:
        loss = criterion(preds, data.float())
    else:
        loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@nut_function
def Convert2numpy(sample):
    X = np.concatenate(sample[3:], axis=1)
    return sample[0], sample[1], sample[2], X


@nut_processor
def SessionMinusSeizure(sample):
    same_sess = lambda x: (x.metadata['pid'], x.metadata['sid'])
    for session in sample >> ChunkBy(same_sess, list):
        meta, id, label, data = session >> Convert2numpy() >> Unzip()
        label, data = sample_negatives(np.array(label), np.array(data))
        for l, d in zip(label, data):
            yield meta[0], l, d




