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


@nut_processor
def MakeBatch(samples, CFG, batchsize, test = False):
    for batch in samples >> Chunk(batchsize):
        if test:
            meta, szrids, targets, data = batch >> Unzip()
        else:
            meta, targets, data = batch >> Unzip()
        data_batch = torch.tensor(data).to(DEVICE)
        if not CFG.sequence_model:
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
def TrainBatch(batch, net, optimizer, criterion):
    targets, data = batch
    preds = net(data)
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@nut_function
def Convert2numpy(sample):
    X = np.concatenate(sample[3:], axis=1)
    return sample[0], sample[1], sample[2], X





