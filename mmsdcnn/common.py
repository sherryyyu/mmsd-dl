'''
Licensed Materials - Property of IBM
(C) Copyright IBM Corp. 2020. All Rights Reserved.

US Government Users Restricted Rights - Use, duplication or
disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

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
from mmsdcnn.constants import PARAMS, DEVICE
from mmsdcommon.metrics import roc_auc_score
from mmsdcommon.preprocess import normalise_acc, normalise_eda, normalise_bvp, normalise_hr


def probabilities(pred):
    return torch.softmax(pred, 1)

def to_numpy(x):
    return x.detach().cpu().numpy()

@nut_processor
def MakeBatch(samples, batchsize):
    for batch in samples >> Chunk(batchsize):
        targets, data = batch >> Unzip()
        data_batch = torch.tensor(data).permute(0,2,1).to(DEVICE)   # change channel location for pytorch compatibility
        tar_batch = torch.tensor(targets).to(DEVICE)

        yield tar_batch, data_batch

@nut_function
def PredBatch(batch, net):
    targets, data = batch
    preds = net(data)
    probs = probabilities(preds)
    preds = torch.max(probs, 1)[1].view_as(targets)

    return to_numpy(targets), to_numpy(preds), to_numpy(probs)


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
    X = np.concatenate(sample[1:], axis=1)
    y = sample[0]
    return y, X

@nut_function
def Normalise(sample):
    hr = normalise_hr(sample.hr)
    bvp = normalise_bvp(sample.bvp)
    eda = normalise_eda(sample.eda)
    acc = normalise_acc(sample.acc)
    sample = sample._replace(hr=hr)
    sample = sample._replace(eda=eda)
    sample = sample._replace(bvp=bvp)
    sample = sample._replace(acc=acc)
    return sample