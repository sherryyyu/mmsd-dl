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
   Define some common functions for loading data
'''

import time
import os
from pathlib import Path

import numpy as np
import torch
from nutsflow import *
from nutsml import PrintType, PlotLines
from sklearn.metrics import roc_auc_score

from mmsdcommon.data import load_metadata,  gen_session,  gen_window
from mmsdcommon.cross_validate import leave1out
from mmsdcommon.preprocess import remove_non_motor
from mmsdcommon.util import num_channels
from mmsdcnn.network import create_network
from mmsdcnn.constants import PARAMS, DEVICE


# def sample_imbalance(sampling, label, data):
# from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler
#     data_shape = data.shape
#
#     # It seems like the imblearn library only accepts data of shape 2,
#     # therefore we reshape the data to shape 2 before passing into the sampler and retrieve the shape after
#     if len(data_shape) == 3:
#         data = np.reshape(data, [data_shape[0], data_shape[1] * data_shape[2]])
#
#     if sampling == 'over':
#         sampler = RandomOverSampler()
#     elif sampling == 'smote':
#         sampler = SMOTE()
#     elif sampling == 'under':
#         sampler = RandomUnderSampler()
#     elif sampling == 'adasyn':
#         sampler = ADASYN()
#     else:
#         print('Error: invalid sampler name, using Random Under Sampling.')
#         sampler = RandomUnderSampler()
#     data, label = sampler.fit_resample(data, label)
#
#     # Retrieve the original data shape if the original shape is 3
#     if len(data_shape) == 3:
#         data = np.reshape(data, [-1, data_shape[1], data_shape[2]])
#     return label, data

def merge_modalities(session):
    label = np.array([sample.label for sample in session])
    data = np.array([np.concatenate(sample[1:], axis=1) for sample in session])
    return label, data

@nut_processor
def Preprocess(sessions):
    return sessions >> gen_window(10, 0.75, 0)  >> remove_non_motor(0.1)

@nut_function
def Convert2numpy(sample, nb_classes=2):
    X = np.concatenate(sample[1:], axis=1)
    # y = one_hot(sample[0], nb_classes)
    y = sample[0]
    return y, X


def one_hot(y, nb_classes):
    return np.array([1 if i==y else 0 for i in range(nb_classes)])


def to_numpy(x):
    return x.detach().cpu().numpy()

def probabilities(pred):
    return torch.softmax(pred, 1)

@nut_processor
def MakeBatch(samples, batchsize):
    for batch in samples >> Chunk(batchsize):
        targets, data = batch >> Unzip()
        data_batch = torch.tensor(data).permute(0,2,1).to(DEVICE)
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
    return loss.item(), auc(preds, targets)


def auc(pred, target):
    probs = probabilities(pred)
    preds = probs[:, 1]
    return safe_auc(to_numpy(target), to_numpy(preds))

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        print('roc_auc_score failed')
        return 0.5

def train_cnn(net, trainset, fdir):
    if PARAMS.verbose > 1:
        plotlines = PlotLines((0, 1, 2), layout=(3, 1), figsize=(8, 12),
                              titles=('loss', 'train-auc', 'val-auc'))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), PARAMS.lr)

    for epoch in range(PARAMS.n_epochs) >> PrintProgress(PARAMS.n_epochs):
        start = time.time()
        net.train()
        losses, aucs =  (gen_session(trainset, fdir) >> Preprocess() >> Convert2numpy()
               >> MakeBatch(PARAMS.batch_size) >> TrainBatch(net, optimizer, criterion) >> Unzip())

        loss, auc = np.mean(losses), np.mean(aucs)

        if PARAMS.verbose:
            msg = "Epoch {:d}..{:d}  {:s} : loss {:.4f}  auc {:.2f}"
            elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start))
            print(msg.format(epoch, PARAMS.n_epochs, elapsed, loss, auc))

def evaluate(net, testset, fdir):
    net.eval()
    with torch.no_grad():
        tars, preds, probs = (gen_session(testset, fdir) >> Preprocess() >> Convert2numpy()
                              >> MakeBatch(PARAMS.batch_size) >>
                              PredBatch(net) >> Unzip())
        tars = tars >> Flatten() >> Collect()
        preds = preds >> Flatten() >> Collect()
        probs = probs >> Flatten() >> Get(1) >> Collect()
        auc = roc_auc_score(tars, probs)
    return tars, preds, probs, auc

if __name__ == '__main__':
    fdir = os.path.join(Path.home(), 'datasets', 'wristband_data')

    modalities = ['EDA', 'ACC', 'BVP', 'HR']
    metadata_df = load_metadata(os.path.join(fdir, 'metadata.csv'), n=None, modalities=modalities, szr_sess_only=True)
    folds = leave1out(metadata_df, 'patient')
    nb_classes = 2

    net = create_network(num_channels(modalities), nb_classes)

    print('Number of folds:', len(folds))
    for i, (train, test) in enumerate(folds):
        print(f'Fold {i+1}: loading train and test data... ')

        train_cnn(net, train, fdir)
        tars, preds, probs, auc = evaluate(net, test, fdir)

        break

