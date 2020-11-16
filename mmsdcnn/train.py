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

from mmsdcommon.data import load_metadata,  gen_session, list_modalities, gen_window
from mmsdcommon.cross_validate import leave1out
from mmsdcommon.preprocess import remove_non_motor
from mmsdcnn.network import create_network
from mmsdcnn.constants import PARAMS, DEVICE
from pathlib import Path
import os
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from nutsflow import *
from nutsml import PrintType
import torch


def sample_imbalance(sampling, label, data):
    data_shape = data.shape

    # It seems like the imblearn library only accepts data of shape 2,
    # therefore we reshape the data to shape 2 before passing into the sampler and retrieve the shape after
    if len(data_shape) == 3:
        data = np.reshape(data, [data_shape[0], data_shape[1] * data_shape[2]])

    if sampling == 'over':
        sampler = RandomOverSampler()
    elif sampling == 'smote':
        sampler = SMOTE()
    elif sampling == 'under':
        sampler = RandomUnderSampler()
    elif sampling == 'adasyn':
        sampler = ADASYN()
    else:
        print('Error: invalid sampler name, using Random Under Sampling.')
        sampler = RandomUnderSampler()
    data, label = sampler.fit_resample(data, label)

    # Retrieve the original data shape if the original shape is 3
    if len(data_shape) == 3:
        data = np.reshape(data, [-1, data_shape[1], data_shape[2]])
    return label, data

def merge_modalities(session):
    label = np.array([sample.label for sample in session])
    data = np.array([np.concatenate(list(sample[1:]), axis=1) for sample in session])
    return label, data


@nut_function
def preprocess_session_nut(session):
    wins = [session] >> PrintType() >> gen_window(10, 0.75, 0)  >> remove_non_motor(0.1) >> Collect()
    y_sess, x_sess = merge_modalities(wins)
    yield y_sess, x_sess

def preprocess_session():
    for session in gen_session(fold,fdir,train=True):
        wins = [session] >> PrintType() >> gen_window(10, 0.75, 0)  >> remove_non_motor(0.1)  >> Collect()
        y_sess, x_sess = merge_modalities(wins)
        yield y_sess, x_sess

def to_numpy(x):
    return x.detach().cpu().numpy()

def probabilities(pred):
    return torch.softmax(pred, 1)

@nut_processor
def MakeBatch(samples, batchsize):
    for batch in samples >> Chunk(batchsize):
        data, targets = batch >> Unzip()
        data_batch = torch.tensor(data).to(DEVICE)
        tar_batch = torch.tensor(targets).to(DEVICE)
        yield data_batch, tar_batch


@nut_function
def PredBatch(batch, net):
    data, targets = batch
    pred = net(data)
    probs = probabilities(pred)
    preds = torch.max(probs, 1)[1].view_as(targets)
    return to_numpy(targets), to_numpy(preds), to_numpy(probs)


@nut_function
def TrainBatch(batch, net, optimizer, criterion):
    data, labels = batch
    pred = net(data)
    loss = criterion(pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train_cnn(net, y_train, x_train):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=PARAMS['lr'])

    print(x_train.shape, y_train.shape)
    x_train = [x_train[i, :, :] for i in range(x_train.shape[0])]
    samples = x_train >> Zip(y_train)

    for epoch in range(PARAMS['num_epoch']):
        print('epoch ', epoch)
        samples >> MakeBatch(PARAMS['batch_size']) >> TrainBatch(net, optimizer, criterion) >> Consume()





if __name__ == '__main__':
    fdir = os.path.join(Path.home(), 'datasets', 'wristband_data')

    modalities = ['EDA', 'ACC', 'BVP']
    metadata_df = load_metadata(os.path.join(fdir, 'metadata.csv'), n=2, modalities=modalities, szr_sess_only=True)
    folds, num = leave1out(metadata_df, 'patient')

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

    net = create_network(640, 2)

    print('Number of folds:', num)
    for i, fold in enumerate(folds):
        print(f'Fold {i+1}: loading train and test data... ')

        # gen_session(fold, fdir, train=True) >> preprocess_session_nut() >> PrintType() >> Consume()

        out = preprocess_session() >> PrintType() >> Collect()

        # merge all sessions
        y_train = np.concatenate([i[0] for i in out])
        x_train = np.concatenate([i[1] for i in out])
        # over sample
        label, data = sample_imbalance('smote', y_train, x_train)
        # one-hot
        y_train = enc.fit_transform(np.reshape(y_train, (len(y_train), 1)))
        print(y_train.shape, x_train.shape)

        train_cnn(net, y_train, x_train)




        # # windows = gen_session(fold, fdir, train=True) >> PrintType() >> gen_window(10, 0.75, 0) >> Collect()
        # windows = gen_session(fold,fdir,train=True) >> PrintType()  >> gen_window(10, 0.75, 0) >> remove_non_motor(0.1) >> Collect()
        # y, x = samples2ndarrays(windows)
        # print(x.shape)
        # y_under, x_under = sample_imbalance('over', y, x)
        # print(y_under.shape, x_under.shape)



        # enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        # y_train = enc.fit_transform(np.reshape(y_train, (len(y_train), 1)))
        break

        # train the CNN model

