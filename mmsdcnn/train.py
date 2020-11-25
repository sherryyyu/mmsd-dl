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
   Define some functions for neural network training
'''

import time
import os
from pathlib import Path

import numpy as np
import torch
from nutsflow import *
from nutsml import PrintType, PlotLines

from mmsdcommon.metrics import sen_far_count, roc_curve, roc_auc_score
from mmsdcommon.data import load_metadata,  gen_session,  gen_window
from mmsdcommon.cross_validate import leave1out
from mmsdcommon.preprocess import remove_non_motor
from mmsdcommon.util import num_channels
from mmsdcnn.network import create_network
from mmsdcnn.constants import PARAMS
from mmsdcnn.common import MakeBatch, TrainBatch, Convert2numpy, Normalise
from mmsdcnn.evaluate import evaluate
from mmsdcommon.print import PrintAll


def sample_imbalance(sampling, label, data):
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
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
        raise ValueError('Invalid over/undersampling type!')
    data, label = sampler.fit_resample(data, label)

    # Retrieve the original data shape if the original shape is 3
    if len(data_shape) == 3:
        data = np.reshape(data, [-1, data_shape[1], data_shape[2]])
    return label, data


def has_szr(dataset):
    '''Check if the dataset contains seizures, can be slow if dataset is big.'''
    y = (gen_session(dataset, fdir) >>
         gen_window(PARAMS.win_len, 0, 0) >>
         remove_non_motor(PARAMS.motor_threshold) >>
         Get(0) >>
         Collect())
    all_zeros = not np.array(y).any()
    return not all_zeros


@nut_processor
def Preprocess(sessions):
    for session in sessions:
        label, data = ([session] >>
                       Normalise() >>
                       PrintAll() >>
                       gen_window(PARAMS.win_len, 0.75, 0) >>
                       remove_non_motor(PARAMS.motor_threshold) >>
                       Convert2numpy() >>
                       Unzip())

        label, data = sample_imbalance('smote', np.array(label), np.array(data))
        for l,d in zip(label, data):
            yield l,d


def train_cnn(net, trainset, fdir, i):
    if PARAMS.verbose > 1:
        p_path = os.path.join(fdir, 'plots', 'fold%d' % i)
        plotlines = PlotLines((0, 1), layout=(2, 1),
                              figsize=(8, 12),
                              titles=('loss', 'test-auc'),
                              filepath=p_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), PARAMS.lr)
    train_cache = Cache(os.path.join(fdir, PARAMS.cachedir, 'train', 'fold%d' % i), PARAMS.cacheclear)
    test_cache = Cache(os.path.join(fdir, PARAMS.cachedir, 'test', 'fold%d' % i), PARAMS.cacheclear)



    #auc  = operating_pts = sens = fars = None
    for epoch in range(PARAMS.n_epochs) >> PrintProgress(PARAMS.n_epochs):
        start = time.time()
        net.train()

        loss = (gen_session(trainset, fdir) >> PrintType() >>
                Preprocess() >> train_cache >> MakeBatch(PARAMS.batch_size) >>
                TrainBatch(net, optimizer, criterion) >> Mean())

        if PARAMS.verbose:
            msg = "Epoch {:d}..{:d}  {:s} : loss {:.4f}"
            elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start))
            print(msg.format(epoch, PARAMS.n_epochs, elapsed, loss))

        tars, probs = evaluate(net, test, fdir, test_cache)

        operating_pts, sens, fars = sen_far_count(tars, probs, PARAMS.preictal_len, PARAMS.postictal_len)
        sen, far = roc_curve(sens, fars)
        auc = roc_auc_score(sen, far)


        if PARAMS.verbose:
            print('sensitivity', sen)
            print('FAR',far)
            print(f'auc {auc}')
            i = len(operating_pts)//2
            print(f'Operating point = {operating_pts[i]:.2f},' +
                   'SEN = {sens[i][0]}/{sens[i][1]},' +
                   'FAR = {fars[i][0]}/{fars[i][1]}')

        if PARAMS.verbose > 1:
            plotlines((loss, auc))

    return auc, operating_pts, sens, fars



if __name__ == '__main__':
    fdir = os.path.join(Path.home(), PARAMS.datadir)
    motor_patients = ['C241', 'C242', 'C245', 'C290', 'C423', 'C433']
    metadata_df = load_metadata(os.path.join(fdir, 'metadata.csv'), n=5,
                                modalities=PARAMS.modalities, szr_sess_only=True,
                                patient_subset=motor_patients)
    folds = leave1out(metadata_df, 'patient')
    nb_classes = 2

    sen_all = []
    far_all = []
    operating_pts = None
    for i, (train, test) in enumerate(folds):
        print(f"Fold {i+1}/{len(folds)}: loading train patients {train['patient'].unique()} and test patients {test['patient'].unique()}... ")
        net = create_network(num_channels(PARAMS.modalities), nb_classes)

        # disabled because it's slow
        # assert has_szr(test), 'Test set contains no seizure, check train-test split or patient set!'

        auc, operating_pts, det_szrs, fars = train_cnn(net, train, fdir, i)
        sen_all.append(det_szrs)
        far_all.append(fars)

    sens = []
    fars = []
    for i, op in enumerate(operating_pts):
        s = list(zip(*[sen_all[j][i] for j in range(len(folds))]))
        sen = sum(s[0])/sum(s[1])
        f = list(zip(*[far_all[j][i] for j in range(len(folds))]))
        far = sum(f[0])/sum(f[1])
        sens.append(sen)
        fars.append(far)
        print(f'op = {op:.2f}, sen = {sen}, far = {far}')

    print('auc = ', roc_auc_score(sens, fars))




