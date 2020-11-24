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
from nutsml import PrintType, PlotLines, PrintColType

from mmsdcommon.data import load_metadata,  gen_session,  gen_window
from mmsdcommon.cross_validate import leave1out
from mmsdcommon.preprocess import remove_non_motor
from mmsdcommon.util import num_channels
from mmsdcnn.network import create_network
from mmsdcnn.constants import PARAMS, DEVICE
from mmsdcnn.common import MakeBatch, TrainBatch, Convert2numpy, Normalise
from mmsdcnn.evaluate import evaluate

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

# def merge_modalities(session):
#     label = np.array([sample.label for sample in session])
#     data = np.array([np.concatenate(sample[1:], axis=1) for sample in session])
#     return label, data

# @nut_processor
# def Preprocess(sessions):
#     return sessions >> gen_window(PARAMS.win_len, 0.75, 0)  >> remove_non_motor(PARAMS.motor_threshold)


def has_szr(dataset):
    '''Check if the dataset contains seizures, can be slow if dataset is big.'''
    y = gen_session(dataset, fdir) >> gen_window(PARAMS.win_len, 0, 0) >> remove_non_motor(PARAMS.motor_threshold) >> Get(0) >> Collect()
    all_zeros = not np.array(y).any()
    return not all_zeros


@nut_processor
def Preprocess(sessions):
    for session in sessions:
        label, data = ([session] >> Normalise() >>  gen_window(PARAMS.win_len, 0.75, 0)
                       >> remove_non_motor(PARAMS.motor_threshold) >> Convert2numpy() >> Unzip())

        label, data = sample_imbalance('under', np.array(label), np.array(data))
        for i, l in enumerate(label):
            yield l, data[i]


def train_cnn(net, trainset, fdir, i):
    if PARAMS.verbose > 1:
        plotlines = PlotLines((0, 1, 2), layout=(3, 1), figsize=(8, 12), titles=('loss', 'test-auc'))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), PARAMS.lr)
    train_cache = Cache(os.path.join(fdir, PARAMS.cachedir, 'train', 'fold%d' % i), PARAMS.cacheclear)
    test_cache = Cache(os.path.join(fdir, PARAMS.cachedir, 'test', 'fold%d' % i), PARAMS.cacheclear)


    auc = detected_szr = num_szr = num_false_alarms = num_non_seizure_intervals = None
    for epoch in range(PARAMS.n_epochs) >> PrintProgress(PARAMS.n_epochs):
        start = time.time()
        net.train()

        losses = (gen_session(trainset, fdir) >> PrintType() >> Preprocess() >> train_cache
                  >> MakeBatch(PARAMS.batch_size) >> TrainBatch(net, optimizer, criterion) >> Collect())

        loss = np.mean(losses)

        if PARAMS.verbose:
            msg = "Epoch {:d}..{:d}  {:s} : loss {:.4f}"
            elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start))
            print(msg.format(epoch, PARAMS.n_epochs, elapsed, loss))

        auc, (detected_szr, num_szr, num_false_alarms, num_non_seizure_intervals) = evaluate(net, test, fdir, test_cache)

        if PARAMS.verbose:
            print(f'auc {auc}, detected {detected_szr}/{num_szr}, FAR {num_false_alarms}/{num_non_seizure_intervals}')

    return auc, detected_szr, num_szr, num_false_alarms, num_non_seizure_intervals



if __name__ == '__main__':
    fdir = os.path.join(Path.home(), PARAMS.datadir)

    motor_patients = ['C241', 'C242', 'C245', 'C290', 'C423', 'C433']

    metadata_df = load_metadata(os.path.join(fdir, 'metadata.csv'), n=5, modalities=PARAMS.modalities, szr_sess_only=True,
                                patient_subset=motor_patients)
    folds = leave1out(metadata_df, 'patient')
    nb_classes = 2



    avg_auc = total_det_szr = total_szr = total_fa = total_int = 0
    for i, (train, test) in enumerate(folds):
        print(f"Fold {i+1}/{len(folds)}: loading train patients {train['patient'].unique()} and test patients {test['patient'].unique()}... ")
        net = create_network(num_channels(PARAMS.modalities), nb_classes)

        # disabled because it's slow
        # assert has_szr(test), 'Test set contains no seizure, check train-test split or patient set!'

        auc, detected_szr, num_szr, num_false_alarms, num_non_seizure_intervals = train_cnn(net, train, fdir, i)
        print(f'auc {auc}, detected {detected_szr}/{num_szr}, FAR {num_false_alarms}/{num_non_seizure_intervals}')

        avg_auc += auc
        total_det_szr += detected_szr
        total_szr += num_szr
        total_fa += num_false_alarms
        total_int += num_non_seizure_intervals


    avg_auc = avg_auc/len(folds)
    sensitivity = total_det_szr/total_szr
    far = total_fa/total_int
    print(f'Average metrics: avg_auc {avg_auc}, sen {sensitivity} ({total_det_szr}/{total_szr}), FAR {far} ({total_fa}/{total_int})')



