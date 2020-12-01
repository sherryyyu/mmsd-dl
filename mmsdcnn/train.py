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


from mmsdcommon.data import load_metadata, gen_session, gen_window
from mmsdcommon.cross_validate import leave1out
from mmsdcommon.preprocess import remove_non_motor, sample_imbalance
from mmsdcommon.util import num_channels
from mmsdcnn.network import create_network
from mmsdcnn.constants import PARAMS
from mmsdcnn.common import MakeBatch, TrainBatch, Convert2numpy, Normalise
from mmsdcnn.evaluate import evaluate
from mmsdcnn.util import print_metrics, print_all_folds

@nut_processor
def SampleImb(sample, sampling):
    same_sess = lambda x: (x.metadata['pid'], x.metadata['sid'])
    for session in sample >> ChunkBy(same_sess, list):
        meta, label, data = session >> Convert2numpy() >> Unzip()
        label, data = sample_imbalance(sampling, np.array(label),
                                       np.array(data))
        for l, d in zip(label, data):
            yield meta[0], l, d


def has_szr(dataset, data_dir):
    '''Check if the dataset contains seizures, can be slow if dataset is big.'''
    y = (gen_session(dataset, data_dir)
         >> gen_window(PARAMS.win_len, 0, 0)
         >> remove_non_motor(PARAMS.motor_threshold)
         >> Get(1) >> Collect())
    all_zeros = not np.array(y).any()
    return not all_zeros


def optimise(nb_classes, trainset, rootdir):
    folds = leave1out(trainset, 'patient')

    overall, best_net, best_auc = [], None, 0
    for i, (train, val) in enumerate(folds):
        print(f"Fold {i + 1}/{len(folds)}: loading train patients "
              f"{train['patient'].unique()} "
              f"and validation patients {val['patient'].unique()}... ")
        # disabled because it's slow
        # assert has_szr(val, data_dir), 'Val set contains no seizure, check train-test split or patient set!'

        net = create_network(num_channels(PARAMS.modalities), nb_classes)
        metrics = train_network(net, train, val, rootdir, i)
        overall.append(
            (metrics['sen_cnt'], metrics['far_cnt'], metrics['thresholds']))

        if metrics['auc'] > best_auc:
            # TODO: change this placeholder save to save_wgts(net)
            best_net, best_auc = net, metrics['auc']

    print_all_folds(overall, len(folds))

    return best_net


def train_network(net, trainset, valset, fdir, i):
    p_path = os.path.join(fdir, PARAMS.plotdir, 'fold%d' % i)
    plotlines = PlotLines((0, 1), layout=(2, 1), figsize=(8, 12),
                          titles=('loss', 'val-auc'), filepath=p_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), PARAMS.lr)
    train_cache = Cache(
        os.path.join(fdir, PARAMS.cachedir, 'train', 'fold%d' % i),
        PARAMS.cacheclear)
    val_cache = Cache(
        os.path.join(fdir, PARAMS.cachedir, 'val', 'fold%d' % i),
        PARAMS.cacheclear)

    data_dir = os.path.join(fdir, PARAMS.datadir)

    for epoch in range(PARAMS.n_epochs) >> PrintProgress(PARAMS.n_epochs):
        start = time.time()
        net.train()

        loss = (gen_session(trainset, data_dir) >> PrintType()
                >> Normalise()
                >> gen_window(PARAMS.win_len, 0.75, 0)
                >> remove_non_motor(PARAMS.motor_threshold)
                >> SampleImb('under') >> train_cache
                >> MakeBatch(PARAMS.batch_size)
                >> TrainBatch(net, optimizer, criterion)
                >> Mean())

        if PARAMS.verbose:
            msg = "Epoch {:d}..{:d}  {:s} : loss {:.4f}"
            elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start))
            print(msg.format(epoch, PARAMS.n_epochs, elapsed, loss))

        metrics = evaluate(net, valset, data_dir, val_cache)
        plotlines((loss, metrics['auc']))

        if PARAMS.verbose > 1:
            print_metrics(metrics)

    return metrics


if __name__ == '__main__':
    rootdir = Path.home()
    data_dir = os.path.join(rootdir, PARAMS.datadir)
    motor_patients = ['C241', 'C242', 'C245', 'C290', 'C423', 'C433']
    metadata_df = load_metadata(os.path.join(data_dir, 'metadata.csv'),
                                n=None, modalities=PARAMS.modalities,
                                szr_sess_only=True,
                                patient_subset=motor_patients)
    folds = leave1out(metadata_df, 'patient')
    nb_classes = 2

    for i, (train, test) in enumerate(folds):
        net = optimise(nb_classes, train, rootdir)

        test_cache = Cache(
            os.path.join(rootdir, PARAMS.cachedir, 'test', 'tfold%d' % i),
            PARAMS.cacheclear)
        metrics = evaluate(net, test, data_dir, test_cache)
        break


