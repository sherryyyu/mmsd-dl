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

import numpy as np
import torch
from nutsflow import *
from nutsml import PrintType, PlotLines

from mmsdcommon.data import load_metadata, gen_session, gen_window
from mmsdcommon.cross_validate import leave1out
from mmsdcommon.preprocess import remove_non_motor, sample_imbalance
from mmsdcommon.util import num_channels, PrintAll
from mmsdcnn.network import create_network
from mmsdcnn.constants import CFG
from mmsdcnn.common import MakeBatch, TrainBatch, Convert2numpy, Normalise
from mmsdcnn.evaluate import evaluate
from mmsdcnn.util import print_metrics, print_all_folds
from mmsdcnn.network import save_wgts, load_wgts

@nut_processor
def BalanceSession(sample, sampling):
    same_sess = lambda x: (x.metadata['pid'], x.metadata['sid'])
    for session in sample >> ChunkBy(same_sess, list):
        meta, label, data = session >> Convert2numpy() >> Unzip()
        label, data = sample_imbalance(sampling, np.array(label),
                                       np.array(data))
        for l, d in zip(label, data):
            yield meta[0], l, d


def has_szr(dataset, data_dir):
    '''Check if the dataset contains seizures, can be slow if dataset is big.'''
    print('val check')
    y = (gen_session(dataset, data_dir)
         >> gen_window(CFG.win_len, 0, 0)
         >> remove_non_motor(CFG.motor_threshold)
         >> Get(1) >> Collect())
    all_zeros = not np.array(y).any()
    return not all_zeros


def optimise(nb_classes, trainset):
    folds = leave1out(trainset, 'patient')
    overall, best_auc = [], 0
    for i, (train, val) in enumerate(folds):
        print(f"Fold {i + 1}/{len(folds)}: loading train patients "
              f"{train['patient'].unique()} "
              f"and validation patients {val['patient'].unique()}... ")

        net = create_network(num_channels(CFG.modalities), nb_classes)
        metrics = train_network(net, train, val, i)
        overall.append(
            (metrics['sen_cnt'], metrics['far_cnt'],
             metrics['thresholds'], metrics['auc']))

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            save_wgts(net)

    print_all_folds(overall, len(folds))

    return best_auc


def train_network(net, trainset, valset, i):
    p_path = os.path.join(CFG.plotdir, 'fold%d' % i)
    plotlines = PlotLines((0, 1), layout=(2, 1), figsize=(8, 12),
                          titles=('loss', 'val-auc'), filepath=p_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), CFG.lr)

    train_cache = Cache(CFG.traincachedir + str(i), CFG.cacheclear)
    val_cache = Cache(CFG.valcachedir + str(i), CFG.cacheclear)

    for epoch in range(CFG.n_epochs) >> PrintProgress(CFG.n_epochs):
        start = time.time()
        net.train()

        loss = (gen_session(trainset, CFG.datadir) >> PrintType()
                >> Normalise() >> PrintAll()
                >> gen_window(CFG.win_len, 0.75, 0)
                >> remove_non_motor(CFG.motor_threshold)
                >> BalanceSession('smote') >> train_cache
                >> Shuffle(1000)
                >> MakeBatch(CFG.batch_size)
                >> TrainBatch(net, optimizer, criterion)
                >> Mean())

        if CFG.verbose:
            msg = "Epoch {:d}..{:d}  {:s} : loss {:.4f}"
            elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start))
            print(msg.format(epoch, CFG.n_epochs, elapsed, loss))

        metrics = evaluate(net, valset, CFG.datadir, val_cache)
        plotlines((loss, metrics['auc']))

        if CFG.verbose > 1:
            print_metrics(metrics)

    return metrics


if __name__ == '__main__':
    motor_patients = ['C241', 'C242', 'C245', 'C290', 'C423', 'C433']
    # motor_patients = ['C189', 'C241', 'C242',  'C305']
    metapath = os.path.join(CFG.datadir, 'metadata.csv')
    metadata_df = load_metadata(metapath, n=None,
                                modalities=CFG.modalities,
                                szr_sess_only=True,
                                patient_subset=motor_patients)
    folds = leave1out(metadata_df, 'patient')
    nb_classes = 2

    for i, (train, test) in enumerate(folds):
        best_auc = optimise(nb_classes, train)
        test_cache = Cache(CFG.testcachedir + str(i), CFG.cacheclear)
        net = create_network(num_channels(CFG.modalities), nb_classes)
        load_wgts(net)
        metrics = evaluate(net, test, CFG.datadir, test_cache)
        break


