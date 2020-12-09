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

from mmsdcommon.data import load_metadata, gen_session, GenWindow
from mmsdcommon.cross_validate import leave1out
from mmsdcommon.preprocess import FilterNonMotor, sample_imbalance
from mmsdcommon.util import num_channels, PrintAll
from mmsdcnn.network import create_network
from mmsdcnn.constants import CFG
from mmsdcnn.common import MakeBatch, TrainBatch, Convert2numpy, Normalise
from mmsdcnn.evaluate import evaluate
from mmsdcnn.util import print_metrics, print_all_folds
from mmsdcnn.network import save_wgts, load_wgts, save_ckp, load_ckp


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
    y = (gen_session(dataset, data_dir)
         >> GenWindow(CFG.win_len, 0, 0)
         >> FilterNonMotor(CFG.motor_threshold)
         >> Get(1) >> Collect())
    all_zeros = not np.array(y).any()
    return not all_zeros


def optimise(nb_classes, trainset):
    folds = leave1out(trainset, 'patient')
    all_metrics, best_auc = [], 0
    for i, (train, val) in enumerate(folds):
        print(f"Fold {i + 1}/{len(folds)}: loading train patients "
              f"{train['patient'].unique()} "
              f"and validation patients {val['patient'].unique()}... ")

        net = create_network(num_channels(CFG.modalities), nb_classes)
        metrics, best_auc = train_network(net, train, val, best_auc, i)
        all_metrics.append(
            (metrics['sen_cnt'], metrics['far_cnt'],
             metrics['thresholds'],
             metrics['auc']))

    print_all_folds(all_metrics, len(folds))

    return best_auc


def create_cache(cfg, fold_no, is_train):
    cachedir = cfg.traincachedir if is_train else cfg.valcachedir
    return Cache(cachedir + str(fold_no), CFG.cacheclear)


def get_state(i, epoch, best_auc, metrics, net, optimizer):
    state = {
        'fold_no': i,
        'epoch': epoch + 1,
        'best_auc': best_auc,
        'metrics': metrics,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    return state


def train_network(net, trainset, valset, best_auc, fold_no):
    p_path = os.path.join(CFG.plotdir, 'fold%d' % fold_no)
    plotlines = PlotLines((0, 1), layout=(2, 1), figsize=(8, 12),
                          titles=('loss', 'val-auc'), filepath=p_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), CFG.lr)

    state = load_ckp(CFG.ckpdir, net, optimizer, best_auc, fold_no)
    net, optimizer, start_epoch, best_auc, metrics = state

    train_cache = create_cache(CFG, fold_no, True)
    val_cache = create_cache(CFG, fold_no, False)

    n_sessions = len(trainset.index)

    for epoch in range(start_epoch, CFG.n_epochs):
        t = Timer()
        net.train()

        loss = (gen_session(trainset, CFG.datadir)
                >> PrintProgress(n_sessions)
                # >> PrintType()
                >> Normalise()
                >> GenWindow(CFG.win_len, CFG.win_overlap)
                >> FilterNonMotor(CFG.motor_threshold)
                >> BalanceSession('under') >> train_cache
                >> Shuffle(50)
                >> MakeBatch(CFG.batch_size)
                >> TrainBatch(net, optimizer, criterion)
                >> Mean())

        metrics = evaluate(net, valset, CFG.datadir, val_cache)

        if CFG.verbose:
            msg = "Epoch {:d}..{:d}  {:s} : loss {:.4f} val-auc {:.4f}"
            print(msg.format(epoch, CFG.n_epochs, str(t), loss, metrics['auc']))

        state = get_state(fold_no, epoch, best_auc, metrics, net, optimizer)
        save_ckp(state, CFG.ckpdir, fold_no)

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            save_wgts(net)

        plotlines((loss, metrics['auc']))

        if CFG.verbose > 1:
            print_metrics(metrics)

    return metrics, best_auc


if __name__ == '__main__':
    motor_patients = ['C241', 'C242', 'C245', 'C290', 'C423', 'C433']
    # motor_patients = ['C189', 'C241', 'C242',  'C305']
    metapath = os.path.join(CFG.datadir, 'metadata.csv')
    metadata_df = load_metadata(metapath, n=5,
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
