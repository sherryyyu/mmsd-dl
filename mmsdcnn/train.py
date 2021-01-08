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
from torch.utils.tensorboard import SummaryWriter

from mmsdcommon.data import load_metadata, gen_session, GenWindow
from mmsdcommon.cross_validate import leave1out
from mmsdcommon.preprocess import (FilterNonMotor, sample_imbalance,
                                   NormaliseRaw, FilterSzrFree)
from mmsdcommon.util import num_channels, metrics2print, print_all_folds

from mmsdcnn.network import create_network
from mmsdcnn.constants import CFG
from mmsdcnn.common import MakeBatch, TrainBatch, Convert2numpy
from mmsdcnn.evaluate import evaluate
from mmsdcnn.util import print_metrics
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


def optimise(nb_classes, trainset, n_fold):
    folds = leave1out(trainset, 'patient')
    all_metrics, best_auc = [], 0
    for i, (train, val) in enumerate(folds):
        print(f"Fold {i + 1}/{len(folds)}: loading train patients "
              f"{train['patient'].unique()} "
              f"and validation patients {val['patient'].unique()}... ")

        net = create_network(num_channels(CFG.modalities), nb_classes)
        i_fold = str(n_fold) + '-' + str(i)
        metrics, best_auc = train_network(net, train, val, best_auc, i_fold)
        all_metrics.append(metrics2print(metrics))

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


def log2tensorboard(writer, epoch, loss, metrics):
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("AUC/val", metrics['auc'], epoch)


def train_network(net, trainset, valset, best_auc, fold_no):
    writer = SummaryWriter()

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

        loss = (gen_session(trainset, CFG.datadir, relabelling=CFG.szr_types)
                >> PrintProgress(n_sessions)
                >> FilterSzrFree()
                >> NormaliseRaw()
                >> GenWindow(CFG.win_len, CFG.win_step)
                >> BalanceSession('under')
                >> train_cache
                >> Shuffle(50)
                >> MakeBatch(CFG.batch_size)
                >> TrainBatch(net, optimizer, criterion)
                >> Mean())

        metrics = evaluate(net, valset, CFG.datadir, val_cache)
        log2tensorboard(writer, epoch, loss, metrics)

        if CFG.verbose:
            msg = "Epoch {:d}..{:d}  {:s} : loss {:.4f} val-auc {:.4f}"
            print(msg.format(epoch, CFG.n_epochs, str(t), loss, metrics['auc']))

        state = get_state(fold_no, epoch, best_auc, metrics, net, optimizer)
        save_ckp(state, CFG.ckpdir, fold_no)

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            save_wgts(net)

        if CFG.verbose > 1:
            print_metrics(metrics)
    writer.close()
    return metrics, best_auc


if __name__ == '__main__':
    # gtc_patients = ['C242', 'C245', 'C290', 'C333', 'C380', 'C387']
    gtc_patients = ['C309', 'C290', 'C333', 'C380', 'C387']
    FBTC = ['C189', 'C192', 'C225', 'C226', 'C232', 'C234', 'C241', 'C242',
           'C245', 'C296', 'C417', 'C421']
    metapath = os.path.join(CFG.datadir, 'metadata.csv')
    metadata_df = load_metadata(metapath, n=None,
                                modalities=CFG.modalities,
                                szr_sess_only=True,
                                patient_subset=FBTC)
    folds = leave1out(metadata_df, 'patient')
    nb_classes = 2

    testp_metrics = []
    for i, (train, test) in enumerate(folds):
        best_auc = optimise(nb_classes, train, i)
        net = create_network(num_channels(CFG.modalities), nb_classes)
        metrics, _ = train_network(net, train, test, 0, i)
        testp_metrics.append(metrics2print(metrics))
        break

    print('LOO test results:')
    print_all_folds(testp_metrics, len(folds))