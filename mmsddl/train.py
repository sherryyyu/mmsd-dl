'''
Author:
    Sherry Yu
Initial Version:
    Nov-2020
Function:
   Define some functions for neural network training
'''


import os
import sys
# print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0,os.getcwd())

import time
import numpy as np
import torch
from nutsflow import *
from nutsml import PrintType, PlotLines
# from torch.utils.tensorboard import SummaryWriter

from joblib import Parallel, delayed
import multiprocessing as mp

from mmsdcommon.data import load_metadata, gen_session, GenWindow
from mmsdcommon.cross_validate import *
from mmsdcommon.preprocess import (FilterNonMotor, sample_imbalance,
                                   NormaliseRaw, FilterSzrFree)
from mmsdcommon.util import num_channels, metrics2print, print_all_folds,save_all_folds

from mmsddl.network import create_network
from mmsddl.get_cfg import get_CFG
from mmsddl.common import MakeBatch, TrainBatch, Convert2numpy
from mmsddl.evaluate import evaluate
from mmsddl.util import print_metrics
from mmsddl.network import save_wgts, load_wgts, save_ckp, load_ckp


@nut_processor
def BalanceSession(sample, sampling):
    same_sess = lambda x: (x.metadata['pid'], x.metadata['sid'])
    for session in sample >> ChunkBy(same_sess, list):
        meta, id, label, data = session >> Convert2numpy() >> Unzip()
        label, data = sample_imbalance(sampling, np.array(label),
                                       np.array(data))
        for l, d in zip(label, data):
            yield meta[0], l, d


def optimise(CFG, nb_classes, trainset, n_fold):
    folds = leave1out(trainset, 'patient')
    # folds = crossfold(trainset, 'patient', 3)

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


def create_cache(CFG, fold_no, is_train):
    cachedir = CFG.traincachedir if is_train else CFG.valcachedir
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


def train_network(CFG, net, trainset, valset, best_auc, fold_no, total_folds):

    # tensorboard off
    # writer = SummaryWriter()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), CFG.lr)

    state = load_ckp(CFG.ckpdir, net, optimizer, best_auc, fold_no)
    net, optimizer, start_epoch, best_auc, metrics = state

    train_cache = create_cache(CFG, fold_no, True)
    val_cache = create_cache(CFG, fold_no, False)

    for epoch in range(start_epoch, CFG.n_epochs):
        t = Timer()
        net.train()

        loss = (gen_session(trainset, CFG.datadir, relabelling=CFG.szr_types)
                >> FilterSzrFree()
                >> NormaliseRaw()
                >> GenWindow(CFG.win_len, CFG.win_step)
                >> BalanceSession('under')
                >> train_cache
                >> Shuffle(50)
                >> MakeBatch(CFG, CFG.batch_size)
                >> TrainBatch(net, optimizer, criterion)
                >> Mean())

        metrics = evaluate(CFG, net, valset, CFG.datadir, val_cache)

        # tensorboard off
        # log2tensorboard(writer, epoch, loss, metrics)

        # print('Evaluating: ', valset['patient'].unique())

        if CFG.verbose:
            msg = "Fold {:d}/{:d} Epoch {:d}/{:d}  {:s} : loss {:.4f} val-auc {:.4f}"
            print(CFG.szr_types[0], CFG.modalities, msg.format(fold_no, total_folds, epoch, CFG.n_epochs, str(t), loss, metrics['auc']))

        state = get_state(fold_no, epoch, best_auc, metrics, net, optimizer)
        save_ckp(state, CFG.ckpdir, fold_no)

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            save_wgts(net)

        if CFG.verbose > 1:
            print_metrics(metrics)

    # tensorboard off
    #writer.close()

    if start_epoch>=CFG.n_epochs:
        metrics = evaluate(CFG, net, valset, CFG.datadir, val_cache)
        if CFG.verbose:
            msg = "Fold {:d}/{:d} Epoch {:d}/{:d} val-auc {:.4f}"
            print(CFG.szr_types[0], CFG.modalities, msg.format(fold_no, total_folds,  start_epoch, CFG.n_epochs, metrics['auc']))

    return metrics, best_auc


def train_fold(i, train, test, cfg, nb_classes):
    print(f"Fold {i + 1}/{len(folds)}: loading train patients "
          f"{train['patient'].unique()} "
          f"and test patients {test['patient'].unique()}... ")

    net = create_network(cfg, num_channels(cfg.modalities), nb_classes)
    metrics, _ = train_network(cfg, net, train, test, 0, i)
    return metrics2print(metrics)


if __name__ == '__main__':
    cfg = get_CFG()

    metapath = os.path.join(cfg.datadir, 'metadata.csv')

    metadata_df = load_metadata(metapath, n=None,
                                modalities=cfg.modalities,
                                szr_sess_only=True,
                                patient_subset=cfg.patients)

    if cfg.crossfold == -1:
        folds = leave1out(metadata_df, 'patient')
    else:
        folds = crossfold(metadata_df, 'patient', cfg.crossfold)

    # cross_folds = crossfold(metadata_df, 'patient',3)
    nb_classes = 2

    # for parallel training, which seems work, but not really parallel for GPU tasks
    # index_folds = []
    # for i, (train, test) in enumerate(folds):
    #     index_folds.append((i, train, test))
    #
    # n_job = min(mp.cpu_count(), cfg.max_cpu)
    # testp_metrics = Parallel(n_jobs=n_job, prefer="threads")(
    #     delayed(train_fold)(i, train, test, cfg, nb_classes)
    #     for i, train, test in index_folds)


    # for non parallel training
    testp_metrics = []
    for i, (train, test) in enumerate(folds):
        # best_auc = optimise(cfg, nb_classes, train, i)

        # print(f"Fold {i + 1}/{len(folds)}: loading train patients "
        #       f"{train['patient'].unique()} "
        #       f"and test patients {test['patient'].unique()}... ")

        net = create_network(cfg, num_channels(cfg.modalities), nb_classes)
        metrics, _ = train_network(cfg, net, train, test, 0, i,len(folds))
        testp_metrics.append(metrics2print(metrics))

    print('LOO test results:')
    results = print_all_folds(testp_metrics, len(folds), cfg, cfg.metric_results_dir)
    save_all_folds(cfg.metric_results_dir, results, cfg)