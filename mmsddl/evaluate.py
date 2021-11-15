'''
Author:
    Sherry Yu
Initial Version:
    Nov-2020
Function:
   Define some  functions for evaluating the neural network model
'''


import os
import sys
#print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0,os.getcwd())

import numpy as np
import torch
import torch.nn as nn
from nutsflow import *
from nutsml import PrintType, PrintColType
from mmsdcommon.data import gen_session, GenWindow
from mmsdcommon.preprocess import FilterNonMotor, NormaliseRaw, BandpassBvp
from mmsddl.common import MakeBatch, PredBatch, PredBatchAE, Convert2numpy, SessionMinusSeizure
from mmsdcommon.metrics import szr_metrics
import matplotlib.pyplot as plt

from mmsddl.get_cfg import get_CFG


def val_loss(labels, losses):
    labels = np.array(labels)
    losses = np.array(losses)
    norm_losses = losses[labels < 1]
    print(len(norm_losses), np.sum(labels))
    return np.mean(norm_losses)


def evaluate(cfg, net, testset, fdir, test_cache):
    net.eval()
    win_step = 10
    # print('Evaluating: ', testset['patient'].unique())
    with torch.no_grad():
        szrids, tars, preds, probs = (gen_session(testset, fdir,
                                                  relabelling=cfg.szr_types)
                              # >> BandpassBvp()
                              >> NormaliseRaw()
                              >> GenWindow(cfg.win_len, win_step)
                              >> Convert2numpy() >> test_cache
                              >> MakeBatch(cfg, cfg.batch_size, test=True)
                              >> PredBatch(net) >> Unzip())

        tars = tars >> Flatten() >> Clone(cfg.win_len) >> Collect()
        szrids = szrids >> Flatten() >> Clone(cfg.win_len) >> Collect()
        probs = (probs >> Flatten() >> Get(1)
                 >> Clone(cfg.win_len) >> Collect())

    return szr_metrics(szrids, tars, probs, cfg.preictal_len, cfg.postictal_len,
                       single_wrst=cfg.sing_wrst)


def val_loss_AE(cfg, net, val, val_cache):
    net.eval()
    win_step = 10
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        szrids, tars, preds, losses = (val
                                       # >> GenWindow(cfg.win_len, win_step)
                                         >> Convert2numpy()
                                         >> val_cache
                                         >> MakeBatch(cfg, 1, test=True)
                                         >> PredBatchAE(net,
                                                        criterion) >> Unzip())
        assert np.sum(tars) < 1, 'has seizures in validation'
        return np.mean(losses), np.std(losses), np.max(losses)


def eval_AE(cfg, net, test, max_loss):
    net.eval()
    win_step = 10
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        szrids, tars, preds, losses = (test >> GenWindow(cfg.win_len, win_step)
                                       >> Convert2numpy()
                                       >> MakeBatch(cfg, 1, test=True)
                                       >> PredBatchAE(net,
                                                      criterion) >> Unzip())

        tars = tars >> Flatten() >> Clone(cfg.win_len) >> Collect()
        szrids = szrids >> Flatten() >> Clone(cfg.win_len) >> Collect()
        losses = losses >> Clone(cfg.win_len) >> Collect()
        # print(tars, losses)
        # x = [i for i in range(len(tars))]
        # plt.plot(x, np.array(tars) * 0.05, losses)
        # plt.show()

        threshold = np.concatenate([np.arange(1.2, 1, -0.04),np.arange(1, 0.9, -0.02)])
        threshold = threshold * max_loss
        return szr_metrics(szrids, tars, losses, cfg.preictal_len, cfg.postictal_len,
                           single_wrst=cfg.sing_wrst, thresholds=threshold)


def evaluate_AE(cfg, net, testset, fdir, test_cache,
                trainset, train_cache, calculate_threshold = False):
    net.eval()
    win_step = 10
    # print('Evaluating: ', testset['patient'].unique())
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        szrids, tars, preds, losses = (gen_session(testset, fdir,
                                                     relabelling=cfg.szr_types)
                                         # >> BandpassBvp()
                                         >> NormaliseRaw()
                                         >> GenWindow(cfg.win_len, win_step)
                                         >> Convert2numpy()
                                         >> test_cache
                                         >> MakeBatch(cfg, 1, test=True)
                                         >> PredBatchAE(net,
                                                        criterion) >> Unzip())

        tars = tars >> Flatten() >> Collect()
        loss = val_loss(tars, losses)
        szrids = szrids >> Flatten() >> Clone(cfg.win_len) >> Collect()
        preds = preds >> Flatten() >> Collect()
        print(len(tars), np.array(preds).shape, len(losses))
        print('val', max(losses), np.mean(losses))

        if calculate_threshold is True:
            # threshold = train_max_loss(cfg, net, trainset, fdir)
            # print(threshold)
            pass
        else:
            threshold = 0.1
        # y_pred = preds_from_losses(losses, threshold)

        if calculate_threshold:
            x = [i for i in range(len(tars))]
            tars = np.array(tars)*0.05
            plt.plot(x, tars, losses)
            # print(np.sum(y_pred))
            # print(np.sum(tars))
            # plt.plot(losses)

            plt.show()

        y_pred = y_pred >> Clone(cfg.win_len) >> Collect()
        tars = tars >> Clone(cfg.win_len) >> Collect()
        print(len(tars), len(y_pred))



        return szr_metrics(szrids, tars, y_pred, cfg.preictal_len,
                           cfg.postictal_len,
                           single_wrst=cfg.sing_wrst), loss


def preds_from_losses(losses, threshold):
    return [0 if loss <= threshold else 1 for loss in losses]


# def validation_loss(cfg, net, testset, fdir, test_cache):
#     net.eval()
#     win_step = 10
#     # print('Evaluating: ', testset['patient'].unique())
#     criterion = torch.nn.MSELoss()
#     with torch.no_grad():
#         szrids, labels, preds, losses = (gen_session(testset, fdir,
#                                                   relabelling=cfg.szr_types)
#                               # >> BandpassBvp()
#                               >> NormaliseRaw()
#                               >> GenWindow(cfg.win_len, win_step)
#                               >> Convert2numpy()
#                               >> test_cache
#                               >> MakeBatch(cfg, 1, test=True)
#                               >> PredBatchAE(net, criterion) >> Unzip())
#
#         labels = labels >> Flatten() >> Collect()
#         szrids = szrids >> Flatten() >> Clone(cfg.win_len) >> Collect()
#         preds = preds >> Flatten() >> Collect()
#         print(len(labels), np.array(preds).shape, len(losses))
#         print('val',max(losses), np.mean(losses))
#     return val_loss(labels, losses)


# def train_max_loss(cfg, net, trainset, fdir):
#     net.eval()
#     win_step = 10
#     # print('Evaluating: ', testset['patient'].unique())
#     criterion = torch.nn.MSELoss()
#     with torch.no_grad():
#         _, _, _, tr_losses = (gen_session(trainset, fdir,
#                                                      relabelling=cfg.szr_types)
#                                          # >> BandpassBvp()
#                                          >> NormaliseRaw()
#                                          >> GenWindow(cfg.win_len, win_step)
#                                          >> Convert2numpy()
#                                          >> MakeBatch(cfg, 1, test=True)
#                                          >> PredBatchAE(net,
#                                                         criterion) >> Unzip())
#
#     return np.max(tr_losses)

if __name__ == '__main__':
    import os
    from nutsml import PrintType
    from mmsdcommon.data import load_metadata
    from mmsddl.train import create_network, num_channels, create_cache
    from mmsddl.network import load_wgts



    cfg = get_CFG()


    # fdir = os.path.join('/Users/shuangyu/datasets/bch', 'wristband_redcap_data')
    # relabelling = ['Tonic-clonic']
    # fdir = os.path.join('/Users/shuangyu/datasets/bch/wristband_REDCap_202102')
    metapath = cfg.datadir

    relabelling = 'same'

    p_set = load_metadata(os.path.join(metapath, 'metadata.csv'),
                          patient_subset=cfg.patients, modalities=cfg.modalities)

    net = create_network(cfg, num_channels(cfg.modalities), 2)
    load_wgts(net)
    val_cache = create_cache(cfg, 'tt', False)
    metrics = evaluate(cfg, net, p_set, metapath, val_cache)
    print(metrics['auc'])
