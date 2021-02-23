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
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0,os.getcwd())

import numpy as np
import torch
from nutsflow import *
from nutsml import PrintType, PrintColType
from mmsdcommon.data import gen_session, GenWindow
from mmsdcommon.preprocess import FilterNonMotor, NormaliseRaw
from mmsddl.common import MakeBatch, PredBatch, Convert2numpy
from mmsdcommon.metrics import szr_metrics

from mmsddl.get_cfg import get_CFG


def evaluate(cfg, net, testset, fdir, test_cache):
    net.eval()
    win_step = 10
    print('Evaluating: ', testset['patient'].unique())
    with torch.no_grad():
        szrids, tars, preds, probs = (gen_session(testset, fdir,
                                                  relabelling=cfg.szr_types)
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
