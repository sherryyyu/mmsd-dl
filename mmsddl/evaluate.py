'''
Author:
    Sherry Yu
Initial Version:
    Nov-2020
Function:
   Define some  functions for evaluating the neural network model
'''

import numpy as np
import torch
from nutsflow import *
from nutsml import PrintType, PrintColType
from mmsdcommon.data import gen_session, GenWindow
from mmsdcommon.preprocess import FilterNonMotor, NormaliseRaw
from mmsddl.common import MakeBatch, PredBatch, Convert2numpy
from mmsdcommon.metrics import szr_metrics

from mmsddl.get_cfg import get_CFG


def evaluate(CFG, net, testset, fdir, test_cache):
    net.eval()
    win_step = 10
    with torch.no_grad():
        szrids, tars, preds, probs = (gen_session(testset, fdir,
                                          relabelling=CFG.szr_types)
                              >> NormaliseRaw()
                              >> GenWindow(CFG.win_len, win_step)
                              >> Convert2numpy() >> test_cache
                              >> MakeBatch(CFG, CFG.batch_size, test=True)
                              >> PredBatch(net) >> Unzip())

        tars = tars >> Flatten() >> Clone(CFG.win_len) >> Collect()
        szrids = szrids >> Flatten() >> Clone(CFG.win_len) >> Collect()
        probs = (probs >> Flatten() >> Get(1)
                 >> Clone(CFG.win_len) >> Collect())

    return szr_metrics(szrids, tars, probs, CFG.preictal_len, CFG.postictal_len,
                       single_wrst=False)

if __name__ == '__main__':
    import os
    from nutsml import PrintType
    from mmsdcommon.data import load_metadata
    from mmsddl.train import create_network, num_channels, create_cache
    from mmsddl.network import load_wgts



    CFG = get_CFG()


    # fdir = os.path.join('/Users/shuangyu/datasets/bch', 'wristband_redcap_data')
    # relabelling = ['Tonic-clonic']
    # fdir = os.path.join('/Users/shuangyu/datasets/bch/wristband_REDCap_202102')
    metapath = CFG.datadir

    relabelling = 'same'

    if len(CFG.patients)==0:
        p_set = load_metadata(os.path.join(metapath, 'metadata.csv'),
                           modalities=CFG.modalities)
    else:
        p_set = load_metadata(os.path.join(metapath, 'metadata.csv'),
                              patient_subset=CFG.patients, modalities=CFG.modalities)

    net = create_network(CFG, num_channels(CFG.modalities), 2)
    load_wgts(net)
    val_cache = create_cache(CFG, 'tt', False)
    metrics = evaluate(CFG, net, p_set, metapath, val_cache)
    print(metrics['auc'])
