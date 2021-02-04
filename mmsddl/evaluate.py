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
   Define some  functions for evaluating the neural network model
'''

import numpy as np
import torch
from nutsflow import *
from nutsml import PrintType, PrintColType
from mmsdcommon.data import gen_session, GenWindow
from mmsdcommon.preprocess import FilterNonMotor, NormaliseRaw
from mmsddl.constants import CFG
from mmsddl.common import MakeBatch, PredBatch, Convert2numpy
from mmsdcommon.metrics import szr_metrics


def evaluate(net, testset, fdir, test_cache):
    net.eval()
    win_step = 10
    with torch.no_grad():
        tars, preds, probs = (gen_session(testset, fdir,
                                          relabelling=CFG.szr_types)
                              >> NormaliseRaw()
                              >> GenWindow(CFG.win_len, win_step)
                              >> Convert2numpy() >> test_cache
                              >> MakeBatch(CFG.batch_size)
                              >> PredBatch(net) >> Unzip())

        tars = tars >> Flatten() >> Clone(CFG.win_len) >> Collect()
        probs = (probs >> Flatten() >> Get(1)
                 >> Clone(CFG.win_len) >> Collect())
    return szr_metrics(tars, probs, CFG.preictal_len, CFG.postictal_len)

