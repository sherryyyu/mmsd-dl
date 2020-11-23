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
from mmsdcommon.metrics import roc_auc_score, seizure_metrics
from mmsdcommon.data import gen_session,  gen_window
from mmsdcommon.preprocess import remove_non_motor
from mmsdcnn.constants import PARAMS
from mmsdcnn.common import MakeBatch, PredBatch, Convert2numpy


def evaluate(net, testset, fdir, test_cache):
    net.eval()
    with torch.no_grad():
        tars, preds, probs = (gen_session(testset, fdir)
                              >> gen_window(PARAMS.win_len, 0, 0) >> remove_non_motor(PARAMS.motor_threshold) >> Convert2numpy() >> test_cache
                              >> MakeBatch(PARAMS.batch_size) >> PredBatch(net) >> Unzip())

        tars = tars  >> Flatten() >> Clone(PARAMS.win_len) >> Collect()
        preds = preds >> Flatten() >> Clone(PARAMS.win_len) >> Collect()
        probs = probs >> Flatten() >> Get(1) >> Clone(PARAMS.win_len) >> Collect()

        auc = roc_auc_score(probs, tars)

    return auc, seizure_metrics(tars, probs, 0.5, 60)

