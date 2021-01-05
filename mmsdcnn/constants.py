"""
Licensed Materials - Property of IBM
(C) Copyright IBM Corp. 2020. All Rights Reserved.

US Government Users Restricted Rights - Use, duplication or
disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

Author:
    Sherry Yu
Initial Version:
    Nov-2020
Function:
   Constants
"""

import torch
from nutsml.config import Config
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = '/Users/shuangyu/datasets/bch/'
DATADIR = 'wristband_redcap_data'
gtc = ['Tonic-clonic']
# ROOT = '/Users/shuangyu/datasets/'
# DATADIR = 'wristband_data'


# ROOT = '/fast1/'  # On ER01

CFG = Config(
    n_epochs=15,
    lr=1e-3,
    batch_size=32,
    verbose=1,
    win_len=10,
    win_step=2,
    rootdir=ROOT,
    datadir=ROOT + DATADIR,
    traincachedir=ROOT + 'cache/train/fold',
    valcachedir=ROOT + 'cache/val/fold',
    testcachedir=ROOT + 'cache/test/fold',
    plotdir=ROOT + 'plots',
    ckpdir='checkpoints',
    modalities=['EDA', 'ACC', 'BVP', 'HR'],
    szr_types = gtc,
    preictal_len=60,
    postictal_len=60,
    motor_threshold=0.1,
    operating_pt=0.5,
    cacheclear=True)
