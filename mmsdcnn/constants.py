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
ROOT = '/Users/shuangyu/datasets/'

# ROOT = '/fast1/'  # On ER01

CFG = Config(
          n_epochs = 50,
          lr= 1e-3,
          batch_size= 256,
          win_len = 10,
          verbose = 2,
          rootdir = ROOT,
          datadir = ROOT+'wristband_data',
          traincachedir= ROOT+'cache/train/fold',
          valcachedir= ROOT+'cache/val/fold',
          testcachedir= ROOT+'cache/test/fold',
          plotdir = ROOT+'plots',
          modalities = ['EDA', 'ACC', 'BVP', 'HR'],
          preictal_len = 60,
          postictal_len = 0,
          motor_threshold = 0.1,
          operating_pt = 0.5,
          cacheclear=True)

