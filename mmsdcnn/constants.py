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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PARAMS = Config(n_epochs =  2,
          lr= 1e-3,
          batch_size= 256,
          win_len = 10,
          verbose = 2,
          datadir = 'datasets/wristband_data',
          cachedir= 'cache',
          modalities = ['EDA', 'ACC', 'BVP', 'HR'],
          preictal_time = 60,
          motor_threshold = 0.1,
          cacheclear=True)

