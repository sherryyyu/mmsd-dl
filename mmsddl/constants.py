"""
Author:
    Sherry Yu
Initial Version:
    Nov-2020
Function:
   Constants
"""

import torch
from nutsflow.config import Config
from pathlib import Path
import os
import platform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if platform.system() == 'Linux':
    #ROOT = '/slow1/out_datasets/bch/'
    ROOT = '/fast2/'
elif platform.system() == 'Darwin':
    # ROOT = '/Users/shuangyu/datasets/bch/'
    ROOT = os.path.join(Path.home(), 'datasets','bch')
else:
    print('Unknown OS platform %s' % platform.system())
    exit()

DATADIR = 'wristband_REDCap_202102_szr_cluster_win0'
# DATADIR = 'wristband_redcap_data'

gtc = ['Tonic-clonic']
fbtc = ['FBTC']
focal_myoclonic = ['focal,Myoclonic']
automatisms = ['Automatisms']
epileptic_spasms = ['Epileptic spasms']
gnr_tonic = ['gnr,Tonic']
behaviour_arrast = ['Behavior arrest']
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
    datadir=os.path.join(ROOT, DATADIR),
    traincachedir=os.path.join(ROOT, 'cache/train/fold'),
    valcachedir=os.path.join(ROOT, 'cache/val/fold'),
    testcachedir=os.path.join(ROOT, 'cache/test/fold'),
    plotdir=os.path.join(ROOT, 'plots'),
    ckpdir=os.path.join(ROOT, 'checkpoints'),
    modalities=['EDA', 'ACC'],
    szr_types = gtc,
    preictal_len=60,
    postictal_len=60,
    motor_threshold=0.1,
    sequence_model=False,
    cacheclear=True)
