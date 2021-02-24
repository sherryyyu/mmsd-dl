#!/bin/bash

# for all
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'szr' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'szr' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'szr' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'szr' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'szr' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'szr' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'szr' --n_epochs 30 --modalities EDA,BVP &

# for FBTC
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'FBTC' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'FBTC' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'FBTC' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'FBTC' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'FBTC' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'FBTC' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'FBTC' --n_epochs 30 --modalities EDA,BVP &

# for gnr,Tonic
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities EDA,BVP &

# for focal,Tonic
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Tonic' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Tonic' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Tonic' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Tonic' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Tonic' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Tonic' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Tonic' --n_epochs 30 --modalities EDA,BVP &

# for focal,subclinical
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,subclinical' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,subclinical' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,subclinical' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,subclinical' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,subclinical' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,subclinical' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,subclinical' --n_epochs 30 --modalities EDA,BVP &

# for focal,Automatisms
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities EDA,BVP &

# for focal,Behavior arrest
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities EDA,BVP &

# for gnr,Epileptic spasms
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities EDA,BVP &

# for gnr,Tonic-clonic
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities ACC,EDA,BVP
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 0 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities EDA,BVP


######################################

# for all
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'szr' --n_epochs 30 --modalities ACC --crossfold 10 &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'szr' --n_epochs 30 --modalities EDA --crossfold 10 &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'szr' --n_epochs 30 --modalities BVP --crossfold 10 &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'szr' --n_epochs 30 --modalities ACC,EDA --crossfold 10 &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'szr' --n_epochs 30 --modalities ACC,EDA,BVP--crossfold 10 &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'szr' --n_epochs 30 --modalities ACC,BVP --crossfold 10 &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'szr' --n_epochs 30 --modalities EDA,BVP --crossfold 10

# for FBTC
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'FBTC' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'FBTC' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'FBTC' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'FBTC' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'FBTC' --n_epochs 30 --modalities ACC,EDA,BVP
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'FBTC' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'FBTC' --n_epochs 30 --modalities EDA,BVP


# for focal,Tonic
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Tonic' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Tonic' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Tonic' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Tonic' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Tonic' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Tonic' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Tonic' --n_epochs 30 --modalities EDA,BVP


# for focal,subclinical
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,subclinical' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,subclinical' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,subclinical' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,subclinical' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,subclinical' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,subclinical' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,subclinical' --n_epochs 30 --modalities EDA,BVP


# for focal,Automatisms
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Automatisms' --n_epochs 30 --modalities EDA,BVP &


# for focal,Behavior arrest
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Behavior arrest' --n_epochs 30 --modalities EDA,BVP &

# for focal | motor | Clonic
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Clonic' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Clonic' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Clonic' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Clonic' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Clonic' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Clonic' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'focal,Clonic' --n_epochs 30 --modalities EDA,BVP &


# for gnr,Tonic
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic' --n_epochs 30 --modalities EDA,BVP


# for gnr,Epileptic spasms
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Epileptic spasms' --n_epochs 30 --modalities EDA,BVP &


# for gnr,Tonic-clonic
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities ACC &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities ACC,EDA &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities ACC,EDA,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities ACC,BVP &
python3 mmsddl/train.py --batch_size 256 --lr 0.001 --win_len 10 --win_step 2 --sing_wrst 1 --szr_types 'gnr,Tonic-clonic' --n_epochs 30 --modalities EDA,BVP &

