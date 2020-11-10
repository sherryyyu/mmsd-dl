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
   Define some common functions for loading data
'''

from mmsdcommon.data import load_metadata, load_sessions, make_windows
from mmsdcommon.cross_validata import split_leave_one_patient_out
from mmsdcommon.preprocess import remove_non_motor_windows
from pathlib import Path
import os
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def sample_imbalance(sampling, data, label):
    data_shape = data.shape

    # It seems like the imblearn library only accepts data of shape 2,
    # therefore we reshape the data to shape 2 before passing into the sampler and retrieve the shape after
    if len(data_shape) == 3:
        data = np.reshape(data, [data_shape[0], data_shape[1] * data_shape[2]])

    if sampling == 'over':
        sampler = RandomOverSampler()
    elif sampling == 'smote':
        sampler = SMOTE()
    elif sampling == 'under':
        sampler = RandomUnderSampler()
    elif sampling == 'adasyn':
        sampler = ADASYN()
    else:
        print('Error: invalid sampler name, using Random Under Sampling.')
        sampler = RandomUnderSampler()
    data, label = sampler.fit_resample(data, label)

    # Retrieve the original data shape if the original shape is 3
    if len(data_shape) == 3:
        data = np.reshape(data, [-1, data_shape[1], data_shape[2]])
    return data, label

def preprocess(data_sess, label_sess, channel_lookup):
    # TODO: insert BVP bandpass filter here

    print('Preprocess: make windows...')
    data_sess, label_sess = make_windows(data_sess, label_sess, window_length=10, label_win_type=0.1)

    # remove non-motor windows
    data_sess, label_sess = remove_non_motor_windows(data_sess, label_sess, channel_lookup, 0.1)

    # merge all sessions into one numpy array
    data = np.concatenate(data_sess)
    label = np.concatenate(label_sess)

    return data, label


if __name__ == '__main__':
    fdir = os.path.join(Path.home(), 'datasets', 'wristband_data')

    modalities = ['EDA', 'ACC', 'BVP']
    metadata_df = load_metadata(os.path.join(fdir, 'metadata.csv'), n=3, modalities=modalities, szr_sess_only=True)
    folds, num = split_leave_one_patient_out(metadata_df)

    print('Number of folds:', num)
    for i, fold in enumerate(folds):
        print(f'Fold {i+1}: loading train and test data... ')
        x_train_sess, y_train_sess, x_test_sess, y_test_sess, channel_lookup = load_sessions(fold, fdir)
        print(f'Loaded {len(x_train_sess)} train sessions and {len(x_test_sess)} test sessions.')

        x_train, y_train = preprocess(x_train_sess, y_train_sess, channel_lookup)

        print('Training data shape before under sampling', x_train.shape, y_train.shape)
        x_train, y_train = sample_imbalance('under', x_train, y_train)
        print('After under sampling', x_train.shape, y_train.shape)

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_train = enc.fit_transform(np.reshape(y_train, (len(y_train), 1)))
        nb_classes = y_train.shape[1]

        # train the CNN model

