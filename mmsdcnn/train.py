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

from mmsdcommon.data import split_leave_one_patient_out, load_data
from pathlib import Path
import os
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
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


if __name__ == '__main__':
    fdir = os.path.join(Path.home(), 'datasets', 'wristband_data')

    modalities = ['EDA', 'ACC', 'BVP']
    folds, num = split_leave_one_patient_out(os.path.join(fdir, 'metadata.csv'), 5, modalities)
    print('Number of folds:', num)
    # print(data[0][0].groupby(['patient', 'session'])['fid'].apply(list))
    for fold in folds:
        print('Loading train and test data... ')
        x_train, y_train, x_test, y_test, mods_list = load_data(fold, fdir, label_win_type=0)
        print('Training data shape before undersampling', x_train.shape, y_train.shape)
        x_train, y_train = sample_imbalance('under', x_train, y_train)
        print('After undersampling', x_train.shape, y_train.shape)
        break
