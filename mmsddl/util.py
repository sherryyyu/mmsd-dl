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
   Define some utility functions
'''

import numpy as np

def one_hot(y, nb_classes):
    return np.array([1 if i==y else 0 for i in range(nb_classes)])


def print_metrics(metrics):
    print('sensitivity', metrics['sens'])
    print('FAR', metrics['fars'])
    print(f'AUC', metrics['auc'])
    sen_cnt, far_cnt, thresholds = metrics['sen_cnt'], metrics['far_cnt'], \
                                   metrics['thresholds']
    i = len(thresholds) // 2
    print(f'Operating point = {thresholds[i]:.2f}, '
          f'SEN = {sen_cnt[i][0]}/{sen_cnt[i][1]}, '
          f'FAR = {far_cnt[i][0]}/{far_cnt[i][1]}')

