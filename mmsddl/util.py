'''
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


def early_stopping(cfg, auc, es_cnt, es_best_auc, loss=False):
    '''
    Calculate early stopping parameters

    :param bool loss: when false, the early stopping criteria is AUC-ROC.
    When true, the criteria is loss, which means auc is loss, and es_best_auc
    is the lowest loss
    '''
    stop_training = False
    if cfg.early_stopping:
        if es_best_auc is None:
            es_best_auc = auc
        else:
            if not loss:
                if auc - es_best_auc > cfg.min_delta:
                    es_best_auc = auc
                    es_cnt = 1
                else:
                    if es_cnt >= cfg.patience:
                        stop_training = True
                    es_cnt += 1
            else:
                # the name here might be confusing by auc I mean loss
                if es_best_auc - auc > cfg.min_delta:
                    es_best_auc = auc
                    es_cnt = 1
                else:
                    if es_cnt >= cfg.patience:
                        stop_training = True
                    es_cnt += 1

    return stop_training, es_cnt, es_best_auc