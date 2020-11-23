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