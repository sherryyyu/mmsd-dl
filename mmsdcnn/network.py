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
   Define the neural network models
'''

import torch.nn as nn
from mmsdcnn.constants import DEVICE

class HAR_model(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(      # input: 640 * 5
            nn.Conv1d(input_size, 64, 5),   # 636 * 64
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),           # 632 * 64
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),           # 628 * 64
            nn.ReLU(),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
        	nn.Dropout(),
        	nn.Linear(40192, 128),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(128, num_classes),
        	)

    def forward(self, x):
        print(x.size())
        x = self.features(x)
        print(x.size())
        x = x.view(x.size(0), 40192)
        out = self.classifier(x)
        return out

def create_network(input_dim, num_classes):
    model = HAR_model(input_dim, num_classes)
    model.to(DEVICE)
    model = model.float()
    return model