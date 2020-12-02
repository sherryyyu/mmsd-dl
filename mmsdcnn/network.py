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

import torch
import torch.nn as nn
from mmsdcnn.constants import DEVICE
from pytorch_model_summary import summary
from time import sleep


class HAR_model(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(      # input: 640 * input_size
            nn.Conv1d(input_size, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2)
            )
        # Classify output, fully connected layers
        # TODO: try global average pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
        	nn.Linear(64*320, 50),
        	nn.ReLU(),
        	nn.Linear(50, num_classes),
        	)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def print_summary(net, input_dim):
    print(summary(net, torch.zeros((1, input_dim, 640)), show_input=False))

def create_network(input_dim, num_classes):
    model = HAR_model(input_dim, num_classes)
    model.to(DEVICE)
    print_summary(model, input_dim)
    return model.double()

def save_wgts(net, filepath='weights.pt'):
    for i in range(5):
        try:
            torch.save(net.state_dict(), filepath)
            break
        except:
            print('Could not save weights file. Retrying...', i)
            sleep(1)


def load_wgts(net, filepath='weights.pt'):
    net.load_state_dict(torch.load(filepath))



if __name__ == '__main__':
    net = create_network(6, 2)