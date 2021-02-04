'''
Author:
    Sherry Yu
Initial Version:
    Nov-2020
Function:
   Define the neural network models
'''

import torch
import torch.nn as nn
from mmsddl.constants import DEVICE, CFG
from pytorch_model_summary import summary
from time import sleep
import os


class HAR_model(nn.Module):
    """1D CNN Model for human-activity-recognition."""

    def __init__(self, input_size, num_classes):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(  # input: 640 * input_size
            nn.Conv1d(input_size, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 320, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x.float())
        x = self.gap(x)
        return x


class LSTMClassifier(nn.Module):
    """Vanilla LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0, c0 = self.reset_state(x)
        out, _ = self.lstm(x.float(), (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

    def reset_state(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(DEVICE)
        return h0, c0


def print_summary(net, input_dim):
    if CFG.sequence_model:
        # RNN: batch, seq_len, input_dim
        t = torch.zeros((1, 640, input_dim)).to(DEVICE)
    else:
        # CNN: batch, input_dim, seq_len
        t = torch.zeros((1, input_dim, 640)).to(DEVICE)
    print('input  dim', t.size())
    print(summary(net, t, show_input=False))


def create_network(input_dim, num_classes):
    if CFG.sequence_model:
        model = LSTMClassifier(input_dim, 200, 1, num_classes)
    else:
        model = HAR_model(input_dim, num_classes)
    model.to(DEVICE)
    if CFG.verbose > 1:
        print_summary(model, input_dim)
    return model


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


def save_ckp(state, ckpdir, fold_no):
    '''Save checkpoint'''
    if not os.path.exists(ckpdir):
        os.makedirs(ckpdir)
    torch.save(state, f'{ckpdir}/{fold_no}')


def load_ckp(ckpdir, net, optimizer, best_auc, fold_no):
    '''Load checkpoint'''
    chppath = f'{ckpdir}/{fold_no}'
    if os.path.isfile(chppath):
        checkpoint = torch.load(chppath)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_auc = checkpoint['best_auc']
        return net, optimizer, checkpoint['epoch'], best_auc, checkpoint[
            'metrics']
    else:
        return net, optimizer, 0, best_auc, None


if __name__ == '__main__':
    net = create_network(6, 2)
