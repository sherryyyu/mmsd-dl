'''
Author:
    Sherry Yu
Initial Version:
    Nov-2020
Function:
   Define the neural network models
'''

import os
import torch
import torch.nn as nn
from mmsddl.get_cfg import DEVICE
from pytorch_model_summary import summary
from time import sleep
from mmsdcommon.util import num_channels, max_fs


class HAR_model(nn.Module):
    """1D CNN Model for human-activity-recognition."""

    def __init__(self, input_size, num_classes):
        super().__init__()

        n_filter = 64

        # Extract features, 1D conv layers
        self.features = nn.Sequential(  # input: 640 * input_size
            nn.Conv1d(input_size, n_filter, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(n_filter, n_filter, 3, padding=1),
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
            nn.Linear(n_filter, num_classes),
        )

    def forward(self, x):
        x = self.features(x.float())
        x = self.gap(x)
        return x


class LSTM(nn.Module):
    """Vanilla LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, 16)
        self.linear = nn.Linear(16, output_dim)

    def forward(self, x):
        h0, c0 = self.reset_state(x)
        lstm_out, _ = self.lstm(x.float(), (h0, c0))
        l_out = self.linear1(lstm_out[:, -1, :])
        out = self.linear(l_out)
        return out

    def reset_state(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(DEVICE)
        return h0, c0


class CNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        n_filter = 5

        # Extract features, 1D conv layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, n_filter, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        ) # out: (step, n_filter * new_fs / 2)

    def forward(self, x):
        x = self.cnn(x.float())
        return x


class CNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes, cfg):
        super(CNNLSTM, self).__init__()
        self.cnn = CNN(input_size)
        self.lstm = nn.LSTM(
            input_size=5 * max_fs(cfg.modalities) // 2,
            hidden_size=50,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(50, num_classes)

    def forward(self, x):
        batch_size, timesteps, S, C = x.size()
        c_in = x.view(batch_size * timesteps, S, C).permute(0, 2, 1)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = self.linear(r_out[:, -1, :])

        return r_out2


class ConvAE(nn.Module):
    """1D CNN based Autoencoder model."""

    def __init__(self, input_size):
        super().__init__()

        n_filter = 128

        self.encode = nn.Sequential(  # input: 640 * input_size
            nn.Conv1d(input_size, n_filter, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(n_filter, n_filter//2, 3, padding=1),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose1d(n_filter // 2, n_filter//2, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose1d(n_filter // 2, n_filter, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose1d(n_filter, input_size, 3, padding=1),
        )

    def forward(self, x):
        x = self.encode(x.float())
        x = self.decode(x)
        return x


def print_summary(cfg, net, input_dim):
    t = None
    if cfg.network == 'lstm':
        # RNN: batch, seq_len, input_dim
        # t = torch.zeros((1, 640, input_dim)).to(DEVICE)
        t = torch.zeros(1, cfg.win_len//cfg.subsequence,
                        cfg.subsequence * max_fs(cfg.modalities) * input_dim).to(DEVICE)
    elif cfg.network == 'cnn':
        # CNN: batch, input_dim, seq_len
        t = torch.zeros(1, input_dim,
                         cfg.win_len * max_fs(cfg.modalities)).to(DEVICE)
    elif cfg.network == 'cnnlstm':
        t = torch.zeros(1, 10, 32, 3)
    elif cfg.network == 'convae':
        # encoder: batch, input_dim, seq_len
        t = torch.zeros(1, input_dim,
                        cfg.win_len * max_fs(cfg.modalities)).to(DEVICE)
    print('input  dim', t.size())
    print(summary(net, t, show_input=False))


def create_network(cfg, num_classes):
    model = None
    n_channel = num_channels(cfg.modalities)
    if cfg.network == 'lstm':
        input_dim = n_channel * max_fs(cfg.modalities) * cfg.subsequence
        model = LSTM(input_dim, 50, 1, num_classes)
    elif cfg.network == 'cnn':
        model = HAR_model(n_channel, num_classes)
    elif cfg.network == 'cnnlstm':
        model = CNNLSTM(n_channel, num_classes, cfg)
    elif cfg.network == 'convae':
        model = ConvAE(n_channel)
    model.to(DEVICE)
    if cfg.verbose > 1:
        print_summary(cfg, model, n_channel)
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
        es_cnt = checkpoint['es_cnt']
        es_best_auc = checkpoint['es_best_auc']
        return net, optimizer, checkpoint[
            'epoch'], best_auc, es_cnt, es_best_auc, checkpoint[
                   'metrics']
    else:
        return net, optimizer, 0, best_auc, 0, None, None


if __name__ == '__main__':
    from mmsddl.get_cfg import get_CFG
    cfg = get_CFG()
    net = create_network(cfg, 2)
