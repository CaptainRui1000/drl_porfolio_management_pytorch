import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as nn_f
from torch import double

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, n_signals, window_length, n_assets, seed, fc1, fc2, predict_type, batch_size):
        """Initialize parameters and build model.

        Args:
            n_signals (int): Number of signals per asset
            window_length (int): Number of days in sliding window
            n_assets (int): Number of assets in portfolio not counting cash
            seed (int): Random seed
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """

        super(Actor, self).__init__()

        self.n_signals = n_signals

        if predict_type == 'cnn':
            self.a0 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a1 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a2 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a3 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a4 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a5 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a6 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a7 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a8 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a9 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a10 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
        elif predict_type == 'lstm':
            self.a0 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a1 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a2 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a3 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a4 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a5 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a6 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a7 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a8 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a9 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
        elif predict_type == 'gru':
            self.a0 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a1 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a2 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a3 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a4 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a5 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a6 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a7 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a8 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a9 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
        elif predict_type == 'rnn':
            self.a0 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a1 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a2 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a3 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a4 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a5 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a6 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a7 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a8 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a9 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
        elif predict_type == 'r_gcn --lstm':
            self.a0 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a1 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a2 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a3 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a4 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a5 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a6 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a7 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a8 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a9 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
        elif predict_type == 'r_gcn --cnn':
            self.a0 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a1 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a2 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a3 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a4 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a5 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a6 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a7 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a8 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a9 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
        elif predict_type == 'r_gcn --combine':
            self.a0 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a1 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a2 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a3 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a4 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a5 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a6 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a7 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a8 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a9 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
        elif predict_type == 'r_gcn --stack':
            self.a0 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a1 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a2 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a3 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a4 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a5 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a6 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a7 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a8 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a9 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')

        if fc2 == 0:
            self.fc_out = nn.Linear(fc1 * n_assets, n_assets)
        else:
            self.fc_out = nn.Linear(fc2 * n_assets, n_assets)
        self.reset_parameters()

    def reset_parameters(self):

        self.a0.reset_parameters()
        self.a1.reset_parameters()
        self.a2.reset_parameters()
        self.a3.reset_parameters()
        self.a4.reset_parameters()
        self.a5.reset_parameters()
        self.a6.reset_parameters()
        self.a7.reset_parameters()
        self.a8.reset_parameters()
        self.a9.reset_parameters()

        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_out.bias.data.fill_(0.1)

    def forward(self, state):
        """Build a actor network."""

        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9 = torch.split(state, self.n_signals, 1)
        x0 = self.a0(s0)
        # print('Actor:')
        # print('0:')
        x1 = self.a1(s1)
        # print('1:')
        x2 = self.a2(s2)
        # print('2:')
        x3 = self.a3(s3)
        # print('3:')
        x4 = self.a4(s4)
        # print('4:')
        x5 = self.a5(s5)
        # print('5:')
        x6 = self.a6(s6)
        # print('6:')
        x7 = self.a7(s7)
        # print('7:')
        x8 = self.a8(s8)
        # print('8:')
        x9 = self.a9(s9)
        # print('9:')

        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9), 1)

        return self.fc_out(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, n_signals, window_length, n_assets, seed, fc1, fc2, predict_type, batch_size):
        """Initialize parameters and build model.

        Args:
            n_signals (int): Number of signals per asset
            window_length (int): Number of days in sliding window
            n_assets (int): Number of assets in portfolio not counting cash
            seed (int): Random seed
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """

        super(Critic, self).__init__()
        self.n_signals = n_signals

        if n_assets != 10:
            print("ERROR:  Only operational to 10 assets per issue #3.")
            raise RuntimeError

        if predict_type == 'cnn':
            self.a0 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a1 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a2 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a3 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a4 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a5 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a6 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a7 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a8 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
            self.a9 = CNNAssetModel(n_signals, window_length, seed, fc1, fc2)
        elif predict_type == 'lstm':
            self.a0 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a1 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a2 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a3 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a4 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a5 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a6 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a7 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a8 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a9 = LSTMAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
        elif predict_type == 'gru':
            self.a0 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a1 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a2 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a3 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a4 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a5 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a6 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a7 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a8 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a9 = GRUAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
        elif predict_type == 'rnn':
            self.a0 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a1 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a2 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a3 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a4 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a5 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a6 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a7 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a8 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
            self.a9 = RNNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size)
        elif predict_type == 'r_gcn --lstm':
            self.a0 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a1 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a2 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a3 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a4 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a5 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a6 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a7 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a8 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
            self.a9 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm')
        elif predict_type == 'r_gcn --cnn':
            self.a0 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a1 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a2 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a3 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a4 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a5 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a6 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a7 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a8 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
            self.a9 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='cnn')
        elif predict_type == 'r_gcn --combine':
            self.a0 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a1 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a2 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a3 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a4 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a5 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a6 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a7 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a8 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
            self.a9 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='combine')
        elif predict_type == 'r_gcn --stack':
            self.a0 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a1 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a2 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a3 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a4 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a5 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a6 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a7 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a8 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')
            self.a9 = RGCNAssetModel(n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='stack')

        if fc2 == 0:
            self.fc_out = nn.Linear(fc1 * n_assets + n_assets, n_assets)
        else:
            self.fc_out = nn.Linear(fc2 * n_assets + n_assets, n_assets)
        self.reset_parameters()

    def reset_parameters(self):

        self.a0.reset_parameters()
        self.a1.reset_parameters()
        self.a2.reset_parameters()
        self.a3.reset_parameters()
        self.a4.reset_parameters()
        self.a5.reset_parameters()
        self.a6.reset_parameters()
        self.a7.reset_parameters()
        self.a8.reset_parameters()
        self.a9.reset_parameters()

        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_out.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9 = torch.split(state, self.n_signals, 1)
        x0 = self.a0(s0)
        # print('Critic:')
        # print('0:')
        x1 = self.a1(s1)
        # print('1:')
        x2 = self.a2(s2)
        # print('2:')
        x3 = self.a3(s3)
        # print('3:')
        x4 = self.a4(s4)
        # print('4:')
        x5 = self.a5(s5)
        # print('5:')
        x6 = self.a6(s6)
        # print('6:')
        x7 = self.a7(s7)
        # print('7:')
        x8 = self.a8(s8)
        # print('8:')
        x9 = self.a9(s9)
        # print('9:')

        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, action), 1)

        return self.fc_out(x)


class CNNAssetModel(nn.Module):
    def __init__(self, n_signals, window_length, seed, fc1, fc2):
        """Network built for each asset.

        Args:
            n_signals (int): Number of signals per asset
            window_length (int): Number of days in sliding window
            seed (int): Random seed
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """
        super(CNNAssetModel, self).__init__()
        out_channels = n_signals
        kernel_size = 3
        self.use_fc2 = fc2 > 0
        self.n_signals = n_signals
        self.conv1d_out = (window_length - kernel_size + 1) * out_channels
        self.seed = torch.manual_seed(seed)

        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)

        self.conv1d = nn.Conv1d(n_signals, out_channels, kernel_size=kernel_size)
        # self.bn = nn.BatchNorm1d(out_channels)

        self.fc1 = nn.Linear(self.conv1d_out, fc1)
        if self.use_fc2:
            self.fc2 = nn.Linear(fc1, fc2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1d.weight)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0.1)
        if self.use_fc2:
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc2.bias.data.fill_(0.1)

    def forward(self, state):
        x = self.drop1(self.relu(self.conv1d(state)))
        x = x.contiguous().view(-1, self.conv1d_out)
        x = self.relu(self.fc1(x))
        if self.use_fc2:
            x = self.drop2(self.relu(self.fc2(x)))
        return x


class LSTMAssetModel(nn.Module):
    def __init__(self, n_signals, window_length, seed, fc1, fc2, batch_size):
        super(LSTMAssetModel, self).__init__()
        self.use_fc2 = fc2 > 0
        self.n_signals = n_signals
        self.window_length = window_length
        self.seed = torch.manual_seed(seed)
        self.batch_size = batch_size
        self.lstm_out = self.n_signals * self.window_length
        self.hidden = None

        self.input_size = 10
        self.hidden_size = 10
        self.num_layers = 1

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)

        self.relu = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(self.lstm_out, fc1)
        if self.use_fc2:
            self.fc2 = nn.Linear(fc1, fc2)
        self.reset_parameters()

    def reset_parameters(self):
        self.hidden = None
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0.1)
        if self.use_fc2:
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc2.bias.data.fill_(0.1)

    def forward(self, state):
        # print('state.shape:', state.shape)

        # if self.hidden is None:
        #     # batch_size = state.shape[0]
        #     # print('batch size: ', self.batch_size)
        #     self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device=device),
        #                    torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device=device))
        # print('hidden.shape:', self.hidden[0].shape, self.hidden[1].shape)

        output, hidden = self.lstm(state, self.hidden)
        output = torch.flatten(output, 1)
        # print('output.shape:', output.shape)
        output = self.drop1(self.relu(self.fc1(output)))
        if self.use_fc2:
            output = self.drop2(self.relu(self.fc2(output)))
        return output


class GRUAssetModel(nn.Module):
    def __init__(self, n_signals, window_length, seed, fc1, fc2, batch_size):
        super(GRUAssetModel, self).__init__()
        self.use_fc2 = fc2 > 0
        self.n_signals = n_signals
        self.window_length = window_length
        self.seed = torch.manual_seed(seed)
        self.batch_size = batch_size
        self.gru_out = self.n_signals * self.window_length
        self.hidden = None

        self.input_size = 10
        self.hidden_size = 10
        self.num_layers = 1

        self.relu = nn.ReLU(inplace=True)

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True)

        self.fc1 = nn.Linear(self.gru_out, fc1)
        if self.use_fc2:
            self.fc2 = nn.Linear(fc1, fc2)
        self.reset_parameters()

    def reset_parameters(self):
        self.hidden = None
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0.1)
        if self.use_fc2:
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc2.bias.data.fill_(0.1)

    def forward(self, state):

        output, hidden = self.gru(state, self.hidden)
        output = torch.flatten(output, 1)
        # print('output.shape:', output.shape)
        output = self.relu(self.fc1(output))
        if self.use_fc2:
            output = self.relu(self.fc2(output))
        return output


class RNNAssetModel(nn.Module):
    def __init__(self, n_signals, window_length, seed, fc1, fc2, batch_size):
        super(RNNAssetModel, self).__init__()
        self.use_fc2 = fc2 > 0
        self.n_signals = n_signals
        self.window_length = window_length
        self.seed = torch.manual_seed(seed)
        self.batch_size = batch_size
        self.rnn_out = self.n_signals * self.window_length
        self.hidden = None

        self.input_size = 10
        self.hidden_size = 10
        self.num_layers = 1

        self.relu = nn.ReLU(inplace=True)

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True)

        self.fc1 = nn.Linear(self.rnn_out, fc1)
        if self.use_fc2:
            self.fc2 = nn.Linear(fc1, fc2)
        self.reset_parameters()

    def reset_parameters(self):
        self.hidden = None
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0.1)
        if self.use_fc2:
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc2.bias.data.fill_(0.1)

    def forward(self, state):

        output, hidden = self.rnn(state, self.hidden)
        output = torch.flatten(output, 1)
        # print('output.shape:', output.shape)
        output = self.relu(self.fc1(output))
        if self.use_fc2:
            output = self.relu(self.fc2(output))
        return output


class RGCNAssetModel(nn.Module):
    def __init__(self, n_signals, window_length, seed, fc1, fc2, batch_size, emb_type='lstm'):
        super(RGCNAssetModel, self).__init__()
        self.use_fc2 = fc2 > 0
        self.n_signals = n_signals
        self.window_length = window_length
        self.seed = torch.manual_seed(seed)
        self.batch_size = batch_size
        self.emb_type = emb_type
        self.relation = torch.tensor(np.load('relation-2022-Aug.npy')).to(device)
        self.n_relations = self.relation.shape[-1]

        self.n_assets = 10
        self.hidden_size = 10
        self.num_layers = 1

        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)

        if self.emb_type == 'lstm':
            self.hidden = None
            self.lstm_out = self.n_signals * self.window_length
            self.lstm = nn.LSTM(input_size=self.window_length, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True)
            self.fc0 = nn.Linear(self.lstm_out, self.n_assets)
        elif self.emb_type == 'cnn':
            out_channels = n_signals
            kernel_size = 3
            self.conv1d_out = (window_length - kernel_size + 1) * out_channels
            self.conv1d = nn.Conv1d(n_signals, out_channels, kernel_size=kernel_size)
            self.fc0 = nn.Linear(self.conv1d_out, self.n_assets)
        elif self.emb_type == 'combine' or 'stack':
            self.hidden = None
            self.lstm_out = self.n_signals * self.window_length
            self.lstm = nn.LSTM(input_size=self.window_length, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True)
            self.fc00 = nn.Linear(self.lstm_out, self.n_assets)

            out_channels = n_signals
            kernel_size = 3
            self.conv1d_out = (window_length - kernel_size + 1) * out_channels
            self.conv1d = nn.Conv1d(n_signals, out_channels, kernel_size=kernel_size)
            self.fc01 = nn.Linear(self.conv1d_out, self.n_assets)

        self.r_gcn = RelationalGraphConvLayer(input_size=self.n_assets, output_size=self.n_assets,
                                              num_bases=self.num_layers, num_rel=self.n_relations)
        self.fc1 = nn.Linear(self.n_assets, fc1)
        if self.use_fc2:
            self.fc2 = nn.Linear(fc1, fc2)
        self.reset_parameters()

    def reset_parameters(self):
        if self.emb_type == 'lstm':
            self.hidden = None
            self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
            self.fc0.bias.data.fill_(0.1)
        elif self.emb_type == 'cnn':
            nn.init.xavier_uniform_(self.conv1d.weight)
            self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
            self.fc0.bias.data.fill_(0.1)
        elif self.emb_type == 'combine' or 'stack':
            self.hidden = None
            self.fc00.weight.data.uniform_(*hidden_init(self.fc00))
            self.fc00.bias.data.fill_(0.1)

            nn.init.xavier_uniform_(self.conv1d.weight)
            self.fc01.weight.data.uniform_(*hidden_init(self.fc01))
            self.fc01.bias.data.fill_(0.1)

        self.r_gcn.reset_parameters()
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0.1)
        if self.use_fc2:
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc2.bias.data.fill_(0.1)

    def forward(self, state):
        if self.emb_type == 'lstm':
            output, hidden = self.lstm(state, self.hidden)
            output = torch.flatten(output, 1)
            output0 = self.drop1(self.relu(self.fc0(output)))
            output1 = self.r_gcn(output0, self.relation)
            output = torch.add(output0, output1)
        elif self.emb_type == 'cnn':
            output = self.drop1(self.relu(self.conv1d(state)))
            output = output.contiguous().view(-1, self.conv1d_out)
            output0 = self.relu(self.fc0(output))
            output1 = self.r_gcn(output0, self.relation)
            output = torch.add(output0, output1)
        elif self.emb_type == 'combine':
            output0, hidden = self.lstm(state, self.hidden)
            output0 = torch.flatten(output0, 1)
            output0 = self.drop1(self.relu(self.fc00(output0)))

            output1 = self.drop1(self.relu(self.conv1d(state)))
            output1 = output1.contiguous().view(-1, self.conv1d_out)
            output1 = self.relu(self.fc01(output1))

            output = torch.add(output0, output1)
        elif self.emb_type == 'stack':
            output = self.drop1(self.relu(self.conv1d(state)))
            output = output.contiguous().view(-1, self.conv1d_out)
            output = self.relu(self.fc01(output))
            output, hidden = self.lstm(output, self.hidden)
            output0 = torch.flatten(output, 1)
            # output0 = self.drop1(self.relu(self.fc00(output)))
            output1 = self.r_gcn(output0, self.relation)
            output = torch.add(output0, output1)

        output = self.drop1(self.relu(self.fc1(output)))
        if self.use_fc2:
            output = self.relu(self.fc2(output))
        return output


class RelationalGraphConvLayer(nn.Module):
    def __init__(self, input_size, output_size, num_bases, num_rel):
        super(RelationalGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.cuda = False if device == 'cpu' else True

        # R-GCN weights
        if num_bases > 0:
            self.w_bases = nn.Parameter(
                torch.FloatTensor(self.num_bases, self.input_size, self.output_size)
            )
            self.w_rel = nn.Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        else:
            self.w = nn.Parameter(
                torch.FloatTensor(self.num_rel, self.input_size, self.output_size)
            )
        # R-GCN bias
        # if bias:
        #     self.bias = nn.Parameter(torch.FloatTensor(self.output_size))
        # else:
        self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.w.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)

    def forward(self, a, x):
        # print('a:', a.shape)
        # print('x:', x.shape)

        a = torch.as_tensor(a, dtype=double).to(device)
        x = torch.as_tensor(x, dtype=double).to(device)
        x = x.to(device)
        self.w = (
            torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
            if self.num_bases > 0
            else self.w
        )
        weights = self.w.view(
            self.w.shape[0] * self.w.shape[1], self.w.shape[2]
        )
        # shape(r*input_size, output_size)
        # Each relations * Weight
        supports = []
        for i in range(self.num_rel):
            if x is not None:
                supports.append(torch.sparse.mm(a, x[:, :, i]))
            else:
                supports.append(a)

        tmp = torch.cat(supports, dim=1)
        out = torch.mm(tmp.float(), weights)  # shape(#node, output_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out

