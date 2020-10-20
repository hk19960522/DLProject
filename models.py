from utils import *
import torch.nn as nn
import torch.nn.functional as F

# all the models
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, input_embedded_size, rnn_size, num_rnn_layer, output_size):

        self.input_size = input_size
        self.input_embedded_size = input_embedded_size
        self.rnn_size = rnn_size
        self.num_rnn_layer = num_rnn_layer
        self.output_size = output_size

        self.input_embedded_layer = nn.Linear(self.input_size, self.input_embedded_size)
        self.lstm = nn.LSTM(self.input_embedded_size, self.rnn_size, self.num_rnn_layer)

        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        self.ac_fn = F.relu

        pass

    def forward(self, obs_traj):
        '''
        obs_traj format: 
        '''
        pass
