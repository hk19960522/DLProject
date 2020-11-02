from utils import *
import torch.nn as nn
import torch.nn.functional as F

# all the models
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, input_embedded_size, rnn_size, num_rnn_layer, output_size):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.input_embedded_size = input_embedded_size
        self.rnn_size = rnn_size
        self.num_rnn_layer = num_rnn_layer
        self.output_size = output_size

        self.input_embedded_layer = nn.Linear(self.input_size, self.input_embedded_size)
        self.lstm = nn.LSTM(self.input_embedded_size, self.rnn_size, self.num_rnn_layer)

        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        self.input_Ac = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.output_seq = nn.Sequential(
            nn.Linear(self.input_embedded_size, self.rnn_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.output_size)
        )
        pass

    def forward(self, obs_traj):
        '''
        obs_traj format: seq_len, batch, input_size
        '''
        embedded_obs_traj = self.dropout(self.input_Ac(self.input_embedded_layer(obs_traj)))
        hidden_tuple = self.init_hidden_state(obs_traj.shape)
        seq_len = obs_traj.shape[0]

        for idx in range(seq_len):
            curr_obs_traj = obs_traj[idx, :, :]
            curr_embedded_obs_traj = \
                self.dropout(self.input_Ac(self.input_embedded_layer(curr_obs_traj)))





        output = self.output_seq(embedded_obs_traj)


        logging.debug(f'Embedded_obs_traj shape:{embedded_obs_traj.shape}\n'
                      f'Output shape:{output.shape}')
        return output

    def init_hidden_state(self, data_shape):
        hidden_state = torch.zeros(data_shape[1], data_shape[2])
        cell_state = torch.zeros(data_shape[1], data_shape[2])
        return (hidden_state, cell_state)

