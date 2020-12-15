from utils import *
import torch.nn as nn
import torch.nn.functional as F

# all the models
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, input_embedded_size, rnn_size, num_rnn_layer, output_size, pred_len):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.input_embedded_size = input_embedded_size
        self.rnn_size = rnn_size
        self.num_rnn_layer = num_rnn_layer
        self.output_size = output_size
        self.pred_len = pred_len

        self.input_embedded_layer = nn.Linear(self.input_size, self.input_embedded_size)
        self.lstm = nn.LSTM(self.input_embedded_size, self.rnn_size, self.num_rnn_layer)

        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        self.input_Ac = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.output_seq = nn.Sequential(
            nn.Linear(self.rnn_size, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(16, self.output_size)
        )

    def forward(self, obs_traj):
        '''
        obs_traj format: seq_len, batch, input_size
        '''
        embedded_obs_traj = self.dropout(self.input_Ac(self.input_embedded_layer(obs_traj)))
        seq_len = obs_traj.shape[0]
        batch_size = obs_traj.shape[1]
        # logging.debug(f'type of batch size: {type(batch_size.shape)}')

        # embedded_obs_traj size: [seq_len, batch, input_embedded_size]
        embedded_obs_traj = \
            self.dropout(self.input_Ac(self.input_embedded_layer(obs_traj)))
        logging.debug(f'embedded_obs_traj size: {embedded_obs_traj.shape}')

        next_hidden_list = []
        output_list = []
        # use lstm to encode embedded trajectory
        hidden = self.init_hidden_state(batch_size)
        for frame_idx in range(seq_len):
            # single pedestrian's trajectory
            # with 8 frames (default)
            curr_embedded_obs_traj = embedded_obs_traj[frame_idx, :, :]
            '''
            because lstm input need 3 dim, 
            but curr_embedded_obs_traj only have 2 dim [batch, feature]
            so use unsqueeze to add dim to [seq_len(1), batch, feature]
            '''
            curr_embedded_obs_traj = curr_embedded_obs_traj.unsqueeze(0)
            # print(f'size: {curr_embedded_obs_traj.shape}')
            # output_data, next_hidden = self.lstm(curr_embedded_obs_traj, (hidden[0][idx], hidden[1][idx]))
            _, hidden = self.lstm(curr_embedded_obs_traj, hidden)

        # hidden will be used in decoder
        logging.debug(f'hidden size: {hidden[0].shape}')
        # decoder
        # decode from the last data of embedded trajectory
        pred_traj = []
        curr_pred_traj = obs_traj[-1, :, :]
        curr_embedded_pred_traj = self.dropout(self.input_Ac(self.input_embedded_layer(curr_pred_traj)))
        # logging.debug(f'Curr embedded pred traj size : {curr_embedded_pred_traj.shape}')
        for frame_idx in range(self.pred_len):
            next_embedded_pred_traj, hidden = self.lstm(curr_embedded_pred_traj.unsqueeze(0), hidden)

            # logging.debug(f'next_embedded_pred_traj size: {next_embedded_pred_traj.shape}')
            displacement = self.output_seq(next_embedded_pred_traj)
            # logging.debug(f'displacement shape:{displacement.shape}')
            next_pred_traj = curr_pred_traj + displacement.squeeze(0)
            pred_traj.append(next_pred_traj)
            curr_embedded_pred_traj = self.dropout(self.input_Ac(self.input_embedded_layer(next_pred_traj)))

        output = torch.stack(pred_traj, 0)

        logging.debug(f'Embedded_obs_traj shape:{embedded_obs_traj.shape}\n'
                     f'Output shape:{output.shape}')
        return output

    def init_hidden_state(self, batch_size):
        hidden_state = torch.zeros((self.num_rnn_layer, batch_size, self.rnn_size)).cuda()
        cell_state = torch.zeros((self.num_rnn_layer, batch_size, self.rnn_size)).cuda()
        return hidden_state, cell_state



