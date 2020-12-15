from utils import *
from loader import *
from models import *
import torch.nn as nn
import torch.optim as optim
import math


# train process model
class TrainingProcess:
    def __init__(self, args):
        self.args = args

        # prepare data
        # training data
        train_set_dir = get_dataset_dir(0, 0)
        self.train_dataset, self.train_loader = data_loader(args, train_set_dir)

        # validation data
        val_set_dir = get_dataset_dir(0, 1)
        self.val_dataset, self.val_loader = data_loader(args, val_set_dir)

        # test data
        test_set_dir = get_dataset_dir(0, 2)
        self.test_dataset, self.test_loader = data_loader(args, test_set_dir)

        # prepare model
        self.model = SimpleLSTM(self.args.input_size,
                                self.args.input_embed_size,
                                self.args.rnn_size,
                                2,
                                2,
                                self.args.pred_length)
        self.optim_SimpleLSTM = optim.Adam(self.model.parameters(),
                                           lr=self.args.learning_rate)

        self.model = self.model.cuda()
        self.L2_series = []
        self.ADE_series = []
        self.FDE_series = []
        pass

    def main(self):
        # TODO: the main training process
        logging.info(f'\n\n======================\n'
                     f'Training Start: \n')

        iter_per_epoch = math.ceil(len(self.train_dataset) / self.args.batch_size)
        logging.debug(f'\nIter time:{iter_per_epoch}\n'
                      f'Dataset size:{len(self.train_dataset)}\n'
                      f'Batch size:{self.args.batch_size}')

        for epoch in range(self.args.num_epochs):

            loss = 0
            ADE = 0
            FDE = 0
            for _iter, batch in enumerate(self.train_loader):
                '''
                if _iter == 0:
                    logging.debug(f'Iter {_iter}, data shape: {len(batch)}')
                    logging.debug(f'{batch[0].shape}')
                '''
                '''
                batch format:
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, seq_start_end
                '''

                batch_loss, batch_ADE, batch_FDE = self.train_step(batch)
                loss += batch_loss.item()
                ADE += batch_ADE.item()
                FDE += batch_FDE.item()
            loss = loss / len(self.train_loader)
            ADE = ADE / len(self.train_loader)
            FDE = FDE / len(self.train_loader)
            if epoch % 10 == 0:
                logging.info(f'Epoch {epoch}:\n'
                             f'Epoch loss: {loss}\n'
                             f'ADE: {ADE}\n'
                             f'FDE: {FDE}')
            self.L2_series.append(loss)
            self.ADE_series.append(ADE)
            self.FDE_series.append(FDE)

        write_result(self.L2_series, self.ADE_series, self.FDE_series)

    def train_step(self, batch):
        # TODO: the training step
        batch_obs_traj, batch_pred_traj, batch_obs_traj_rel, batch_pred_traj_rel, seq_start_end = batch

        self.model.zero_grad()
        L2_loss = torch.zeros(1).cuda()
        ADE_loss = torch.zeros(1).cuda()
        FDE_loss = torch.zeros(1).cuda()

        batch_cnt = 0
        for (start, end) in seq_start_end:
            batch_cnt += 1
            # split into single data
            obs_traj = batch_obs_traj[:, start: end]
            pred_traj = batch_pred_traj[:, start: end]
            obs_traj_rel = batch_obs_traj_rel[:, start: end]
            pred_traj_rel = batch_pred_traj_rel[:, start: end]

            output = self.model(obs_traj.cuda())
            L2_loss_part, ADE_loss_part, FDE_loss_part = self.get_loss(output, pred_traj, end-start)
            L2_loss = L2_loss + L2_loss_part
            ADE_loss = ADE_loss + ADE_loss_part
            FDE_loss = FDE_loss + FDE_loss_part

        L2_loss = L2_loss / batch_cnt
        ADE_loss = ADE_loss / batch_cnt
        FDE_loss = FDE_loss / batch_cnt
        L2_loss.backward()
        self.optim_SimpleLSTM.step()
        return L2_loss, ADE_loss, FDE_loss

    def get_loss(self, pred_traj, corr_pred_traj, batch_size):
        pred_traj = pred_traj.cuda()
        corr_pred_traj = corr_pred_traj.cuda()
        L2_loss = ((pred_traj - corr_pred_traj) ** 2).sum()
        L2_loss = L2_loss / batch_size

        ADE_loss = ((pred_traj - corr_pred_traj) ** 2).sum(2).sqrt().mean()
        FDE_loss = ((pred_traj[-1] - corr_pred_traj[-1]) ** 2).sum(1).sqrt().mean()
        return L2_loss, ADE_loss, FDE_loss
