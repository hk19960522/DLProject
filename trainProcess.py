from utils import *
from loader import *
import torch.nn as nn
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
            logging.info(f'Epoch {epoch}:')
            for _iter, batch in enumerate(self.train_loader):
                if _iter == 0:
                    logging.debug(f'Iter {_iter}, data shape: {len(batch)}')
                    logging.debug(f'{batch[0].shape}')
                pass
            break
            pass

        pass

    def train(self):
        # TODO: the training step
        pass

