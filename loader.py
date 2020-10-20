from torch.utils.data import DataLoader
from dataProcessor import *


def data_loader(args, path):
    '''
    TODO:
    Use default value first, will modify it
    '''
    data_set = TrajectoryDataSet(
        path
    )

    loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=seq_collate
    )
    return data_set, loader
