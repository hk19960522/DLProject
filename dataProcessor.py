from utils import *
from torch.utils.data import Dataset


# all data processor (dataloader, preprocessor...etc)
# I want to play video games........................

class TrajectoryDataSet(Dataset):
    """  Data set  """
    def __init__(self, data_dir, obs_len=8, pred_len=8, data_type=0):
        super(TrajectoryDataSet, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_type = data_type  # usually, 0 for train, 1 for val, 2 for test

        data = self.read_row_data()



    def read_row_data(self):
        '''
        just dubug first:
        '''

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []

        for path in all_files:
            # file_data format: [ frame id ,ped id, ped x, ped y, frame id, ped id, ......]
            file_data = read_file(path)
            #  get all frame id
            frame_id_list = np.unique(file_data[:, 0])
            # logging.debug(f'Frame ID:\n{frame_id_list}')
            for frame_id in frame_id_list:
                # get all peds data in each frame


            pass
        return all_files

    pass

class dataLoader():
    pass


def read_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split()
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def get_dataset_dir(id, data_type = 0):
    data_dir_list = ['test1', 'univ', 'zara1', 'zara2', 'eth', 'hotel', 'raw']
    type_list = ['train', 'val', 'test']

    if id >= len(data_dir_list):
        id = 0
    if data_type >= len(type_list):
        data_type = 0

    data_dir = '.\\datasets\\'+data_dir_list[id]+'\\'+type_list[data_type]
    # logging.debug(f'Data dir: {data_dir}')
    return data_dir

