from utils import *
from torch.utils.data import Dataset


# all data processor (dataloader, preprocessor...etc)
# I want to play video games........................

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(1, 0, 2)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(1, 0, 2)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(1, 0, 2)
    seq_start_end = torch.LongTensor(seq_start_end)
    output = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,  seq_start_end
    ]

    return tuple(output)


class TrajectoryDataSet(Dataset):
    """  Data set  """
    def __init__(self, data_dir, obs_len=8, pred_len=8, data_type=0, min_ped = 1):
        super(TrajectoryDataSet, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.data_type = data_type  # usually, 0 for train, 1 for val, 2 for test
        self.min_ped = min_ped

        '''
        just dubug first:
        '''
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        num_peds_in_seq = []
        seq_list = []
        rel_seq_list = []

        for path in all_files:
            # file_data format: [ [frame id ,ped id, ped x, ped y], [ frame id, ped id, ......] ]
            file_data = read_file(path)
            # logging.debug(f'File data:\n{file_data}')

            # get all frame id and pedestrian id
            frame_id_list = np.unique(file_data[:, 0]).tolist()
            all_frame_data = []
            # logging.debug(f'Frame ID:\n{frame_id_list}')
            # test_part(frame_id_list)

            ped_id_list = np.unique(file_data[:, 1]).astype(int)
            all_ped_data = {}
            # logging.debug(f'Ped ID:\n{ped_id_list}')

            for frame_id in frame_id_list:
                # get all peds data in each frame
                # data_in_frame = file_data[frame_id == file_data[:, 0], :]
                all_frame_data.append(file_data[frame_id == file_data[:, 0], :])

            num_sequences = len(frame_id_list) - self.seq_len + 1
            # logging.debug(f'Num Seq: {num_sequences}')
            for idx in range(num_sequences):
                # check whether the sequence is continuous
                curr_seq_frame_id = np.asarray(frame_id_list[idx: idx+self.seq_len])
                curr_seq_frame_id = curr_seq_frame_id[1:] - curr_seq_frame_id[:-1]
                if len(np.unique(curr_seq_frame_id)) > 1:
                    continue

                # curr_seq_data format: [ [ frame id, ped id, x, y], [ frame id, ped id, x, y] ]
                curr_seq_data = np.concatenate(
                    all_frame_data[idx: idx+self.seq_len], axis=0
                )
                peds_id_in_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_id_in_seq), self.seq_len, 2))
                curr_rel_seq = np.zeros((len(peds_id_in_seq), self.seq_len, 2))
                # logging.debug(f'Current Sequence data: \n{curr_seq_data}')

                num_peds_cnt = 0
                for ped_idx, ped_id in enumerate(peds_id_in_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # logging.debug(f'Ped ID: {ped_id}, sequence: \n{curr_ped_seq}')

                    ped_front = frame_id_list.index(curr_ped_seq[0, 0]) - idx
                    ped_end = frame_id_list.index(curr_ped_seq[-1, 0]) - idx + 1
                    if ped_end - ped_front != self.seq_len:
                        # TODO: maybe will add threshold to record or fix the sequences
                        continue

                    # make relative position
                    curr_ped_pos_seq = curr_ped_seq[:, 2:]
                    rel_curr_ped_pos_seq = np.zeros(curr_ped_pos_seq.shape)
                    rel_curr_ped_pos_seq[1:, :] = curr_ped_pos_seq[1:, :] - curr_ped_pos_seq[:-1, :]
                    # logging.debug(f'Ped ID: {ped_id}, sequence: \n{curr_ped_pos_seq}\n{rel_curr_ped_pos_seq}')

                    _idx = num_peds_cnt
                    curr_seq[_idx, :, ped_front: ped_end] = curr_ped_pos_seq
                    curr_rel_seq[_idx, :, ped_front: ped_end] = rel_curr_ped_pos_seq

                    num_peds_cnt += 1

                # logging.debug(f'Curr seq: {curr_seq}')
                if num_peds_cnt > self.min_ped:
                    num_peds_in_seq.append(num_peds_cnt)
                    seq_list.append(curr_seq[: num_peds_cnt])
                    rel_seq_list.append(curr_seq[: num_peds_cnt])

        '''
        Before do the np.concatenate, the data format is: [file_id, seq_id, ....]
        After the np.concatenate, the data format is: [seq_id, ....]
        '''
        # logging.debug(f'Shape: {num_peds_in_seq}')
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        rel_seq_list = np.concatenate(rel_seq_list, axis=0)

        # logging.debug(f'Shape: {seq_list.shape}')
        # convert nparray to tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :self.obs_len, :]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, self.obs_len:, :]).type(torch.float)
        self.obs_rel_traj = torch.from_numpy(
            rel_seq_list[:, :self.obs_len, :]).type(torch.float)
        self.pred_rel_traj = torch.from_numpy(
            rel_seq_list[:, self.obs_len:, :]).type(torch.float)
        cum_start_idex = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end_idx = [
            (start, end)
            for start, end in zip(cum_start_idex, cum_start_idex[1:])
        ]
        logging.debug(f'Total sequences: {self.num_seq}')

        '''
        for ped_id in ped_id_list:
            if not all_ped_data.get(ped_id):
                all_ped_data[ped_id] = []
        '''

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end_idx[index]
        output = [
            self.obs_traj[start: end, :], self.pred_traj[start: end, :],
            self.obs_rel_traj[start: end, :], self.pred_rel_traj[start: end, :]
        ]
        return output


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


def test_part(_list):
    gap = 10
    cnt = 0
    last_idx = 0
    all_continue_part = []
    cumsum = 0
    for idx in range(1, len(_list)):
        if _list[idx] - _list[idx-1] != gap:
            # print(f'NO GAP')
            cnt += 1
            all_continue_part.append([_list[last_idx], _list[idx-1]])
            cumsum += _list[idx-1] - _list[last_idx] + 10

            last_idx = idx

    cumsum += _list[-1] - _list[last_idx] + 10
    print(f'list size: {len(_list)}\n'
          f'all_continue_part: {all_continue_part}'
          f'consum: {cumsum}')
