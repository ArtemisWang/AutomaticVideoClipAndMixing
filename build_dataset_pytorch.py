import torch as t
from torch.utils import data
import data_helper
import numpy as np
import pickle
import random

class QuoraDataset_train(data.Dataset):
    def __init__(self, input_path, char_path, flags_path):
        sample_num = 1000
        with open(input_path, 'rb') as handle:
            data = pickle.load(handle)
            all_num = len(data['p_inputs'])
            index = random.sample(list(range(all_num)), sample_num)
            self.p_inputs = [data['p_inputs'][i] for i in index]
            self.q_inputs = [data['q_inputs'][i] for i in index]
            self.targets = [data['targets'][i] for i in index]
            # self.p_inputs = data['p_inputs']
            # self.q_inputs = data['q_inputs']
            # self.targets = data['targets']
        with open(char_path, 'rb') as handle:
            data = pickle.load(handle)
            self.p_inputs_char = [data['p_inputs_char'][i] for i in index]
            self.q_inputs_char = [data['q_inputs_char'][i] for i in index]
            # self.p_inputs_char = data['p_inputs_char']
            # self.q_inputs_char = data['q_inputs_char']
        with open(flags_path, 'rb') as handle:
            data = pickle.load(handle)
            self.p_flags = [data['p_flags'][i] for i in index]
            self.q_flags = [data['q_flags'][i] for i in index]
            # self.p_flags = data['p_flags']
            # self.q_flags = data['q_flags']

    def __getitem__(self, index):
        seq_len = 30
        char_len = 10
        p_inputs = t.from_numpy(np.array(self.p_inputs[index])).narrow(0, 0, seq_len)
        q_inputs = t.from_numpy(np.array(self.q_inputs[index])).narrow(0, 0, seq_len)
        targets = t.from_numpy(np.array(self.targets[index]))
        p_inputs_char = t.from_numpy(np.array(self.p_inputs_char[index])).narrow(0, 0, seq_len).narrow(1, 0, char_len)
        q_inputs_char = t.from_numpy(np.array(self.q_inputs_char[index])).narrow(0, 0, seq_len).narrow(1, 0, char_len)
        p_flags = np.array(self.p_flags[index])[:, np.newaxis]
        p_flags = t.from_numpy(p_flags.astype(np.float32)).narrow(0, 0, seq_len)
        q_flags = np.array(self.q_flags[index])[:, np.newaxis]
        q_flags = t.from_numpy(q_flags.astype(np.float32)).narrow(0, 0, seq_len)
        return p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags
    def __len__(self):
        return len(self.p_inputs)

class QuoraDataset_test(data.Dataset):
    def __init__(self, input_path, char_path, flags_path):
        with open(input_path, 'rb') as handle:
            data = pickle.load(handle)
            all_num = len(data['p_inputs'])
            self.p_inputs = data['p_inputs']
            self.q_inputs = data['q_inputs']
            self.targets = data['targets']
        with open(char_path, 'rb') as handle:
            data = pickle.load(handle)
            self.p_inputs_char = data['p_inputs_char']
            self.q_inputs_char = data['q_inputs_char']
        with open(flags_path, 'rb') as handle:
            data = pickle.load(handle)
            self.p_flags = data['p_flags']
            self.q_flags = data['q_flags']

    def __getitem__(self, index):
        seq_len = 30
        char_len = 10
        p_inputs = t.from_numpy(np.array(self.p_inputs[index])).narrow(0, 0, seq_len)
        q_inputs = t.from_numpy(np.array(self.q_inputs[index])).narrow(0, 0, seq_len)
        targets = t.from_numpy(np.array(self.targets[index]))
        p_inputs_char = t.from_numpy(np.array(self.p_inputs_char[index])).narrow(0, 0, seq_len).narrow(1, 0, char_len)
        q_inputs_char = t.from_numpy(np.array(self.q_inputs_char[index])).narrow(0, 0, seq_len).narrow(1, 0, char_len)
        p_flags = np.array(self.p_flags[index])[:, np.newaxis]
        p_flags = t.from_numpy(p_flags.astype(np.float32)).narrow(0, 0, seq_len)
        q_flags = np.array(self.q_flags[index])[:, np.newaxis]
        q_flags = t.from_numpy(q_flags.astype(np.float32)).narrow(0, 0, seq_len)
        return p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags
    def __len__(self):
        return len(self.p_inputs)


class QuoraDataset_predict(data.Dataset):
    def __init__(self, p, q_path, q_char_path):
        p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad = data_helper.loadDataset_predict(
            q_path, q_char_path, p)
        self.p_inputs = p_pad
        self.p_inputs_char = p_char_pad
        self.p_flags = p_flags_pad
        self.q_inputs = q_pad
        self.q_inputs_char = q_char_pad
        self.q_flags = q_flags_pad
    def __getitem__(self, index):
        seq_len = 30
        char_len = 10
        p_inputs = t.from_numpy(np.array(self.p_inputs)).narrow(0, 0, seq_len)
        q_inputs = t.from_numpy(np.array(self.q_inputs[index])).narrow(0, 0, seq_len)
        p_inputs_char = t.from_numpy(np.array(self.p_inputs_char)).narrow(0, 0, seq_len).narrow(1, 0, char_len)
        q_inputs_char = t.from_numpy(np.array(self.q_inputs_char[index])).narrow(0, 0, seq_len).narrow(1, 0, char_len)
        p_flags = np.array(self.p_flags[index])[:, np.newaxis]
        p_flags = t.from_numpy(p_flags.astype(np.float32)).narrow(0, 0, seq_len)
        q_flags = np.array(self.q_flags[index])[:, np.newaxis]
        q_flags = t.from_numpy(q_flags.astype(np.float32)).narrow(0, 0, seq_len)
        return p_inputs, q_inputs, p_inputs_char, q_inputs_char, p_flags, q_flags, index
    def __len__(self):
        return len(self.q_inputs)



if __name__ == '__main__':
    dataset = QuoraDataset_train('data/train_data_pad.pkl','data/train_data_char_pad.pkl','data/train_data_flags_pad.pkl')
    p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags = dataset[0]
    print(dataset.__len__())
    print(targets)
