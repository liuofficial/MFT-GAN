import os
import torch
import scipy.io as sio
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, mat_path):

        self.data_path = mat_path
        self.images = self.read_file()

    def __getitem__(self, index):
        data = sio.loadmat(self.images[index])
        # print(data.keys())
        pan = data['P']
        # print(pan.shape)
        lrhs = data['Y']
        lrhs = np.transpose(lrhs, (2, 0, 1)).astype(np.float32)
        lrhs = torch.from_numpy(lrhs)
        pan = np.transpose(pan, (2, 0, 1)).astype(np.float32)
        pan = torch.from_numpy(pan)
        return {'Y': lrhs, 'Z': pan,}

    def __len__(self):
        return len(self.images)

    def read_file(self):
        path_list = []
        for ph in os.listdir(self.data_path):
            path = self.data_path + ph
            path_list.append(path)
        return path_list


