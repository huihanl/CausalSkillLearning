from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *
import numpy as np


class WidowX250s_Dataset(Dataset):

    def __init__(self, args):
        self.dataset_directory = '/nfs/kun1/users/avi/imitation_datasets/' \
                                 'sept25_easygrasp_Widow250GraspEasy-v0_500_noise_0.1_2020-09-25T10-56-36/' \
                                 'sept25_easygrasp_Widow250GraspEasy-v0_500_noise_0.1_2020-09-25T10-56-36_500.npy'
        self.dataset = np.load(self.dataset_directory, allow_pickle=True)
        self.args = args
        self.total_length = len(self.dataset)
        #import ipdb; ipdb.set_trace()
        self.environment_names = ["Widow250GraspEasy-v0"]

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):

        if index >= self.total_length:
            print("Out of bounds of dataset.")
            return None

        trajectory = self.dataset[index]
        traj_state = np.array([trajectory["observations"][i]["state"] for i in range(len(trajectory))])

        data_element = dict()
        data_element['demo'] = traj_state
        data_element['is_valid'] = True
        data_element['environment-name'] = self.environment_names[0]

        return data_element
