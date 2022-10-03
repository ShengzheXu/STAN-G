"""
"""
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from torch.utils.data import Dataset
import torch
import os
import pickle
import time


class ScenarioDataset(Dataset):   
    def __init__(self, dir='tmp'):
        path = os.getcwd()
        self.data_event = np.load(os.path.join(path, dir, 'nn_input', 'train_event.npy')) # load
        self.data_behavior = np.load(os.path.join(path, dir, 'nn_input', 'train_behavior.npy')) # load
        self.data_next = np.load(os.path.join(path, dir, 'nn_input', 'train_next.npy')) # load

        self.data_event = torch.from_numpy(self.data_event).float()
        self.data_behavior = torch.from_numpy(self.data_behavior).float()
        self.data_next = torch.from_numpy(self.data_next).float()

        # self.data_description = np.loadtxt(os.path.join(path, 'tmp', 'nn_input', 'train_desc'), dtype=np.int) # load
        with open(os.path.join(path, dir, 'nn_input', 'train_desc.pickle'), 'rb') as filehandle:
            self.data_description = pickle.load(filehandle)

    def description(self):
        print('event', self.data_event.shape)
        print('behavior', self.data_behavior.shape)
        print('next', self.data_next.shape)
        return self.data_description

    def __len__(self):
        return len(self.data_next)

    def __getitem__(self, idx):
        return (self.data_behavior[idx], self.data_event[idx], self.data_next[idx])


class BehaviorContextLoader(object):
    
    def __init__(self, short_seq_len=10, long_time_delta=60, long_seq_len=50, dim=22) -> None:
        """Load and preprocess real-world datasets.
        Args:
        - delta_te_len: time delta as second
        - seq_len: sequence length

        Returns:
        - data: preprocessed data.
        """
        self.short_seq_len = short_seq_len

        self.long_time_delta = long_time_delta
        self.long_seq_len = long_seq_len

        self.dim = dim
        
        self.reset()
   
    def reset(self):
        self.mem_buffer = [-np.zeros(self.dim)]*self.long_seq_len #deque([])
        # self.mem_buffer = [-np.ones(self.dim)]*self.long_seq_len #deque([])
        self.time_buffer = [0] * self.long_seq_len #deque([])
        self.time_stamp = 0

    def padding_and_sampling(self):
        """
        Args:
        - self.dim: event_dim
        - self.mem_buffer (mem_snapshot): can be l_i * m , 
        - self.long_seq_len (max_vol): the receptive field shape

        Returns:
        - result: the 
        """
        pads = np.zeros((self.long_seq_len, self.dim))
        # pads = -np.ones((self.long_seq_len, self.dim)) # for debugging
        
        data = np.array(self.mem_buffer)
        # padding
        # data = pads if len(self.mem_buffer)==0 else np.concatenate((pads, data), axis=0)
        if data.shape[0] == 0:
            data = pads
        elif data.shape[0] <= self.long_seq_len:    
            pads[self.long_seq_len-data.shape[0]:, :data.shape[1]] = data
            data = pads

        # sampling
        idx = np.linspace(0, data.shape[0], self.long_seq_len, endpoint=False, dtype=int)
        # result2 = data[np.random.choice(data.shape[0], max_vol, replace=False), :]
        return data[idx, :]
    
    def sampling_loading_push(self, time_delta, content):
        '''
        Args:
        - time_delta: timestamp in second to move forward
        - content: row
        Returns:
        - _x: long context as [long_seq_len, d_data]
        '''
        # move to new time_stamp and clean buffer base on current timestamp
        # print(time_delta, content.shape, content)
        # input()
        self.time_stamp += time_delta
        st_idx = 0
        # start_time = time.time()
        # if self.time_stamp > 59:
        #     print(self.time_stamp, self.time_buffer[:5], self.time_buffer[-5:])
        #     print('='*60)
        while st_idx<len(self.time_buffer) and self.time_buffer[st_idx] <= self.time_stamp - self.long_time_delta:
            # _, _ = self.time_buffer.popleft(), self.mem_buffer.popleft()
            st_idx += 1
        self.time_buffer = self.time_buffer[st_idx:]
        self.mem_buffer = self.mem_buffer[st_idx:] 
        
        # if self.time_stamp > 59:
        #     print(self.time_stamp, self.time_buffer[:5], self.time_buffer[-5:])
        #     input()
        # print("move buffer--- %s seconds ---" % (time.time() - start_time))
        
        # start_time = time.time()
        # cut data
        _x = self.padding_and_sampling()
        # print("cut data --- %s seconds ---" % (time.time() - start_time))
        
        # add to time and memory buffer
        self.time_buffer.append(self.time_stamp)
        self.mem_buffer.append(content)

        return _x

    def long_context_loading(self, data: np.array, time_delta_col=0, return_snapshot=False):
        temp_data = []
        snapshot = []
        self.reset()
        start_time = time.time()
        for idx, val in enumerate(data):
            buffer = self.sampling_loading_push(val[time_delta_col], val)
            # print(self.time_buffer)
            # print('val', val[time_delta_col])
            # input()
            temp_data.append(buffer)
            snapshot.append(np.append(buffer[:, 1], val[1]).astype(int))
            if idx % 10000 == 0:
                print(f'[loaded] {idx} long context in {time.time() - start_time} seconds')
                start_time = time.time()

        return temp_data, snapshot if return_snapshot else temp_data

    def short_context_loading(self, data: np.array):
        """Load short context.
        Args:
        - data: Numpy array with the values from a a Dataset

        Returns:
        - temp_data: X
        - temp_next: y
        """
        # Flip the data to make chronological data
        # ori_data = data[::-1]
        ori_data = data

        # Preprocess the dataset
        temp_data = []
        temp_next = []
        # Pad Suffix zeros for marginal distribution conditions
        pads = np.zeros((self.short_seq_len, ori_data.shape[1]))
        ori_data = np.concatenate((pads, ori_data), axis=0)
        # Cut data by sequence length
        for i in range(0, len(ori_data) - self.short_seq_len):
            _x = ori_data[i:i + self.short_seq_len]
            _y = ori_data[i+self.short_seq_len]
            temp_data.append(_x)
            temp_next.append(_y)

        return temp_data, temp_next