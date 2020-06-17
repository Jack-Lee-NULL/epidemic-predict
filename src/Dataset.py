#
# author: Jingquan Lee
# date: 2020-06-04
# email: m113152143642@163.com
#
#

import os

import numpy as np
import torch

class Dataset:

    def __init__(self, args):
        if not type(args) == tuple:
            citys = ['A', 'B', 'C', 'D', 'E']
            days = 45
            density = np.loadtxt(args.density_file, delimiter=',', comments='city', dtype=str)
            infection = np.loadtxt(args.infections_file, delimiter=',', dtype=str)
            self.batch_size = args.batch_size
            self.A = torch.Tensor(np.loadtxt(args.seir_A_file, delimiter=','))
            self.density = []
            self.mean_density = []
            self.infection = []
            self.A_regions = []
            for c in citys:
                pos = np.argwhere(density[:, 0]==c).flatten()
                d = density[pos, 2:].astype('float32')
                self.density.append(torch.Tensor(d[:, 0].reshape(-1, days).T[:, :, np.newaxis]))
                self.mean_density.append(torch.Tensor((d[:, 0]/d[:, -1]).reshape(-1, days).T[:, :, np.newaxis]))
                pos = np.argwhere(infection[:, 0]==c).flatten()
                d = infection[pos, 1:].astype('float32')
                self.infection.append(torch.Tensor(d[:, -1].reshape(-1, days).T[:, :, np.newaxis]))
                transfer = np.loadtxt(os.path.join(args.migration_dir, c+'_migration.csv'), 
                                delimiter=',')
                num_of_regions = len(d)//days
                adj = np.zeros((days, num_of_regions, num_of_regions))
                for i in transfer:
                    j = (i[0]-21200500)%100+(i[0]-21200500)//100*31-1
                    adj[int(j), int(i[2]), int(i[1])] = i[-1]
                self.A_regions.append(torch.Tensor(adj))
            self.density = self.data_expand(self.density)
            self.infection = self.data_expand(self.infection)
            self.mean_density = self.data_expand(self.mean_density)
            self.A_regions = self.data_expand(self.A_regions)
            indexs = list(range(len(self.density)))
            np.random.shuffle(indexs)
            self.density = [self.density[i] for i in indexs]
            self.infection = [self.infection[i] for i in indexs]
            self.mean_density = [self.mean_density[i] for i in indexs]
            self.A_regions = [self.A_regions[i] for i in indexs]
        else:
            self.density = args[0]
            self.mean_density = args[1]
            self.infection = args[2]
            self.A_regions = args[3]
            self.A = args[4]

    def split(self, n1, n2):
        train_size = int(len(self.density)*n1/(n1+n2))
        train_data = (self.density[:train_size], self.mean_density[:train_size],
                        self.infection[:train_size], self.A_regions[:train_size], self.A)
        validate_data = (self.density[train_size:], self.mean_density[train_size:],
                        self.infection[train_size:], self.A_regions[train_size:], self.A)
        return Dataset(train_data), Dataset(validate_data)

    def data_expand(self, x, encode_least=10, decode_least=1):
        print('data expand')
        output = []
        for d in x:
            time_l = d.shape[0]
            print(time_l)
            for i in range(encode_least, time_l-decode_least):
                for j in range(decode_least, time_l-i):
                    record_encode, record_decode = [], []
                    for k in range(time_l-i-j):
                        output.append([d[np.newaxis, k:k+i], d[np.newaxis, k+i:k+i+j]])
        return output

    def __iter__(self):
        states = list(range(len(self.density)))
        np.random.shuffle(states)
        for state in states:
            yield self.density[state],\
                  self.mean_density[state],\
                  self.infection[state],\
                  self.A_regions[state]
            state = state + 1

    def __len__(self):
        return len(self.density)

    def __getitem__(self, idx):
        return self.density[idx], self.mean_density[idx], self.infection[idx]

class TestDataset:

    def __init__(self, args):
        if not type(args) == tuple:
            citys = ['A', 'B', 'C', 'D', 'E']
            days = 45
            density = np.loadtxt(args.density_file, delimiter=',', comments='city', dtype=str)
            infection = np.loadtxt(args.infections_file, delimiter=',', dtype=str)
            self.A = torch.Tensor(np.loadtxt(args.seir_A_file, delimiter=','))
            self.wan_data = torch.Tensor(np.loadtxt(os.path.join(args.wan_data_dir, 'city_all_norm.csv'), delimiter=',')[:, np.newaxis, :])
            self.transfer = []
            self.density = []
            self.mean_density = []
            self.infection = []
            self.A_regions = []
            for c in citys:
                pos = np.argwhere(density[:, 0]==c).flatten()
                d = density[pos, 2:].astype('float32')
                self.density.append(torch.Tensor(d[:, 0].reshape(-1, days).T[np.newaxis, :, :, np.newaxis]))
                self.mean_density.append(torch.Tensor((d[:, 0]/d[:, -1]).reshape(-1, days).T[np.newaxis, :, :, np.newaxis]))
                pos = np.argwhere(infection[:, 0]==c).flatten()
                d = infection[pos, 1:].astype('float32')
                self.infection.append(torch.Tensor(d[:, -1].reshape(-1, days).T[np.newaxis, :, :, np.newaxis]))
                transfer = np.loadtxt(os.path.join(args.migration_dir, c+'_migration.csv'), 
                                delimiter=',')
                num_of_regions = len(d)//days
                adj = np.zeros((days, num_of_regions, num_of_regions))
                for i in transfer:
                    j = (i[0]-21200500)%100+(i[0]-21200500)//100*31-1
                    adj[int(j), int(i[2]), int(i[1])] = i[-1]
                self.A_regions.append(torch.Tensor(adj[np.newaxis, :]))
                adj = np.zeros((num_of_regions, num_of_regions))
                transfer = np.loadtxt(os.path.join(args.migration_dir, 'tranfer_a_day_'+c+'.csv'), 
                                delimiter=',')
                for i in transfer:
                    adj[int(i[2]), int(i[1])] = i[-1]
                self.transfer.append(torch.Tensor(adj[np.newaxis, np.newaxis, :]))


    def __iter__(self):
        states = list(range(len(self.density)))
        for state in states:
            yield self.density[state], self.mean_density[state],\
                  self.infection[state], self.A_regions[state],\
                  self.transfer[state]
            state = state + 1

    def __len__(self):
        return len(self.density)

    def __getitem__(self, idx):
        return self.density[idx], self.mean_density[idx], self.infection[idx]
