
import os
import argparse

import torch
import numpy as np

from MyLSTM import LSTMPre
from seir import model
from Dataset import TestDataset
from train import *

def main(args):
    citys = ['A', 'B', 'C', 'D', 'E']
    wan_model = struct_wan_model()
    li_model = model([3, 36, 25], [5, 48], [25, 48, 25], 48)
    wan_model.load_state_dict(torch.load(args.wan_save_path, map_location=lambda storage, loc: storage))
    li_model.load_state_dict(torch.load(args.li_save_path, map_location=lambda storage, loc: storage))
    dataset = TestDataset(args)
    output = []

    li_ys, seir_features, transfers = [], [], []
    for i, (density, mean_density, infection, A_regions, transfer) in enumerate(dataset):
        density = density_normalize(density)
        mean_density = mean_density_normalize(mean_density)
        infection = infection_normalize(infection)
        label = [density[:, 1:], mean_density[:, 1:], infection[:, 1:]]
        x = torch.cat([density[:, 0], 
                mean_density[:, 0], 
                infection[:, 0]], dim=-1)
        li_y, seir_feature = encode(li_model, A=dataset.A, A_regions=A_regions[:, 1:], x=x,
                        mean_density=mean_density[:, 0], label=label, num_steps=44)
        li_ys.append([density[:, -1:], mean_density[:, -1:], infection[:, -1:]])
        transfers.append(transfer)
        seir_features.append(seir_feature)
    wan_y, wan_hc = wan_encode(wan_model, wan=dataset.wan_data)

    output = [[], [], [], [], []]
    for t in range(30):
        for i, (li_y, seir_feature, transfer) in enumerate(zip(li_ys, seir_features, transfers)):
            label = li_y
            li_ys[i], seir_features[i] = decode(li_model, seir_feature=seir_feature,
                      A=dataset.A, A_regions=transfer, label=li_y, wan=wan_y[-1, 0, i*2])
            print(wan_y[-1, 0, i*2])
            infections = infection_recover(li_ys[i][-1])
            print(i, infections.shape)
            output[i].append(infections)
        infections = [infection_normalize(torch.sum(infection_recover(li_y[-1][:, -1:]), dim=-2)) for li_y in li_ys]
        infections.append(wan_y[-1:, :, :])
        x = torch.cat(infections, dim=-1)
        wan_y, wan_hc = wan_model.output(x, wan_hc)

    result = []
    for i, infections in enumerate(output):
        for k, inf in enumerate(infections):
            for j, output_infection in enumerate(inf[0, 0]):
                result.append([citys[i], str(j), str((k+45-31)%30+(k+45-31)//30*100+1+21200600), str(float(output_infection[0])*float(output_infection[0]>0))])
    np.savetxt('output.csv', result, delimiter=',', fmt='%s')

def struct_wan_model():
    inp_dim = 15  # 输入数据的维度，[感染数，流入量，流出量]
    out_dim = 10  # 输出数据的维度，只预测 [流入量，流出量]
    mid_dim = 20  # LSTM三个门的网络宽度，即LSTM输出的张量维度，可调参数，类似于隐神经元数量
    mid_layers = 2  # LSTM内部各个门使用的全连接层数量，一般设置为1或2，可调参数
    net = LSTMPre(inp_dim, out_dim, mid_dim, mid_layers)
    return net

def wan_encode(wan_model, **kwargs):
    with torch.no_grad():
        wan_y, hc = wan_model.output(kwargs['wan'], (torch.zeros(2, 1, 20), torch.zeros(2, 1, 20)))
    return wan_y, hc

def encode(li_model, **kwargs):
    with torch.no_grad():
        print(kwargs['A'].shape, kwargs['A_regions'].shape, kwargs['x'].shape, kwargs['mean_density'].shape)
        li_y, seir_feature = li_model(kwargs['A'], kwargs['A_regions'], kwargs['x'], kwargs['mean_density'],
                                kwargs['label'], kwargs['num_steps'], use_label=True,
                                resume=False)
    return li_y, seir_feature
    
def wan_decode(wan_model, x, hc):
    pass

def decode(li_model, **kwargs):
    with torch.no_grad():
        seir_feature = kwargs['seir_feature']
        mean_density = kwargs['label'][1]
        wan_y = kwargs['wan']
        city_input = wan_y
        A_regions = city_input*kwargs['A_regions']
        aaa = seir_feature
        li_y, seir_feature = li_model(kwargs['A'], A_regions, seir_feature, mean_density,
                None, 1, use_label=False,
                resume=True)
    return li_y, seir_feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--li_save_path', type=str, default='./log-xxxn/model_1990')
    parser.add_argument('--wan_save_path', type=str, default='./wan/model/networkA.pth')
    parser.add_argument('--wan_data_dir', type=str, default='./wan/data')
    parser.add_argument('--density_file', type=str, default='../preprocess/loc_inf.csv')
    parser.add_argument('--infections_file', type=str, default='../preprocess/infection.csv')
    parser.add_argument('--seir_A_file', type=str, default='../preprocess/seir_A.csv')
    parser.add_argument('--migration_dir', type=str, default='../preprocess/')
    args = parser.parse_args()
    main(args)
