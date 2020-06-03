#
# author: Jingquan Lee
# date: 2020-06-02
# email: m13152143642@163.com
#

import torch
import numpy as np

class seir_base(torch.nn.Module):
    """
    achievement of seir basic formulations by graph convolution network
    """

    def __init__(self, channel, act='ReLU'):
        """
        Args:
          -channel: input channel == output channel
          -act: activation function, now offer RelU and sigmoid
        """
        super(seir_base, self).__init__()
        self.add_module('state_fc', torch.nn.Linear(channel, channel, bias=True))
        if act=='ReLU':
            self.add_module('state_active', torch.nn.ReLU())
        elif act=='sigmoid':
            self.add_module('state_active', torch.nn.Sigmoid())
        else:
            raise InputError('param: act', 'the value is valide')

    def forward(self, A, A_param, x):
        """
        Args:
          -A: adjacent matrix, define the link between each state, shape is 
              (5, 5)
          -A_param: weight adjacent matrix, shape is either (1, 5, 5) or (num_of_regions, 5, 5)
          -x: seir feature, five states(seird), shape could be (num_of_regions, 5, c)
        Return:
          -output: a seir feature, shape is (num_of_regions, 5, c)
        """
        A = torch.unsqueeze(A, dim=-1)*A_param
        x = torch.matmul(A, x)
        output = self.state_fc(x.reshape(-1, x.shape[-1]))
        output = output.reshape(x.shape)
        output = self.state_active(output) #TODO Do we need batch normalization?
        return output

class seir_model(torch.nn.Module):
    """
    completely achievement of seir
    """

    def __init__(self, seir_channels, adj_feature_channel):
        """
        Args:
          -channels: len(channels)==2, channels[0] is the channel of seir feature 
              which is static, channels[1] is the channle of middle layer.
        """
        super(seir_model, self).__init__()
        self.add_module('compute_adj', torch.nn.Sequential(
                torch.nn.Linear(seir_channels[0]*5+adj_feature_channel, seir_channels[1], bias=True),
                torch.nn.BatchNorm1d(seir_channels[1]),
                torch.nn.ReLU(),
                torch.nn.Linear(seir_channels[1], 5*5, bias=True),
                torch.nn.Sigmoid()))
        self.add_module('seir', seir_base(seir_channels[0]))
        self.add_module('mse_loss', torch.nn.MSELoss(reduce=False))

    def forward(self, A, x, adj_feature):
        """
        Args:
          -A: adjacent matrix, define the link between each state, shape is 
              (5, 5)
          -x: seir feature, five states(seird), shape could be (num_of_regions, 5, c)
        Return:
          -output: a seir feature, shape is (num_of_regions, 5, c)
        """
        A_param = self.compute_adj(torch.cat([x.reshape(-1, x.shape[-1]*x.shape[-2]),
                adj_feature], dim=-1))
        A_param = A_param.reshape(-1, 5, 5)
        output = self.seir(A, A_param, x)
        return output

class regions_gcn(torch.nn.Module):
    def __init__(self, channels):
        super(regions_gcn, self).__init__()
        self.add_module('fc', torch.nn.Sequential(
                torch.nn.Linear(channels[0], channels[1], bias=False),
                torch.nn.BatchNorm1d(channels[1]),
                torch.nn.ReLU(),
                torch.nn.Linear(channels[1], channels[2], bias=True),
                torch.nn.ReLU()))

    def forward(self, A, x):
        x = torch.matmul(A, self.fc(x))
        return x

class model(torch.nn.Module):
    def __init__(self, init_channels, seir_channels, region_channels, density_channel):
        super(model, self).__init__()
        self.S, self.I, self.E, self.R, self.D = (0, 1, 2, 3, 4)
        seir_channel = init_channels[2]//5
        self.add_module('compute_init_seir', torch.nn.Sequential(
                torch.nn.Linear(init_channels[0], init_channels[1], bias=False),
                torch.nn.BatchNorm1d(init_channels[1]),
                torch.nn.ReLU(),
                torch.nn.Linear(init_channels[1], init_channels[2], bias=True),
                torch.nn.Sigmoid()
                ))
        self.add_module('seir', seir_model(seir_channels, 1))
        self.add_module('relate_regions', regions_gcn(region_channels))
        self.add_module('compute_seir', torch.nn.Sequential(
                torch.nn.Linear(seir_channel*5+2, seir_channel*2, bias=False),
                torch.nn.BatchNorm1d(seir_channel*2),
                torch.nn.ReLU(),
                torch.nn.Linear(seir_channel*2, seir_channel, bias=True),
                torch.nn.ReLU()))
        self.add_module('compute_infection', torch.nn.Sequential(
                torch.nn.Linear(init_channels[2]//5, 1)))
        self.add_module('compute_density', torch.nn.Sequential(
                torch.nn.Linear(init_channels[2], density_channel),
                torch.nn.BatchNorm1d(density_channel),
                torch.nn.ReLU(),
                torch.nn.Linear(density_channel, 2)))

    def forward(self, A, A_regions, x, mean_density,
            label, num_steps, use_label=False, resume=False):
        """
        Args:
          -A: adjacent matrix of seir, define the process of compute.
          -x: initial input which contain initial density, mean density and infection(first day)
              or input of middle time(seir feature whose shape is (num_of_regions, 5, c))
          -label: label of mean_density, density and infection.
          -num_steps: run num_steps
          -use_label: whether use label as input of each time, if not, network use the predictions
              as the input of next step.
          -resume: whether continue to predict by previous step, which needs seir feature as input
              if not, restart from fisrt step, which needs initial input
        Returns:
          -densitys: prediction of density (len(densitys)==num_steps)
          -mean_densitys: prediction of density (len(mean_densitys)==num_steps)
          -infections: prediction of infections (len(infections)==num_steps)
          -seir_feature: the final seir_feature.
        """
        densitys = []
        mean_densitys = []
        infections = []
        self.num_of_regions = A_regions.shape[-1]
        if resume:
            seir_feature = x
        else:
            seir_feature = self.compute_init_seir(x)
            seir_feature = seir.reshape(x.shape[1], 5, -1)
        for i in range(num_steps):
            seir_feature = self.relate_regions(A_regions, 
                    seir_feature.reshape(self.num_of_regions, -1))
            seir_feature = seir_feature.reshape(self.num_of_regions, 5, -1)
            seir_feature = self.seir(A, seir_feature, mean_density)
            infection = self.compute_infection(seir_feature[:, self.I, :])
            d = self.compute_density(seir_feature.reshape(self.num_of_regions, -1))
            density, mean_density = d[:, 0], d[:, 1]
            if use_label:
                seir_feature = torch.cat([seir_feature.reshape(self.num_of_regions, -1), 
                        label[0][i+1].unsqueeze(dim=-1), infection[i+1].unsqueeze(dim=-1)], dim=-1)
            else:
                seir_feature = torch.cat([seir_feature.reshape(self.num_of_regions, -1), 
                        density.unsqueeze(dim=-1), infection.unsqueeze(dim=-1)], dim=-1)
            seir_feature = self.compute_seir(seir_feature)
            infections.append(infection)
            mean_densitys.append(mean_density)
            densitys.append(density)
            if use_label:
                mean_density = label[1][i, :, :]
        return (densitys, mean_densitys, infections), seir_feature

    def compute_loss(self, y, label, weight):
        y = torch.cat(y, axis=-1)
        label = torch.cat(label, axis=-1)
        loss = self.mse_loss(y, label)
        loss = torch.matmul(loss, weight)
        loss = torch.mean(loss)
        return loss

    def evaluate(self, y, label):
        e = torch.sqrt(torch.mean(torch.power(torch.log((y+1)/(label+1)), 2)))
        return e

if __name__ == "__main__":
    m = model([3, 36, 10], [2, 36], [10, 36, 10], 36)
    param = m.parameters()
    print(sum(p.numel() for p in param))
