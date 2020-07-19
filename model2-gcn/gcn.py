import torch
import numpy as np
import pandas as pd
import random
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import GraphConv


def dataset(data_onehot, data_density, data_edge):
    """
    定义图结构
    :param data_onehot: 对每个地区进行one hot编码，尺寸是 [城市的区域数*903]
    :param data_density: 将density数据作为模型的标签，尺寸是 [城市的区域数*1]
    :param data_edge: 定义图结构的边信息
    :return: 整个网络的输入图结构信息
    """
    x = torch.Tensor(data_onehot, dtype=torch.float)
    y = torch.Tensor(data_density, dtype=torch.float)
    edge_index = torch.Tensor(data_edge, dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)

    return data


def norm(x):
    """
    对部分特征进行log归一化处理
    :param x: 需要归一化的输入特征
    :return: 归一化后的数据
    """
    return (math.log(x+1) + 2) / 10


def denorm(x):
    """
    对输出的日感染病人数进行恢复
    :param x: 需要去归一化的输出特征
    :return: 日感染病人数(整数)
    """
    return round(math.exp(x * 10 - 2))


'''def train_net(data):
    """
    训练整个网络模型
    :param data:
    :return:
    """
    data = data.to(device)
    optimizer.zero_grad()
    output, embedding = model(data)
    label = data.y.to(device)
    loss = crit(output, label)
    loss.backward()
    optimizer.step()
    loss = loss.item()

    return loss, embedding, model'''


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GraphConv(embed_dim, 32)
        self.conv2 = GraphConv(32, 8)
        self.lin1 = torch.nn.Linear(9, 1)
        self.act1 = torch.nn.ReLU()

    def forward(self, data, x1, edge_weight):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.log_softmax(self.conv2(x, edge_index, edge_weight), dim=1)
        embedding = x

        x = torch.cat((x, x1), 1)

        x = self.lin1(x)
        x = self.act1(x)

        return x, embedding


if __name__ == '__main__':
    # 这里需要设置训练集和测试集
    # 训练集是用density有值的那部分
    # 测试集用未知density的部分做
    # 目的是获得未知的density，同时会提取网络的中间层embedding做特征

    density = pd.read_csv("./DATA/city_A/den_rst.csv", header=None)
    density = np.array(density.values.tolist())
    density = np.delete(density, 0, axis=1)
    for i in range(np.shape(density)[0]-1):
        for j in range(np.shape(density)[1]):
            density[i+1, j] = norm(density[i+1, j])

    infection = pd.read_csv("./DATA/city_A/inf_rst.csv", header=None)
    infection = np.array(infection.values.tolist())
    infection = np.delete(infection, 0, axis=1)
    for i in range(np.shape(infection)[0]-1):
        for j in range(np.shape(infection)[1]):
            infection[i+1, j] = norm(infection[i+1, j])

    edge = pd.read_csv("./DATA/tranfer_a_day_A.csv")
    edge = np.array(edge.values.tolist())

    one_hot = pd.read_csv("./DATA/city_A/one_hot.csv", header=None)
    one_hot = np.array(one_hot.values.tolist())
    one_hot = np.delete(one_hot, 0, axis=0)
    one_hot = np.delete(one_hot, 0, axis=1)

    data_edge = np.transpose(edge[:, (1, 2)])  # 每天的数据是一致的
    for i in range(np.shape(data_edge)[1]):
        for j in range(2):
            data_edge[j, i].strip("A_")
    edge_weight = np.transpose(edge[:, 3])
    edge_weight = torch.Tensor(edge_weight, dtype=torch.float)  # 每天的数据是一致的

    for i in range(np.shape(edge_weight)[1]):
        edge_weight[0, i] = norm(edge_weight[0, i])

    data_onehot = one_hot[:118, :]  # 每一天的数据也是一致的

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = 118  # 根据不同城市的地区数设置
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    crit = torch.nn.BCELoss()  # 用交叉熵函数来计算损失
    model.train()
    # 划分训练集
    # 这里需要按照数据集大小设置循环 epoch 和 batch_size
    batch_size = 8
    epoch = 32
    loss_result = np.zeros((epoch, 1))
    for k in range(epoch):
        loss_all = 0
        for i in range(batch_size):
            index = random.randint(0, np.shape(density)[1] - 1)
            data_density = density[1:, index]
            for j in range(np.shape(infection)[1]):
                if density[0, index] == infection[0, j]:
                    data_infection = infection[1:, index]
            data = dataset(data_onehot, data_density, data_edge)
            x1 = torch.Tensor(data_infection, dtype=torch.float)  # 将每个区域的新增感染病人数作为补充的输入特征，尺寸是[城市的区域数*1]

            data = data.to(device)
            x1 = x1.to(device)
            edge_weight = edge_weight.to(device)
            optimizer.zero_grad()
            output, _ = model(data, x1, edge_weight)
            label = data.y.to(device)
            loss = crit(output, label)
            loss.backward()
            optimizer.step()
            loss = loss.item()

            loss_all += loss * embed_dim

        loss = loss_all / batch_size
        loss_result[k, 0] = loss
        print('Epoch:', k, 'Loss:', loss)

    np.savetxt("./result/city_A_loss.csv", loss_result, delimiter=",")

    model.eval()
    density_rst = np.zeros((embed_dim, np.shape(infection)[1]))
    embedding_rst = np.zeros((embed_dim, np.shape(infection)[1]*8))
    for i in range(np.shape(infection)[1]):
        data_infection = infection[1:, i]
        data_density = density[1:, 0]  # 该定义不存在实际意义
        data = dataset(data_onehot, data_density, data_edge)
        x1 = torch.Tensor(data_infection, dtype=torch.float)

        data = data.to(device)
        x1 = x1.to(device)
        edge_weight = edge_weight.to(device)

        output, embedding = model(data, x1, edge_weight)
        for j in range(8):
            for k in range(embed_dim):
                embedding_rst[k, j+i*8] = embedding[k, j]

        for j in range(embed_dim):
            density_rst[j, i] = output[j, i]

    np.savetxt("./result/city_A_embedding.csv", embedding_rst, delimiter=",")
    np.savetxt("./result/city_A_density.csv", density_rst, delimiter=",")
