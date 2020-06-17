import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt


def trainModule():
    # 记得改名字：
    dataPath = './data/cityA.csv'  # 数据位置（注意数据格式已经更改，打开查看）
    savePath = "./model/networkA.pth"  # 模型保存位置

    # 模型参数：
    inp_dim = 2  # 输入数据的维度，[感染数，流入量，流出量]
    out_dim = 3  # 输出数据的维度，只预测 [流入量，流出量]
    mid_dim = 20  # LSTM三个门的网络宽度，即LSTM输出的张量维度，可调参数，类似于隐神经元数量
    mid_layers = 2  # LSTM内部各个门使用的全连接层数量，一般设置为1或2，可调参数
    batch_size = 20  # 批训练的数据量，不要大于或接近总数据量，可调参数
    learning_rate = 0.01  # 学习效率，可调参数
    episodes = 100  # 训练轮次

    data = loadData(dataPath)
    dataX = data[:-1, :]  # 输入数据，也就是 [感染数，流入量，流出量]
    dataY = data[+1:, 5:]  # 标签，也就是 [流入量，流出量]
    assert dataX.shape[1] == inp_dim

    # 训练集的数量，可调参数，下边两个语句一个是分训练集的，一个是45天全用的：
    trainSize = int(len(data) * 0.75)  # 前75%做训练
    # trainSize = len(data)

    # 分出训练集
    trainX = dataX[:trainSize]
    trainY = dataY[:trainSize]
    trainX = trainX.reshape((trainSize, inp_dim))
    trainY = trainY.reshape((trainSize, out_dim))

    # 建立模型，不用改的部分
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LSTMPre(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # t训练，不用改的部分
    varX = torch.tensor(trainX, dtype=torch.float32, device=device)
    varY = torch.tensor(trainY, dtype=torch.float32, device=device)

    batchVarX = list()
    batchVarY = list()

    for i in range(batch_size):
        j = trainSize - i
        batchVarX.append(varX[j:])
        batchVarY.append(varY[j:])

    batchVarX = pad_sequence(batchVarX)
    batchVarY = pad_sequence(batchVarY)

    with torch.no_grad():
        weights = np.tanh(np.array(len(trainY)) * (np.e / len(trainY)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Start to train")
    for e in range(episodes):
        out = net(batchVarX)

        loss = lossFunction(out, batchVarY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("episode: ", e, ", loss: ", loss.item())

    # 截止到现在已经完成了训练并且模型已经保存好了
    torch.save(net.state_dict(), savePath)

    # 验证部分，如果全用来训练了可以直接注释如下的部分
    net.load_state_dict(torch.load(savePath, map_location=lambda storage, loc: storage))
    net = net.eval()

    testX = dataX.copy()
    testX[trainSize:, 0] = 0
    testX = testX[:, np.newaxis, :]
    testX = torch.tensor(testX, dtype=torch.float32, device=device)

    for i in range(trainSize, len(data) - 2):
        testY = net(testX[:i])
        testX[i, 0, 5] = testY[-1, 0, 0]
        testX[i, 0, 6] = testY[-1, 0, 1]
        testX[i, 0, 7] = testY[-1, 0, 2]
        testX[i, 0, 8] = testY[-1, 0, 3]
        testX[i, 0, 9] = testY[-1, 0, 4]
        testX[i, 0, 10] = testY[-1, 0, 5]
        testX[i, 0, 11] = testY[-1, 0, 6]
        testX[i, 0, 12] = testY[-1, 0, 7]
        testX[i, 0, 13] = testY[-1, 0, 8]
        testX[i, 0, 14] = testY[-1, 0, 9]
    # predIn = testX[1:, 0, 1:2]
    # predOut = testX[1:, 0, 1:2]
    # predIn = predIn.cpu().data.numpy()
    # predOut = predOut.cpu().data.numpy()

    predData = testX[1:, 0, 5:14]
    predData = predData.cpu().data.numpy()

    # 画图部分，不需要的话直接注释掉
    """
    plt.figure()
    subFigure1 = plt.subplot(2, 1, 1)
    subFigure2 = plt.subplot(2, 1, 2)

    plt.sca(subFigure1)
    plt.plot(predIn, 'r', label="prediction")
    plt.plot(dataY[:, 0], 'b', label="real", alpha=0.3)
    plt.plot([trainSize, trainSize], [-1, 2], color='k', label='train | evaluation')
    plt.legend(loc='best')

    plt.sca(subFigure2)
    plt.plot(predOut, 'r', label="prediction")
    plt.plot(dataY[:, 1], 'b', label="real", alpha=0.3)
    plt.plot([trainSize, trainSize], [-1, 2], color='k', label='train | evaluation')
    plt.legend(loc='best')

    plt.show()
    """

class LSTMPre(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(LSTMPre, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)
        self.regression = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Tanh(), nn.Linear(mid_dim, out_dim))

    def forward(self, x):
        y = self.rnn(x)[0]

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.regression(y)
        y = y.view(seq_len, batch_size, -1)

        return y

    def output(self, x, hc):
        y, hc = self.rnn(x, hc)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.regression(y)
        y = y.view(seq_len, batch_size, -1)

        return y, hc


def loadData(dataPath):
    dataLoad = np.loadtxt(dataPath, delimiter=",")
    return dataLoad


if __name__ == "__main__":
    trainModule()
    # loadData()
