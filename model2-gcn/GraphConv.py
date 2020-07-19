import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        """"""
        h = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return aggr_out + self.lin(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
