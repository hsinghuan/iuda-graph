import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv



class TwoLayerGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return x

    def get_input_dim(self):
        return self.in_dim

    def get_hidden_dim(self):
        return self.hidden_dim