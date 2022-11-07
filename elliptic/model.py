import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv



class TwoLayerGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout)
        output = self.linear2(x1)
        # x = F.softmax(x, dim=1)
        features = x1
        return output, features
