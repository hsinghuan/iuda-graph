import torch.nn as nn
import torch.nn.functional as F

class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout)
        output = self.linear2(x1)
        # x = F.softmax(x, dim=1)
        features = x1
        return output, features

class OneLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)

    def reset_parameters(self):
        self.linear1.reset_parameters()

    def forward(self, x):
        output = self.linear1(x)
        features = x
        return output, features