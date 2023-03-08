from copy import deepcopy
import torch
import torch.nn as nn
from .augment import DropNodes


class DGI(nn.Module):
    def __init__(self, encoder, emb_dim, p, device="cpu", encoder2=None):
        super(DGI, self).__init__()
        self.device = device
        self.encoder1 = encoder
        self.encoder2 = deepcopy(encoder2) if encoder2 else deepcopy(encoder)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(emb_dim)
        self.dn = DropNodes(p)
        print("Drop node augmentor:", self.dn)

    def forward(self, data, samp_bias1=None, samp_bias2=None, single_graph_mode=None): # mode only has to specify for single graph situations

        # h_0 = self.encoder(data.x, data.edge_index) # node num * emb dim
        data_v1 = data.to(self.device) # data.to(self.device)
        data_v2 = self.dn(deepcopy(data)).to(self.device)
        mask = data_v2.node_mask
        if 'train_mask' in data and single_graph_mode == 'train':
            mask = torch.logical_and(mask, data.train_mask)
        elif 'val_mask' in data and single_graph_mode == 'val':
            mask = torch.logical_and(mask, data.val_mask)

        h_1 = self.encoder1(data_v1.x, data_v1.edge_index)[mask] # node num * emb dim
        c_1 = self.read(h_1)  # emb dim
        c_1 = self.sigm(c_1)  # emb dim


        h_2 = self.encoder2(data_v2.x, data_v2.edge_index)[mask] # node num * emb dim
        c_2 = self.read(h_2) # emb dim
        c_2 = self.sigm(c_2) # emb dim

        shuf_idx = torch.randperm(data.x.shape[0])
        h_3 = self.encoder1(data_v1.x[shuf_idx], data_v1.edge_index)[mask] # node num * emb dim (negative sample)
        h_4 = self.encoder2(data_v2.x[shuf_idx], data_v2.edge_index)[mask] # node num * emb dim (negative sample)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2, mask.sum().item()




class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, feat, msk=None): # feat: node num * emb dim, msk: node num
        if msk is None:
            return torch.mean(feat, 0) # emb dim
        else:
            return torch.mean(feat[msk], 0) # emb dim

class Discriminator(nn.Module):
    # the original implementation takes (batch size, node num, emb dim) as input
    # we get rid of the dim 0 since batch size is always 1
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        """
        :param c1: graph repr of augmented view 1, shape: feat dim
        :param c2: graph repr of augmented view 2, shape: feat dim
        :param h1: node repr of augmented view 1, shape: node num v1 * feat dim
        :param h2: node repr of augmented view 2, shape: node num v2 * feat dim
        :param h3: node repr of negative sample 1, shape: node num v1 * feat dim
        :param h4: node repr of negative sample 2, shape: node num v2 * feat dim
        :return:
        """
        c_x1 = torch.unsqueeze(c1, 0) # shape: 1 * feat_dim
        c_x1 = c_x1.expand_as(h1).contiguous() # shape: node num v1 * feat dim
        c_x2 = torch.unsqueeze(c2, 0) # shape: 1 * feat_dim
        c_x2 = c_x2.expand_as(h2).contiguous() # shape: node num v2 * feat dim
        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 1) # node repr 2 * graph repr 1 -> node num v2
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 1) # node repr 1 * graph repr 2 -> node num v1

        # negative
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 1) # node repr 4 * graph repr 1 -> node num v2
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 1) # node repr 3 * graph repr 2 -> node num v1

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 0) # 2 * (node num v1 + node num v2)
        return logits