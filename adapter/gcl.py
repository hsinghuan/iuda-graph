"""
Code modified from https://github.com/Shen-Lab/GraphCL
"""
import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from .augment import DropNodes, Subgraph



class MultiGraphContrastiveLearner():
    def __init__(self, encoder, emb_dim, aug1_p, aug2_p, device="cpu"):
        self.device = device
        self.model = DGI(encoder, emb_dim, aug1_p, aug2_p, device=device).to(self.device)  #

    def _pretrain_train(self, model, train_loader, optimizer, loss_fn):
        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            logits, _, _, node_num = model(data)
            lbl_1 = torch.ones(node_num * 2)
            lbl_2 = torch.zeros(node_num * 2)
            lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
            loss = loss_fn(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        return total_loss

    def _pretrain_val(self, model, val_loader, loss_fn):
        total_loss = 0
        for data in val_loader:
            data = data.to(self.device)
            logits, _, _, node_num = model(data)
            lbl_1 = torch.ones(node_num * 2)
            lbl_2 = torch.zeros(node_num * 2)
            lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
            loss = loss_fn(logits, lbl)
            total_loss += loss.item()
        total_loss /= len(val_loader)
        return total_loss

    def _pretrain_run(self, train_loader, val_loader, lr, args):
        model = deepcopy(self.model)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        b_xent = nn.BCEWithLogitsLoss()

        best_loss = np.inf
        patience = 10
        staleness = 0

        for e in range(1, args.pretrain_epochs + 1):
            train_loss = self._pretrain_train(model, train_loader, optimizer, b_xent)
            val_loss = self._pretrain_val(model, val_loader, b_xent)

            print(f"LR:{lr} Epoch:{e} Train Loss:{train_loss} Val Loss:{val_loss}")
            self.writer.add_scalar("Pretrain Loss/train", train_loss, e)
            self.writer.add_scalar("Pretrain Loss/val", val_loss, e)


            if val_loss < best_loss:
                best_loss = val_loss
                staleness = 0
            else:
                staleness += 1

            if staleness > patience:
                break

        return model, best_loss

    def pretrain(self, train_loader, val_loader, lr_list, stage_name, args, subdir_name=""):
        # print("Before pretraining")
        # for name, param in self.model.encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break

        performance_dict = dict()
        for lr in lr_list:
            run_name = f'pretrain_{str(lr)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name,
                             run_name))
            model, loss = self._pretrain_run(train_loader, val_loader, lr, args)
            performance_dict[lr] = {'model': model, 'loss': loss}

        best_loss = np.inf
        best_model = None
        for lr, perf_dict in performance_dict.items():
            if perf_dict['loss'] < best_loss:
                best_loss = perf_dict['loss']
                best_model = perf_dict['model']
            print(f"pretrain lr: {lr} loss: {perf_dict['loss']}")

        pretrained_encoder = deepcopy(best_model.encoder)
        # print("After pretraining")
        # for name, param in pretrained_encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        return pretrained_encoder


class SingleGraphContrastiveLearner(): # for arxiv
    def __init__(self, encoder, emb_dim, aug1_p, aug2_p, device="cpu"):
        self.device = device
        self.model = DGI(encoder, emb_dim, aug1_p, aug2_p, device=device).to(self.device)  #
        self.aug1_p = aug1_p
        self.aug2_p = aug2_p

    def _pretrain_train(self, model, cluster_tgt_loader, optimizer, loss_fn):
        total_loss = 0
        total_node = 0
        for data in cluster_tgt_loader:
            data = data.to(self.device)
            logits, _, _, node_num = model(data)
            lbl_1 = torch.ones(node_num * 2)
            lbl_2 = torch.zeros(node_num * 2)
            lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
            loss = loss_fn(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(lbl)
            total_node += len(lbl)

        return total_loss / total_node

    def _pretrain_run(self, cluster_tgt_loader, lr, args):
        model = deepcopy(self.model)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        b_xent = nn.BCEWithLogitsLoss()

        best_loss = np.inf
        patience = 10
        staleness = 0

        for e in range(1, args.pretrain_epochs + 1):
            loss = self._pretrain_train(model, cluster_tgt_loader, optimizer, b_xent)

            print(f"LR:{lr} Epoch:{e} Train Loss:{loss}")
            self.writer.add_scalar("Pretrain Loss/train", loss, e)

            if loss < best_loss:
                best_loss = loss
                staleness = 0
            else:
                staleness += 1

            if staleness > patience:
                break

        return model, best_loss

    def pretrain(self, cluster_tgt_loader, lr_list, stage_name, args, subdir_name=""): # tgt contains src in arxiv
        # print("Before pretraining")
        # for name, param in self.model.encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        performance_dict = dict()
        for lr in lr_list:
            run_name = f'gcl_pretrain_{str(lr)}_{str(self.aug1_p)}_{str(self.aug2_p)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name,
                             run_name))
            model, loss = self._pretrain_run(cluster_tgt_loader, lr, args)
            performance_dict[lr] = {'model': model, 'loss': loss}

        best_loss = np.inf
        best_model = None
        for lr, perf_dict in performance_dict.items():
            if perf_dict['loss'] < best_loss:
                best_loss = perf_dict['loss']
                best_model = perf_dict['model']
            print(f"pretrain lr: {lr} loss: {perf_dict['loss']}")

        pretrained_encoder = deepcopy(best_model.encoder)
        # print("After pretraining")
        # for name, param in pretrained_encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        return pretrained_encoder





class DGI(nn.Module):
    def __init__(self, encoder, emb_dim, aug1_p, aug2_p, device="cpu"):
        super(DGI, self).__init__()
        self.device = device
        self.encoder = encoder
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(emb_dim)
        # follow GraphCL's recommendation, use node drop and subgraph
        self.aug1 = DropNodes(aug1_p)
        self.aug2 = DropNodes(aug2_p) # Subgraph(0.2)
        print("Augmentation 1:", self.aug1)
        print("Augmentation 2:", self.aug2)
    def forward(self, data, samp_bias1=None, samp_bias2=None):

        # h_0 = self.encoder(data.x, data.edge_index) # node num * emb dim

        data_v1 = self.aug1(deepcopy(data)).to(self.device) # view 1
        data_v2 = self.aug2(deepcopy(data)).to(self.device) # view 2
        msk = torch.logical_and(data_v1.node_mask, data_v2.node_mask)  # only focus on the common nodes

        h_1 = self.encoder(data_v1.x, data_v1.edge_index)[msk] # left node num * emb dim
        c_1 = self.read(h_1)  # emb dim
        c_1 = self.sigm(c_1)  # emb dim

        h_2 = self.encoder(data_v2.x, data_v2.edge_index)[msk] # left node num * emb dim
        c_2 = self.read(h_2) # emb dim
        c_2 = self.sigm(c_2) # emb dim

        shuf_idx = torch.randperm(data.x.shape[0])
        h_3 = self.encoder(data_v1.x[shuf_idx], data_v1.edge_index)[msk] # node num * emb dim (negative sample)
        h_4 = self.encoder(data_v2.x[shuf_idx], data_v2.edge_index)[msk] # node num * emb dim (negative sample)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2, msk.sum().item()

    # # Detach the return variables
    # def embed(self, data, msk):
    #     h_1 = self.encoder(data.x, data.edge_index)
    #     c = self.read(h_1, msk)
    #
    #     return h_1.detach(), c.detach()


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
