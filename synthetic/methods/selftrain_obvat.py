import os
from copy import deepcopy
import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from pytorch_adapt.validators import BNMValidator, IMValidator


class OBVATSelfTrainer():
    def __init__(self, model, src_train_loader=None, src_val_loader=None, device="cpu"):
        self.device = device
        self.set_model(model)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        # self.validator = BNMValidator()
        self.validator = IMValidator()

    def _adapt_train_epoch(self, model, tgt_train_loader, optimizer, gamma=1):



        model.train()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader)) if self.src_train_loader else len(tgt_train_loader)
        src_iter = iter(self.src_train_loader) if self.src_train_loader else None
        tgt_iter = iter(tgt_train_loader)
        total_src_loss = 0
        total_tgt_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_lds = 0
        total_src_logits = []
        total_tgt_logits = []

        for _ in range(len_dataloader):

            if self.src_train_loader:
                src_data = src_iter.next().to(self.device)
                src_node_num = src_data.x.shape[0]
                src_lds = self._obvat_loss(model, src_data, gamma=gamma) * src_node_num
                src_y, _ = model(src_data.x, src_data.edge_index)
                src_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_data.y, reduction='sum')
                total_src_logits.append(src_y)
            else:
                src_lds = torch.tensor(0.0)
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))

            tgt_data, pseudo_tgt_label, pseudo_tgt_mask = tgt_iter.next()
            pseudo_tgt_label = torch.squeeze(pseudo_tgt_label, dim=0).to(self.device)
            pseudo_tgt_mask = torch.squeeze(pseudo_tgt_mask, dim=0).to(self.device)
            tgt_data = tgt_data.to(self.device)
            tgt_node_num = pseudo_tgt_mask.sum().item()
            tgt_lds = self._obvat_loss(model, tgt_data, gamma=gamma) * tgt_node_num
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1)[pseudo_tgt_mask],
                                  pseudo_tgt_label[pseudo_tgt_mask], reduction='sum')
            total_tgt_logits.append(tgt_y)

            loss = src_loss + tgt_loss + src_lds + tgt_lds

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_lds += (src_lds + tgt_lds).item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num

        total_loss = (total_src_loss + total_tgt_loss + total_lds) / (total_src_node + total_tgt_node)  if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_tgt_loss = total_tgt_loss / total_tgt_node if total_tgt_node > 0 else 0.
        total_lds = total_lds / (total_src_node + total_tgt_node)  if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_lds, total_src_logits, total_tgt_logits


    def _adapt_test_epoch(self, model, tgt_val_loader, gamma=1):
        model.eval()

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader)) if self.src_val_loader else len(tgt_val_loader)
        src_iter = iter(self.src_val_loader) if self.src_val_loader else None
        tgt_iter = iter(tgt_val_loader)

        total_src_loss = 0
        total_tgt_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_lds = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):

            if self.src_val_loader:
                src_data = src_iter.next().to(self.device)
                src_node_num = src_data.x.shape[0]
                src_lds = self._obvat_loss(model, src_data, gamma=gamma) * src_node_num
                src_y, _ = model(src_data.x, src_data.edge_index)
                src_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_data.y, reduction='sum')
                total_src_logits.append(src_y)
            else:
                src_lds = torch.tensor(0.0)
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))

            tgt_data, pseudo_tgt_label, pseudo_tgt_mask = tgt_iter.next()
            pseudo_tgt_label = torch.squeeze(pseudo_tgt_label, dim=0).to(self.device)
            pseudo_tgt_mask = torch.squeeze(pseudo_tgt_mask, dim=0).to(self.device)
            tgt_data = tgt_data.to(self.device)
            tgt_node_num = pseudo_tgt_mask.sum().item()
            tgt_lds = self._obvat_loss(model, tgt_data, gamma=gamma) * tgt_node_num
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1)[pseudo_tgt_mask],
                                  pseudo_tgt_label[pseudo_tgt_mask], reduction='sum')
            total_tgt_logits.append(tgt_y)

            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_lds += (src_lds + tgt_lds).item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num

        total_loss = (total_src_loss + total_tgt_loss + total_lds) / (total_src_node + total_tgt_node) if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_tgt_loss = total_tgt_loss / total_tgt_node if total_tgt_node > 0 else 0.
        total_lds = total_lds / (total_src_node + total_tgt_node)  if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_lds, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, gamma, args):
        thres = 0.0
        model = deepcopy(self.model)
        tgt_train_datalist = []
        for data in tgt_train_loader:
            data = data.to(self.device)
            pseudo_y_hard_label, pseudo_mask = self._pseudo_label(model, data, thres)
            tgt_train_datalist.append((data, pseudo_y_hard_label, pseudo_mask))
        tgt_pseudo_train_loader = DataLoader(dataset=tgt_train_datalist, batch_size=1, shuffle=True)

        tgt_val_datalist = []
        for data in tgt_val_loader:
            data = data.to(self.device)
            pseudo_y_hard_label, pseudo_mask = self._pseudo_label(model, data, thres)
            tgt_val_datalist.append((data, pseudo_y_hard_label, pseudo_mask))
        tgt_pseudo_val_loader = DataLoader(dataset=tgt_val_datalist, batch_size=1, shuffle=True)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)

        # train with pseudo labels
        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 10
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_tgt_loss, train_lds, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                model, tgt_pseudo_train_loader, optimizer, gamma=gamma)
            val_loss, val_src_loss, val_tgt_loss, val_lds, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                model,
                tgt_pseudo_val_loader, gamma=gamma)
            train_src_score = self.validator(target_train={'logits': train_src_logits})
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_src_score = self.validator(target_train={'logits': val_src_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_score = val_tgt_score
                best_model = deepcopy(model)
                staleness = 0
            else:
                staleness += 1
            print(
                f'Gamma: {gamma} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} Train LDS: {round(train_lds, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)} Val LDS: {round(val_lds, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Label Loss/val", val_src_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Label Loss/train", train_tgt_loss, e)
            self.writer.add_scalar("Target Label Loss/val", val_tgt_loss, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("LDS/train", train_lds, e)
            self.writer.add_scalar("LDS/val", val_lds, e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score


    def adapt(self, tgt_train_loader, tgt_val_loader, gamma_list, stage, args):
        performance_dict = dict()
        for gamma in gamma_list:
            run_name = f'{args.method}_{str(gamma)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, args.shift, str(stage[0]) + "_" + str(stage[1]) + "_" + str(stage[2]),
                             run_name))
            model, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, gamma, args)
            performance_dict[gamma] = {'tgt_model': model,
                                            'tgt_val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for gamma, perf_dict in performance_dict.items():
            if perf_dict['tgt_val_score'] > best_val_score:
                best_val_score = perf_dict['tgt_val_score']
                best_model = perf_dict['tgt_model']
            print(f"gamma: {gamma} val_score: {perf_dict['tgt_val_score']}")
        self.set_model(best_model)

    def _pseudo_label(self, model, data, thres=0):
        model.eval()
        pseudo_y, _ = model(data.x, data.edge_index)
        pseudo_y = F.softmax(pseudo_y, dim=1)

        # if self.propagate:
        #     adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
        #                        sparse_sizes=(data.x.shape[0], data.x.shape[0]))
        #     adj_t = adj.t()
        #     deg = adj_t.sum(dim=1).to(torch.float)
        #     deg_inv_sqrt = deg.pow_(-0.5)
        #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #     DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        #     pseudo_y = self.prop(pseudo_y, DAD)

        pseudo_y_confidence, pseudo_y_hard_label = torch.max(pseudo_y, dim=1)
        pseudo_mask = pseudo_y_confidence > thres
        return pseudo_y_hard_label, pseudo_mask

    def _obvat_loss(self, model, data, gamma=1, iter=10):
        with torch.no_grad():
            pred, _ = model(data.x, data.edge_index)
        d = torch.rand(data.x.shape).sub(0.5).to(data.x.device)
        d = torch.autograd.Variable(d, requires_grad=True)
        optimizer = torch.optim.Adam(params=[d], lr=1e-2)
        for _ in range(iter):
            pred_hat, _ = model(data.x + d, data.edge_index)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            frob_norm = torch.norm(d)
            loss = - adv_distance + gamma * frob_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        d.requires_grad = False
        pred_hat, _ = model(data.x + d, data.edge_index)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        lds = F.kl_div(logp_hat, pred, reduction='batchmean')
        return lds

    def set_model(self, model):
        self.model = deepcopy(model).to(self.device)

    def get_model(self):
        return self.model
