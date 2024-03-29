import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.nn.models import LabelPropagation
from pytorch_adapt.validators import DiversityValidator
from .adapter import FullModelMultigraphAdapter, FullModelSinglegraphAdapter

class MultigraphClassBalancedSelfTrainer(FullModelMultigraphAdapter):
    def __init__(self, model, src_train_loader=None, src_val_loader=None, num_class=2, device="cpu", propagate=False):
        super().__init__(model, src_train_loader, src_val_loader, device)
        self.num_class = num_class
        self.validator = DiversityValidator() #  change to a validator that doesn't use confidence measure
        self.propagate = propagate
        if self.propagate:
            print("Use Label Propagation As Consistency Regularization")
            self.prop = LabelPropagation(50, 0.6)

    def _adapt_train_epoch(self, model, tgt_train_loader, optimizer, reg_weight):
        model.train()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader)) if self.src_train_loader else len(tgt_train_loader)
        src_iter = iter(self.src_train_loader) if self.src_train_loader else None
        tgt_iter = iter(tgt_train_loader)


        total_src_loss = 0
        total_tgt_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            if self.src_train_loader:
                src_data = src_iter.next().to(self.device)
                src_y, _ = model(src_data.x, src_data.edge_index)
                if 'mask' in src_data: # e.g. elliptic (not all nodes are labeled)
                    src_mask = src_data.mask
                else:
                    src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask], reduction='sum')
                src_node_num = src_data.x.shape[0]
                total_src_logits.append(src_y)
            else:
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))


            tgt_data, pseudo_tgt_label, pseudo_tgt_mask = tgt_iter.next()
            pseudo_tgt_label = torch.squeeze(pseudo_tgt_label, dim=0).to(self.device)
            pseudo_tgt_mask = torch.squeeze(pseudo_tgt_mask, dim=0).to(self.device)
            tgt_data = tgt_data.to(self.device)
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1)[pseudo_tgt_mask],
                                         pseudo_tgt_label[pseudo_tgt_mask], reduction='sum')

            mrkld = torch.sum(- F.log_softmax(tgt_y, dim=1)[pseudo_tgt_mask] / self.num_class)
            tgt_loss += mrkld * reg_weight


            tgt_node_num = tgt_data.x[pseudo_tgt_mask].shape[0]
            total_tgt_logits.append(tgt_y[pseudo_tgt_mask])

            loss = src_loss + tgt_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num

        total_loss = (total_src_loss + total_tgt_loss) / (total_src_node + total_tgt_node) if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_tgt_loss = total_tgt_loss / total_tgt_node if total_tgt_node > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_src_logits, total_tgt_logits, total_tgt_node

    @torch.no_grad()
    def _adapt_test_epoch(self, model, tgt_val_loader, reg_weight):
        model.eval()

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader)) if self.src_val_loader else len(tgt_val_loader)
        src_iter = iter(self.src_val_loader) if self.src_val_loader else None
        tgt_iter = iter(tgt_val_loader)

        total_src_loss = 0
        total_tgt_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            if self.src_train_loader:
                src_data = src_iter.next().to(self.device)
                src_y, _ = model(src_data.x, src_data.edge_index)
                if 'mask' in src_data: # e.g. elliptic (not all nodes are labeled)
                    src_mask = src_data.mask
                else:
                    src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask], reduction='sum')
                src_node_num = src_data.x.shape[0]
                total_src_logits.append(src_y)
            else:
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))

            tgt_data, pseudo_tgt_label, pseudo_tgt_mask = tgt_iter.next()
            pseudo_tgt_label = torch.squeeze(pseudo_tgt_label, dim=0).to(self.device)
            pseudo_tgt_mask = torch.squeeze(pseudo_tgt_mask, dim=0).to(self.device)
            tgt_data = tgt_data.to(self.device)
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1)[pseudo_tgt_mask],
                                  pseudo_tgt_label[pseudo_tgt_mask], reduction='sum')

            mrkld = torch.sum(- F.log_softmax(tgt_y, dim=1)[pseudo_tgt_mask] / self.num_class)
            tgt_loss += mrkld * reg_weight

            tgt_node_num = tgt_data.x[pseudo_tgt_mask].shape[0]
            total_tgt_logits.append(tgt_y[pseudo_tgt_mask])

            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num

        total_loss = (total_src_loss + total_tgt_loss) / (total_src_node + total_tgt_node) if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_tgt_loss = total_tgt_loss / total_tgt_node if total_tgt_node > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_src_logits, total_tgt_logits, total_tgt_node

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, args, p_min=0.1, p_max=0.5, p_inc=0.05, reg_weight=None):
        model = deepcopy(self.model)
        p_list = [p_min + i * p_inc for i in range(int((p_max - p_min) // p_inc) + 2)]
        total_e = 0
        # print("reg weight", reg_weight)

        for p in p_list:
            # pseudo label
            tgt_train_datalist = []
            for data in tgt_train_loader:
                data = data.to(self.device)
                pseudo_y_hard_label, pseudo_mask = self._pseudo_label(model, data, p)
                tgt_train_datalist.append((data, pseudo_y_hard_label, pseudo_mask))
            tgt_pseudo_train_loader = DataLoader(dataset=tgt_train_datalist, batch_size=1, shuffle=True)

            tgt_val_datalist = []
            for data in tgt_val_loader:
                data = data.to(self.device)
                pseudo_y_hard_label, pseudo_mask = self._pseudo_label(model, data, p)
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
                total_e += 1
                train_loss, train_src_loss, train_tgt_loss, train_src_logits, train_tgt_logits, train_tgt_nodenum = self._adapt_train_epoch(
                    model, tgt_pseudo_train_loader, optimizer, reg_weight)
                val_loss, val_src_loss, val_tgt_loss, val_src_logits, val_tgt_logits, val_tgt_nodenum = self._adapt_test_epoch(
                    model,
                    tgt_pseudo_val_loader,
                    reg_weight)
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
                    f'p: {round(p, 3)} Epoch: {total_e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)}')

                self.writer.add_scalar("Total Loss/train", train_loss, total_e)
                self.writer.add_scalar("Total Loss/val", val_loss, total_e)
                self.writer.add_scalar("Source Loss/train", train_src_loss, total_e)
                self.writer.add_scalar("Source Loss/val", val_src_loss, total_e)
                self.writer.add_scalar("Target Loss/train", train_tgt_loss, total_e)
                self.writer.add_scalar("Target Loss/val", val_tgt_loss, total_e)
                self.writer.add_scalar("Source Score/train", train_src_score, total_e)
                self.writer.add_scalar("Source Score/val", val_src_score, total_e)
                self.writer.add_scalar("Target Score/train", train_tgt_score, total_e)
                self.writer.add_scalar("Target Score/val", val_tgt_score, total_e)
                self.writer.add_scalar("Target Node Num/train", train_tgt_nodenum, total_e)
                self.writer.add_scalar("Target Node Num/val", val_tgt_nodenum, total_e)
                if staleness > patience:
                    break

            model = deepcopy(best_model)

        return model, best_val_score

    def adapt(self, tgt_train_loader, tgt_val_loader, reg_weight_list, stage_name, args, subdir_name=""):
        performance_dict = dict()
        for reg_weight in reg_weight_list:
            run_name = f'{args.method}_{str(args.p_min).replace(".", "")}_{str(args.p_max).replace(".", "")}_{str(args.p_inc).replace(".", "")}_{str(reg_weight).replace(".", "")}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name,
                             run_name))
            model, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, args,
                                                                    p_min=args.p_min, p_max=args.p_max,
                                                                    p_inc=args.p_inc, reg_weight=reg_weight)
            performance_dict[reg_weight] = {'model': model,
                                            'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for reg_weight, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_model = perf_dict['model']

        self.set_model(best_model)

    @torch.no_grad()
    def _pseudo_label(self, model, data, p):
        model.eval()
        pseudo_y, _ = model(data.x, data.edge_index)
        pseudo_y = F.softmax(pseudo_y, dim=1)

        if self.propagate:
            adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                               sparse_sizes=(data.x.shape[0], data.x.shape[0]))
            adj_t = adj.t()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            pseudo_y = self.prop(pseudo_y, DAD)
        pseudo_y_confidence, pseudo_y_hard_label = torch.max(pseudo_y, dim=1)
        pseudo_mask = torch.zeros_like(pseudo_y_hard_label, dtype=torch.bool)
        # for each class, sort the confidence from high to low, and mark the top p portion as True in the pseudo mask
        for cls in range(torch.max(pseudo_y_hard_label) + 1):
            cls_num = (pseudo_y_hard_label == cls).sum().item()
            cls_confidence = pseudo_y_confidence[pseudo_y_hard_label == cls]  # the confidence of those predicted as cls
            cls_idx = torch.arange(len(pseudo_y_hard_label))[
                pseudo_y_hard_label == cls]  # the true indices of those predicted as cls
            sorted_confidence_idx = torch.argsort(cls_confidence, descending=True)
            top_p_confident_cls_idx = cls_idx[sorted_confidence_idx][:int(cls_num * p) + 1]
            # print(pseudo_y_confidence[top_p_confident_cls_idx])
            pseudo_mask[top_p_confident_cls_idx] = True

        return pseudo_y_hard_label, pseudo_mask



class SinglegraphClassBalancedSelfTrainer(FullModelSinglegraphAdapter):
    def __init__(self, model, src_data=None, num_class=2, device="cpu", propagate=False):
        super().__init__(model, src_data, device)
        self.num_class = num_class
        self.validator = DiversityValidator()  # change to a validator that doesn't use confidence measure
        self.propagate = propagate
        if self.propagate:
            print("Use Label Propagation As Consistency Regularization")
            self.prop = LabelPropagation(50, 0.6)

    def _adapt_train_epoch(self, model, tgt_data, pseudo_tgt_label, pseudo_tgt_mask, optimizer, reg_weight):
        model.train()

        if self.src_data:
            src_y, _ = model(self.src_data.x, self.src_data.edge_index)
            src_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.train_mask], dim=1), self.src_data.y[self.src_data.train_mask], reduction='sum')
            src_node_num = self.src_data.train_mask.sum().item()
        else:
            src_loss, src_node_num = torch.tensor(0.0), 0
            src_y = torch.tensor([[]])

        pseudo_tgt_train_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.train_mask)
        tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
        tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1)[pseudo_tgt_train_mask],
                              pseudo_tgt_label[pseudo_tgt_train_mask], reduction='sum')

        mrkld = torch.sum(- F.log_softmax(tgt_y, dim=1)[pseudo_tgt_train_mask] / self.num_class)
        tgt_loss += mrkld * reg_weight
        tgt_node_num = pseudo_tgt_train_mask.sum().item()

        loss = src_loss + tgt_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item() / (src_node_num + tgt_node_num) if (src_node_num + tgt_node_num) > 0 else 0.
        train_src_loss = src_loss.item() / src_node_num if src_node_num > 0 else 0.
        train_tgt_loss = tgt_loss.item() / tgt_node_num if tgt_node_num > 0 else 0.
        return train_loss, train_src_loss, train_tgt_loss, src_y[self.src_data.train_mask], tgt_y[tgt_data.train_mask], tgt_node_num

    @torch.no_grad()
    def _adapt_test_epoch(self, model, tgt_data, pseudo_tgt_label, pseudo_tgt_mask, reg_weight):
        model.eval()

        if self.src_data:
            src_y, _ = model(self.src_data.x, self.src_data.edge_index)
            src_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.val_mask], dim=1),
                                  self.src_data.y[self.src_data.val_mask], reduction='sum')
            src_node_num = self.src_data.val_mask.sum().item()
        else:
            src_loss, src_node_num = torch.tensor(0.0), 0
            src_y = torch.tensor([[]])

        pseudo_tgt_val_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.val_mask)
        tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
        tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1)[pseudo_tgt_val_mask],
                              pseudo_tgt_label[pseudo_tgt_val_mask], reduction='sum')

        mrkld = torch.sum(- F.log_softmax(tgt_y, dim=1)[pseudo_tgt_val_mask] / self.num_class)
        tgt_loss += mrkld * reg_weight
        tgt_node_num = pseudo_tgt_val_mask.sum().item()

        loss = src_loss + tgt_loss


        val_loss = loss.item() / (src_node_num + tgt_node_num) if (src_node_num + tgt_node_num) > 0 else 0.
        val_src_loss = src_loss.item() / src_node_num if src_node_num > 0 else 0.
        val_tgt_loss = tgt_loss.item() / tgt_node_num if tgt_node_num > 0 else 0.
        return val_loss, val_src_loss, val_tgt_loss, src_y[self.src_data.val_mask], tgt_y[tgt_data.val_mask], tgt_node_num

    def _adapt_train_test(self, tgt_data, args, p_min=0.1, p_max=0.5, p_inc=0.05, reg_weight=None):
        model = deepcopy(self.model)
        p_list = [p_min + i * p_inc for i in range(int((p_max - p_min) // p_inc) + 2)]
        total_e = 0
        # print("reg weight", reg_weight)

        for p in p_list:
            pseudo_y_hard_label, pseudo_mask = self._pseudo_label(model, tgt_data, p)
            optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)

            # train with pseudo labels
            best_val_loss = np.inf
            best_val_score = None
            best_model = None
            patience = 10
            staleness = 0
            for e in range(1, args.adapt_epochs + 1):
                total_e += 1
                train_loss, train_src_loss, train_tgt_loss, train_src_logits, train_tgt_logits, train_tgt_nodenum = self._adapt_train_epoch(
                    model, tgt_data, pseudo_y_hard_label, pseudo_mask, optimizer, reg_weight)
                val_loss, val_src_loss, val_tgt_loss, val_src_logits, val_tgt_logits, val_tgt_nodenum = self._adapt_test_epoch(
                    model,
                    tgt_data, pseudo_y_hard_label, pseudo_mask,
                    reg_weight)
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
                    f'p: {round(p, 3)} Epoch: {total_e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)}')

                self.writer.add_scalar("Total Loss/train", train_loss, total_e)
                self.writer.add_scalar("Total Loss/val", val_loss, total_e)
                self.writer.add_scalar("Source Loss/train", train_src_loss, total_e)
                self.writer.add_scalar("Source Loss/val", val_src_loss, total_e)
                self.writer.add_scalar("Target Loss/train", train_tgt_loss, total_e)
                self.writer.add_scalar("Target Loss/val", val_tgt_loss, total_e)
                self.writer.add_scalar("Source Score/train", train_src_score, total_e)
                self.writer.add_scalar("Source Score/val", val_src_score, total_e)
                self.writer.add_scalar("Target Score/train", train_tgt_score, total_e)
                self.writer.add_scalar("Target Score/val", val_tgt_score, total_e)
                self.writer.add_scalar("Target Node Num/train", train_tgt_nodenum, total_e)
                self.writer.add_scalar("Target Node Num/val", val_tgt_nodenum, total_e)
                if staleness > patience:
                    break

            model = deepcopy(best_model)

        return model, best_val_score


    def adapt(self, tgt_data, reg_weight_list, stage_name, args, subdir_name=""):
        tgt_data = tgt_data.to(self.device)
        performance_dict = dict()
        for reg_weight in reg_weight_list:
            run_name = f'{args.method}_{str(args.p_min).replace(".", "")}_{str(args.p_max).replace(".", "")}_{str(args.p_inc).replace(".", "")}_{str(reg_weight).replace(".", "")}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name,
                             run_name))
            model, val_score = self._adapt_train_test(tgt_data, args,
                                                      p_min=args.p_min, p_max=args.p_max,
                                                      p_inc=args.p_inc, reg_weight=reg_weight)
            performance_dict[reg_weight] = {'model': model,
                                            'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for reg_weight, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_model = perf_dict['model']

        self.set_model(best_model)

    @torch.no_grad()
    def _pseudo_label(self, model, data, p):
        model.eval()
        pseudo_y, _ = model(data.x, data.edge_index)
        pseudo_y = F.softmax(pseudo_y, dim=1)

        if self.propagate:
            adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                               sparse_sizes=(data.x.shape[0], data.x.shape[0]))
            adj_t = adj.t()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            pseudo_y = self.prop(pseudo_y, DAD)
        pseudo_y_confidence, pseudo_y_hard_label = torch.max(pseudo_y, dim=1)
        pseudo_mask = torch.zeros_like(pseudo_y_hard_label, dtype=torch.bool)
        # for each class, sort the confidence from high to low, and mark the top p portion as True in the pseudo mask
        for cls in range(torch.max(pseudo_y_hard_label) + 1):
            cls_num = (pseudo_y_hard_label == cls).sum().item()
            cls_confidence = pseudo_y_confidence[pseudo_y_hard_label == cls]  # the confidence of those predicted as cls
            cls_idx = torch.arange(len(pseudo_y_hard_label))[
                pseudo_y_hard_label == cls]  # the true indices of those predicted as cls
            sorted_confidence_idx = torch.argsort(cls_confidence, descending=True)
            top_p_confident_cls_idx = cls_idx[sorted_confidence_idx][:int(cls_num * p) + 1]
            # print(pseudo_y_confidence[top_p_confident_cls_idx])
            pseudo_mask[top_p_confident_cls_idx] = True

        return pseudo_y_hard_label, pseudo_mask