import os
import collections
import math
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.nn.models import LabelPropagation
from torch_sparse import SparseTensor
from pytorch_adapt.validators import IMValidator
from .augment import WeakAugmentor, StrongAugmentor


class FixMatchAdapter():
    def __init__(self, model, src_train_loader=None, src_val_loader=None, device="cpu", propagate=False):
        self.device = device
        self.set_model(model)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        self.validator = IMValidator()
        self.propagate = propagate
        if self.propagate:
            print("Use Label Propagation As Consistency Regularization")
            self.prop = LabelPropagation(50, 0.6)
        self.weak_augment = WeakAugmentor()
        self.strong_augment = StrongAugmentor()

    def _adapt_train_epoch(self, model, tgt_train_loader, optimizer, thres, con_tradeoff, scheduler=None):
        model.train()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader)) if self.src_train_loader else len(tgt_train_loader)
        src_iter = iter(self.src_train_loader) if self.src_train_loader else None
        tgt_iter = iter(tgt_train_loader)
        total_loss = 0
        total_src_loss = 0
        total_con_loss = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            if self.src_train_loader:
                src_data = src_iter.next().to(self.device)
                src_y, _ = model(src_data.x, src_data.edge_index)
                src_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_data.y)
                total_src_logits.append(src_y)
            else:
                src_loss = torch.tensor(0.0)
                total_src_logits.append(torch.tensor([[]]))

            tgt_data = tgt_iter.next().to(self.device)
            # weak augment
            pseudo_tgt_label, pseudo_tgt_mask = self._pseudo_label(model, self.weak_augment(deepcopy(tgt_data)), thres)

            # strong augment
            # print("Non zero elements before strong augment:", torch.count_nonzero(tgt_data.x))
            # print("tgt data x", tgt_data.x[0])
            tgt_data = self.strong_augment(deepcopy(tgt_data))
            # print("Non zero elements after strong augment:", torch.count_nonzero(tgt_data.x))
            # print("tgt data x", tgt_data.x[0])
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            if 'node_mask' in tgt_data:
                pseudo_tgt_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.node_mask)
            con_loss = F.nll_loss(F.log_softmax(tgt_y[pseudo_tgt_mask], dim=1),
                                  pseudo_tgt_label[pseudo_tgt_mask]) if pseudo_tgt_mask.sum().item() != 0 else torch.tensor(0.)


            total_tgt_logits.append(tgt_y)

            loss = src_loss + con_loss * con_tradeoff
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            total_src_loss += src_loss.item()
            total_con_loss += con_loss.item()

        total_loss /= len_dataloader
        total_src_loss /= len_dataloader
        total_con_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_con_loss, total_src_logits, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, model, tgt_val_loader, thres, con_tradeoff):
        model.eval()

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader)) if self.src_val_loader else len(
            tgt_val_loader)
        src_iter = iter(self.src_val_loader) if self.src_val_loader else None
        tgt_iter = iter(tgt_val_loader)
        total_loss = 0
        total_src_loss = 0
        total_con_loss = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            if self.src_val_loader:
                src_data = src_iter.next().to(self.device)
                src_y, _ = model(src_data.x, src_data.edge_index)
                src_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_data.y)
                total_src_logits.append(src_y)
            else:
                src_loss = torch.tensor(0.0)
                total_src_logits.append(torch.tensor([[]]))

            tgt_data = tgt_iter.next().to(self.device)
            # weak augment
            pseudo_tgt_label, pseudo_tgt_mask = self._pseudo_label(model, self.weak_augment(deepcopy(tgt_data)), thres)
            # strong augment
            tgt_data = self.strong_augment(deepcopy(tgt_data))
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            if 'node_mask' in tgt_data:
                pseudo_tgt_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.node_mask)
            con_loss = F.nll_loss(F.log_softmax(tgt_y[pseudo_tgt_mask], dim=1),
                                  pseudo_tgt_label[pseudo_tgt_mask]) if pseudo_tgt_mask.sum().item() != 0 else torch.tensor(0.)


            total_tgt_logits.append(tgt_y)
            loss = src_loss + con_loss * con_tradeoff

            total_loss += loss.item()
            total_src_loss += src_loss.item()
            total_con_loss += con_loss.item()

        total_loss /= len_dataloader
        total_src_loss /= len_dataloader
        total_con_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_con_loss, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, thres, con_tradeoff, args):
        model = deepcopy(self.model)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)
        # optimizer = torch.optim.SGD(list(model.parameters()), lr=args.adapt_lr)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.adapt_epochs)

        # train with pseudo labels
        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 10
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_con_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                model, tgt_train_loader, optimizer, thres, con_tradeoff)
            val_loss, val_src_loss, val_con_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                model,
                tgt_val_loader, thres, con_tradeoff)
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
                f'Thres: {thres} Consistency Tradeoff: {con_tradeoff} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Consistency Loss: {round(train_con_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Consistency Loss: {round(val_con_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Label Loss/val", val_src_loss, e)
            self.writer.add_scalar("Target Consistency Loss/train", train_con_loss, e)
            self.writer.add_scalar("Target Consistency Loss/val", val_con_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            # self.writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score

    def _pseudo_label(self, model, data, thres):
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
        pseudo_mask = pseudo_y_confidence > thres
        return pseudo_y_hard_label, pseudo_mask

    def adapt(self, tgt_train_loader, tgt_val_loader, threshold_list, con_tradeoff_list, stage, args):
        performance_dict = collections.defaultdict(dict)
        for thres in threshold_list:
            for con_tradeoff in con_tradeoff_list:
                lp = "lp" if args.label_prop else ""
                run_name = f'{args.method}_{lp}_{str(thres)}_{str(con_tradeoff)}_{str(args.model_seed)}'

                self.writer = SummaryWriter(
                    os.path.join(args.log_dir, args.shift, str(stage[0]) + "_" + str(stage[1]) + "_" + str(stage[2]),
                                 run_name))
                model, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, thres, con_tradeoff, args)
                performance_dict[thres][con_tradeoff] = {'tgt_model': model, 'tgt_val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for thres, thres_dict in performance_dict.items():
            for con_tradeoff, ckpt_dict in thres_dict.items():
                if ckpt_dict['tgt_val_score'] > best_val_score:
                    best_val_score = ckpt_dict['tgt_val_score']
                    best_model = ckpt_dict['tgt_model']
                print(f"thres: {thres} consistency tradeoff: {con_tradeoff} val_score: {ckpt_dict['tgt_val_score']}")
        self.set_model(best_model)

    def set_model(self, model):
        self.model = deepcopy(model).to(self.device)

    def get_model(self):
        return self.model

# def get_cosine_schedule_with_warmup(optimizer,
#                                     num_warmup_steps,
#                                     num_training_steps,
#                                     num_cycles=7. / 16.,
#                                     last_epoch=-1):
#     def _lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         no_progress = float(current_step - num_warmup_steps) / \
#                       float(max(1, num_training_steps - num_warmup_steps))
#         return max(0., math.cos(math.pi * num_cycles * no_progress))
#
#     return LambdaLR(optimizer, _lr_lambda, last_epoch)