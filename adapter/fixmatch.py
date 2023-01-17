import os
from copy import deepcopy
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from pytorch_adapt.validators import IMValidator
from .adapter import FullModelMultigraphAdapter, FullModelSinglegraphAdapter
from .augment import WeakAugmentor, StrongAugmentor

class MultigraphFixMatchAdapter(FullModelMultigraphAdapter):
    def __init__(self, model, src_train_loader=None, src_val_loader=None, device="cpu", propagate=False, weak_p=0.1, strong_p=0.2):
        super().__init__(model, src_train_loader, src_val_loader, device)
        self.validator = IMValidator()
        self.propagate = propagate
        if self.propagate:
            print("Use Label Propagation As Consistency Regularization")
            self.prop = LabelPropagation(50, 0.6)
        self.weak_augment = WeakAugmentor(dropedge_p=weak_p, dropfeat_p=weak_p)
        # self.strong_augment = WeakAugmentor(dropedge_p=strong_p, dropfeat_p=strong_p)
        self.strong_augment = StrongAugmentor(dropnode_p=strong_p, dropedge_p=strong_p, dropfeat_p=strong_p, addedge_p=strong_p)

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
                if 'mask' in src_data: # e.g. elliptic (not all nodes are labeled)
                    src_mask = src_data.mask
                else:
                    src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask])
                total_src_logits.append(src_y)
            else:
                src_loss = torch.tensor(0.0)
                total_src_logits.append(torch.tensor([[]]))

            tgt_data = tgt_iter.next().to(self.device)
            # weak augment
            pseudo_tgt_label, pseudo_tgt_mask, pseudo_tgt_logits = self._pseudo_label(model, self.weak_augment(deepcopy(tgt_data)), thres)
            total_tgt_logits.append(pseudo_tgt_logits)
            # strong augment
            tgt_data = self.strong_augment(deepcopy(tgt_data))
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)

            # if 'node_mask' in tgt_data:
            #     print("number of nodes:", tgt_data.x.shape[0])
            #     print("pseudo tgt mask shape:", pseudo_tgt_mask.shape)
            #     print("tgt data node mask shape:", tgt_data.node_mask.shape)
            if 'node_mask' in tgt_data:
                pseudo_tgt_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.node_mask)
            con_loss = F.nll_loss(F.log_softmax(tgt_y[pseudo_tgt_mask], dim=1),
                                  pseudo_tgt_label[
                                      pseudo_tgt_mask]) if pseudo_tgt_mask.sum().item() != 0 else torch.tensor(0.)


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

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader)) if self.src_val_loader else len(tgt_val_loader)
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
                if 'mask' in src_data: # e.g. elliptic (not all nodes are labeled)
                    src_mask = src_data.mask
                else:
                    src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask])
                total_src_logits.append(src_y)
            else:
                src_loss = torch.tensor(0.0)
                total_src_logits.append(torch.tensor([[]]))

            tgt_data = tgt_iter.next().to(self.device)
            # weak augment
            pseudo_tgt_label, pseudo_tgt_mask, pseudo_tgt_logits = self._pseudo_label(model, self.weak_augment(deepcopy(tgt_data)), thres)
            total_tgt_logits.append(pseudo_tgt_logits)

            # strong augment
            tgt_data = self.strong_augment(deepcopy(tgt_data))
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            if 'node_mask' in tgt_data:
                pseudo_tgt_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.node_mask)
            con_loss = F.nll_loss(F.log_softmax(tgt_y[pseudo_tgt_mask], dim=1),
                                  pseudo_tgt_label[
                                      pseudo_tgt_mask]) if pseudo_tgt_mask.sum().item() != 0 else torch.tensor(0.)

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
            self.writer.add_scalar("Source Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Loss/val", val_src_loss, e)
            self.writer.add_scalar("Target Loss/train", train_con_loss, e)
            self.writer.add_scalar("Target Loss/val", val_con_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            # self.writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score



    def adapt(self, tgt_train_loader, tgt_val_loader, threshold_list, con_tradeoff_list, stage_name, args, subdir_name=""):
        performance_dict = collections.defaultdict(dict)
        for thres in threshold_list:
            for con_tradeoff in con_tradeoff_list:
                lp = "lp" if args.label_prop else ""
                run_name = f'{args.method}_{lp}_{str(thres)}_{str(con_tradeoff)}_{str(args.model_seed)}'

                self.writer = SummaryWriter(
                    os.path.join(args.log_dir, subdir_name, stage_name, run_name))
                model, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, thres, con_tradeoff, args)
                performance_dict[thres][con_tradeoff] = {'model': model, 'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for thres, thres_dict in performance_dict.items():
            for con_tradeoff, ckpt_dict in thres_dict.items():
                if ckpt_dict['val_score'] > best_val_score:
                    best_val_score = ckpt_dict['val_score']
                    best_model = ckpt_dict['model']
                print(f"thres: {thres} consistency tradeoff: {con_tradeoff} val_score: {ckpt_dict['val_score']}")
        self.set_model(best_model)


    def _pseudo_label(self, model, data, thres):
        model.eval()
        pseudo_y_logits, _ = model(data.x, data.edge_index)
        pseudo_y = F.softmax(pseudo_y_logits, dim=1)

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
        return pseudo_y_hard_label, pseudo_mask, pseudo_y_logits



class SinglegraphFixMatchAdapter(FullModelSinglegraphAdapter):
    def __init__(self, model, src_data=None, device="cpu", propagate=False, weak_p=0.1, strong_p=0.2):
        super().__init__(model, src_data, device)
        self.validator = IMValidator()
        self.propagate = propagate
        if self.propagate:
            print("Use Label Propagation As Consistency Regularization")
            self.prop = LabelPropagation(50, 0.6)
        self.weak_augment = WeakAugmentor(dropedge_p=weak_p, dropfeat_p=weak_p)
        # self.strong_augment = WeakAugmentor(dropedge_p=strong_p, dropfeat_p=strong_p)
        self.strong_augment = StrongAugmentor(dropnode_p=strong_p, dropedge_p=strong_p, dropfeat_p=strong_p, addedge_p=strong_p)

    def _adapt_train_epoch(self, model, tgt_data, optimizer, thres, con_tradeoff, scheduler=None):
        model.train()

        if self.src_data:
            src_y, _ = model(self.src_data.x, self.src_data.edge_index)
            src_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.train_mask], dim=1), self.src_data.y[self.src_data.train_mask])
            src_logits = src_y[self.src_data.train_mask]
        else:
            src_loss = torch.tensor(0.0)
            src_logits = torch.tensor([[]])

        # weak augment
        pseudo_tgt_label, pseudo_tgt_mask, pseudo_tgt_logits = self._pseudo_label(model, self.weak_augment(deepcopy(tgt_data)), thres)
        # strong augment
        tgt_data = self.strong_augment(deepcopy(tgt_data))
        tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
        pseudo_tgt_train_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.train_mask)
        if 'node_mask' in tgt_data:
            pseudo_tgt_train_mask = torch.logical_and(pseudo_tgt_train_mask, tgt_data.node_mask)

        con_loss = F.nll_loss(F.log_softmax(tgt_y[pseudo_tgt_train_mask], dim=1),
                              pseudo_tgt_label[pseudo_tgt_train_mask]) if pseudo_tgt_train_mask.sum().item() != 0 else torch.tensor(0.)


        loss = src_loss + con_loss * con_tradeoff
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()


        return loss.item(), src_loss.item(), con_loss.item(), src_logits, pseudo_tgt_logits[tgt_data.train_mask]

    @torch.no_grad()
    def _adapt_test_epoch(self, model, tgt_data, thres, con_tradeoff):
        model.eval()

        if self.src_data:
            src_y, _ = model(self.src_data.x, self.src_data.edge_index)
            src_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.val_mask], dim=1),
                                  self.src_data.y[self.src_data.val_mask])
            src_logits = src_y[self.src_data.val_mask]
        else:
            src_loss = torch.tensor(0.0)
            src_logits = torch.tensor([[]])

        # weak augment
        pseudo_tgt_label, pseudo_tgt_mask, pseudo_tgt_logits = self._pseudo_label(model,
                                                                                  self.weak_augment(deepcopy(tgt_data)),
                                                                                  thres)
        # strong augment
        tgt_data = self.strong_augment(deepcopy(tgt_data))
        tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
        tgt_logits = tgt_y[tgt_data.val_mask]
        pseudo_tgt_val_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.val_mask)
        if 'node_mask' in tgt_data:
            pseudo_tgt_val_mask = torch.logical_and(pseudo_tgt_val_mask, tgt_data.node_mask)

        con_loss = F.nll_loss(F.log_softmax(tgt_y[pseudo_tgt_val_mask], dim=1),
                              pseudo_tgt_label[
                                  pseudo_tgt_val_mask]) if pseudo_tgt_val_mask.sum().item() != 0 else torch.tensor(
            0.)

        loss = src_loss + con_loss * con_tradeoff

        return loss.item(), src_loss.item(), con_loss.item(), src_logits, pseudo_tgt_logits[tgt_data.val_mask]



    def _adapt_train_test(self, tgt_data, thres, con_tradeoff, args):
        model = deepcopy(self.model)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)
        # optimizer = torch.optim.SGD(list(model.parameters()), lr=args.adapt_lr)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.adapt_epochs)

        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 10
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_con_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                model, tgt_data, optimizer, thres, con_tradeoff)
            val_loss, val_src_loss, val_con_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                model,
                tgt_data, thres, con_tradeoff)
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
            self.writer.add_scalar("Source Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Loss/val", val_src_loss, e)
            self.writer.add_scalar("Target Loss/train", train_con_loss, e)
            self.writer.add_scalar("Target Loss/val", val_con_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            # self.writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score



    def adapt(self, tgt_data, threshold_list, con_tradeoff_list, stage_name, args, subdir_name=""):
        tgt_data = tgt_data.to(self.device)
        performance_dict = collections.defaultdict(dict)
        for thres in threshold_list:
            for con_tradeoff in con_tradeoff_list:
                lp = "lp" if args.label_prop else ""
                run_name = f'{args.method}_{lp}_{str(thres)}_{str(con_tradeoff)}_{str(args.model_seed)}'

                self.writer = SummaryWriter(os.path.join(args.log_dir, subdir_name, stage_name, run_name))
                model, val_score = self._adapt_train_test(tgt_data, thres, con_tradeoff, args)
                performance_dict[thres][con_tradeoff] = {'model': model, 'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for thres, thres_dict in performance_dict.items():
            for con_tradeoff, ckpt_dict in thres_dict.items():
                if ckpt_dict['val_score'] > best_val_score:
                    best_val_score = ckpt_dict['val_score']
                    best_model = ckpt_dict['model']
                print(f"thres: {thres} consistency tradeoff: {con_tradeoff} val_score: {ckpt_dict['val_score']}")
        self.set_model(best_model)


    def _pseudo_label(self, model, data, thres):
        model.eval()
        pseudo_y_logits, _ = model(data.x, data.edge_index)
        pseudo_y = F.softmax(pseudo_y_logits, dim=1)

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
        return pseudo_y_hard_label, pseudo_mask, pseudo_y_logits
