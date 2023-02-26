import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from pytorch_adapt.validators import IMValidator
from .adapter import FullModelMultigraphAdapter, FullModelSinglegraphAdapter


class MultigraphGST(FullModelMultigraphAdapter):
    def __init__(self, model, device="cpu"):
        super().__init__(model, src_train_loader=None, src_val_loader=None, device=device)
        # self.validator = BNMValidator()
        self.validator = IMValidator()

    def _adapt_train_epoch(self, model, tgt_train_loader, optimizer):
        model.train()


        total_tgt_loss = 0
        total_tgt_logits = []
        for tgt_data, pseudo_tgt_label, pseudo_tgt_mask in tgt_train_loader:
            pseudo_tgt_label = torch.squeeze(pseudo_tgt_label, dim=0).to(self.device)
            pseudo_tgt_mask = torch.squeeze(pseudo_tgt_mask, dim=0).to(self.device)
            tgt_data = tgt_data.to(self.device)
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1)[pseudo_tgt_mask],
                                  pseudo_tgt_label[pseudo_tgt_mask], reduction='mean')

            total_tgt_logits.append(tgt_y) # use all node's logits predictions

            optimizer.zero_grad()
            tgt_loss.backward()
            optimizer.step()

            total_tgt_loss += tgt_loss.item()

        total_tgt_loss = total_tgt_loss / len(tgt_train_loader)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_tgt_loss, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, model, tgt_val_loader):
        model.eval()

        total_tgt_loss = 0
        total_tgt_logits = []
        for tgt_data, pseudo_tgt_label, pseudo_tgt_mask in tgt_val_loader:
            pseudo_tgt_label = torch.squeeze(pseudo_tgt_label, dim=0).to(self.device)
            pseudo_tgt_mask = torch.squeeze(pseudo_tgt_mask, dim=0).to(self.device)
            tgt_data = tgt_data.to(self.device)
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1)[pseudo_tgt_mask],
                                  pseudo_tgt_label[pseudo_tgt_mask], reduction='mean')

            total_tgt_logits.append(tgt_y)  # use all node's logits predictions

            total_tgt_loss += tgt_loss.item()

        total_tgt_loss = total_tgt_loss / len(tgt_val_loader)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_tgt_loss, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, thres, args):
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
        patience = 20
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_tgt_loss, train_tgt_logits = self._adapt_train_epoch(
                model, tgt_pseudo_train_loader, optimizer)
            val_tgt_loss, val_tgt_logits = self._adapt_test_epoch(
                model,
                tgt_pseudo_val_loader)
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            if val_tgt_loss < best_val_loss:
                best_val_loss = val_tgt_loss
                best_val_score = val_tgt_score
                best_model = deepcopy(model)
                staleness = 0
            else:
                staleness += 1
            print(
                f'Thres: {thres} Epoch: {e} Train Tgt Loss: {round(train_tgt_loss, 3)} \n Val Tgt Loss: {round(val_tgt_loss, 3)}')

            self.writer.add_scalar("Target Loss/train", train_tgt_loss, e)
            self.writer.add_scalar("Target Loss/val", val_tgt_loss, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score

    @torch.no_grad()
    def _pseudo_label(self, model, data, thres):
        model.eval()
        pseudo_y, _ = model(data.x, data.edge_index)
        pseudo_y = F.softmax(pseudo_y, dim=1)
        pseudo_y_confidence, pseudo_y_hard_label = torch.max(pseudo_y, dim=1)
        pseudo_mask = pseudo_y_confidence > thres
        return pseudo_y_hard_label, pseudo_mask

    def adapt(self, tgt_train_loader, tgt_val_loader, threshold_list, stage_name, args, subdir_name=""):
        performance_dict = dict()
        for thres in threshold_list:
            lp = "lp" if args.label_prop else ""
            run_name = f'{args.method}_{lp}_{str(thres)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name,
                             run_name))
            model, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, thres, args)
            performance_dict[thres] = {'model': model, 'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for thres, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_model = perf_dict['model']
            print(f"thres: {thres} val_score: {perf_dict['val_score']}")
        self.set_model(best_model)



class SinglegraphGST(FullModelSinglegraphAdapter):
    def __init__(self, model, device="cpu"):
        super().__init__(model, src_data=None, device=device)
        self.validator = IMValidator()

    def _adapt_train_epoch(self, model, tgt_data, pseudo_tgt_label, pseudo_tgt_mask, optimizer):
        model.train()

        pseudo_tgt_train_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.train_mask)
        tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
        tgt_loss = F.nll_loss(F.log_softmax(tgt_y[pseudo_tgt_train_mask], dim=1), pseudo_tgt_label[pseudo_tgt_train_mask], reduction='mean')
        tgt_logits = tgt_y[tgt_data.train_mask]

        optimizer.zero_grad()
        tgt_loss.backward()
        optimizer.step()

        return tgt_loss.item(), tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, model, tgt_data, pseudo_tgt_label, pseudo_tgt_mask):
        model.eval()

        pseudo_tgt_val_mask = torch.logical_and(pseudo_tgt_mask, tgt_data.val_mask)
        tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
        tgt_loss = F.nll_loss(F.log_softmax(tgt_y[pseudo_tgt_val_mask], dim=1),
                              pseudo_tgt_label[pseudo_tgt_val_mask], reduction='mean')
        tgt_logits = tgt_y[tgt_data.val_mask]


        return tgt_loss.item(), tgt_logits

    def _adapt_train_test(self, tgt_data, thres, args):
        model = deepcopy(self.model)
        pseudo_y_hard_label, pseudo_mask = self._pseudo_label(model, tgt_data, thres)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)

        # train with pseudo labels
        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 20
        staleness = 0

        for e in range(1, args.adapt_epochs + 1):
            train_tgt_loss, train_tgt_logits = self._adapt_train_epoch(model, tgt_data, pseudo_y_hard_label, pseudo_mask, optimizer)
            val_tgt_loss, val_tgt_logits = self._adapt_test_epoch(model, tgt_data, pseudo_y_hard_label, pseudo_mask)
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            if val_tgt_loss < best_val_loss:
                best_val_loss = val_tgt_loss
                best_val_score = val_tgt_score
                best_model = deepcopy(model)
                staleness = 0
            else:
                staleness += 1
            print(
                f'Thres: {thres} Epoch: {e} Train Tgt Loss: {round(train_tgt_loss, 3)} \n Val Tgt Loss: {round(val_tgt_loss, 3)}')

            self.writer.add_scalar("Target Loss/train", train_tgt_loss, e)
            self.writer.add_scalar("Target Loss/val", val_tgt_loss, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score


    @torch.no_grad()
    def _pseudo_label(self, model, data, thres):
        model.eval()
        pseudo_y, _ = model(data.x, data.edge_index)
        pseudo_y = F.softmax(pseudo_y, dim=1)
        pseudo_y_confidence, pseudo_y_hard_label = torch.max(pseudo_y, dim=1)
        pseudo_mask = pseudo_y_confidence > thres
        return pseudo_y_hard_label, pseudo_mask

    def adapt(self, tgt_data, threshold_list, stage_name, args, subdir_name=""):
        tgt_data = tgt_data.to(self.device)
        performance_dict = dict()
        for thres in threshold_list:
            lp = "lp" if args.label_prop else ""
            run_name = f'{args.method}_{lp}_{str(thres)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(os.path.join(args.log_dir, subdir_name, stage_name, run_name))
            model, val_score = self._adapt_train_test(tgt_data, thres, args)
            performance_dict[thres] = {'model': model, 'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for thres, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_model = perf_dict['model']
            print(f"thres: {thres} val_score: {perf_dict['val_score']}")
        self.set_model(best_model)
