"""
This EMATeacher implementation is modified from https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/self_training/mean_teacher.py
"""

import os
from copy import deepcopy
import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.typing import Adj
from pytorch_adapt.validators import IMValidator
from .adapter import FullModelMultigraphAdapter, FullModelSinglegraphAdapter


class MultigraphMeanTeacherAdapter(FullModelMultigraphAdapter):
    def __init__(self, model, src_train_loader=None, src_val_loader=None, device="cpu"):
        super().__init__(model, src_train_loader, src_val_loader, device)
        self.validator = IMValidator()
        self.alpha = 0.999
        self.warm_up_epochs = 10

    def _adapt_train_epoch(self, model, teacher, tgt_train_loader, e, global_step, optimizer, con_trade_off):
        model.train()
        teacher.train()
        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader)) if self.src_train_loader else len(
            tgt_train_loader)
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
            # consistency loss
            with torch.no_grad():
                tgt_y_teacher, _ = teacher(tgt_data.x, tgt_data.edge_index)

            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            total_tgt_logits.append(tgt_y)
            con_loss = sigmoid_warm_up(e, self.warm_up_epochs) * F.mse_loss(F.softmax(tgt_y, dim=1), F.softmax(tgt_y_teacher, dim=1))
            loss = src_loss + con_trade_off * con_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update teacher
            teacher.set_alpha(min(self.alpha, 1 - 1 / global_step))
            teacher.update()
            update_bn(model, teacher.teacher)
            # print("Global step:", global_step, "Alpha:", min(self.alpha, 1 - 1 / global_step), "sigmoid warm up:", sigmoid_warm_up(e, self.warm_up_epochs))
            total_loss += loss.item()
            total_src_loss += src_loss.item()
            total_con_loss += con_loss.item()
            global_step += 1

        total_loss /= len_dataloader
        total_src_loss /= len_dataloader
        total_con_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_con_loss, total_src_logits, total_tgt_logits, global_step

    @torch.no_grad()
    def _adapt_test_epoch(self, model, teacher, tgt_val_loader, e, con_trade_off):
        model.eval()
        teacher.eval()

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader)) if self.src_train_loader else len(
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
                if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                    src_mask = src_data.mask
                else:
                    src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask])
                total_src_logits.append(src_y)
            else:
                src_loss = torch.tensor(0.0)
                total_src_logits.append(torch.tensor([[]]))

            tgt_data = tgt_iter.next().to(self.device)
            # consistency loss
            with torch.no_grad():
                tgt_y_teacher, _ = teacher(tgt_data.x, tgt_data.edge_index)

            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            total_tgt_logits.append(tgt_y)
            con_loss = sigmoid_warm_up(e, self.warm_up_epochs) * F.mse_loss(F.softmax(tgt_y, dim=1),
                                                                            F.softmax(tgt_y_teacher, dim=1))
            loss = src_loss + con_trade_off * con_loss

            total_loss += loss.item()
            total_src_loss += src_loss.item()
            total_con_loss += con_loss.item()

        total_loss /= len_dataloader
        total_src_loss /= len_dataloader
        total_con_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_con_loss, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, con_trade_off, args):
        model = deepcopy(self.model)
        teacher = EMATeacher(model, alpha=self.alpha)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)

        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 10
        staleness = 0
        global_step = 1

        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_con_loss, train_src_logits, train_tgt_logits, global_step = self._adapt_train_epoch(
                model, teacher, tgt_train_loader, e, global_step, optimizer, con_trade_off=con_trade_off)
            val_loss, val_src_loss, val_con_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                model, teacher,
                tgt_val_loader, e, con_trade_off=con_trade_off)
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
                f'Consistency Trade Off: {con_trade_off} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Con Loss: {round(train_con_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Con Loss: {round(val_con_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Loss/val", val_src_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("Consistency Loss/train", train_con_loss, e)
            self.writer.add_scalar("Consistency Loss/val", val_con_loss, e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score


    def adapt(self, tgt_train_loader, tgt_val_loader, con_trade_off_list, stage_name, args, subdir_name=""):
        performance_dict = dict()
        for con_trade_off in con_trade_off_list:
            run_name = f'{args.method}_{str(con_trade_off)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name,
                             run_name))
            model, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, con_trade_off, args)
            performance_dict[con_trade_off] = {'model': model, 'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for con_trade_off, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_model = perf_dict['model']
            print(f"Consistency Tradeoff: {con_trade_off} val_score: {perf_dict['val_score']}")
        self.set_model(best_model)



class SinglegraphMeanTeacherAdapter(FullModelSinglegraphAdapter):
    def __init__(self, model, src_data=None, device="cpu"):
        super().__init__(model, src_data, device)
        self.validator = IMValidator()
        self.alpha = 0.999
        self.warm_up_epochs = 10

    def _adapt_train_epoch(self, model, teacher, tgt_data, e, global_step, optimizer, con_trade_off):
        model.train()
        teacher.train()


        if self.src_data:
            src_y, _ = model(self.src_data.x, self.src_data.edge_index)
            src_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.train_mask], dim=1), self.src_data.y[self.src_data.train_mask])
            src_logits = src_y[self.src_data.train_mask]
        else:
            src_loss = torch.tensor(0.0)
            src_logits = torch.tensor([[]])

        # consistency loss
        with torch.no_grad():
            tgt_y_teacher, _ = teacher(tgt_data.x, tgt_data.edge_index)

        tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
        tgt_logits = tgt_y[tgt_data.train_mask]
        con_loss = sigmoid_warm_up(e, self.warm_up_epochs) * F.mse_loss(F.softmax(tgt_y[tgt_data.train_mask], dim=1), F.softmax(tgt_y_teacher[tgt_data.train_mask], dim=1))
        loss = src_loss + con_trade_off * con_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher
        teacher.set_alpha(min(self.alpha, 1 - 1 / global_step))
        teacher.update()
        update_bn(model, teacher.teacher)
        # print("Global step:", global_step, "Alpha:", min(self.alpha, 1 - 1 / global_step), "sigmoid warm up:", sigmoid_warm_up(e, self.warm_up_epochs
        global_step += 1


        return loss.item(), src_loss.item(), con_loss.item(), src_logits, tgt_logits, global_step

    @torch.no_grad()
    def _adapt_test_epoch(self, model, teacher, tgt_data, e, con_trade_off):
        model.eval()
        teacher.eval()


        if self.src_data:
            src_y, _ = model(self.src_data.x, self.src_data.edge_index)
            src_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.val_mask], dim=1), self.src_data.y[self.src_data.val_mask])
            src_logits = src_y[self.src_data.val_mask]
        else:
            src_loss = torch.tensor(0.0)
            src_logits = torch.tensor([[]])

        # consistency loss
        with torch.no_grad():
            tgt_y_teacher, _ = teacher(tgt_data.x, tgt_data.edge_index)

        tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
        tgt_logits = tgt_y[tgt_data.val_mask]
        con_loss = sigmoid_warm_up(e, self.warm_up_epochs) * F.mse_loss(F.softmax(tgt_y[tgt_data.val_mask], dim=1), F.softmax(tgt_y_teacher[tgt_data.val_mask], dim=1))
        loss = src_loss + con_trade_off * con_loss


        return loss.item(), src_loss.item(), con_loss.item(), src_logits, tgt_logits 

    def _adapt_train_test(self, tgt_data, con_trade_off, args):
        model = deepcopy(self.model)
        teacher = EMATeacher(model, alpha=self.alpha)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)

        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 10
        staleness = 0
        global_step = 1

        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_con_loss, train_src_logits, train_tgt_logits, global_step = self._adapt_train_epoch(
                model, teacher, tgt_data, e, global_step, optimizer, con_trade_off=con_trade_off)
            val_loss, val_src_loss, val_con_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                model, teacher,
                tgt_data, e, con_trade_off=con_trade_off)
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
                f'Consistency Trade Off: {con_trade_off} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Con Loss: {round(train_con_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Con Loss: {round(val_con_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Loss/val", val_src_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("Consistency Loss/train", train_con_loss, e)
            self.writer.add_scalar("Consistency Loss/val", val_con_loss, e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score


    def adapt(self, tgt_data, con_trade_off_list, stage_name, args, subdir_name=""):
        tgt_data = tgt_data.to(self.device)
        performance_dict = dict()
        for con_trade_off in con_trade_off_list:
            run_name = f'{args.method}_{str(con_trade_off)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(os.path.join(args.log_dir, subdir_name, stage_name, run_name))
            model, val_score = self._adapt_train_test(tgt_data, con_trade_off, args)
            performance_dict[con_trade_off] = {'model': model, 'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for con_trade_off, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_model = perf_dict['model']
            print(f"Consistency Tradeoff: {con_trade_off} val_score: {perf_dict['val_score']}")
        self.set_model(best_model)








def set_requires_grad(net, requires_grad=False):
    """
    Set requires_grad=False for all the parameters to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad

class EMATeacher(object):
    r"""
    Exponential moving average model from `Mean teachers are better role models: Weight-averaged consistency targets
    improve semi-supervised deep learning results (NIPS 2017) <https://arxiv.org/abs/1703.01780>`_
    """

    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        self.teacher = deepcopy(model)
        set_requires_grad(self.teacher, False)

    def set_alpha(self, alpha: float):
        assert alpha >= 0
        self.alpha = alpha

    def update(self):
        for teacher_param, param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data = self.alpha * teacher_param + (1 - self.alpha) * param

    def __call__(self, x: torch.Tensor, edge_index: Adj):
        return self.teacher(x, edge_index)

    def train(self, mode: Optional[bool] = True):
        self.teacher.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.teacher.state_dict()

    def load_state_dict(self, state_dict):
        self.teacher.load_state_dict(state_dict)

    @property
    def module(self):
        return self.teacher.module


def update_bn(model, ema_model):
    """
    Replace batch normalization statistics of the teacher model with that of the student model
    """
    for m2, m1 in zip(ema_model.named_modules(), model.named_modules()):
        if ('bn' in m2[0]) and ('bn' in m1[0]):
            bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
            bn2['running_mean'].data.copy_(bn1['running_mean'].data)
            bn2['running_var'].data.copy_(bn1['running_var'].data)
            bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)


def sigmoid_warm_up(current_epoch, warm_up_epochs: int):
    """Exponential warm up function from `Temporal Ensembling for Semi-Supervised Learning
    (ICLR 2017) <https://arxiv.org/abs/1610.02242>`_.
    """
    assert warm_up_epochs >= 0
    if warm_up_epochs == 0:
        return 1.0
    else:
        current_epoch = np.clip(current_epoch, 0.0, warm_up_epochs)
        process = 1.0 - current_epoch / warm_up_epochs
        return float(np.exp(-5.0 * process * process))
