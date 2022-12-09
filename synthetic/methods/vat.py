import os
from copy import deepcopy
import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
# from torch_sparse import SparseTensor
from pytorch_adapt.validators import BNMValidator, IMValidator


class VirtualAdversarialTrainer():
    def __init__(self, model, src_train_loader=None, src_val_loader=None, device="cpu"):
        self.device = device
        self.set_model(model)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        # self.validator = BNMValidator()
        self.validator = IMValidator()

    def _adapt_train_epoch(self, model, tgt_train_loader, optimizer, xi=1e-6, eps=1.0, ip=1):
        model.train()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader)) if self.src_train_loader else len(tgt_train_loader)
        src_iter = iter(self.src_train_loader) if self.src_train_loader else None
        tgt_iter = iter(tgt_train_loader)
        total_src_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_lds = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            vat_loss = VATLoss(xi=xi, eps=eps, ip=ip)
            if self.src_train_loader:
                src_data = src_iter.next().to(self.device)
                src_node_num = src_data.x.shape[0]
                src_lds = vat_loss(model, src_data) * src_node_num
                src_y, _ = model(src_data.x, src_data.edge_index)
                src_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_data.y, reduction='sum')
                total_src_logits.append(src_y)
            else:
                src_lds = torch.tensor(0.0)
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))

            tgt_data = tgt_iter.next().to(self.device)
            tgt_node_num = tgt_data.x.shape[0]
            tgt_lds = vat_loss(model, tgt_data) * tgt_node_num
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            total_tgt_logits.append(tgt_y)

            loss = src_loss + src_lds + tgt_lds

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_src_loss += src_loss.item()
            total_lds += (src_lds + tgt_lds).item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num

        total_loss = (total_src_loss + total_lds) / (total_src_node + total_tgt_node)  if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_lds = total_lds / (total_src_node + total_tgt_node)  if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_lds, total_src_logits, total_tgt_logits


    def _adapt_test_epoch(self, model, tgt_val_loader, xi=1e-6, eps=1.0, ip=1):
        model.eval()

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader)) if self.src_val_loader else len(tgt_val_loader)
        src_iter = iter(self.src_val_loader) if self.src_val_loader else None
        tgt_iter = iter(tgt_val_loader)

        total_src_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_lds = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            vat_loss = VATLoss(xi=xi, eps=eps, ip=ip)
            if self.src_val_loader:
                src_data = src_iter.next().to(self.device)
                src_node_num = src_data.x.shape[0]
                src_lds = vat_loss(model, src_data) * src_node_num
                src_y, _ = model(src_data.x, src_data.edge_index)
                src_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_data.y, reduction='sum')
                total_src_logits.append(src_y)
            else:
                src_lds = torch.tensor(0.0)
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))

            tgt_data = tgt_iter.next().to(self.device)
            tgt_node_num = tgt_data.x.shape[0]
            tgt_lds = vat_loss(model, tgt_data) * tgt_node_num
            tgt_y, _ = model(tgt_data.x, tgt_data.edge_index)
            total_tgt_logits.append(tgt_y)

            total_src_loss += src_loss.item()
            total_lds += (src_lds + tgt_lds).item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num

        total_loss = (total_src_loss + total_lds) / (total_src_node + total_tgt_node) if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_lds = total_lds / (total_src_node + total_tgt_node)  if (total_src_node + total_tgt_node) > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_lds, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, eps, args):
        model = deepcopy(self.model)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)


        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 10
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_lds, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                model, tgt_train_loader, optimizer, eps=eps)
            val_loss, val_src_loss, val_lds, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                model,
                tgt_val_loader, eps=eps)
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
                f'Eps: {eps} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train LDS: {round(train_lds, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val LDS: {round(val_lds, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Label Loss/val", val_src_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("LDS/train", train_lds, e)
            self.writer.add_scalar("LDS/val", val_lds, e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score


    def adapt(self, tgt_train_loader, tgt_val_loader, eps_list, stage, args):
        performance_dict = dict()
        for eps in eps_list:
            run_name = f'{args.method}_{str(eps)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, args.shift, str(stage[0]) + "_" + str(stage[1]) + "_" + str(stage[2]),
                             run_name))
            model, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, eps, args)
            performance_dict[eps] = {'tgt_model': model,
                                            'tgt_val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for eps, perf_dict in performance_dict.items():
            if perf_dict['tgt_val_score'] > best_val_score:
                best_val_score = perf_dict['tgt_val_score']
                best_model = perf_dict['tgt_model']
            print(f"eps: {eps} val_score: {perf_dict['tgt_val_score']}")
        self.set_model(best_model)

    def set_model(self, model):
        self.model = deepcopy(model).to(self.device)

    def get_model(self):
        return self.model


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

class VATLoss(nn.Module):
    def __init__(self, xi=1e-6, eps=1.0, ip=1):
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, data):
        with torch.no_grad():
            pred, _ = model(data.x, data.edge_index)
            pred = F.softmax(pred)
        # prepare random unit tensor
        d = torch.rand(data.x.shape).sub(0.5).to(data.x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat, _ = model(data.x + self.xi * d, data.edge_index)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat, _ = model(data.x + r_adv, data.edge_index)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')
        return lds
