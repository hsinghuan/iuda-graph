from copy import deepcopy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import IMValidator
from .adapter import DecoupledMultigraphAdapter, DecoupledSinglegraphAdapter

class MultigraphDeepCORALAdapter(DecoupledMultigraphAdapter):
    def __init__(self, encoder, classifier, src_train_loader, src_val_loader, device="cpu"):
        super().__init__(encoder, classifier, src_train_loader, src_val_loader, device)
        self.coral_loss = CorrelationAlignmentLoss()
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, tgt_train_loader, optimizer, coral_tradeoff):
        encoder.train()
        classifier.train()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        total_loss = 0
        total_cls_loss = 0
        total_transfer_loss = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            src_data = src_iter.next().to(self.device)
            tgt_data = tgt_iter.next().to(self.device)
            src_f = encoder(src_data.x, src_data.edge_index)
            src_y, _ = classifier(src_f)
            tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
            tgt_y, _ = classifier(tgt_f)
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                src_mask = src_data.mask
            else:
                src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
            cls_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask])
            transfer_loss =  self.coral_loss(src_f, tgt_f)
            loss = cls_loss + coral_tradeoff * transfer_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_transfer_loss += transfer_loss.item()
            total_src_logits.append(src_y)
            total_tgt_logits.append(tgt_y)

        total_loss /= len_dataloader
        total_cls_loss /= len_dataloader
        total_transfer_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_cls_loss, total_transfer_loss, total_src_logits, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, tgt_val_loader, coral_tradeoff):
        encoder.eval()
        classifier.eval()

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader))
        src_iter = iter(self.src_val_loader)
        tgt_iter = iter(tgt_val_loader)
        total_loss = 0
        total_cls_loss = 0
        total_transfer_loss = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            src_data = src_iter.next().to(self.device)
            tgt_data = tgt_iter.next().to(self.device)
            src_f = encoder(src_data.x, src_data.edge_index)
            src_y, _ = classifier(src_f)
            tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
            tgt_y, _ = classifier(tgt_f)
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                src_mask = src_data.mask
            else:
                src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
            cls_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask])
            transfer_loss = self.coral_loss(src_f, tgt_f)
            loss = cls_loss + coral_tradeoff * transfer_loss

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_transfer_loss += transfer_loss.item()
            total_src_logits.append(src_y)
            total_tgt_logits.append(tgt_y)

        total_loss /= len_dataloader
        total_cls_loss /= len_dataloader
        total_transfer_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_cls_loss, total_transfer_loss, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, coral_tradeoff, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.adapt_lr)

        best_val_loss = np.inf
        best_val_score = None
        best_encoder, best_classifier = None, None
        patience = 20
        staleness = 0
        for e in range(1, args.adapt_epochs):
            train_loss, train_cls_loss, train_coral_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, tgt_train_loader, optimizer, coral_tradeoff)
            val_loss, val_cls_loss, val_coral_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(encoder,
                                                                                                           classifier,
                                                                                                           tgt_val_loader,
                                                                                                           coral_tradeoff)
            train_src_score = self.validator(target_train={'logits': train_src_logits})
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_src_score = self.validator(target_train={'logits': val_src_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_score = val_tgt_score
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)
                staleness = 0
            else:
                staleness += 1
            print(
                f'Coral Tradeoff {coral_tradeoff} Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_cls_loss, 3)} Train Transfer Loss: {round(train_coral_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Cls Loss: {round(val_cls_loss, 3)} Val Transfer Loss: {round(val_coral_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Loss/train", train_cls_loss, e)
            self.writer.add_scalar("Source Loss/val", val_cls_loss, e)
            self.writer.add_scalar("Coral Loss/train", train_coral_loss, e)
            self.writer.add_scalar("Coral Loss/val", val_coral_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)

            if staleness > patience:
                break

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)
        return encoder, classifier, best_val_score

    def adapt(self, tgt_train_loader, tgt_val_loader, coral_tradeoff_list, stage_name, args, subdir_name=""):
        performance_dict = dict()
        for coral_tradeoff in coral_tradeoff_list:
            run_name = f'{args.method}_{str(coral_tradeoff)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name, run_name))
            encoder, classifier, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, coral_tradeoff, args)
            performance_dict[coral_tradeoff] = {'encoder': encoder, 'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        best_encoder = None
        best_classifier = None
        for coral_tradeoff, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_encoder = perf_dict['encoder']
                best_classifier = perf_dict['classifier']
            print(f"Coral tradeoff: {coral_tradeoff} val_score: {perf_dict['val_score']}")
        self.set_encoder_classifier(best_encoder, best_classifier)



class SinglegraphDeepCORALAdapter(DecoupledSinglegraphAdapter):
    def __init__(self, encoder, classifier, src_data, device="cpu"):
        super().__init__(encoder, classifier, src_data, device)
        self.coral_loss = CorrelationAlignmentLoss()
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, tgt_data, optimizer, coral_tradeoff):
        encoder.train()
        classifier.train()

        src_f = encoder(self.src_data.x, self.src_data.edge_index)
        src_y, _ = classifier(src_f)
        tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
        tgt_y, _ = classifier(tgt_f)
        cls_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.train_mask], dim=1), self.src_data.y[self.src_data.train_mask])
        transfer_loss = self.coral_loss(src_f[self.src_data.train_mask], tgt_f[tgt_data.train_mask])
        loss = cls_loss + coral_tradeoff * transfer_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), cls_loss.item(), transfer_loss.item(), src_y[self.src_data.train_mask], tgt_y[tgt_data.train_mask]

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, tgt_data, coral_tradeoff):
        encoder.eval()
        classifier.eval()

        src_f = encoder(self.src_data.x, self.src_data.edge_index)
        src_y, _ = classifier(src_f)
        tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
        tgt_y, _ = classifier(tgt_f)
        cls_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.val_mask], dim=1),
                              self.src_data.y[self.src_data.val_mask])
        transfer_loss = self.coral_loss(src_f[self.src_data.val_mask], tgt_f[tgt_data.val_mask])
        loss = cls_loss + coral_tradeoff * transfer_loss

        return loss.item(), cls_loss.item(), transfer_loss.item(), src_y[self.src_data.val_mask], tgt_y[tgt_data.val_mask]

    def _adapt_train_test(self, tgt_data, coral_tradeoff, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.adapt_lr)

        best_val_loss = np.inf
        best_val_score = None
        best_encoder, best_classifier = None, None
        patience = 20
        staleness = 0
        for e in range(1, args.adapt_epochs):
            train_loss, train_cls_loss, train_coral_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, tgt_data, optimizer, coral_tradeoff)
            val_loss, val_cls_loss, val_coral_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(encoder,
                                                                                                               classifier,
                                                                                                               tgt_data,
                                                                                                               coral_tradeoff)
            train_src_score = self.validator(target_train={'logits': train_src_logits})
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_src_score = self.validator(target_train={'logits': val_src_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_score = val_tgt_score
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)
                staleness = 0
            else:
                staleness += 1
            print(
                f'Coral Tradeoff {coral_tradeoff} Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_cls_loss, 3)} Train Transfer Loss: {round(train_coral_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Cls Loss: {round(val_cls_loss, 3)} Val Transfer Loss: {round(val_coral_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Loss/train", train_cls_loss, e)
            self.writer.add_scalar("Source Loss/val", val_cls_loss, e)
            self.writer.add_scalar("Coral Loss/train", train_coral_loss, e)
            self.writer.add_scalar("Coral Loss/val", val_coral_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)

            if staleness > patience:
                break

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)
        return encoder, classifier, best_val_score

    def adapt(self, tgt_data, coral_tradeoff_list, stage_name, args, subdir_name=""):
        tgt_data = tgt_data.to(self.device)
        performance_dict = dict()
        for coral_tradeoff in coral_tradeoff_list:
            run_name = f'{args.method}_{str(coral_tradeoff)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name, run_name))
            encoder, classifier, val_score = self._adapt_train_test(tgt_data, coral_tradeoff,
                                                                    args)
            performance_dict[coral_tradeoff] = {'encoder': encoder, 'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        best_encoder = None
        best_classifier = None
        for coral_tradeoff, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_encoder = perf_dict['encoder']
                best_classifier = perf_dict['classifier']
            print(f"Coral tradeoff: {coral_tradeoff} val_score: {perf_dict['val_score']}")
        self.set_encoder_classifier(best_encoder, best_classifier)



class CorrelationAlignmentLoss(nn.Module):
    r"""The `Correlation Alignment Loss` in
    `Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016) <https://arxiv.org/pdf/1607.01719.pdf>`_.
    Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by
    .. math::
        C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S))
    .. math::
        C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T))
    where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
    source and target samples, respectively. We use :math:`d` to denote feature dimension, use
    :math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
    given by
    .. math::
        l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
    Shape:
        - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
        - Outputs: scalar.
    """

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff