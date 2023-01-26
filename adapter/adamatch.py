import os
from copy import deepcopy
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.models import LabelPropagation
from pytorch_adapt.validators import IMValidator
from .adapter import FullModelMultigraphAdapter, FullModelSinglegraphAdapter
from .augment import WeakAugmentor, StrongAugmentor

class MultigraphAdaMatchAdapter(FullModelMultigraphAdapter):
    def __init__(self, model, src_train_loader, src_val_loader, device="cpu", propagate=False, weak_p=0.1, strong_p=0.2):
        super().__init__(model, src_train_loader, src_val_loader, device)
        self.validator = IMValidator()
        self.propagate = propagate
        if self.propagate:
            print("Use Label Propagation As Consistency Regularization")
            self.prop = LabelPropagation(50, 0.6)
        self.weak_augment = WeakAugmentor(dropedge_p=weak_p, dropfeat_p=weak_p)
        self.strong_augment = StrongAugmentor(dropnode_p=strong_p, dropedge_p=strong_p, dropfeat_p=strong_p, addedge_p=strong_p)

    def _adapt_train_epoch(self, model, tgt_train_loader, optimizer, tau, mu):
        model.train()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader)) if self.src_train_loader else len(
            tgt_train_loader)
        src_iter = iter(self.src_train_loader) if self.src_train_loader else None
        tgt_iter = iter(tgt_train_loader)

        total_loss = 0
        total_src_loss = 0
        total_tgt_loss = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            # augment graphs
            src_data = src_iter.next().to(self.device)
            src_data_w = self.weak_augment(deepcopy(src_data))
            src_data_s = self.strong_augment(deepcopy(src_data))
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                src_mask = src_data.mask
            else:
                src_mask = torch.ones(src_data.y.shape[0], dtype=torch.bool).to(self.device)

            tgt_data = tgt_iter.next().to(self.device)
            tgt_data_w = self.weak_augment(deepcopy(tgt_data))
            tgt_data_s = self.strong_augment(deepcopy(tgt_data))

            # pass src weak, src strong, tgt weak, and tgt strong graphs through model (batch statistics updated)
            logits_combined_src_w, _ = model(src_data_w.x, src_data_w.edge_index)  # weak augmentor doesn't drop nodes
            logits_combined_src_s, _ = model(src_data_s.x, src_data_s.edge_index)

            logits_combined_src_w = logits_combined_src_w[src_mask]
            if 'node_mask' in src_data_s:
                logits_combined_src_s = logits_combined_src_s[torch.logical_and(src_data_s.node_mask, src_mask)]
            else:
                logits_combined_src_s = logits_combined_src_s[src_mask]
            logits_combined_tgt_w, _ = model(tgt_data_w.x, tgt_data_w.edge_index)
            total_tgt_logits.append(logits_combined_tgt_w)
            logits_combined_tgt_s, _ = model(tgt_data_s.x, tgt_data_s.edge_index)
            if 'node_mask' in tgt_data_s:
                logits_combined_tgt_s = logits_combined_tgt_s[tgt_data_s.node_mask]
            logits_src_p = torch.cat([logits_combined_src_w, logits_combined_src_s], 0)
            # pass src weak and src strong graphs through model to get src weak and src strong logits (batch statistics not updated)
            self._disable_batchnorm_tracking(model)
            logits_src_w, _ = model(src_data_w.x, src_data_w.edge_index)
            logits_src_s, _ = model(src_data_s.x, src_data_s.edge_index)

            logits_src_w = logits_src_w[src_mask]
            if 'node_mask' in src_data_s:
                logits_src_s = logits_src_s[torch.logical_and(src_data_s.node_mask, src_mask)]
            else:
                logits_src_s = logits_src_s[src_mask]
            logits_src_pp = torch.cat([logits_src_w, logits_src_s], 0)

            self._enable_batchnorm_tracking(model)

            # random logit interpolation
            lambd = torch.rand_like(logits_src_p).to(self.device)
            final_logits_src = (lambd * logits_src_p) + ((1 - lambd) * logits_src_pp)
            total_src_logits.append(final_logits_src[:logits_src_w.size(0)])
            # distribution alignment
            ## softmax for logits of weakly augmented source nodes/edges
            prob_src = F.softmax(final_logits_src[:logits_src_w.size(0)], 1)

            ## softmax for logits of weakly augmented target nodes/edges
            prob_tgt = F.softmax(logits_combined_tgt_w, 1)
            ## align target label distribtion to source label distribution
            expectation_ratio = (1e-6 + torch.mean(prob_src, dim=0)) / (1e-6 + torch.mean(prob_tgt, dim=0))
            final_prob = F.normalize((prob_tgt * expectation_ratio), p=1,
                                     dim=1)  # L1 normalization s.t. final prob is still a distribution
            # relative confidence thresholding
            row_wise_max, _ = torch.max(prob_src, dim=1)
            final_sum = torch.mean(row_wise_max, dim=0)  # average confidence of top-1 predictions
            c_tau = tau * final_sum
            max_values, _ = torch.max(final_prob, dim=1)
            mask = (max_values >= c_tau).float()

            # compute src loss
            if 'node_mask' in src_data_s:
                src_labels_s = src_data_s.y[torch.logical_and(src_data_s.node_mask, src_mask)]
            else:
                src_labels_s = src_data_s.y[src_mask]

            src_loss = self._compute_source_loss(final_logits_src[:logits_src_w.size(0)],
                                                 final_logits_src[logits_src_w.size(0):], src_data_w.y[src_mask], src_labels_s)

            # compute tgt loss
            final_pseudolabels = torch.max(final_prob, 1)[1]  # argmax
            if 'node_mask' in tgt_data_s:
                final_pseudolabels = final_pseudolabels[tgt_data_s.node_mask]
                mask = mask[tgt_data_s.node_mask]
            tgt_loss = self._compute_target_loss(final_pseudolabels, logits_combined_tgt_s, mask)

            loss = src_loss + mu * tgt_loss

            # backpropagate and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()

        total_loss /= len_dataloader
        total_src_loss /= len_dataloader
        total_tgt_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_src_logits, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, model, tgt_val_loader, tau, mu):
        model.eval()

        len_dataloader = min(len(self.src_train_loader), len(tgt_val_loader)) if self.src_train_loader else len(tgt_val_loader)
        src_iter = iter(self.src_train_loader) if self.src_train_loader else None
        tgt_iter = iter(tgt_val_loader)

        total_loss = 0
        total_src_loss = 0
        total_tgt_loss = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            # augment graphs
            src_data = src_iter.next().to(self.device)
            src_data_w = self.weak_augment(deepcopy(src_data))
            src_data_s = self.strong_augment(deepcopy(src_data))
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                src_mask = src_data.mask
            else:
                src_mask = torch.ones(src_data.y.shape[0], dtype=torch.bool).to(self.device)

            tgt_data = tgt_iter.next().to(self.device)
            tgt_data_w = self.weak_augment(deepcopy(tgt_data))
            tgt_data_s = self.strong_augment(deepcopy(tgt_data))

            # pass src weak, src strong, tgt weak, and tgt strong graphs through model (batch statistics updated)
            logits_combined_src_w, _ = model(src_data_w.x, src_data_w.edge_index)  # weak augmentor doesn't drop nodes
            logits_combined_src_s, _ = model(src_data_s.x, src_data_s.edge_index)

            logits_combined_src_w = logits_combined_src_w[src_mask]
            if 'node_mask' in src_data_s:
                logits_combined_src_s = logits_combined_src_s[torch.logical_and(src_data_s.node_mask, src_mask)]
            else:
                logits_combined_src_s = logits_combined_src_s[src_mask]

            logits_combined_tgt_w, _ = model(tgt_data_w.x, tgt_data_w.edge_index)
            total_tgt_logits.append(logits_combined_tgt_w)
            logits_combined_tgt_s, _ = model(tgt_data_s.x, tgt_data_s.edge_index)
            if 'node_mask' in tgt_data_s:
                logits_combined_tgt_s = logits_combined_tgt_s[tgt_data_s.node_mask]
            logits_src_p = torch.cat([logits_combined_src_w, logits_combined_src_s], 0)
            # pass src weak and src strong graphs through model to get src weak and src strong logits (batch statistics not updated)
            self._disable_batchnorm_tracking(model)
            logits_src_w, _ = model(src_data_w.x, src_data_w.edge_index)
            logits_src_s, _ = model(src_data_s.x, src_data_s.edge_index)

            logits_src_w = logits_src_w[src_mask]
            if 'node_mask' in src_data_s:
                logits_src_s = logits_src_s[torch.logical_and(src_data_s.node_mask, src_mask)]
            else:
                logits_src_s = logits_src_s[src_mask]

            logits_src_pp = torch.cat([logits_src_w, logits_src_s], 0)

            self._enable_batchnorm_tracking(model)

            # random logit interpolation
            lambd = torch.rand_like(logits_src_p).to(self.device)
            final_logits_src = (lambd * logits_src_p) + ((1 - lambd) * logits_src_pp)
            total_src_logits.append(final_logits_src[:logits_src_w.size(0)])
            # distribution alignment
            ## softmax for logits of weakly augmented source nodes/edges
            prob_src = F.softmax(final_logits_src[:logits_src_w.size(0)], 1)

            ## softmax for logits of weakly augmented target nodes/edges
            prob_tgt = F.softmax(logits_combined_tgt_w, 1)
            ## align target label distribtion to source label distribution
            expectation_ratio = (1e-6 + torch.mean(prob_src, dim=0)) / (1e-6 + torch.mean(prob_tgt, dim=0))
            final_prob = F.normalize((prob_tgt * expectation_ratio), p=1,
                                     dim=1)  # L1 normalization s.t. final prob is still a distribution
            # relative confidence thresholding
            row_wise_max, _ = torch.max(prob_src, dim=1)
            final_sum = torch.mean(row_wise_max, dim=0)  # average confidence of top-1 predictions
            c_tau = tau * final_sum
            max_values, _ = torch.max(final_prob, dim=1)
            mask = (max_values >= c_tau).float()

            # compute src loss
            if 'node_mask' in src_data_s:
                src_labels_s = src_data_s.y[torch.logical_and(src_data_s.node_mask, src_mask)]
            else:
                src_labels_s = src_data_s.y[src_mask]

            src_loss = self._compute_source_loss(final_logits_src[:logits_src_w.size(0)],
                                                 final_logits_src[logits_src_w.size(0):], src_data_w.y[src_mask],
                                                 src_labels_s)

            # compute tgt loss
            final_pseudolabels = torch.max(final_prob, 1)[1]  # argmax
            if 'node_mask' in tgt_data_s:
                final_pseudolabels = final_pseudolabels[tgt_data_s.node_mask]
                mask = mask[tgt_data_s.node_mask]
            tgt_loss = self._compute_target_loss(final_pseudolabels, logits_combined_tgt_s, mask)

            loss = src_loss + mu * tgt_loss

            total_loss += loss.item()
            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()

        total_loss /= len_dataloader
        total_src_loss /= len_dataloader
        total_tgt_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, tau, mu, args):
        model = deepcopy(self.model)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)

        # train with pseudo labels
        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 20
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_tgt_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                model, tgt_train_loader, optimizer, tau, mu)
            val_loss, val_src_loss, val_tgt_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(model,
                                                                                                          tgt_val_loader,
                                                                                                          tau, mu)
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
                f'Tau: {tau} Mu: {mu} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Loss/val", val_src_loss, e)
            self.writer.add_scalar("Target Loss/train", train_tgt_loss, e)
            self.writer.add_scalar("Target Loss/val", val_tgt_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score

    def adapt(self, tgt_train_loader, tgt_val_loader, tau_list, mu_list, stage_name, args, subdir_name=""):
        performance_dict = collections.defaultdict(dict)
        for tau in tau_list:
            for mu in mu_list:
                lp = "lp" if args.label_prop else ""
                run_name = f'{args.method}_{lp}_{str(tau)}_{str(mu)}_{str(args.model_seed)}'

                self.writer = SummaryWriter(
                    os.path.join(args.log_dir, subdir_name, stage_name, run_name))
                model, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, tau, mu, args)
                performance_dict[tau][mu] = {'model': model, 'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for tau, tau_dict in performance_dict.items():
            for mu, ckpt_dict in tau_dict.items():
                if ckpt_dict['val_score'] > best_val_score:
                    best_val_score = ckpt_dict['val_score']
                    best_model = ckpt_dict['model']
                print(f"Tau: {tau} Mu: {mu} val_score: {ckpt_dict['val_score']}")
        self.set_model(best_model)

    @staticmethod
    def _compute_source_loss(logits_weak, logits_strong, labels_weak, labels_strong):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        loss_function = nn.CrossEntropyLoss()  # default: `reduction="mean"`
        weak_loss = loss_function(logits_weak, labels_weak)
        strong_loss = loss_function(logits_strong, labels_strong)

        # return weak_loss + strong_loss
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach()  # remove from backpropagation

        loss = loss_function(logits_strong, pseudolabels)

        return (loss * mask).mean()

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)


class SinglegraphAdaMatchAdapter(FullModelSinglegraphAdapter):
    def __init__(self, model, src_data, device="cpu", propagate=False, weak_p=0.1, strong_p=0.2):
        super().__init__(model, src_data, device)
        self.validator = IMValidator()
        self.propagate = propagate
        if self.propagate:
            print("Use Label Propagation As Consistency Regularization")
            self.prop = LabelPropagation(50, 0.6)
        self.weak_augment = WeakAugmentor(dropedge_p=weak_p, dropfeat_p=weak_p)
        self.strong_augment = StrongAugmentor(dropnode_p=strong_p, dropedge_p=strong_p, dropfeat_p=strong_p, addedge_p=strong_p)

    def _adapt_train_epoch(self, model, tgt_data, optimizer, tau, mu):
        model.train()


        # augment graphs
        src_data_w = self.weak_augment(deepcopy(self.src_data))
        src_data_s = self.strong_augment(deepcopy(self.src_data))

        tgt_data_w = self.weak_augment(deepcopy(tgt_data))
        tgt_data_s = self.strong_augment(deepcopy(tgt_data))

        # pass src weak, src strong, tgt weak, and tgt strong graphs through model (batch statistics updated)
        logits_combined_src_w, _ = model(src_data_w.x, src_data_w.edge_index)  # weak augmentor doesn't drop nodes
        logits_combined_src_s, _ = model(src_data_s.x, src_data_s.edge_index)

        logits_combined_src_w = logits_combined_src_w[self.src_data.train_mask]
        if 'node_mask' in src_data_s:
            logits_combined_src_s = logits_combined_src_s[torch.logical_and(src_data_s.node_mask, self.src_data.train_mask)]
        else:
            logits_combined_src_s = logits_combined_src_s[self.src_data.train_mask]
        logits_combined_tgt_w, _ = model(tgt_data_w.x, tgt_data_w.edge_index)
        tgt_logits = logits_combined_tgt_w[tgt_data.train_mask]
        logits_combined_tgt_s, _ = model(tgt_data_s.x, tgt_data_s.edge_index)
        if 'node_mask' in tgt_data_s:
            logits_combined_tgt_s = logits_combined_tgt_s[torch.logical_and(tgt_data_s.node_mask, tgt_data_s.train_mask)]
        logits_src_p = torch.cat([logits_combined_src_w, logits_combined_src_s], 0)
        # pass src weak and src strong graphs through model to get src weak and src strong logits (batch statistics not updated)
        self._disable_batchnorm_tracking(model)
        logits_src_w, _ = model(src_data_w.x, src_data_w.edge_index)
        logits_src_s, _ = model(src_data_s.x, src_data_s.edge_index)

        logits_src_w = logits_src_w[self.src_data.train_mask]
        if 'node_mask' in src_data_s:
            logits_src_s = logits_src_s[torch.logical_and(src_data_s.node_mask, self.src_data.train_mask)]
        else:
            logits_src_s = logits_src_s[self.src_data.train_mask]
        logits_src_pp = torch.cat([logits_src_w, logits_src_s], 0)

        self._enable_batchnorm_tracking(model)

        # random logit interpolation
        lambd = torch.rand_like(logits_src_p).to(self.device)
        final_logits_src = (lambd * logits_src_p) + ((1 - lambd) * logits_src_pp)
        src_logits = final_logits_src[:logits_src_w.size(0)]
        # distribution alignment
        ## softmax for logits of weakly augmented source nodes/edges
        prob_src = F.softmax(final_logits_src[:logits_src_w.size(0)], 1)

        ## softmax for logits of weakly augmented target nodes/edges
        prob_tgt = F.softmax(logits_combined_tgt_w, 1)
        ## align target label distribtion to source label distribution
        expectation_ratio = (1e-6 + torch.mean(prob_src, dim=0)) / (1e-6 + torch.mean(prob_tgt, dim=0))
        final_prob = F.normalize((prob_tgt * expectation_ratio), p=1,
                                 dim=1)  # L1 normalization s.t. final prob is still a distribution
        # relative confidence thresholding
        row_wise_max, _ = torch.max(prob_src, dim=1)
        final_sum = torch.mean(row_wise_max, dim=0)  # average confidence of top-1 predictions
        c_tau = tau * final_sum
        max_values, _ = torch.max(final_prob, dim=1)
        mask = (max_values >= c_tau).float()

        # compute src loss
        if 'node_mask' in src_data_s:
            src_labels_s = src_data_s.y[torch.logical_and(src_data_s.node_mask, self.src_data.train_mask)]
        else:
            src_labels_s = src_data_s.y[self.src_data.train_mask]

        src_loss = self._compute_source_loss(final_logits_src[:logits_src_w.size(0)],
                                             final_logits_src[logits_src_w.size(0):], src_data_w.y[self.src_data.train_mask], src_labels_s)

        # compute tgt loss
        final_pseudolabels = torch.max(final_prob, 1)[1]  # argmax
        if 'node_mask' in tgt_data_s:
            final_pseudolabels = final_pseudolabels[torch.logical_and(tgt_data_s.node_mask, tgt_data.train_mask)]
            mask = mask[torch.logical_and(tgt_data_s.node_mask, tgt_data.train_mask)]
        tgt_loss = self._compute_target_loss(final_pseudolabels, logits_combined_tgt_s, mask)

        loss = src_loss + mu * tgt_loss

        # backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), src_loss.item(), tgt_loss.item(), src_logits, tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, model, tgt_data, tau, mu):
        model.eval()

        # augment graphs
        src_data_w = self.weak_augment(deepcopy(self.src_data))
        src_data_s = self.strong_augment(deepcopy(self.src_data))

        tgt_data_w = self.weak_augment(deepcopy(tgt_data))
        tgt_data_s = self.strong_augment(deepcopy(tgt_data))

        # pass src weak, src strong, tgt weak, and tgt strong graphs through model (batch statistics updated)
        logits_combined_src_w, _ = model(src_data_w.x, src_data_w.edge_index)  # weak augmentor doesn't drop nodes
        logits_combined_src_s, _ = model(src_data_s.x, src_data_s.edge_index)

        logits_combined_src_w = logits_combined_src_w[self.src_data.val_mask]
        if 'node_mask' in src_data_s:
            logits_combined_src_s = logits_combined_src_s[
                torch.logical_and(src_data_s.node_mask, self.src_data.val_mask)]
        else:
            logits_combined_src_s = logits_combined_src_s[self.src_data.val_mask]
        logits_combined_tgt_w, _ = model(tgt_data_w.x, tgt_data_w.edge_index)
        tgt_logits = logits_combined_tgt_w[tgt_data.val_mask]
        logits_combined_tgt_s, _ = model(tgt_data_s.x, tgt_data_s.edge_index)
        if 'node_mask' in tgt_data_s:
            logits_combined_tgt_s = logits_combined_tgt_s[
                torch.logical_and(tgt_data_s.node_mask, tgt_data_s.val_mask)]
        logits_src_p = torch.cat([logits_combined_src_w, logits_combined_src_s], 0)
        # pass src weak and src strong graphs through model to get src weak and src strong logits (batch statistics not updated)
        self._disable_batchnorm_tracking(model)
        logits_src_w, _ = model(src_data_w.x, src_data_w.edge_index)
        logits_src_s, _ = model(src_data_s.x, src_data_s.edge_index)

        logits_src_w = logits_src_w[self.src_data.val_mask]
        if 'node_mask' in src_data_s:
            logits_src_s = logits_src_s[torch.logical_and(src_data_s.node_mask, self.src_data.val_mask)]
        else:
            logits_src_s = logits_src_s[self.src_data.val_mask]
        logits_src_pp = torch.cat([logits_src_w, logits_src_s], 0)

        self._enable_batchnorm_tracking(model)

        # random logit interpolation
        lambd = torch.rand_like(logits_src_p).to(self.device)
        final_logits_src = (lambd * logits_src_p) + ((1 - lambd) * logits_src_pp)
        src_logits = final_logits_src[:logits_src_w.size(0)]
        # distribution alignment
        ## softmax for logits of weakly augmented source nodes/edges
        prob_src = F.softmax(final_logits_src[:logits_src_w.size(0)], 1)

        ## softmax for logits of weakly augmented target nodes/edges
        prob_tgt = F.softmax(logits_combined_tgt_w, 1)
        ## align target label distribtion to source label distribution
        expectation_ratio = (1e-6 + torch.mean(prob_src, dim=0)) / (1e-6 + torch.mean(prob_tgt, dim=0))
        final_prob = F.normalize((prob_tgt * expectation_ratio), p=1,
                                 dim=1)  # L1 normalization s.t. final prob is still a distribution
        # relative confidence thresholding
        row_wise_max, _ = torch.max(prob_src, dim=1)
        final_sum = torch.mean(row_wise_max, dim=0)  # average confidence of top-1 predictions
        c_tau = tau * final_sum
        max_values, _ = torch.max(final_prob, dim=1)
        mask = (max_values >= c_tau).float()

        # compute src loss
        if 'node_mask' in src_data_s:
            src_labels_s = src_data_s.y[torch.logical_and(src_data_s.node_mask, self.src_data.val_mask)]
        else:
            src_labels_s = src_data_s.y[self.src_data.val_mask]

        src_loss = self._compute_source_loss(final_logits_src[:logits_src_w.size(0)],
                                             final_logits_src[logits_src_w.size(0):],
                                             src_data_w.y[self.src_data.val_mask], src_labels_s)

        # compute tgt loss
        final_pseudolabels = torch.max(final_prob, 1)[1]  # argmax
        if 'node_mask' in tgt_data_s:
            final_pseudolabels = final_pseudolabels[torch.logical_and(tgt_data_s.node_mask, tgt_data.val_mask)]
            mask = mask[torch.logical_and(tgt_data_s.node_mask, tgt_data.val_mask)]
        tgt_loss = self._compute_target_loss(final_pseudolabels, logits_combined_tgt_s, mask)

        loss = src_loss + mu * tgt_loss

        return loss.item(), src_loss.item(), tgt_loss.item(), src_logits, tgt_logits

    def _adapt_train_test(self, tgt_data, tau, mu, args):
        model = deepcopy(self.model)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.adapt_lr)

        # train with pseudo labels
        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 20
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_tgt_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                model, tgt_data, optimizer, tau, mu)
            val_loss, val_src_loss, val_tgt_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(model,
                                                                                                          tgt_data,
                                                                                                          tau, mu)
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
                f'Tau: {tau} Mu: {mu} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Loss/val", val_src_loss, e)
            self.writer.add_scalar("Target Loss/train", train_tgt_loss, e)
            self.writer.add_scalar("Target Loss/val", val_tgt_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            if staleness > patience:
                break

        model = deepcopy(best_model)

        return model, best_val_score

    def adapt(self, tgt_data, tau_list, mu_list, stage_name, args, subdir_name=""):
        tgt_data = tgt_data.to(self.device)
        performance_dict = collections.defaultdict(dict)
        for tau in tau_list:
            for mu in mu_list:
                lp = "lp" if args.label_prop else ""
                run_name = f'{args.method}_{lp}_{str(tau)}_{str(mu)}_{str(args.model_seed)}'

                self.writer = SummaryWriter(os.path.join(args.log_dir, subdir_name, stage_name, run_name))
                model, val_score = self._adapt_train_test(tgt_data, tau, mu, args)
                performance_dict[tau][mu] = {'model': model, 'val_score': val_score}

        best_val_score = -np.inf
        best_model = None
        for tau, tau_dict in performance_dict.items():
            for mu, ckpt_dict in tau_dict.items():
                if ckpt_dict['val_score'] > best_val_score:
                    best_val_score = ckpt_dict['val_score']
                    best_model = ckpt_dict['model']
                print(f"Tau: {tau} Mu: {mu} val_score: {ckpt_dict['val_score']}")
        self.set_model(best_model)

    @staticmethod
    def _compute_source_loss(logits_weak, logits_strong, labels_weak, labels_strong):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        loss_function = nn.CrossEntropyLoss()  # default: `reduction="mean"`
        weak_loss = loss_function(logits_weak, labels_weak)
        strong_loss = loss_function(logits_strong, labels_strong)

        # return weak_loss + strong_loss
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach()  # remove from backpropagation

        loss = loss_function(logits_strong, pseudolabels)

        return (loss * mask).mean()

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)