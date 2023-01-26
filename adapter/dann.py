from copy import deepcopy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import IMValidator
from .grl import GradientReverseLayer
from .adapter import DecoupledMultigraphAdapter, DecoupledSinglegraphAdapter


class MultigraphDANNAdapter(DecoupledMultigraphAdapter):
    def __init__(self, encoder, classifier, src_train_loader, src_val_loader, emb_dim, device="cpu"):
        super().__init__(encoder, classifier, src_train_loader, src_val_loader, device)
        self.emb_dim = emb_dim
        self.grl = GradientReverseLayer()
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, domain_classifier, tgt_train_loader, optimizer, e, epochs, lamb):
        encoder.train()
        classifier.train()
        domain_classifier.train()
        p = e / epochs
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * lamb
        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        total_loss = 0
        total_loss_src_label = 0
        total_loss_src_domain = 0
        total_loss_tgt_domain = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            src_data = src_iter.next().to(self.device)
            src_size = src_data.x.shape[0]
            src_dom_label = torch.zeros(src_size).long().to(self.device)
            src_f = encoder(src_data.x, src_data.edge_index)
            src_cls_output, _ = classifier(src_f)
            src_dom_output = domain_classifier(self.grl(src_f, alpha))
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                src_mask = src_data.mask
            else:
                src_mask = torch.ones(src_cls_output.shape[0], dtype=torch.bool)
            loss_src_label = F.nll_loss(F.log_softmax(src_cls_output[src_mask], dim=1), src_data.y[src_mask])
            loss_src_domain = F.nll_loss(F.log_softmax(src_dom_output, dim=1), src_dom_label)

            tgt_data = tgt_iter.next().to(self.device)
            tgt_size = tgt_data.x.shape[0]
            tgt_dom_label = torch.ones(tgt_size).long().to(self.device)
            tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
            tgt_dom_output = domain_classifier(self.grl(tgt_f, alpha))
            loss_tgt_domain = F.nll_loss(F.log_softmax(tgt_dom_output, dim=1), tgt_dom_label)

            loss = loss_src_label + loss_src_domain + loss_tgt_domain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tgt_cls_output, _ = classifier(tgt_f)
            total_loss += loss.item()
            total_loss_src_label += loss_src_label.item()
            total_loss_src_domain += loss_src_domain.item()
            total_loss_tgt_domain += loss_tgt_domain.item()
            total_src_logits.append(src_cls_output)
            total_tgt_logits.append(tgt_cls_output)

        total_loss /= len_dataloader
        total_loss_src_label /= len_dataloader
        total_loss_src_domain /= len_dataloader
        total_loss_tgt_domain /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_loss_src_label, total_loss_src_domain, total_loss_tgt_domain, total_src_logits, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, domain_classifier, tgt_val_loader, e, epochs, lamb):
        encoder.eval()
        classifier.eval()
        domain_classifier.eval()
        p = e / epochs
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * lamb
        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader))
        src_iter = iter(self.src_val_loader)
        tgt_iter = iter(tgt_val_loader)
        total_loss = 0
        total_loss_src_label = 0
        total_loss_src_domain = 0
        total_loss_tgt_domain = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            src_data = src_iter.next().to(self.device)
            src_size = src_data.x.shape[0]
            src_dom_label = torch.zeros(src_size).long().to(self.device)
            src_f = encoder(src_data.x, src_data.edge_index)
            src_cls_output, _ = classifier(src_f)
            src_dom_output = domain_classifier(self.grl(src_f, alpha))
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                src_mask = src_data.mask
            else:
                src_mask = torch.ones(src_cls_output.shape[0], dtype=torch.bool)
            loss_src_label = F.nll_loss(F.log_softmax(src_cls_output[src_mask], dim=1), src_data.y[src_mask])
            loss_src_domain = F.nll_loss(F.log_softmax(src_dom_output, dim=1), src_dom_label)

            tgt_data = tgt_iter.next().to(self.device)
            tgt_size = tgt_data.x.shape[0]
            tgt_dom_label = torch.ones(tgt_size).long().to(self.device)
            tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
            tgt_dom_output = domain_classifier(self.grl(tgt_f, alpha))
            loss_tgt_domain = F.nll_loss(F.log_softmax(tgt_dom_output, dim=1), tgt_dom_label)

            loss = loss_src_label + loss_src_domain + loss_tgt_domain

            tgt_cls_output, _ = classifier(tgt_f)
            total_loss += loss.item()
            total_loss_src_label += loss_src_label.item()
            total_loss_src_domain += loss_src_domain.item()
            total_loss_tgt_domain += loss_tgt_domain.item()
            total_src_logits.append(src_cls_output)
            total_tgt_logits.append(tgt_cls_output)

        total_loss /= len_dataloader
        total_loss_src_label /= len_dataloader
        total_loss_src_domain /= len_dataloader
        total_loss_tgt_domain /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_loss_src_label, total_loss_src_domain, total_loss_tgt_domain, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, lamb, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        domain_classifier = DomainClassifier(self.emb_dim, self.emb_dim, 2).to(self.device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()) + list(domain_classifier.parameters()),
            lr=args.adapt_lr)
        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_loss_src_label, train_loss_src_domain, train_loss_tgt_domain, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, domain_classifier, tgt_train_loader, optimizer, e, args.adapt_epochs, lamb)
            val_loss, val_loss_src_label, val_loss_src_domain, val_loss_tgt_domain, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                encoder, classifier, domain_classifier, tgt_val_loader, e, args.adapt_epochs, lamb)
            train_src_score = self.validator(target_train={'logits': train_src_logits})
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_src_score = self.validator(target_train={'logits': val_src_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            # val_score = self.validator(target_train={'logits': torch.cat([val_src_logits, val_tgt_logits])})

            if val_tgt_score > best_val_score:
                # if val_loss_src_label < best_val_loss_src_label:
                best_val_score = val_tgt_score
                # best_val_loss_src_label = val_loss_src_label
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)

            print(
                f'Lambda {lamb} Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_loss_src_label, 3)} Train Src Domain Cls Loss: {round(train_loss_src_domain, 3)} Train Tgt Domain Cls Loss: {round(train_loss_tgt_domain, 3)} \n \
                                     Val Loss: {round(val_loss, 3)} Val Cls Loss: {round(val_loss_src_label, 3)} Val Src Domain Cls Loss: {round(val_loss_src_domain, 3)} Val Tgt Domain Cls Loss: {round(val_loss_tgt_domain, 3)}')
            # writer
            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_loss_src_label, e)
            self.writer.add_scalar("Source Label Loss/val", val_loss_src_label, e)
            self.writer.add_scalar("Source Domain Cls Loss/train", train_loss_src_domain, e)
            self.writer.add_scalar("Source Domain Cls Loss/val", val_loss_src_domain, e)
            self.writer.add_scalar("Target Domain Cls Loss/train", train_loss_tgt_domain, e)
            self.writer.add_scalar("Target Domain Cls Loss/val", val_loss_tgt_domain, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, best_val_score

    def adapt(self, tgt_train_loader, tgt_val_loader, lambda_coeff_list, stage_name, args, subdir_name=""):
        performance_dict = dict()
        for lamb in lambda_coeff_list:
            run_name = f'{args.method}_{str(lamb)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name, run_name))
            encoder, classifier, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, lamb, args)
            performance_dict[lamb] = {'encoder': encoder, 'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        # best_val_loss_src_label = np.inf
        best_encoder = None
        best_classifier = None
        for lamb, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                # if perf_dict['val_loss_src_label'] < best_val_loss_src_label:
                best_val_score = perf_dict['val_score']
                # best_val_loss_src_label = perf_dict['val_loss_src_label']
                best_encoder = perf_dict['encoder']
                best_classifier = perf_dict['classifier']
            print(f"Lambda: {lamb} val_score: {perf_dict['val_score']}")
        self.set_encoder_classifier(best_encoder, best_classifier)


class SinglegraphDANNAdapter(DecoupledSinglegraphAdapter):
    def __init__(self, encoder, classifier, src_data, emb_dim, device="cpu"):
        super().__init__(encoder, classifier, src_data, device)
        self.emb_dim = emb_dim
        self.grl = GradientReverseLayer()
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, domain_classifier, tgt_data, optimizer, e, epochs, lamb):
        encoder.train()
        classifier.train()
        domain_classifier.train()
        p = e / epochs
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * lamb
        src_size = self.src_data.x.shape[0]
        src_dom_label = torch.zeros(src_size).long().to(self.device)
        src_f = encoder(self.src_data.x, self.src_data.edge_index)
        src_cls_output, _ = classifier(src_f)
        src_dom_output = domain_classifier(self.grl(src_f, alpha))
        loss_src_label = F.nll_loss(F.log_softmax(src_cls_output[self.src_data.train_mask], dim=1),
                                    self.src_data.y[self.src_data.train_mask])
        loss_src_domain = F.nll_loss(F.log_softmax(src_dom_output[self.src_data.train_mask], dim=1),
                                     src_dom_label[self.src_data.train_mask])

        tgt_size = tgt_data.x.shape[0]
        tgt_dom_label = torch.ones(tgt_size).long().to(self.device)
        tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
        tgt_dom_output = domain_classifier(self.grl(tgt_f, alpha))
        loss_tgt_domain = F.nll_loss(F.log_softmax(tgt_dom_output[tgt_data.train_mask], dim=1),
                                     tgt_dom_label[tgt_data.train_mask])

        loss = loss_src_label + loss_src_domain + loss_tgt_domain
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tgt_cls_output, _ = classifier(tgt_f)
        return loss.item(), loss_src_label.item(), loss_src_domain.item(), loss_tgt_domain.item(), src_cls_output[
            self.src_data.train_mask], tgt_cls_output[tgt_data.train_mask]

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, domain_classifier, tgt_data, e, epochs, lamb):
        encoder.eval()
        classifier.eval()
        domain_classifier.eval()
        p = e / epochs
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * lamb
        src_size = self.src_data.x.shape[0]
        src_dom_label = torch.zeros(src_size).long().to(self.device)
        src_f = encoder(self.src_data.x, self.src_data.edge_index)
        src_cls_output, _ = classifier(src_f)
        src_dom_output = domain_classifier(self.grl(src_f, alpha))
        loss_src_label = F.nll_loss(F.log_softmax(src_cls_output[self.src_data.val_mask], dim=1),
                                    self.src_data.y[self.src_data.val_mask])
        loss_src_domain = F.nll_loss(F.log_softmax(src_dom_output[self.src_data.val_mask], dim=1),
                                     src_dom_label[self.src_data.val_mask])

        tgt_size = tgt_data.x.shape[0]
        tgt_dom_label = torch.ones(tgt_size).long().to(self.device)
        tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
        tgt_dom_output = domain_classifier(self.grl(tgt_f, alpha))
        loss_tgt_domain = F.nll_loss(F.log_softmax(tgt_dom_output[tgt_data.val_mask], dim=1),
                                     tgt_dom_label[tgt_data.val_mask])

        loss = loss_src_label + loss_src_domain + loss_tgt_domain

        tgt_cls_output, _ = classifier(tgt_f)
        return loss.item(), loss_src_label.item(), loss_src_domain.item(), loss_tgt_domain.item(), src_cls_output[
            self.src_data.val_mask], tgt_cls_output[tgt_data.val_mask]

    def _adapt_train_test(self, tgt_data, lamb, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        domain_classifier = DomainClassifier(self.emb_dim, self.emb_dim, 2).to(self.device)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()) + list(domain_classifier.parameters()), lr=args.adapt_lr)
        best_val_score = -np.inf
        # best_val_loss_src_label = np.inf
        best_encoder, best_classifier = None, None
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_loss_src_label, train_loss_src_domain, train_loss_tgt_domain, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, domain_classifier, tgt_data, optimizer, e, args.adapt_epochs, lamb)
            val_loss, val_loss_src_label, val_loss_src_domain, val_loss_tgt_domain, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                encoder, classifier, domain_classifier, tgt_data, e, args.adapt_epochs, lamb)
            train_src_score = self.validator(target_train={'logits': train_src_logits})
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_src_score = self.validator(target_train={'logits': val_src_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            # val_score = self.validator(target_train={'logits': torch.cat([val_src_logits, val_tgt_logits])})

            if val_tgt_score > best_val_score:
            # if val_loss_src_label < best_val_loss_src_label:
                best_val_score = val_tgt_score
                # best_val_loss_src_label = val_loss_src_label
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)

            print(
                f'Lambda {lamb} Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_loss_src_label, 3)} Train Src Domain Cls Loss: {round(train_loss_src_domain, 3)} Train Tgt Domain Cls Loss: {round(train_loss_tgt_domain, 3)} \n \
                             Val Loss: {round(val_loss, 3)} Val Cls Loss: {round(val_loss_src_label, 3)} Val Src Domain Cls Loss: {round(val_loss_src_domain, 3)} Val Tgt Domain Cls Loss: {round(val_loss_tgt_domain, 3)}')
            # writer
            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_loss_src_label, e)
            self.writer.add_scalar("Source Label Loss/val", val_loss_src_label, e)
            self.writer.add_scalar("Source Domain Cls Loss/train", train_loss_src_domain, e)
            self.writer.add_scalar("Source Domain Cls Loss/val", val_loss_src_domain, e)
            self.writer.add_scalar("Target Domain Cls Loss/train", train_loss_tgt_domain, e)
            self.writer.add_scalar("Target Domain Cls Loss/val", val_loss_tgt_domain, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, best_val_score


    def adapt(self, tgt_data, lambda_coeff_list, stage_name, args, subdir_name=""):
        tgt_data = tgt_data.to(self.device)

        performance_dict = dict()
        for lamb in lambda_coeff_list:
            run_name = f'{args.method}_{str(lamb)}_{str(args.model_seed)}'
            self.writer = SummaryWriter(
                os.path.join(args.log_dir, subdir_name, stage_name, run_name))
            encoder, classifier, val_score = self._adapt_train_test(tgt_data, lamb, args)
            performance_dict[lamb] = {'encoder': encoder, 'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        # best_val_loss_src_label = np.inf
        best_encoder = None
        best_classifier = None
        for lamb, perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
            # if perf_dict['val_loss_src_label'] < best_val_loss_src_label:
                best_val_score = perf_dict['val_score']
                # best_val_loss_src_label = perf_dict['val_loss_src_label']
                best_encoder = perf_dict['encoder']
                best_classifier = perf_dict['classifier']
            print(f"Lambda: {lamb} val_score: {perf_dict['val_score']}")
        self.set_encoder_classifier(best_encoder, best_classifier)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.linear2(x)
        return x