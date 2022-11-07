from copy import deepcopy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .grl import GradientReverseLayer
from pytorch_adapt.validators import IMValidator


class IWDANN():
    def __init__(self, encoder, classifier, src_data, emb_dim, device="cpu", oracle=False):
        self.device = device
        self.set_encoder_classifier(encoder, classifier)
        self.src_data = src_data.to(self.device)
        self.grl = GradientReverseLayer()
        self.emb_dim = emb_dim
        self.class_num = 40
        self.validator = IMValidator()
        self.oracle = oracle
        self.source_label_distribution = (
                    torch.sum(F.one_hot(self.src_data.y, num_classes=self.class_num), dim=0) / self.src_data.y.shape[
                0]).to(self.device)
        self.class_weights = 1.0 / self.source_label_distribution

    def _adapt_train_epoch(self, encoder, classifier, domain_classifier, tgt_data, optimizer, e, epochs, lambda_coeff):
        encoder.train()
        classifier.train()
        domain_classifier.train()

        ys_onehot = F.one_hot(self.src_data.y, num_classes=self.class_num)
        ys_onehot = ys_onehot.type(torch.FloatTensor).to(self.device)
        if self.oracle:
            weights = torch.mm(ys_onehot, self.true_weights)

        p = e / epochs
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * lambda_coeff
        src_size = self.src_data.x.shape[0]
        src_dom_label = torch.zeros(src_size).long().to(self.device)
        src_f = encoder(self.src_data.x, self.src_data.edge_index)
        src_cls_output, _ = classifier(src_f)
        src_dom_output = domain_classifier(self.grl(src_f, alpha))
        loss_src_label = torch.mean(F.nll_loss(F.log_softmax(src_cls_output[self.src_data.train_mask], dim=1),
                                    self.src_data.y[self.src_data.train_mask], weight=self.class_weights, reduction='none') * weights) / self.class_num
        loss_src_domain = torch.mean(F.nll_loss(F.log_softmax(src_dom_output[self.src_data.train_mask], dim=1),
                                     src_dom_label[self.src_data.train_mask], reduction='none') * weights)

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
        return loss.item(), loss_src_label.item(), loss_src_domain.item(), loss_tgt_domain.item(), src_cls_output[self.src_data.train_mask], tgt_cls_output[tgt_data.train_mask]

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, domain_classifier, tgt_data, e, epochs, lambda_coeff):
        encoder.eval()
        classifier.eval()
        domain_classifier.eval()

        ys_onehot = F.one_hot(self.src_data.y, num_classes=self.class_num)
        ys_onehot = ys_onehot.type(torch.FloatTensor).to(self.device)
        if self.oracle:
            weights = torch.mm(ys_onehot, self.true_weights)

        p = e / epochs
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * lambda_coeff

        src_size = self.src_data.x.shape[0]
        src_dom_label = torch.zeros(src_size).long().to(self.device)
        src_f = encoder(self.src_data.x, self.src_data.edge_index)
        src_cls_output, _ = classifier(src_f)
        src_dom_output = domain_classifier(self.grl(src_f, alpha))
        loss_src_label = torch.mean(F.nll_loss(F.log_softmax(src_cls_output[self.src_data.val_mask], dim=1),
                                               self.src_data.y[self.src_data.val_mask], weight=self.class_weights,
                                               reduction='none') * weights) / self.class_num
        loss_src_domain = torch.mean(F.nll_loss(F.log_softmax(src_dom_output[self.src_data.val_mask], dim=1),
                                                src_dom_label[self.src_data.val_mask], reduction='none') * weights)

        tgt_size = tgt_data.x.shape[0]
        tgt_dom_label = torch.ones(tgt_size).long().to(self.device)
        tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
        tgt_dom_output = domain_classifier(self.grl(tgt_f, alpha))
        loss_tgt_domain = F.nll_loss(F.log_softmax(tgt_dom_output[tgt_data.val_mask], dim=1),
                                     tgt_dom_label[tgt_data.val_mask])

        loss = loss_src_label + loss_src_domain + loss_tgt_domain
        tgt_logits, _ = classifier(tgt_f)

        return loss.item(), loss_src_label.item(), loss_src_domain.item(), loss_tgt_domain.item(), src_cls_output[self.src_data.val_mask], tgt_logits[tgt_data.val_mask]

    def _adapt_train_test(self, tgt_data, lambda_coeff, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        domain_classifier = DomainClassifier(self.emb_dim, self.emb_dim // 4, 2).to(self.device)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()) + list(domain_classifier.parameters()), lr=args.adapt_lr)
        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for e in range(1, args.adapt_epochs):
            train_loss, train_loss_src_label, train_loss_src_domain, train_loss_tgt_domain, train_src_logits, train_tgt_logits = self._adapt_train_epoch(encoder, classifier, domain_classifier, tgt_data, optimizer, e, args.adapt_epochs, lambda_coeff)
            val_loss, val_loss_src_label, val_loss_src_domain, val_loss_tgt_domain, val_src_logits, val_tgt_logits = self._adapt_test_epoch(encoder, classifier, domain_classifier, tgt_data, e, args.adapt_epochs, lambda_coeff)
            val_score = self.validator(target_train={'logits': val_tgt_logits})
            val_src_score = self.negative_entropy_from_logits(
                val_src_logits)  # self.validator(target_train={'logits': val_src_logits})
            train_src_score = self.negative_entropy_from_logits(
                train_src_logits)  # self.validator(target_train={'logits': train_src_logits})
            train_score = self.negative_entropy_from_logits(
                train_tgt_logits)  # self.validator(target_train={'logits': train_tgt_logits})
            if val_score > best_val_score:
                best_val_score = val_score
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)
            print(
                f'Lambda {lambda_coeff} Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_loss_src_label, 3)} Train Src Domain Cls Loss: {round(train_loss_src_domain, 3)} Train Tgt Domain Cls Loss: {round(train_loss_tgt_domain, 3)} \n \
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
            self.writer.add_scalar("Target Score/val", val_score, e)
            self.writer.add_scalar("Target Score/train", train_score, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
        # assign back
        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)
        return encoder, classifier, best_val_score

    def adapt(self, tgt_data, lambda_coeff_list, stage, args):
        tgt_data = tgt_data.to(self.device)
        if self.oracle:
            self.target_label_distribution = torch.sum(F.one_hot(tgt_data.y, num_classes=40), dim=0) / tgt_data.y.shape[0]
            self.true_weights = (self.target_label_distribution / self.source_label_distribution)[:,None]
            self.true_weights.requires_grad = False
            self.true_weights = self.true_weights.to(self.device)
            print("True weight", self.true_weights)

        performance_dict = dict()
        for lambda_coeff in lambda_coeff_list:
            run_name = args.method + "_" + str(lambda_coeff).replace(".","") + "_" + str(args.model_seed)
            self.writer = SummaryWriter(os.path.join(args.log_dir, str(stage[0]) + "_" + str(stage[1]) + "_" + str(stage[2]), run_name))
            encoder, classifier, val_score = self._adapt_train_test(tgt_data, lambda_coeff, args)
            performance_dict[lambda_coeff] = {'tgt_encoder': encoder, 'tgt_classifier': classifier, 'tgt_val_score': val_score}

        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for lambda_coeff, perf_dict in performance_dict.items():
            if perf_dict['tgt_val_score'] > best_val_score:
                best_val_score = perf_dict['tgt_val_score']
                best_encoder = perf_dict['tgt_encoder']
                best_classifier = perf_dict['tgt_classifier']

        self.set_encoder_classifier(best_encoder, best_classifier)


    def set_encoder_classifier(self, encoder, classifier):
        self.encoder = deepcopy(encoder).to(self.device)
        self.classifier = deepcopy(classifier).to(self.device)

    def get_encoder_classifier(self):
        return self.encoder, self.classifier

    def negative_entropy_from_logits(self, logits):
        neg_entropies = torch.sum(
            torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1
        )
        return torch.mean(neg_entropies)

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