from copy import deepcopy
import numpy as np
import os
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, k_hop_subgraph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader
from torch_geometric.data import Data
from pytorch_adapt.validators import IMValidator
from .grl import GradientReverseLayer
from .adapter import DecoupledMultigraphAdapter, DecoupledSinglegraphAdapter


class MultigraphUDAGCNAdapter(DecoupledMultigraphAdapter):
    def __init__(self, encoder, classifier, src_train_loader, src_val_loader, emb_dim, path_len, device="cpu"):
        super().__init__(encoder, classifier, src_train_loader, src_val_loader, device)
        src_train_datalist = []
        for i, data in enumerate(src_train_loader):
            name = "src_tr_" + str(i)
            src_train_datalist.append((data, name))
        self.src_train_loader = DataLoader(dataset=src_train_datalist, batch_size=1, shuffle=True)

        src_val_datalist = []
        for i, data in enumerate(src_val_loader):
            name = "src_val_" + str(i)
            src_val_datalist.append((data, name))
        self.src_val_loader = DataLoader(dataset=src_val_datalist, batch_size=1, shuffle=True)


        self.feat_dim = encoder.get_input_dim()
        self.emb_dim = emb_dim
        self.grl = GradientReverseLayer()
        self.validator = IMValidator()
        self.path_len = path_len

    def _adapt_train_epoch(self, encoder, classifier, ppmi_encoder, domain_model, att_model, tgt_train_loader, optimizer, e, epochs):
        encoder.train()
        classifier.train()
        ppmi_encoder.train()
        domain_model.train()
        att_model.train()

        rate = min((e + 1) / epochs, 0.05)
        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        total_loss = 0
        total_loss_src_label = 0
        total_loss_grl = 0
        total_loss_entropy = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            src_data, src_name = src_iter.next()
            src_data = src_data.to(self.device)
            src_name = src_name[0]
            # print("ENCODER:", encoder(src_data.x, src_data.edge_index))
            # print("PPMI ENCODER:", ppmi_encoder(src_data.x, src_data.edge_index))
            src_f = self.encode(encoder, ppmi_encoder, att_model, src_data, cache_name=src_name)
            src_cls_output, _ = classifier(src_f)
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                loss_src_label = F.nll_loss(F.log_softmax(src_cls_output, dim=1), src_data.y[src_data.mask])
            else:
                loss_src_label = F.nll_loss(F.log_softmax(src_cls_output, dim=1), src_data.y)

            tgt_data, tgt_name = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)
            tgt_name = tgt_name[0]
            tgt_f = self.encode(encoder, ppmi_encoder, att_model, tgt_data, cache_name=tgt_name)
            # print("src f:", src_f)
            # print("tgt f:", tgt_f)
            src_domain_preds = domain_model(self.grl(src_f, rate))
            tgt_domain_preds = domain_model(self.grl(tgt_f, rate))
            src_domain_cls_loss = F.nll_loss(
                F.log_softmax(src_domain_preds, dim=1),
                torch.zeros(src_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )
            tgt_domain_cls_loss = F.nll_loss(
                F.log_softmax(tgt_domain_preds, dim=1),
                torch.ones(tgt_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )

            loss_grl = src_domain_cls_loss + tgt_domain_cls_loss
            loss = loss_src_label + loss_grl

            tgt_logits, _ = classifier(tgt_f)
            tgt_probs = F.softmax(tgt_logits, dim=-1)
            tgt_probs = torch.clamp(tgt_probs, min=1e-9, max=1.0)
            loss_entropy = torch.mean(torch.sum(-tgt_probs * torch.log(tgt_probs), dim=-1))
            loss = loss + loss_entropy * (e / epochs * 0.01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_src_label += loss_src_label.item()
            total_loss_grl += loss_grl.item()
            total_loss_entropy += loss_entropy.item()
            total_src_logits.append(src_cls_output)
            total_tgt_logits.append(tgt_logits)

        total_loss /= len_dataloader
        total_loss_src_label /= len_dataloader
        total_loss_grl /= len_dataloader
        total_loss_entropy /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_loss_src_label, total_loss_grl, total_loss_entropy, total_src_logits, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, ppmi_encoder, domain_model, att_model, tgt_train_loader, e, epochs):
        encoder.eval()
        classifier.eval()
        ppmi_encoder.eval()
        domain_model.eval()
        att_model.eval()

        rate = min((e + 1) / epochs, 0.05)
        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        total_loss = 0
        total_loss_src_label = 0
        total_loss_grl = 0
        total_loss_entropy = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            src_data, src_name = src_iter.next()
            src_data = src_data.to(self.device)
            src_name = src_name[0]
            src_f = self.encode(encoder, ppmi_encoder, att_model, src_data, cache_name=src_name)
            src_cls_output, _ = classifier(src_f)
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                loss_src_label = F.nll_loss(F.log_softmax(src_cls_output, dim=1),
                                            src_data.y[src_data.mask])
            else:
                loss_src_label = F.nll_loss(F.log_softmax(src_cls_output, dim=1), src_data.y)

            tgt_data, tgt_name = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)
            tgt_name = tgt_name[0]
            tgt_f = self.encode(encoder, ppmi_encoder, att_model, tgt_data, cache_name=tgt_name)
            # print("src f:", src_f)
            # print("tgt f:", tgt_f)
            src_domain_preds = domain_model(self.grl(src_f, rate))
            tgt_domain_preds = domain_model(self.grl(tgt_f, rate))
            src_domain_cls_loss = F.nll_loss(
                F.log_softmax(src_domain_preds, dim=1),
                torch.zeros(src_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )
            tgt_domain_cls_loss = F.nll_loss(
                F.log_softmax(tgt_domain_preds, dim=1),
                torch.ones(tgt_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )

            loss_grl = src_domain_cls_loss + tgt_domain_cls_loss
            loss = loss_src_label + loss_grl

            tgt_logits, _ = classifier(tgt_f)
            tgt_probs = F.softmax(tgt_logits, dim=-1)
            tgt_probs = torch.clamp(tgt_probs, min=1e-9, max=1.0)
            loss_entropy = torch.mean(torch.sum(-tgt_probs * torch.log(tgt_probs), dim=-1))
            loss = loss + loss_entropy * (e / epochs * 0.01)


            total_loss += loss.item()
            total_loss_src_label += loss_src_label.item()
            total_loss_grl += loss_grl.item()
            total_loss_entropy += loss_entropy.item()
            total_src_logits.append(src_cls_output)
            total_tgt_logits.append(tgt_logits)

        total_loss /= len_dataloader
        total_loss_src_label /= len_dataloader
        total_loss_grl /= len_dataloader
        total_loss_entropy /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_loss_src_label, total_loss_grl, total_loss_entropy, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, args):
        # name each graph for cache
        tgt_train_datalist = []
        for i, data in enumerate(tgt_train_loader):
            name = "tgt_tr_" + str(i)
            tgt_train_datalist.append((data, name))
        tgt_train_loader = DataLoader(dataset=tgt_train_datalist, batch_size=1, shuffle=True)

        tgt_val_datalist = []
        for i, data in enumerate(tgt_val_loader):
            name = "tgt_val_" + str(i)
            tgt_val_datalist.append((data, name))
        tgt_val_loader = DataLoader(dataset=tgt_val_datalist, batch_size=1, shuffle=True)

        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)

        ppmi_encoder = PPMIGNN(feat_dim=self.feat_dim, emb_dim=self.emb_dim, base_model=None, type="ppmi", path_len=self.path_len).to(self.device)
        domain_model = nn.Sequential(
                            nn.Linear(self.emb_dim, 40),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(40, 2),
                        ).to(self.device)

        att_model = Attention(self.emb_dim).to(self.device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()) + list(ppmi_encoder.parameters()) + list(domain_model.parameters()) + list(att_model.parameters()),
            lr=args.adapt_lr)
        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_loss_src_label, train_loss_grl, train_loss_entropy, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, ppmi_encoder, domain_model, att_model, tgt_train_loader, optimizer, e, args.adapt_epochs)
            val_loss, val_loss_src_label, val_loss_grl, val_loss_entropy, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                encoder, classifier, ppmi_encoder, domain_model, att_model, tgt_val_loader, e, args.adapt_epochs)
            train_src_score = self.validator(target_train={'logits': train_src_logits})
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_src_score = self.validator(target_train={'logits': val_src_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            # val_score = self.validator(target_train={'logits': torch.cat([val_src_logits, val_tgt_logits])})
            if val_tgt_score > best_val_score:
                best_val_score = val_tgt_score
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)

            print(
                f'Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_loss_src_label, 3)} Train GRL Loss: {round(train_loss_grl, 3)} Train Entropy Loss: {round(train_loss_entropy, 3)} \n \
                                     Val Loss: {round(val_loss, 3)} Val Cls Loss: {round(val_loss_src_label, 3)} Val GRL Loss: {round(val_loss_grl, 3)} Val Entropy Loss: {round(val_loss_entropy, 3)}')
            # writer
            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_loss_src_label, e)
            self.writer.add_scalar("Source Label Loss/val", val_loss_src_label, e)
            self.writer.add_scalar("GRL Loss/train", train_loss_grl, e)
            self.writer.add_scalar("GRL Loss/val", val_loss_grl, e)
            self.writer.add_scalar("Entropy Loss/train", train_loss_entropy, e)
            self.writer.add_scalar("Entropy Loss/val", val_loss_entropy, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, best_val_score

    def adapt(self, tgt_train_loader, tgt_val_loader, stage_name, args, subdir_name=""):

        run_name = f'{args.method}_{str(args.model_seed)}'
        self.writer = SummaryWriter(
            os.path.join(args.log_dir, subdir_name, stage_name, run_name))
        encoder, classifier, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, args)
        print("Val Score:", val_score)
        self.set_encoder_classifier(encoder, classifier)

    def encode(self, encoder, ppmi_encoder, att_model, data, cache_name):
        encoder_output = encoder(data.x, data.edge_index)
        # print("x size:", data.x.shape)
        # print("max node index:", data.edge_index.max().item())
        ppmi_output = ppmi_encoder(data.x, data.edge_index, cache_name)
        if 'mask' in data: # e.g. elliptic (not all nodes are labeled)
            encoder_output = encoder_output[data.mask]
            ppmi_output = ppmi_output[data.mask]
        outputs = att_model([encoder_output, ppmi_output])
        return outputs



class SinglegraphUDAGCNAdapter(DecoupledSinglegraphAdapter):
    def __init__(self, encoder, classifier, src_data, emb_dim, path_len, device="cpu"):
        super().__init__(encoder, classifier, src_data, device)
        self.feat_dim = encoder.get_input_dim()
        self.emb_dim = emb_dim
        self.grl = GradientReverseLayer()
        self.validator = IMValidator()
        self.path_len = path_len

    def _adapt_train_epoch(self, encoder, classifier, ppmi_encoder, domain_model, att_model, src_loader, tgt_loader, optimizer, e, epochs):
        encoder.train()
        classifier.train()
        ppmi_encoder.train()
        domain_model.train()
        att_model.train()


        rate = min((e + 1) / epochs, 0.05)
        assert len(src_loader) == len(tgt_loader)
        len_dataloader = min(len(src_loader), len(tgt_loader))
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        total_loss = 0
        total_loss_src_label = 0
        total_loss_grl = 0
        total_loss_entropy = 0
        total_src_logits = []
        total_tgt_logits = []
        for i in range(len_dataloader):
            src_data = src_iter.next()
            src_data = src_data.to(self.device)
            src_f = self.encode(encoder, ppmi_encoder, att_model, src_data, cache_name="src_" + str(i), mask=src_data.train_mask)
            src_cls_output, _ = classifier(src_f)
            loss_src_label = F.nll_loss(F.log_softmax(src_cls_output, dim=1), src_data.y[src_data.train_mask])

            tgt_data = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)
            tgt_f = self.encode(encoder, ppmi_encoder, att_model, tgt_data, cache_name="tgt_" + str(i), mask=tgt_data.train_mask)
            src_domain_preds = domain_model(self.grl(src_f, rate))
            tgt_domain_preds = domain_model(self.grl(tgt_f, rate))
            src_domain_cls_loss = F.nll_loss(
                F.log_softmax(src_domain_preds, dim=1),
                torch.zeros(src_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )
            tgt_domain_cls_loss = F.nll_loss(
                F.log_softmax(tgt_domain_preds, dim=1),
                torch.ones(tgt_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )

            loss_grl = src_domain_cls_loss + tgt_domain_cls_loss
            loss = loss_src_label + loss_grl

            tgt_logits, _ = classifier(tgt_f)
            tgt_probs = F.softmax(tgt_logits, dim=-1)
            tgt_probs = torch.clamp(tgt_probs, min=1e-9, max=1.0)
            loss_entropy = torch.mean(torch.sum(-tgt_probs * torch.log(tgt_probs), dim=-1))
            loss = loss + loss_entropy * (e / epochs * 0.01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_src_label += loss_src_label.item()
            total_loss_grl += loss_grl.item()
            total_loss_entropy += loss_entropy.item()
            total_src_logits.append(src_cls_output)
            total_tgt_logits.append(tgt_logits)

        total_loss /= len_dataloader
        total_loss_src_label /= len_dataloader
        total_loss_grl /= len_dataloader
        total_loss_entropy /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)

        return total_loss, total_loss_src_label, total_loss_grl, total_loss_entropy, total_src_logits, total_tgt_logits


    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, ppmi_encoder, domain_model, att_model, src_loader, tgt_loader, e, epochs):
        encoder.eval()
        classifier.eval()
        ppmi_encoder.eval()
        domain_model.eval()
        att_model.eval()

        rate = min((e + 1) / epochs, 0.05)
        assert len(src_loader) == len(tgt_loader)
        len_dataloader = min(len(src_loader), len(tgt_loader))
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        total_loss = 0
        total_loss_src_label = 0
        total_loss_grl = 0
        total_loss_entropy = 0
        total_src_logits = []
        total_tgt_logits = []
        for i in range(len_dataloader):
            src_data = src_iter.next()
            src_data = src_data.to(self.device)
            src_f = self.encode(encoder, ppmi_encoder, att_model, src_data, cache_name="src_" + str(i),
                                mask=src_data.val_mask)
            src_cls_output, _ = classifier(src_f)
            loss_src_label = F.nll_loss(F.log_softmax(src_cls_output, dim=1), src_data.y[src_data.val_mask])

            tgt_data = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)
            tgt_f = self.encode(encoder, ppmi_encoder, att_model, tgt_data, cache_name="tgt_" + str(i),
                                mask=tgt_data.val_mask)
            src_domain_preds = domain_model(self.grl(src_f, rate))
            tgt_domain_preds = domain_model(self.grl(tgt_f, rate))
            src_domain_cls_loss = F.nll_loss(
                F.log_softmax(src_domain_preds, dim=1),
                torch.zeros(src_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )
            tgt_domain_cls_loss = F.nll_loss(
                F.log_softmax(tgt_domain_preds, dim=1),
                torch.ones(tgt_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )

            loss_grl = src_domain_cls_loss + tgt_domain_cls_loss
            loss = loss_src_label + loss_grl

            tgt_logits, _ = classifier(tgt_f)
            tgt_probs = F.softmax(tgt_logits, dim=-1)
            tgt_probs = torch.clamp(tgt_probs, min=1e-9, max=1.0)
            loss_entropy = torch.mean(torch.sum(-tgt_probs * torch.log(tgt_probs), dim=-1))
            loss = loss + loss_entropy * (e / epochs * 0.01)


            total_loss += loss.item()
            total_loss_src_label += loss_src_label.item()
            total_loss_grl += loss_grl.item()
            total_loss_entropy += loss_entropy.item()
            total_src_logits.append(src_cls_output)
            total_tgt_logits.append(tgt_logits)

        total_loss /= len_dataloader
        total_loss_src_label /= len_dataloader
        total_loss_grl /= len_dataloader
        total_loss_entropy /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)

        return total_loss, total_loss_src_label, total_loss_grl, total_loss_entropy, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, src_loader, tgt_loader, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)

        ppmi_encoder = PPMIGNN(feat_dim=self.feat_dim, emb_dim=self.emb_dim, base_model=None, type="ppmi",
                               path_len=self.path_len, device=self.device).to(self.device)
        domain_model = nn.Sequential(
            nn.Linear(self.emb_dim, 40),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(40, 2),
        ).to(self.device)

        att_model = Attention(self.emb_dim).to(self.device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()) + list(ppmi_encoder.parameters()) + list(
                domain_model.parameters()) + list(att_model.parameters()),
            lr=args.adapt_lr)
        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_loss_src_label, train_loss_grl, train_loss_entropy, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, ppmi_encoder, domain_model, att_model, src_loader, tgt_loader, optimizer, e, args.adapt_epochs)
            val_loss, val_loss_src_label, val_loss_grl, val_loss_entropy, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                encoder, classifier, ppmi_encoder, domain_model, att_model, src_loader, tgt_loader, e, args.adapt_epochs)
            train_src_score = self.validator(target_train={'logits': train_src_logits})
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_src_score = self.validator(target_train={'logits': val_src_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            if val_tgt_score > best_val_score:
                best_val_score = val_tgt_score
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)

            print(
                f'Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_loss_src_label, 3)} Train GRL Loss: {round(train_loss_grl, 3)} Train Entropy Loss: {round(train_loss_entropy, 3)} \n \
                                             Val Loss: {round(val_loss, 3)} Val Cls Loss: {round(val_loss_src_label, 3)} Val GRL Loss: {round(val_loss_grl, 3)} Val Entropy Loss: {round(val_loss_entropy, 3)}')
            # writer
            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_loss_src_label, e)
            self.writer.add_scalar("Source Label Loss/val", val_loss_src_label, e)
            self.writer.add_scalar("GRL Loss/train", train_loss_grl, e)
            self.writer.add_scalar("GRL Loss/val", val_loss_grl, e)
            self.writer.add_scalar("Entropy Loss/train", train_loss_entropy, e)
            self.writer.add_scalar("Entropy Loss/val", val_loss_entropy, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, best_val_score


    def adapt(self, tgt_data, stage_name, args, subdir_name=""):
        # tgt_data = self._extract_k_hop(tgt_data, 2).to(self.device)
        cluster_src_data = ClusterData(self.src_data, num_parts=2)
        cluster_src_loader = ClusterLoader(cluster_src_data, batch_size=1, shuffle=False)
        cluster_tgt_data = ClusterData(self._extract_k_hop(tgt_data, 2), num_parts=2)
        cluster_tgt_loader = ClusterLoader(cluster_tgt_data, batch_size=1, shuffle=False)

        run_name = f'{args.method}_{str(args.model_seed)}'
        self.writer = SummaryWriter(
            os.path.join(args.log_dir, subdir_name, stage_name, run_name))
        encoder, classifier, val_score = self._adapt_train_test(cluster_src_loader, cluster_tgt_loader, args)
        print("Val Score:", val_score)
        self.set_encoder_classifier(encoder, classifier)

    def encode(self, encoder, ppmi_encoder, att_model, data, cache_name, mask=None):
        encoder_output = encoder(data.x, data.edge_index)
        ppmi_output = ppmi_encoder(data.x, data.edge_index, cache_name)
        if mask is not None:
            encoder_output = encoder_output[mask]
            ppmi_output = ppmi_output[mask]
        outputs = att_model([encoder_output, ppmi_output])
        return outputs

    def _extract_k_hop(self, data, k_hop):
        x, edge_index = data.x, data.edge_index
        train_mask, val_mask = data.train_mask, data.val_mask
        train_node_idx = torch.argwhere(train_mask).squeeze(1).tolist()
        val_node_idx = torch.argwhere(val_mask).squeeze(1).tolist()
        # print("before discarding isolated nodes:", len(train_node_idx) + len(val_node_idx))
        connected_train_mask = torch.zeros_like(train_mask, dtype=torch.bool) # only consider nodes that appear in edge_index
        for idx in train_node_idx:
            if idx in edge_index:
                connected_train_mask[idx] = True
        connected_val_mask = torch.zeros_like(val_mask, dtype=torch.bool)
        for idx in val_node_idx:
            if idx in edge_index:
                connected_val_mask[idx] = True
        connected_train_node_idx = torch.argwhere(connected_train_mask).squeeze(1).tolist()
        connected_val_node_idx = torch.argwhere(connected_val_mask).squeeze(1).tolist()
        connected_node_idx = connected_train_node_idx + connected_val_node_idx
        # print("after discarding isolated nodes:", len(connected_node_idx))
        subset, new_edge_index, mapping, edge_mask = k_hop_subgraph(connected_node_idx, k_hop, edge_index,
                                                                        relabel_nodes=True)
        new_x = x[subset]  # crop features vector from original tgt.x
        new_train_mask = torch.zeros_like(subset, dtype=torch.bool)
        new_train_mask[mapping[:len(connected_train_node_idx)]] = True  # first half of the mapping are training nodes
        new_val_mask = torch.zeros_like(subset, dtype=torch.bool)
        new_val_mask[mapping[len(connected_train_node_idx):]] = True  # second half of the mapping are validation nodes
        data = Data(x=new_x, edge_index=new_edge_index, train_mask=new_train_mask,
                        val_mask=new_val_mask)
        return data






class CachedGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 weight=None,
                 bias=None,
                 improved=False,
                 use_bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cache_dict = {}

        # self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        #
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)


        if weight is None:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels).to(torch.float32))
            glorot(self.weight)
        else:
            self.weight = weight
            print("use shared weight")

        if bias is None:
            if use_bias:
                self.bias = Parameter(torch.Tensor(out_channels).to(torch.float32))
            else:
                self.register_parameter('bias', None)
            zeros(self.bias)
        else:
            self.bias = bias
            print("use shared bias")

        # self.reset_parameters()

    # def reset_parameters(self):
    #     glorot(self.weight)
    #     zeros(self.bias)
        # self.cached_result = None
        # self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1./2. # according to the original author's reproducing tips # 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, cache_name="default_cache", edge_weight=None):
        """"""

        x = torch.matmul(x, self.weight)

        if not cache_name in self.cache_dict:
            # print("not cached, cache name:", cache_name)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                     self.improved, x.dtype)
            self.cache_dict[cache_name] = edge_index, norm
        else:
            # print("cached, cache name:", cache_name)
            edge_index, norm = self.cache_dict[cache_name]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class PPMIConv(CachedGCNConv):

    def __init__(self, in_channels, out_channels,
                 weight=None, bias=None, improved=False, use_bias=True,
                 path_len=5,
                 gpu_device="cuda:0",
                 **kwargs):
        super().__init__(in_channels, out_channels, weight, bias, improved, use_bias, **kwargs)
        self.path_len = path_len
        self.gpu_device = gpu_device


    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):

        adj_dict = {}

        def add_edge(a, b):
            if a in adj_dict:
                neighbors = adj_dict[a]
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        cpu_device = torch.device("cpu")
        gpu_device = torch.device(self.gpu_device)
        for a, b in edge_index.t().detach().to(cpu_device).numpy():
            a = int(a)
            b = int(b)
            add_edge(a, b)
            add_edge(b, a)

        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

        def sample_neighbor(a):
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]


        # word_counter = Counter()
        walk_counters = {}

        def norm(counter):
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter

        for _ in tqdm(range(40)):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, self.path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter

                    walk_counter[b] += 1

                    current_a = b

        normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

        prob_sums = Counter()

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                prob_sums[b] += prob

        ppmis = {}

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / self.path_len)
                # print("ppmi before log:", prob / prob_sums[b] * len(prob_sums) / self.path_len)
                ppmis[(a, b)] = ppmi

        new_edge_index = []
        edge_weight = []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)

        edge_index = torch.tensor(new_edge_index).t().to(gpu_device)
        edge_weight = torch.tensor(edge_weight).to(gpu_device)
        # print("edge weight:", edge_weight)

        fill_value = 1./2. # according to the original author's reproducing tips # 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        # print("row:", row)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # print("deg:", deg)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # print("deg inv sqrt:", deg_inv_sqrt)
        # print("norm", (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32))
        return edge_index, (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32)


class PPMIGNN(torch.nn.Module):
    def __init__(self, feat_dim, emb_dim, base_model=None, type="ppmi", device="cuda:0", **kwargs):
        super(PPMIGNN, self).__init__()
        assert type == "ppmi"
        if base_model is None:
            weights = [None, None]
            biases = [None, None]
            hidden_dim = 128
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]
            hidden_dim = base_model.get_hidden_dim()


        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv

        self.conv_layers = nn.ModuleList([
            model_cls(feat_dim, hidden_dim,
                     weight=weights[0],
                     bias=biases[0],
                     gpu_device=device,
                      **kwargs),
            model_cls(hidden_dim, emb_dim,
                     weight=weights[1],
                     bias=biases[1],
                     gpu_device=device,
                      **kwargs)
        ])


    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs
