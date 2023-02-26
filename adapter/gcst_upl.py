import os
from copy import deepcopy
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader
from pytorch_adapt.validators import IMValidator
from .adapter import DecoupledMultigraphAdapter, DecoupledSinglegraphAdapter
from .mvgrl_dn import DGI


class MultigraphGCSTUPL(DecoupledMultigraphAdapter):
    def __init__(self, encoder, classifier, emb_dim, src_train_loader=None, src_val_loader=None, device="cpu"):
        super().__init__(encoder, classifier, src_train_loader, src_val_loader, device)
        # self.validator = BNMValidator()
        self.emb_dim = emb_dim
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, dgi, tgt_train_loader, optimizer, thres, contrast):
        encoder.train()
        classifier.train()
        dgi.train()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader)) if self.src_train_loader else len(tgt_train_loader)
        src_iter = iter(self.src_train_loader) if self.src_train_loader else None
        tgt_iter = iter(tgt_train_loader)
        total_src_loss = 0
        total_tgt_loss = 0
        total_src_contrast_loss = 0
        total_tgt_contrast_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_src_contrast_num = 0
        total_tgt_contrast_num = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            if self.src_train_loader:
                src_data = src_iter.next().to(self.device)
                src_y, _ = classifier(encoder(src_data.x, src_data.edge_index))
                if 'mask' in src_data: # e.g. elliptic (not all nodes are labeled)
                    src_mask = src_data.mask
                else:
                    src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask], reduction='sum')
                src_node_num = src_mask.sum().item()
                total_src_logits.append(src_y)

                src_con_logits, _, _, src_con_node_num = dgi(src_data)
                lbl_1 = torch.ones(src_con_node_num * 2)
                lbl_2 = torch.zeros(src_con_node_num * 2)
                lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
                src_contrast_loss = F.binary_cross_entropy_with_logits(src_con_logits, lbl, reduction='sum')
                src_contrast_num = len(lbl)


            else:
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))
                src_contrast_loss, src_contrast_num = torch.tensor(0.0), 0

            tgt_data = tgt_iter.next().to(self.device)
            tgt_y, _ = classifier(encoder(tgt_data.x, tgt_data.edge_index))
            tgt_loss, tgt_node_num = self._pseudo_loss(tgt_y, thres, reduction='sum')
            total_tgt_logits.append(tgt_y)

            tgt_con_logits, _, _, tgt_con_node_num = dgi(tgt_data)
            lbl_1 = torch.ones(tgt_con_node_num * 2)
            lbl_2 = torch.zeros(tgt_con_node_num * 2)
            lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
            tgt_contrast_loss = F.binary_cross_entropy_with_logits(tgt_con_logits, lbl, reduction='sum')
            tgt_contrast_num = len(lbl)



            loss = src_loss + tgt_loss + contrast * (src_contrast_loss + tgt_contrast_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_src_contrast_loss += src_contrast_loss.item()
            total_tgt_contrast_loss += tgt_contrast_loss.item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num
            total_src_contrast_num += src_contrast_num
            total_tgt_contrast_num += tgt_contrast_num

        total_loss = (total_src_loss + total_tgt_loss + contrast * (total_src_contrast_loss + total_tgt_contrast_loss)) / (total_src_node + total_tgt_node + total_src_contrast_num + total_tgt_contrast_num)  if (total_src_node + total_tgt_node + total_src_contrast_num + total_tgt_contrast_num) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_tgt_loss = total_tgt_loss / total_tgt_node if total_tgt_node > 0 else 0.
        total_src_contrast_loss = total_src_contrast_loss / total_src_contrast_num if total_src_contrast_num > 0 else 0.
        total_tgt_contrast_loss = total_tgt_contrast_loss / total_tgt_contrast_num if total_tgt_contrast_num > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_src_contrast_loss, total_tgt_contrast_loss, total_src_logits, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, dgi, tgt_val_loader, thres, contrast):
        encoder.eval()
        classifier.eval()
        dgi.eval()

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader)) if self.src_val_loader else len(
            tgt_val_loader)
        src_iter = iter(self.src_val_loader) if self.src_val_loader else None
        tgt_iter = iter(tgt_val_loader)
        total_src_loss = 0
        total_tgt_loss = 0
        total_src_contrast_loss = 0
        total_tgt_contrast_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_src_contrast_num = 0
        total_tgt_contrast_num = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            if self.src_val_loader:
                src_data = src_iter.next().to(self.device)
                src_y, _ = classifier(encoder(src_data.x, src_data.edge_index))
                if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                    src_mask = src_data.mask
                else:
                    src_mask = torch.ones(src_y.shape[0], dtype=torch.bool)
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask], reduction='sum')
                src_node_num = src_mask.sum().item()
                total_src_logits.append(src_y)

                src_con_logits, _, _, src_con_node_num = dgi(src_data)
                lbl_1 = torch.ones(src_con_node_num * 2)
                lbl_2 = torch.zeros(src_con_node_num * 2)
                lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
                src_contrast_loss = F.binary_cross_entropy_with_logits(src_con_logits, lbl, reduction='sum')
                src_contrast_num = len(lbl)


            else:
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))
                src_contrast_loss, src_contrast_num = torch.tensor(0.0), 0

            tgt_data = tgt_iter.next().to(self.device)
            tgt_y, _ = classifier(encoder(tgt_data.x, tgt_data.edge_index))
            tgt_loss, tgt_node_num = self._pseudo_loss(tgt_y, thres, reduction='sum')
            total_tgt_logits.append(tgt_y)

            tgt_con_logits, _, _, tgt_con_node_num = dgi(tgt_data)
            lbl_1 = torch.ones(tgt_con_node_num * 2)
            lbl_2 = torch.zeros(tgt_con_node_num * 2)
            lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
            tgt_contrast_loss = F.binary_cross_entropy_with_logits(tgt_con_logits, lbl, reduction='sum')
            tgt_contrast_num = len(lbl)


            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_src_contrast_loss += src_contrast_loss.item()
            total_tgt_contrast_loss += tgt_contrast_loss.item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num
            total_src_contrast_num += src_contrast_num
            total_tgt_contrast_num += tgt_contrast_num

        total_loss = (total_src_loss + total_tgt_loss + contrast * (
                    total_src_contrast_loss + total_tgt_contrast_loss)) / (
                                 total_src_node + total_tgt_node + total_src_contrast_num + total_tgt_contrast_num) if ( total_src_node + total_tgt_node + total_src_contrast_num + total_tgt_contrast_num) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_tgt_loss = total_tgt_loss / total_tgt_node if total_tgt_node > 0 else 0.
        total_src_contrast_loss = total_src_contrast_loss / total_src_contrast_num if total_src_contrast_num > 0 else 0.
        total_tgt_contrast_loss = total_tgt_contrast_loss / total_tgt_contrast_num if total_tgt_contrast_num > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_src_contrast_loss, total_tgt_contrast_loss, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, thres, contrast, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)

        dgi = DGI(encoder, self.emb_dim, p=0.2, device=self.device).to(self.device)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()) + list(dgi.parameters()), lr=args.adapt_lr)

        # train with pseudo labels
        best_val_loss = np.inf
        best_val_score = None
        best_encoder, best_classifier = None, None
        patience = 20
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_tgt_loss, train_src_contrast_loss, train_tgt_contrast_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, dgi, tgt_train_loader, optimizer, thres, contrast)
            val_loss, val_src_loss, val_tgt_loss, val_src_contrast_loss, val_tgt_contrast_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                encoder, classifier, dgi,
                tgt_val_loader, thres, contrast)
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
                f'Thres: {thres} Contrast: {contrast} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Src Contrast Loss: {round(train_src_contrast_loss, 3)} Train Tgt Contrast Loss: {round(train_tgt_contrast_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Src Contrast Loss: {round(val_src_contrast_loss, 3)} Val Tgt Contrast Loss: {round(val_tgt_contrast_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Loss/val", val_src_loss, e)
            self.writer.add_scalar("Target Loss/train", train_tgt_loss, e)
            self.writer.add_scalar("Target Loss/val", val_tgt_loss, e)
            self.writer.add_scalar("Source Contrast Loss/train", train_src_contrast_loss, e)
            self.writer.add_scalar("Source Contrast Loss/val", val_src_contrast_loss, e)
            self.writer.add_scalar("Target Contrast Loss/train", train_tgt_contrast_loss, e)
            self.writer.add_scalar("Target Contrast Loss/val", val_tgt_contrast_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            if staleness > patience:
                break

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, best_val_score


    def adapt(self, tgt_train_loader, tgt_val_loader, threshold_list, contrast_list, stage_name, args, subdir_name=""):
        performance_dict = defaultdict(dict)
        for thres in threshold_list:
            for contrast in contrast_list:
                run_name = f'{args.method}_{str(thres)}_{str(contrast)}_{str(args.model_seed)}'
                self.writer = SummaryWriter(
                    os.path.join(args.log_dir, subdir_name, stage_name,
                                 run_name))
                encoder, classifier, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, thres, contrast, args)
                performance_dict[thres][contrast] = {'encoder': encoder, 'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for thres, perf_dict in performance_dict.items():
            for contrast, ckpt_dict in perf_dict.items():
                if ckpt_dict['val_score'] > best_val_score:
                    best_val_score = ckpt_dict['val_score']
                    best_encoder = ckpt_dict['encoder']
                    best_classifier = ckpt_dict['classifier']
                print(f"thres: {thres} contrast: {contrast} val_score: {ckpt_dict['val_score']}")

        self.set_encoder_classifier(best_encoder, best_classifier)

    def _pseudo_loss(self, logits, thres, reduction='mean'):
        confidence, pseudo_labels = torch.max(F.softmax(logits.detach(), dim=1), dim=1)
        mask = (confidence > thres)
        loss = F.cross_entropy(logits[mask], pseudo_labels[mask], reduction=reduction)
        if reduction == 'mean':
            return loss
        elif reduction == 'sum':
            return loss, mask.sum().item()



class SinglegraphGCSTUPL(DecoupledSinglegraphAdapter):
    def __init__(self, encoder, classifier, emb_dim, src_data=None, device="cpu"):
        super().__init__(encoder, classifier, src_data, device)
        self.emb_dim = emb_dim
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, dgi, src_loader, tgt_loader, optimizer, thres, contrast):
        encoder.train()
        classifier.train()
        dgi.train()

        len_dataloader = min(len(src_loader), len(tgt_loader)) if src_loader else len(tgt_loader)
        src_iter = iter(src_loader) if src_loader else None
        tgt_iter = iter(tgt_loader)
        total_src_loss = 0
        total_tgt_loss = 0
        total_src_contrast_loss = 0
        total_tgt_contrast_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_src_contrast_num = 0
        total_tgt_contrast_num = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            if src_loader:
                src_data = src_iter.next().to(self.device)
                src_y, _ = classifier(encoder(src_data.x, src_data.edge_index))
                src_mask = src_data.train_mask
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask], reduction='sum')
                src_node_num = src_mask.sum().item()
                total_src_logits.append(src_y[src_mask])

                src_con_logits, _, _, src_con_node_num = dgi(src_data)
                lbl_1 = torch.ones(src_con_node_num * 2)
                lbl_2 = torch.zeros(src_con_node_num * 2)
                lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
                src_contrast_loss = F.binary_cross_entropy_with_logits(src_con_logits, lbl, reduction='sum')
                src_contrast_num = len(lbl)

            else:
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))
                src_contrast_loss, src_contrast_num = torch.tensor(0.0), 0

            tgt_data = tgt_iter.next().to(self.device)
            tgt_mask = tgt_data.train_mask
            tgt_y, _ = classifier(encoder(tgt_data.x, tgt_data.edge_index))
            tgt_loss, tgt_node_num = self._pseudo_loss(tgt_y[tgt_mask], thres, reduction='sum')

            total_tgt_logits.append(tgt_y[tgt_mask])

            tgt_con_logits, _, _, tgt_con_node_num = dgi(tgt_data)
            lbl_1 = torch.ones(tgt_con_node_num * 2)
            lbl_2 = torch.zeros(tgt_con_node_num * 2)
            lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
            tgt_contrast_loss = F.binary_cross_entropy_with_logits(tgt_con_logits, lbl, reduction='sum')
            tgt_contrast_num = len(lbl)

            loss = src_loss + tgt_loss + contrast * (src_contrast_loss + tgt_contrast_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_src_contrast_loss += src_contrast_loss.item()
            total_tgt_contrast_loss += tgt_contrast_loss.item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num
            total_src_contrast_num += src_contrast_num
            total_tgt_contrast_num += tgt_contrast_num

        total_loss = (total_src_loss + total_tgt_loss + contrast * (
                    total_src_contrast_loss + total_tgt_contrast_loss)) / (
                                 total_src_node + total_tgt_node + total_src_contrast_num + total_tgt_contrast_num) if (
                                                                                                                                   total_src_node + total_tgt_node + total_src_contrast_num + total_tgt_contrast_num) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_tgt_loss = total_tgt_loss / total_tgt_node if total_tgt_node > 0 else 0.
        total_src_contrast_loss = total_src_contrast_loss / total_src_contrast_num if total_src_contrast_num > 0 else 0.
        total_tgt_contrast_loss = total_tgt_contrast_loss / total_tgt_contrast_num if total_tgt_contrast_num > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_src_contrast_loss, total_tgt_contrast_loss, total_src_logits, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, dgi, src_loader, tgt_loader, thres, contrast):
        encoder.eval()
        classifier.eval()
        dgi.eval()

        len_dataloader = min(len(src_loader), len(tgt_loader)) if src_loader else len(tgt_loader)
        src_iter = iter(src_loader) if src_loader else None
        tgt_iter = iter(tgt_loader)
        total_src_loss = 0
        total_tgt_loss = 0
        total_src_contrast_loss = 0
        total_tgt_contrast_loss = 0
        total_src_node = 0
        total_tgt_node = 0
        total_src_contrast_num = 0
        total_tgt_contrast_num = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            if src_loader:
                src_data = src_iter.next().to(self.device)
                src_y, _ = classifier(encoder(src_data.x, src_data.edge_index))
                src_mask = src_data.val_mask
                src_loss = F.nll_loss(F.log_softmax(src_y[src_mask], dim=1), src_data.y[src_mask], reduction='sum')
                src_node_num = src_mask.sum().item()
                total_src_logits.append(src_y[src_mask])

                src_con_logits, _, _, src_con_node_num = dgi(src_data)
                lbl_1 = torch.ones(src_con_node_num * 2)
                lbl_2 = torch.zeros(src_con_node_num * 2)
                lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
                src_contrast_loss = F.binary_cross_entropy_with_logits(src_con_logits, lbl, reduction='sum')
                src_contrast_num = len(lbl)

            else:
                src_loss, src_node_num = torch.tensor(0.0), 0
                total_src_logits.append(torch.tensor([[]]))
                src_contrast_loss, src_contrast_num = torch.tensor(0.0), 0

            tgt_data = tgt_iter.next().to(self.device)
            tgt_mask = tgt_data.val_mask
            tgt_y, _ = classifier(encoder(tgt_data.x, tgt_data.edge_index))
            tgt_loss, tgt_node_num = self._pseudo_loss(tgt_y[tgt_mask], thres, reduction='sum')

            total_tgt_logits.append(tgt_y[tgt_mask])

            tgt_con_logits, _, _, tgt_con_node_num = dgi(tgt_data)
            lbl_1 = torch.ones(tgt_con_node_num * 2)
            lbl_2 = torch.zeros(tgt_con_node_num * 2)
            lbl = torch.cat([lbl_1, lbl_2], 0).to(self.device)
            tgt_contrast_loss = F.binary_cross_entropy_with_logits(tgt_con_logits, lbl, reduction='sum')
            tgt_contrast_num = len(lbl)



            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_src_contrast_loss += src_contrast_loss.item()
            total_tgt_contrast_loss += tgt_contrast_loss.item()
            total_src_node += src_node_num
            total_tgt_node += tgt_node_num
            total_src_contrast_num += src_contrast_num
            total_tgt_contrast_num += tgt_contrast_num

        total_loss = (total_src_loss + total_tgt_loss + contrast * (
                total_src_contrast_loss + total_tgt_contrast_loss)) / (
                             total_src_node + total_tgt_node + total_src_contrast_num + total_tgt_contrast_num) if (
                                                                                                                           total_src_node + total_tgt_node + total_src_contrast_num + total_tgt_contrast_num) > 0 else 0.
        total_src_loss = total_src_loss / total_src_node if total_src_node > 0 else 0.
        total_tgt_loss = total_tgt_loss / total_tgt_node if total_tgt_node > 0 else 0.
        total_src_contrast_loss = total_src_contrast_loss / total_src_contrast_num if total_src_contrast_num > 0 else 0.
        total_tgt_contrast_loss = total_tgt_contrast_loss / total_tgt_contrast_num if total_tgt_contrast_num > 0 else 0.
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_src_loss, total_tgt_loss, total_src_contrast_loss, total_tgt_contrast_loss, total_src_logits, total_tgt_logits

    def _adapt_train_test(self, src_loader, tgt_loader, thres, contrast, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)

        dgi = DGI(encoder, self.emb_dim, p=0.2, device=self.device).to(self.device)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()) + list(dgi.parameters()), lr=args.adapt_lr)

        # train with pseudo labels
        best_val_loss = np.inf
        best_val_score = None
        best_encoder, best_classifier = None, None
        patience = 20
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_tgt_loss, train_src_contrast_loss, train_tgt_contrast_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, dgi, src_loader, tgt_loader, optimizer, thres, contrast)
            val_loss, val_src_loss, val_tgt_loss, val_src_contrast_loss, val_tgt_contrast_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                encoder, classifier, dgi,
                src_loader, tgt_loader, thres, contrast)
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
                f'Thres: {thres} Contrast: {contrast} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Src Contrast Loss: {round(train_src_contrast_loss, 3)} Train Tgt Contrast Loss: {round(train_tgt_contrast_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Src Contrast Loss: {round(val_src_contrast_loss, 3)} Val Tgt Contrast Loss: {round(val_tgt_contrast_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Loss/train", train_src_loss, e)
            self.writer.add_scalar("Source Loss/val", val_src_loss, e)
            self.writer.add_scalar("Target Loss/train", train_tgt_loss, e)
            self.writer.add_scalar("Target Loss/val", val_tgt_loss, e)
            self.writer.add_scalar("Source Contrast Loss/train", train_src_contrast_loss, e)
            self.writer.add_scalar("Source Contrast Loss/val", val_src_contrast_loss, e)
            self.writer.add_scalar("Target Contrast Loss/train", train_tgt_contrast_loss, e)
            self.writer.add_scalar("Target Contrast Loss/val", val_tgt_contrast_loss, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            if staleness > patience:
                break

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, best_val_score


    def adapt(self, tgt_data, threshold_list, contrast_list, stage_name, args, subdir_name=""):
        if self.src_data:
            cluster_src_data = ClusterData(self.src_data, num_parts=32)
            cluster_src_loader = ClusterLoader(cluster_src_data, batch_size=16, shuffle=True)
        else:
            cluster_src_loader = None
        # cluster_tgt_data = ClusterData(tgt_data, num_parts=32)
        cluster_tgt_data = ClusterData(self._extract_k_hop(tgt_data, 2), num_parts=32)
        cluster_tgt_loader = ClusterLoader(cluster_tgt_data, batch_size=16, shuffle=True)

        performance_dict = defaultdict(dict)
        for thres in threshold_list:
            for contrast in contrast_list:
                run_name = f'{args.method}_{str(thres)}_{str(contrast)}_{str(args.model_seed)}'
                self.writer = SummaryWriter(
                    os.path.join(args.log_dir, subdir_name, stage_name,
                                 run_name))
                encoder, classifier, val_score = self._adapt_train_test(cluster_src_loader, cluster_tgt_loader, thres, contrast, args)
                performance_dict[thres][contrast] = {'encoder': encoder, 'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for thres, perf_dict in performance_dict.items():
            for contrast, ckpt_dict in perf_dict.items():
                if ckpt_dict['val_score'] > best_val_score:
                    best_val_score = ckpt_dict['val_score']
                    best_encoder = ckpt_dict['encoder']
                    best_classifier = ckpt_dict['classifier']
                print(f"thres: {thres} contrast: {contrast} val_score: {ckpt_dict['val_score']}")

        self.set_encoder_classifier(best_encoder, best_classifier)

    def _pseudo_loss(self, logits, thres, reduction='mean'):
        confidence, pseudo_labels = torch.max(F.softmax(logits.detach(), dim=1), dim=1)
        mask = (confidence > thres)
        loss = F.cross_entropy(logits[mask], pseudo_labels[mask], reduction=reduction)
        if reduction == 'mean':
            return loss
        elif reduction == 'sum':
            return loss, mask.sum().item()

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
