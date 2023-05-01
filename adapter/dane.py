from copy import deepcopy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
from torch_geometric.utils import degree, k_hop_subgraph, structured_negative_sampling
from torch_geometric.data import Data
from pytorch_adapt.validators import IMValidator
from .adapter import DecoupledMultigraphAdapter, DecoupledSinglegraphAdapter


class MultigraphDANE(DecoupledMultigraphAdapter):
    def __init__(self, encoder, classifier, src_train_loader, src_val_loader, emb_dim, d_epochs, device="cpu"):
        super().__init__(encoder, classifier, src_train_loader, src_val_loader, device)
        self.emb_dim = emb_dim
        self.d_epochs = d_epochs
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, domain_classifier, tgt_train_loader, optimizer, adv_coeff, ce_coeff, k=5, pos_num=64):
        encoder.train()
        classifier.train()
        domain_classifier.eval()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        total_loss = 0
        total_line_loss = 0
        total_adv_loss = 0
        total_src_label_loss = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            src_data = src_iter.next().to(self.device)
            src_f = encoder(src_data.x, src_data.edge_index)
            tgt_data = tgt_iter.next().to(self.device)
            tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
            # print(_, "compute adv loss")
            src_dom_pred = domain_classifier(src_f)
            tgt_dom_pred = domain_classifier(tgt_f)
            adv_loss = (tgt_dom_pred ** 2).mean() + ((src_dom_pred - 1) ** 2).mean()
            # print(_, "compute line loss")
            src_line_loss = line_unsup_loss(src_f, src_data.edge_index, src_data.x.shape[0], k=k, pos_num=pos_num)
            tgt_line_loss = line_unsup_loss(tgt_f, tgt_data.edge_index, tgt_data.x.shape[0], k=k, pos_num=pos_num)
            line_loss = src_line_loss + tgt_line_loss
            # print(_, "compute src lbl loss")
            src_cls_output, _ = classifier(src_f)
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                src_mask = src_data.mask
            else:
                src_mask = torch.ones(src_cls_output.shape[0], dtype=torch.bool)
            src_label_loss = F.nll_loss(F.log_softmax(src_cls_output[src_mask], dim=1), src_data.y[src_mask])


            loss = line_loss + adv_coeff * adv_loss + ce_coeff * src_label_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tgt_cls_output, _ = classifier(tgt_f)
            total_loss += loss.item()
            total_line_loss += line_loss.item()
            total_adv_loss += adv_loss.item()
            total_src_label_loss += src_label_loss.item()
            total_src_logits.append(src_cls_output)
            total_tgt_logits.append(tgt_cls_output)

        total_loss /= len_dataloader
        total_line_loss /= len_dataloader
        total_adv_loss /= len_dataloader
        total_src_label_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_line_loss, total_adv_loss, total_src_label_loss, total_src_logits, total_tgt_logits

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, domain_classifier, tgt_val_loader, adv_coeff, ce_coeff, k=5, pos_num=64):
        encoder.eval()
        classifier.eval()
        domain_classifier.eval()

        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader))
        src_iter = iter(self.src_val_loader)
        tgt_iter = iter(tgt_val_loader)
        total_loss = 0
        total_line_loss = 0
        total_adv_loss = 0
        total_src_label_loss = 0
        total_src_logits = []
        total_tgt_logits = []
        for _ in range(len_dataloader):
            src_data = src_iter.next().to(self.device)
            src_f = encoder(src_data.x, src_data.edge_index)
            tgt_data = tgt_iter.next().to(self.device)
            tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
            src_dom_pred = domain_classifier(src_f)
            tgt_dom_pred = domain_classifier(tgt_f)

            adv_loss = (tgt_dom_pred ** 2).mean() + ((src_dom_pred - 1) ** 2).mean()

            src_line_loss = line_unsup_loss(src_f, src_data.edge_index, src_data.x.shape[0], k=k, pos_num=pos_num)
            tgt_line_loss = line_unsup_loss(tgt_f, tgt_data.edge_index, tgt_data.x.shape[0], k=k, pos_num=pos_num)
            line_loss = src_line_loss + tgt_line_loss

            src_cls_output, _ = classifier(src_f)
            if 'mask' in src_data:  # e.g. elliptic (not all nodes are labeled)
                src_mask = src_data.mask
            else:
                src_mask = torch.ones(src_cls_output.shape[0], dtype=torch.bool)
            src_label_loss = F.nll_loss(F.log_softmax(src_cls_output[src_mask], dim=1), src_data.y[src_mask])


            loss = line_loss + adv_coeff * adv_loss + ce_coeff * src_label_loss

            tgt_cls_output, _ = classifier(tgt_f)
            total_loss += loss.item()
            total_line_loss += line_loss.item()
            total_adv_loss += adv_loss.item()
            total_src_label_loss += src_label_loss.item()
            total_src_logits.append(src_cls_output)
            total_tgt_logits.append(tgt_cls_output)

        total_loss /= len_dataloader
        total_line_loss /= len_dataloader
        total_adv_loss /= len_dataloader
        total_src_label_loss /= len_dataloader
        total_src_logits = torch.cat(total_src_logits)
        total_tgt_logits = torch.cat(total_tgt_logits)
        return total_loss, total_line_loss, total_adv_loss, total_src_label_loss, total_src_logits, total_tgt_logits

    def _train_domain_classifier(self, encoder, domain_classifier, tgt_train_loader, optimizer_d, batch_size=64):
        encoder.eval()
        domain_classifier.train()
        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        total_loss = 0
        for _ in range(len_dataloader):
            src_data = src_iter.next().to(self.device)
            src_emb = encoder(src_data.x, src_data.edge_index)

            tgt_data = tgt_iter.next().to(self.device)
            tgt_emb = encoder(tgt_data.x, tgt_data.edge_index)

            # src_idx = torch.randint(high=src_emb.shape[0], size=(batch_size*8,))
            # tgt_idx = torch.randint(high=src_emb.shape[0], size=(batch_size*8,))
            src_pred = domain_classifier(src_emb)
            tgt_pred = domain_classifier(tgt_emb)

            loss = (src_pred ** 2).mean() + ((tgt_pred - 1) ** 2).mean()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()
            total_loss += loss.item()

        total_loss /= len_dataloader
        return total_loss



    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, adv_coeff, ce_coeff, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        domain_classifier = NetD(self.emb_dim).to(self.device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=args.adapt_lr)
        optimizer_d = torch.optim.Adam(domain_classifier.parameters(), lr=args.adapt_lr)
        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for e in range(1, args.adapt_epochs + 1):
            # print("update domain classifier")
            for d in range(1, self.d_epochs + 1):
                domain_cls_loss = self._train_domain_classifier(encoder, domain_classifier, tgt_train_loader, optimizer_d)
            # print("update encoder, classifier")
            train_loss, train_line_loss, train_adv_loss, train_src_label_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, domain_classifier, tgt_train_loader, optimizer, adv_coeff, ce_coeff)
            # print("validate encoder, classifier")
            val_loss, val_line_loss, val_adv_loss, val_src_label_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                encoder, classifier, domain_classifier, tgt_val_loader, adv_coeff, ce_coeff)
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
                f'Adv Coeff: {adv_coeff} CE Coeff: {ce_coeff} Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train LINE Loss: {round(train_line_loss, 3)} Train Adv Loss: {round(train_adv_loss, 3)} Train Src Lbl Loss: {round(train_src_label_loss, 3)} \n \
                                     Val Loss: {round(val_loss, 3)} Val LINE Loss: {round(val_line_loss, 3)} Val Adv Loss: {round(val_adv_loss, 3)} Val Src Lbl Loss: {round(val_src_label_loss, 3)}')
            # writer
            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("LINE Loss/train", train_line_loss, e)
            self.writer.add_scalar("LINE Loss/val", val_line_loss, e)
            self.writer.add_scalar("Adv Loss/train", train_adv_loss, e)
            self.writer.add_scalar("Adv Loss/val", val_adv_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_src_label_loss, e)
            self.writer.add_scalar("Source Label Loss/val", val_src_label_loss, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, best_val_score

    def adapt(self, tgt_train_loader, tgt_val_loader, adv_coeff_list, ce_coeff_list, stage_name, args, subdir_name=""):
        performance_dict = dict()
        for adv_coeff in adv_coeff_list:
            for ce_coeff in ce_coeff_list:
                run_name = f'{args.method}_{str(adv_coeff)}_{str(ce_coeff)}_{str(args.model_seed)}'
                self.writer = SummaryWriter(
                    os.path.join(args.log_dir, subdir_name, stage_name, run_name))
                encoder, classifier, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, adv_coeff, ce_coeff, args)
                performance_dict[(adv_coeff, ce_coeff)] = {'encoder': encoder, 'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        best_encoder = None
        best_classifier = None
        for (adv_coeff, ce_coeff), perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_encoder = perf_dict['encoder']
                best_classifier = perf_dict['classifier']
            print(f"Adv Coeff: {adv_coeff} CE Coeff: {ce_coeff} val_score: {perf_dict['val_score']}")
        self.set_encoder_classifier(best_encoder, best_classifier)


class SinglegraphDANE(DecoupledSinglegraphAdapter):
    def __init__(self, encoder, classifier, src_data, emb_dim, d_epochs, device="cpu"):
        super().__init__(encoder, classifier, src_data, device)
        self.emb_dim = emb_dim
        self.d_epochs = d_epochs
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, domain_classifier, tgt_data, optimizer, adv_coeff, ce_coeff, k=5, pos_num=64):
        encoder.train()
        classifier.train()
        domain_classifier.train()

        src_f = encoder(self.src_data.x, self.src_data.edge_index)
        tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
        src_dom_pred = domain_classifier(src_f[self.src_data.train_mask])
        tgt_dom_pred = domain_classifier(tgt_f[tgt_data.train_mask])
        adv_loss = (tgt_dom_pred ** 2).mean() + ((src_dom_pred - 1) ** 2).mean()
        src_line_loss = line_unsup_loss(src_f, self.src_data.edge_index, self.src_data.x.shape[0], k=k, pos_num=pos_num)
        tgt_line_loss = line_unsup_loss(tgt_f, tgt_data.edge_index, tgt_data.x.shape[0], k=k, pos_num=pos_num)
        line_loss = src_line_loss + tgt_line_loss
        src_cls_output, _ = classifier(src_f[self.src_data.train_mask])
        src_label_loss = F.nll_loss(F.log_softmax(src_cls_output, dim=1), self.src_data.y[self.src_data.train_mask])


        loss = line_loss + adv_coeff * adv_loss + ce_coeff * src_label_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tgt_cls_output, _ = classifier(tgt_f[tgt_data.train_mask])

        return loss.item(), line_loss.item(), adv_loss.item(), src_label_loss.item(), src_cls_output, tgt_cls_output

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, domain_classifier, tgt_data, adv_coeff, ce_coeff, k=5, pos_num=64):
        encoder.eval()
        classifier.eval()
        domain_classifier.eval()
        src_f = encoder(self.src_data.x, self.src_data.edge_index)
        tgt_f = encoder(tgt_data.x, tgt_data.edge_index)
        src_dom_pred = domain_classifier(src_f[self.src_data.val_mask])
        tgt_dom_pred = domain_classifier(tgt_f[tgt_data.val_mask])
        adv_loss = (tgt_dom_pred ** 2).mean() + ((src_dom_pred - 1) ** 2).mean()
        src_line_loss = line_unsup_loss(src_f, self.src_data.edge_index, self.src_data.x.shape[0], k=k, pos_num=pos_num)
        tgt_line_loss = line_unsup_loss(tgt_f, tgt_data.edge_index, tgt_data.x.shape[0], k=k, pos_num=pos_num)
        line_loss = src_line_loss + tgt_line_loss
        src_cls_output, _ = classifier(src_f[self.src_data.val_mask])
        src_label_loss = F.nll_loss(F.log_softmax(src_cls_output, dim=1), self.src_data.y[self.src_data.val_mask])

        loss = line_loss + adv_coeff * adv_loss + ce_coeff * src_label_loss

        tgt_cls_output, _ = classifier(tgt_f[tgt_data.val_mask])

        return loss.item(), line_loss.item(), adv_loss.item(), src_label_loss.item(), src_cls_output, tgt_cls_output


    def _train_domain_classifier(self, encoder, domain_classifier, tgt_data, optimizer_d, batch_size=64):
        encoder.eval()
        domain_classifier.train()

        src_emb = encoder(self.src_data.x, self.src_data.edge_index)[self.src_data.train_mask]
        tgt_emb = encoder(tgt_data.x, tgt_data.edge_index)[tgt_data.train_mask]

        # src_idx = torch.randint(high=src_emb.shape[0], size=(batch_size*8,))
        # tgt_idx = torch.randint(high=src_emb.shape[0], size=(batch_size*8,))

        src_pred = domain_classifier(src_emb)
        tgt_pred = domain_classifier(tgt_emb)

        loss = (src_pred ** 2).mean() + ((tgt_pred - 1) ** 2).mean()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        return loss.item()

    def _adapt_train_test(self, tgt_data, adv_coeff, ce_coeff, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        domain_classifier = NetD(self.emb_dim).to(self.device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=args.adapt_lr)
        optimizer_d = torch.optim.Adam(domain_classifier.parameters(), lr=args.adapt_lr)
        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for e in range(1, args.adapt_epochs + 1):
            for d in range(1, self.d_epochs + 1):
                domain_cls_loss = self._train_domain_classifier(encoder, domain_classifier, tgt_data, optimizer_d)
            train_loss, train_line_loss, train_adv_loss, train_src_label_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(
                encoder, classifier, domain_classifier, tgt_data, optimizer, adv_coeff, ce_coeff)
            val_loss, val_line_loss, val_adv_loss, val_src_label_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(
                encoder, classifier, domain_classifier, tgt_data, adv_coeff, ce_coeff)
            train_src_score = self.validator(target_train={'logits': train_src_logits})
            train_tgt_score = self.validator(target_train={'logits': train_tgt_logits})
            val_src_score = self.validator(target_train={'logits': val_src_logits})
            val_tgt_score = self.validator(target_train={'logits': val_tgt_logits})
            # val_score = self.validator(target_train={'logits': torch.cat([val_src_logits, val_tgt_logits])})

            if val_tgt_score > best_val_score: # only record checkpoints after 10
                # if val_loss_src_label < best_val_loss_src_label:
                best_val_score = val_tgt_score
                # best_val_loss_src_label = val_loss_src_label
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)

            print(
                f'Adv Coeff: {adv_coeff} CE Coeff: {ce_coeff} Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train LINE Loss: {round(train_line_loss, 3)} Train Adv Loss: {round(train_adv_loss, 3)} Train Src Lbl Loss: {round(train_src_label_loss, 3)} \n \
                                     Val Loss: {round(val_loss, 3)} Val LINE Loss: {round(val_line_loss, 3)} Val Adv Loss: {round(val_adv_loss, 3)} Val Src Lbl Loss: {round(val_src_label_loss, 3)}')
            # writer
            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("LINE Loss/train", train_line_loss, e)
            self.writer.add_scalar("LINE Loss/val", val_line_loss, e)
            self.writer.add_scalar("Adv Loss/train", train_adv_loss, e)
            self.writer.add_scalar("Adv Loss/val", val_adv_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_src_label_loss, e)
            self.writer.add_scalar("Source Label Loss/val", val_src_label_loss, e)
            self.writer.add_scalar("Target Score/train", train_tgt_score, e)
            self.writer.add_scalar("Target Score/val", val_tgt_score, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, best_val_score


    def adapt(self, tgt_data, adv_coeff_list, ce_coeff_list, stage_name, args, subdir_name=""):
        tgt_data = self._extract_k_hop(tgt_data, 2).to(self.device)

        performance_dict = dict()
        for adv_coeff in adv_coeff_list:
            for ce_coeff in ce_coeff_list:
                run_name = f'{args.method}_{str(adv_coeff)}_{str(ce_coeff)}_{str(args.model_seed)}'
                self.writer = SummaryWriter(
                    os.path.join(args.log_dir, subdir_name, stage_name, run_name))
                encoder, classifier, val_score = self._adapt_train_test(tgt_data, adv_coeff, ce_coeff, args)
                performance_dict[(adv_coeff, ce_coeff)] = {'encoder': encoder, 'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        best_encoder = None
        best_classifier = None
        for (adv_coeff, ce_coeff), perf_dict in performance_dict.items():
            if perf_dict['val_score'] > best_val_score:
                best_val_score = perf_dict['val_score']
                best_encoder = perf_dict['encoder']
                best_classifier = perf_dict['classifier']
            print(f"Adv Coeff: {adv_coeff} CE Coeff: {ce_coeff} val_score: {perf_dict['val_score']}")
        self.set_encoder_classifier(best_encoder, best_classifier)

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



LAMBDA = 10

class NetD(nn.Module):
    def __init__(self, nhid, opt=None):
        super(NetD, self).__init__()
        self.emb_dim = nhid
        if opt is not None:
            self.dis_layers = opt['dis_layers']
            self.dis_hid_dim = opt['dis_hid_dim']
            self.dis_dropout = opt['dis_dropout']
            self.dis_input_dropout = opt['dis_input_dropout']
        else:
            self.dis_layers = 2
            self.dis_hid_dim = 4*nhid
            self.dis_dropout = 0.1
            self.dis_input_dropout = 0.1

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dis_dropout))
        #layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)



    def forward(self, x):
        return self.layers(x).view(-1)

    def calc_gradient_penalty(self, real_data, fake_data, BATCH_SIZE, use_cuda):
        # print "real_data: ", real_data.size(), fake_data.size()
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, real_data.nelement() / BATCH_SIZE).contiguous().view(BATCH_SIZE, self.emb_dim)
        alpha = alpha.cuda() if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if use_cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(
                                  ) if use_cuda else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

def line_unsup_loss(emb: torch.Tensor, edge_index: torch.LongTensor, num_nodes: int, k: int, pos_num: int = None):
    edge_num = edge_index.shape[1]
    if pos_num:
        pos_idx = torch.randint(high=edge_num, size=(pos_num,)) # only pos_num edges will be indexed
    else:
        pos_idx = torch.ones((edge_num,), dtype=torch.long) # all edges will be indexed


    sampled_edge_index = edge_index[:,pos_idx]
    emb_u = emb[sampled_edge_index[0,:]]
    emb_v = emb[sampled_edge_index[1,:]]

    # follow the DANE implementation which randomly samples nodes according to node weight as negative samples
    node_weight = degree(edge_index[0], num_nodes)
    node_weight = node_weight ** 0.75

    emb_neg_list = [emb[torch.multinomial(node_weight, pos_num, replacement=False)] for _ in range(k)]
    # for _ in range(k):
    #     # print(_, "negative sampling")
    #     u_idx, v_idx, neg_idx = structured_negative_sampling(edge_index) # use full edge index to search negative sample
    #     # print("sampled u idx:", u_idx[pos_idx])
    #     # print("sampled v idx:", v_idx[pos_idx])
    #     # print("sampled neg idx:", neg_idx[pos_idx])
    #     emb_neg_list.append(emb[neg_idx[pos_idx]]) # only append the negative samples corresponding to the sampled positive nodes
    # print("computer inner line loss")
    pos = torch.sum(torch.mul(emb_u, emb_v), dim=1)
    neg = [torch.sum(torch.mul(emb_u, emb_neg_list[i]) * (-1), dim=1) for i in range(k)]
    loss = - torch.sum(F.logsigmoid(pos))
    for i in range(k):
        loss = loss - torch.sum(F.logsigmoid(neg[i]))
    return loss / (len(pos_idx) * k)