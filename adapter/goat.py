import collections
import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from pytorch_adapt.validators import IMValidator
from .adapter import DecoupledMultigraphAdapter
from .ot_util import generate_domains


class MultigraphGOATAdapter(DecoupledMultigraphAdapter):
    def __init__(self, encoder, classifier, prev_train_loader=None, prev_val_loader=None, device="cpu"):
        super().__init__(encoder, classifier, None, None, device)
        self.prev_train_loader = prev_train_loader
        self.prev_val_loader = prev_val_loader
        self.validator = IMValidator()

    def _adapt_train_epoch(self, classifier, feat, pseudo_label, confidence_mask, optimizer):
        classifier.train()
        logits, _ = classifier(feat)
        loss = F.nll_loss(F.log_softmax(logits[confidence_mask], dim=1), pseudo_label[confidence_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), logits

    @torch.no_grad()
    def _adapt_test_epoch(self, classifier, feat, pseudo_label, confidence_mask):
        classifier.eval()
        logits, _ = classifier(feat)
        loss = F.nll_loss(F.log_softmax(logits[confidence_mask], dim=1), pseudo_label[confidence_mask])
        return loss.item(), logits


    def _adapt_train_test(self, all_domain_feat, thres, args):
        classifier = deepcopy(self.classifier)
        steps = len(all_domain_feat) - 1
        global_epoch = 0
        for i in range(steps):
            # adaptation within intermediate domains
            train_feat = torch.from_numpy(all_domain_feat[i]).to(self.device)
            print("current number of generated samples for training:", train_feat.shape[0])
            val_feat = torch.from_numpy(all_domain_feat[i+1]).to(self.device)
            print("current number of generated samples for validation:", val_feat.shape[0])
            train_pseudo_label, train_confident_mask = self._pseudo_label(classifier, train_feat, thres)
            val_pseudo_label, val_confident_mask = self._pseudo_label(classifier, val_feat, thres)
            optimizer = torch.optim.Adam(list(classifier.parameters()), lr=args.adapt_lr)

            best_val_loss = np.inf
            best_val_score = None
            best_classifier = None
            patience = 10
            staleness = 0

            for e in range(1, args.adapt_epochs + 1):
                train_loss, train_logits = self._adapt_train_epoch(classifier, train_feat, train_pseudo_label, train_confident_mask, optimizer)
                val_loss, val_logits = self._adapt_test_epoch(classifier, val_feat, val_pseudo_label, val_confident_mask)
                train_score = self.validator(target_train={'logits': train_logits})
                val_score = self.validator(target_train={'logits': val_logits})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_score = val_score
                    best_classifier = deepcopy(classifier)
                    staleness = 0
                else:
                    staleness += 1
                print(
                    f'Thres: {thres} Epoch: {global_epoch} Train Loss: {round(train_loss, 3)} \n Val Loss: {round(val_loss, 3)}')

                self.writer.add_scalar("Total Loss/train", train_loss, global_epoch)
                self.writer.add_scalar("Total Loss/val", val_loss, global_epoch)
                self.writer.add_scalar("Score/train", train_score, global_epoch)
                self.writer.add_scalar("Score/val", val_score, global_epoch)
                if staleness > patience:
                    break

                global_epoch += 1

            classifier = deepcopy(best_classifier)

        return classifier, best_val_score # the validation score of the last feature set



    def adapt(self, tgt_train_loader, tgt_val_loader, threshold_list, interdomain_num_list, stage_name, args, subdir_name=""):
        performance_dict = collections.defaultdict(dict)
        for thres in threshold_list:
            for interdomain_num in interdomain_num_list:
                run_name = f'{args.method}_{str(thres)}_{str(interdomain_num)}_{str(args.model_seed)}'
                self.writer = SummaryWriter(
                    os.path.join(args.log_dir, subdir_name, stage_name,
                                 run_name))
                # compute tgt (t-1) and tgt (t) features
                prev_train_feat = self._encode_feature(self.prev_train_loader)
                prev_val_feat = self._encode_feature(self.prev_val_loader)
                cur_train_feat = self._encode_feature(tgt_train_loader)
                cur_val_feat = self._encode_feature(tgt_val_loader)
                # generate intermediate features
                    # combine prev train and prev val feats to prev feats
                prev_feat = np.concatenate([prev_train_feat, prev_val_feat], axis=0)
                # generate intermediate features to a list, compute the transportation plan from prev feats to cur train feat
                all_domain_feat = generate_domains(interdomain_num, prev_feat, cur_train_feat, entry_cutoff=0.01)
                all_domain_feat.append(cur_val_feat) # the last item is only for validation
                # gradual self training on intermediate features
                classifier, val_score = self._adapt_train_test(all_domain_feat, thres, args)
                performance_dict[thres][interdomain_num] = {'classifier': classifier, 'val_score': val_score}

        best_val_score = -np.inf
        best_classifier = None
        for thres, thres_dict in performance_dict.items():
            for interdomain_num, ckpt_dict in thres_dict.items():
                if ckpt_dict['val_score'] > best_val_score:
                    best_val_score = ckpt_dict['val_score']
                    best_classifier = ckpt_dict['classifier']
                print(f"thres: {thres} interdomain num: {interdomain_num} val_score: {ckpt_dict['val_score']}")

        self.set_encoder_classifier(self.encoder, best_classifier)
        self.set_prev_dataloader(tgt_train_loader, tgt_val_loader)


    @torch.no_grad()
    def _pseudo_label(self, classifier, feat, thres):
        classifier.eval()
        pseudo_y, _ = classifier(feat)
        pseudo_y = F.softmax(pseudo_y, dim=1)
        pseudo_y_confidence, pseudo_y_hard_label = torch.max(pseudo_y, dim=1)
        pseudo_mask = pseudo_y_confidence > thres
        return pseudo_y_hard_label, pseudo_mask

    @torch.no_grad()
    def _encode_feature(self, dataloader):
        # return a N * d pytorch tensor
        feature = []
        for data in dataloader:
            data = data.to(self.device)
            feature.append(self.encoder(data.x, data.edge_index))
        feature = torch.cat(feature)
        return feature.cpu().detach().numpy()

    def set_prev_dataloader(self, train_loader, val_loader):
        self.prev_train_loader = train_loader
        self.prev_val_loader = val_loader
