from copy import deepcopy
import argparse
import os
import pickle
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred.evaluate import Evaluator
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from dataset import temp_partition_arxiv
from utils import set_model_seed, get_device
from shared import train_stage_list, test_stage_list

import sys
sys.path.append("..")
from adapter import *
from model import TwoLayerGraphSAGE, TwoLayerMLP, Model

def train_epoch(encoder, mlp, optimizer, data):
    encoder.train()
    mlp.train()

    out, _ = mlp(encoder(data.x, data.edge_index))
    out = F.log_softmax(out, dim=1)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_epoch(encoder, mlp, data, evaluator):
    encoder.eval()
    mlp.eval()
    # print("data device:", data.x.device)
    # print("mlp device:", next(mlp.parameters()).device)
    # print("encoder device:", next(encoder.parameters()).device)

    out, _ = mlp(encoder(data.x, data.edge_index))
    out = F.log_softmax(out, dim=1)
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
    y_pred = out.argmax(dim=-1, keepdim=True)
    val_acc = evaluator.eval({
        'y_true': data.y[data.val_mask].unsqueeze(1),
        'y_pred': y_pred[data.val_mask],
    })['acc']

    return val_loss, val_acc

def train_source(data, device, evaluator, args, random=False):
    feat_dim = 128
    encoder = TwoLayerGraphSAGE(feat_dim, args.hidden_dim, args.emb_dim).to(device)
    mlp = TwoLayerMLP(args.emb_dim, args.emb_dim // 4, 40).to(device)
    if random:
        return encoder, mlp
    data = data.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(mlp.parameters()), lr=1e-3)

    best_acc = 0
    best_encoder = None
    best_mlp = None
    for e in range(1, args.train_epochs + 1):
        train_loss = train_epoch(encoder, mlp, optimizer, data)
        val_loss, val_acc = test_epoch(encoder, mlp, data, evaluator)
        print(
            f"Epoch:{e}/{args.train_epochs} Train Loss:{round(train_loss, 4)} Val Loss:{round(val_loss, 4)} Val Acc:{round(val_acc, 4)}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_encoder = deepcopy(encoder)
            best_mlp = deepcopy(mlp)

    encoder = deepcopy(best_encoder)
    mlp = deepcopy(best_mlp)
    return encoder, mlp

def main(args):
    set_model_seed(args.model_seed)
    device = get_device(args.gpuID)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.data_dir)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    class_num = 40
    evaluator = Evaluator(name='ogbn-arxiv')
    print("Partition data and train at source")
    src_data = temp_partition_arxiv(data, train_stage_list[0])
    print("Finish partition")
    encoder, mlp = train_source(src_data, device, evaluator, args, random=args.method=="random")
    acc_list = []

    @torch.no_grad()
    def dump_feature_y(data, stage_name):
        encoder.eval()
        mlp.eval()

        data = data.to(device)
        feature = encoder(data.x, data.edge_index)[data.train_mask]
        pred, _ = mlp(feature)
        y = torch.argmax(pred, dim=1)
        feature = feature.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        os.makedirs(os.path.join("feature_y", stage_name), exist_ok=True)

        feat_filename = args.method + "_" + str(args.model_seed) + "_feat.npy"
        y_filename = args.method + "_" + str(args.model_seed) + "_y.npy"
        np.save(os.path.join("feature_y", stage_name, feat_filename), feature)
        np.save(os.path.join("feature_y", stage_name, y_filename), y)

    if args.analyze_feat:
        dump_feature_y(src_data, "_".join([str(e) for e in train_stage_list[0]]))



    if args.method =='gst':
        model = Model(encoder, mlp)
        adapter = SinglegraphGST(model, device=device)
    elif args.method == "cbst" or args.method == "crst":
        model = Model(encoder, mlp)
        adapter = SinglegraphClassBalancedSelfTrainer(model, src_data, class_num, device)
    elif args.method == 'dann':
        adapter = SinglegraphDANNAdapter(encoder, mlp, src_data, args.emb_dim, device)
    elif args.method == 'jan':
        adapter = SinglegraphJANAdapter(encoder, mlp, src_data, device)
    elif args.method == "deep-coral":
        adapter = SinglegraphDeepCORALAdapter(encoder, mlp, src_data, device=device)
    elif args.method == "uda-gcn":
        adapter = SinglegraphUDAGCNAdapter(encoder, mlp, src_data, args.emb_dim, path_len=10, device=device)
    elif args.method == "dane":
        adapter = SinglegraphDANE(encoder, mlp, src_data, args.emb_dim, d_epochs=5, device=device)
    elif args.method == "gcst-fpl":
        adapter = SinglegraphGCSTFPL(encoder, mlp, args.emb_dim, src_data, device=device)
    elif args.method == "gcst-upl":
        adapter = SinglegraphGCSTUPL(encoder, mlp, args.emb_dim, src_data, device=device)
    elif args.method == "gcst-fpl-wo-con":
        adapter = SinglegraphGCSTFPLXCON(encoder, mlp, args.emb_dim, src_data, device=device)
    elif args.method == "gcst-upl-wo-con":
        adapter = SinglegraphGCSTUPLXCON(encoder, mlp, args.emb_dim, src_data, device=device)
    elif args.method == "gcst-fpl-wo-src":
        adapter = SinglegraphGCSTFPL(encoder, mlp, args.emb_dim, device=device)
    elif args.method == "gcst-upl-wo-src":
        adapter = SinglegraphGCSTUPL(encoder, mlp, args.emb_dim, device=device)
    elif args.method == "gcst-wo-pl":
        adapter = SinglegraphGCSTXPL(encoder, mlp, args.emb_dim, device=device)
    elif args.method == "gcst-fpl-direct":
        adapter = SinglegraphGCSTFPLDirect(encoder, mlp, args.emb_dim, src_data, device=device)
    elif args.method == "gcst-upl-direct":
        adapter = SinglegraphGCSTUPLDirect(encoder, mlp, args.emb_dim, src_data, device=device)






    for i, test_stage in enumerate(test_stage_list):
        print("Partition data and test at stage:", test_stage)
        test_loss, test_acc = test_epoch(encoder, mlp, temp_partition_arxiv(data, test_stage).to(device), evaluator)
        acc_list.append(test_acc)

        ckpt_subdir = os.path.join(args.ckpt_dir, str(train_stage_list[i][0]) + "_" + str(train_stage_list[i][1]) + "_" + str(train_stage_list[i][2])) # the checkpoint directory is named by the split the model is trained on, not evaluated on
        os.makedirs(ckpt_subdir, exist_ok=True)
        torch.save({"encoder": encoder,
                    "classifier": mlp},
                   os.path.join(ckpt_subdir, args.method + "_" + str(args.model_seed) + ".pt"))

        if i == len(test_stage_list) - 1:
            break

        train_stage = train_stage_list[i+1]

        if args.method == "fixed":
            continue

        print("Partition data and adapt at stage:", train_stage)
        tgt_data = temp_partition_arxiv(data, train_stage)

        stage_name = "_".join([str(e) for e in train_stage])


        if args.method == "gst":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            adapter.adapt(tgt_data, threshold_list, stage_name, args)
            model = adapter.get_model()
            encoder, mlp = model.get_encoder_classifier()
        elif args.method == "cbst":
            reg_weight_list = [0]
            adapter.adapt(tgt_data, reg_weight_list, stage_name, args)
            model = adapter.get_model()
            encoder, mlp = model.get_encoder_classifier()
        elif args.method == "crst":
            reg_weight_list = [1]
            adapter.adapt(tgt_data, reg_weight_list, stage_name, args)
            model = adapter.get_model()
            encoder, mlp = model.get_encoder_classifier()
        elif args.method == "dann":
            lambda_coeff_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            adapter.adapt(tgt_data, lambda_coeff_list, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "jan":
            jmmd_tradeoff_list = [0.5, 1, 5]
            adapter.adapt(tgt_data, jmmd_tradeoff_list, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "deep-coral":
            coral_tradeoff_list = [0.5, 1, 5]
            adapter.adapt(tgt_data, coral_tradeoff_list, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "uda-gcn":
            adapter.adapt(tgt_data, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-fpl" or args.method == "gcst-fpl-wo-src":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            contrast_list = [0.01, 0.05, 0.1, 0.5, 1]
            adapter.adapt(tgt_data, threshold_list, contrast_list, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-upl" or args.method == "gcst-upl-wo-src":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            contrast_list = [0.01, 0.05, 0.1, 0.5, 1]
            adapter.adapt(tgt_data, threshold_list, contrast_list, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-fpl-wo-con" or args.method == "gcst-upl-wo-con":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            adapter.adapt(tgt_data, threshold_list, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-wo-pl":
            contrast_list = [0.01, 0.05, 0.1, 0.5, 1]
            adapter.adapt(tgt_data, contrast_list, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-fpl-direct" or args.method == "gcst-upl-direct":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            contrast_list = [0.01, 0.05, 0.1, 0.5, 1]
            adapter.adapt(tgt_data, threshold_list, contrast_list, stage_name, args)
            encoder, mlp = adapter.get_adapted_encoder_classifier()
        elif args.method == "dane":
            adv_coeff_list = [0.1, 1] # k1
            ce_coeff_list = [1, 10] # k2
            adapter.adapt(tgt_data, adv_coeff_list, ce_coeff_list, stage_name, args)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "random":
            pass
        else:
            print("Unknown method")
            exit()
        if args.analyze_feat:
            dump_feature_y(tgt_data, stage_name)

    print(acc_list)
    print("Test Acc List:", acc_list, "Avg Acc:", sum(acc_list) / len(acc_list))
    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, args.method + "_" + str(args.model_seed) + "_acc_list"), "wb") as fp:
        pickle.dump(acc_list, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to data directory")
    parser.add_argument("--log_dir", type=str, help="path to log directory", default="runs")
    parser.add_argument("--ckpt_dir", type=str, help="path to model checkpoint directory", default="checkpoints")
    parser.add_argument("--result_dir", type=str, help="path to performance results directory", default="results")
    parser.add_argument("--method", type=str, help="adaptation method")
    parser.add_argument("--train_epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--adapt_epochs", type=int, help="number of adaptation epochs", default=500)
    parser.add_argument("--adapt_lr", type=float, help="adaptation learning rate", default=1e-3)
    parser.add_argument("--p_min", type=float, help="initial ratio of unlabeled data being pseudo-labeled", default=0.2)
    parser.add_argument("--p_max", type=float, help="final ratio of unlabeled data being pseudo-labeled", default=0.5)
    parser.add_argument("--p_inc", type=float, help="incremental value from p_min to p_max", default=0.05)
    parser.add_argument("--analyze_feat", help="whether save features", nargs='?', type=bool, const=1, default=0)
    parser.add_argument("--hidden_dim", type=int, help="GNN hidden layer dimension", default=128)
    parser.add_argument("--emb_dim", type=int, help="embedding dimension", default=256)
    parser.add_argument("--model_seed", type=int, help="random seed", default=42)
    parser.add_argument("--gpuID", type=int, help="which gpu to use", default=0)
    args = parser.parse_args()
    print(args)
    main(args)
