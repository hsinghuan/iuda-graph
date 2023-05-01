import os
import pickle
import argparse
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utils import set_model_seed, get_device
from dataset import load_sbm_dataset
from shared import train_stage_list, test_stage_list
import numpy as np

import sys
sys.path.append("..")
from adapter import *
from model import TwoLayerGraphSAGE, TwoLayerMLP, Model

def train(encoder, mlp, optimizer, loader, device="cpu"):
    encoder.train()
    mlp.train()
    optimizer.zero_grad()

    total_train_loss = 0
    total_node_num = 0
    for data in loader:
        data = data.to(device)
        node_num = data.x.shape[0]
        out, _ = mlp(encoder(data.x, data.edge_index))
        loss = F.nll_loss(F.log_softmax(out, dim=1), data.y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * node_num
        total_node_num += node_num

    total_train_loss /= total_node_num
    return total_train_loss


@torch.no_grad()
def test(encoder, mlp, loader, device="cpu"):
    encoder.eval()
    mlp.eval()
    total_val_loss = 0
    total_correct = 0
    total_node_num = 0
    for data in loader:
        data = data.to(device)
        node_num = data.x.shape[0]
        out, _ = mlp(encoder(data.x, data.edge_index))
        loss = F.nll_loss(F.log_softmax(out, dim=1), data.y)
        y_pred = torch.argmax(out, dim=1)

        total_val_loss += loss.item() * node_num
        total_correct += torch.eq(y_pred, data.y).sum().item()
        total_node_num += node_num

    total_val_loss /= total_node_num
    total_acc = total_correct / total_node_num
    return total_val_loss, total_acc

def train_source(source_stage, device, args, random=False):
    train_data_list = [load_sbm_dataset(args.data_dir, i) for i in range(source_stage[0], source_stage[1])]
    val_data_list = [load_sbm_dataset(args.data_dir, i) for i in range(source_stage[1], source_stage[2])]


    feat_dim = train_data_list[0].x.shape[1]
    hidden_dim = args.hidden_dim
    class_num = 3 if args.shift == "sbmb" or "sbmb_ys" else 2
    emb_dim = args.emb_dim
    encoder = TwoLayerGraphSAGE(feat_dim, hidden_dim, emb_dim).to(device)
    mlp = TwoLayerMLP(emb_dim, emb_dim // 4, class_num).to(device)
    if random:
        return encoder, mlp, None, None
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(mlp.parameters()), lr=1e-3)

    train_loader = DataLoader(dataset=train_data_list, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_data_list, batch_size=1, shuffle=False)

    best_acc = 0
    best_encoder = None
    best_mlp = None
    for e in range(1, args.train_epochs + 1):
        train_loss = train(encoder, mlp, optimizer, train_loader, device)
        val_loss, val_acc = test(encoder, mlp, val_loader, device)
        print(
            f"Epoch:{e}/{args.train_epochs} Train Loss:{round(train_loss, 4)} Val Loss:{round(val_loss, 4)} Val Acc:{round(val_acc, 4)}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_encoder = deepcopy(encoder)
            best_mlp = deepcopy(mlp)
    encoder = deepcopy(best_encoder)
    mlp = deepcopy(best_mlp)
    return encoder, mlp, train_loader, val_loader

def main(args):
    assert args.shift in args.data_dir
    set_model_seed(args.model_seed)
    device = get_device(args.gpuID)

    encoder, mlp, src_train_loader, src_val_loader = train_source(train_stage_list[0], device, args, args.method == "random")
    class_num = 3 if args.shift == "sbmb" or "sbmb_ys" else 2

    @torch.no_grad()
    def dump_feature_y(loader, stage_name, subdir_name):
        encoder.eval()
        mlp.eval()
        feature = []
        y = []
        for data in loader:
            data = data.to(device)
            f = encoder(data.x, data.edge_index)
            pred, _ = mlp(f)
            feature.append(f)
            y.append(torch.argmax(pred, dim=1))
        feature = torch.cat(feature, dim=0).detach().cpu().numpy()
        y = torch.cat(y).detach().cpu().numpy()
        os.makedirs(os.path.join("feature_y", stage_name, subdir_name), exist_ok=True)

        feat_filename = args.method + "_" + str(args.model_seed) + "_feat.npy"
        y_filename = args.method + "_" + str(args.model_seed) + "_y.npy"
        np.save(os.path.join("feature_y", stage_name, subdir_name, feat_filename), feature)
        np.save(os.path.join("feature_y", stage_name, subdir_name, y_filename), y)

    if args.analyze_feat:
        dump_feature_y(src_train_loader, "_".join([str(e) for e in train_stage_list[0]]), args.shift)

    if args.method == "gst":
        model = Model(encoder, mlp)
        adapter = MultigraphGST(model, device=device)
    elif args.method == "cbst" or args.method == "crst":
        model = Model(encoder, mlp)
        adapter = MultigraphClassBalancedSelfTrainer(model, src_train_loader, src_val_loader, class_num, device)
    elif args.method == "dann":
        adapter = MultigraphDANNAdapter(encoder, mlp, src_train_loader, src_val_loader, args.emb_dim, device=device)
    elif args.method == "jan":
        adapter = MultigraphJANAdapter(encoder, mlp, src_train_loader, src_val_loader, device=device)
    elif args.method == "deep-coral":
        adapter = MultigraphDeepCORALAdapter(encoder, mlp, src_train_loader, src_val_loader, device=device)
    elif args.method == "uda-gcn":
        adapter = MultigraphUDAGCNAdapter(encoder, mlp, src_train_loader, src_val_loader, args.emb_dim, path_len=5,
                                          device=device)
    elif args.method == "dane":
        adapter = MultigraphDANE(encoder, mlp, src_train_loader, src_val_loader, args.emb_dim, d_epochs=5, device=device)
    elif args.method == "gcst-fpl":
        adapter = MultigraphGCSTFPL(encoder, mlp, args.emb_dim, src_train_loader, src_val_loader, device=device)
    elif args.method == "gcst-upl":
        adapter = MultigraphGCSTUPL(encoder, mlp, args.emb_dim, src_train_loader, src_val_loader, device=device)
    elif args.method == "gcst-fpl-wo-src":
        adapter = MultigraphGCSTFPL(encoder, mlp, args.emb_dim, device=device)
    elif args.method == "gcst-upl-wo-src":
        adapter = MultigraphGCSTUPL(encoder, mlp, args.emb_dim, device=device)
    elif args.method == "gcst-fpl-wo-con":
        adapter = MultigraphGCSTFPLXCON(encoder, mlp, args.emb_dim, src_train_loader, src_val_loader, device=device)
    elif args.method == "gcst-upl-wo-con":
        adapter = MultigraphGCSTUPLXCON(encoder, mlp, args.emb_dim, src_train_loader, src_val_loader, device=device)
    elif args.method == "gcst-wo-pl":
        adapter = MultigraphGCSTXPL(encoder, mlp, args.emb_dim, src_train_loader, src_val_loader, device=device)
    elif args.method == "gcst-upl-direct":
        adapter = MultigraphGCSTUPLDirect(encoder, mlp, args.emb_dim, src_train_loader, src_val_loader, device=device)
    elif args.method == "gcst-fpl-direct":
        adapter = MultigraphGCSTFPLDirect(encoder, mlp, args.emb_dim, src_train_loader, src_val_loader, device=device)


    test_acc_list = []
    for j, test_stage in enumerate(test_stage_list):
        print("Test Stage:", test_stage)
        test_loader = DataLoader(dataset=[load_sbm_dataset(args.data_dir, i) for i in range(test_stage[0], test_stage[1])], batch_size=1, shuffle=False)
        test_loss, test_acc = test(encoder, mlp, test_loader, device)

        ckpt_subdir = os.path.join(args.ckpt_dir, args.shift,
                                   str(train_stage_list[j][0]) + "_" + str(train_stage_list[j][1]) + "_" + str(
                                       train_stage_list[j][
                                           2]))  # the checkpoint directory is named by the split the model is trained on, not evaluated on
        os.makedirs(ckpt_subdir, exist_ok=True)
        torch.save({"encoder": encoder,
                    "classifier": mlp},
                   os.path.join(ckpt_subdir, args.method + "_" + str(args.model_seed) + ".pt"))

        test_acc_list.append(test_acc)
        if j == len(test_stage_list) - 1:
            break

        print("Adapt Stage:", train_stage_list[j+1])
        tgt_train_loader = DataLoader(dataset=[load_sbm_dataset(args.data_dir, i) for i in range(train_stage_list[j+1][0], train_stage_list[j+1][1])], batch_size=1, shuffle=True)
        tgt_val_loader = DataLoader(dataset=[load_sbm_dataset(args.data_dir, i) for i in range(train_stage_list[j+1][1], train_stage_list[j+1][2])], batch_size=1, shuffle=True)

        stage_name = "_".join([str(e) for e in train_stage_list[j + 1]])


        if args.method == "gst":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            adapter.adapt(tgt_train_loader, tgt_val_loader, threshold_list, stage_name, args, subdir_name=args.shift)
            model = adapter.get_model()
            encoder, mlp = model.get_encoder_classifier()
        elif args.method == "cbst":
            reg_weight_list = [0]
            adapter.adapt(tgt_train_loader, tgt_val_loader, reg_weight_list, stage_name, args, subdir_name=args.shift)
            model = adapter.get_model()
            encoder, mlp = model.get_encoder_classifier()
        elif args.method == "crst":
            reg_weight_list = [0.5]
            adapter.adapt(tgt_train_loader, tgt_val_loader, reg_weight_list, stage_name, args, subdir_name=args.shift)
            model = adapter.get_model()
            encoder, mlp = model.get_encoder_classifier()
        elif args.method == "dann":
            lambda_coeff_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            adapter.adapt(tgt_train_loader, tgt_val_loader, lambda_coeff_list, stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "jan":
            jmmd_tradeoff_list = [0.5, 1, 5]
            adapter.adapt(tgt_train_loader, tgt_val_loader, jmmd_tradeoff_list, stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "deep-coral":
            coral_tradeoff_list = [0.5, 1, 5]
            adapter.adapt(tgt_train_loader, tgt_val_loader, coral_tradeoff_list, stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "uda-gcn":
            adapter.adapt(tgt_train_loader, tgt_val_loader, stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "dane":
            adv_coeff_list = [0.1, 1] # k1
            ce_coeff_list = [1, 10] # k2
            adapter.adapt(tgt_train_loader, tgt_val_loader, adv_coeff_list, ce_coeff_list, stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-fpl" or args.method == "gcst-fpl-wo-src":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            contrast_list = [0.01, 0.05, 0.1, 0.5, 1]
            adapter.adapt(tgt_train_loader, tgt_val_loader, threshold_list, contrast_list,
                          stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-upl" or args.method == "gcst-upl-wo-src":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            contrast_list = [0.01, 0.05, 0.1, 0.5, 1]
            adapter.adapt(tgt_train_loader, tgt_val_loader, threshold_list, contrast_list,
                          stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-fpl-wo-con" or args.method == "gcst-upl-wo-con":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            adapter.adapt(tgt_train_loader, tgt_val_loader, threshold_list,
                          stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-wo-pl":
            contrast_list = [0.01, 0.05, 0.1, 0.5, 1]
            adapter.adapt(tgt_train_loader, tgt_val_loader, contrast_list,
                          stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_encoder_classifier()
        elif args.method == "gcst-upl-direct" or args.method == "gcst-fpl-direct":
            threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            contrast_list = [0.01, 0.05, 0.1, 0.5, 1]
            adapter.adapt(tgt_train_loader, tgt_val_loader, threshold_list, contrast_list,
                          stage_name, args, subdir_name=args.shift)
            encoder, mlp = adapter.get_adapted_encoder_classifier()
        elif args.method == "fixed":
            pass
        elif args.method == "random":
            pass
        if args.analyze_feat:
            dump_feature_y(tgt_train_loader, stage_name, args.shift)

    print("Test Acc List:", test_acc_list)
    print("Avg Acc List:", sum(test_acc_list)/len(test_acc_list))
    os.makedirs(os.path.join(args.result_dir, args.shift), exist_ok=True)
    with open(os.path.join(args.result_dir, args.shift, args.method + "_" + str(args.model_seed) + "_acc_list"), "wb") as fp:
        pickle.dump(test_acc_list, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", type=str, help="block connectivity shift or feature shift")
    parser.add_argument("--data_dir", type=str, help="path to data directory")
    parser.add_argument("--log_dir", type=str, help="path to log directory", default="runs")
    parser.add_argument("--ckpt_dir", type=str, help="path to model checkpoint directory", default="checkpoints")
    parser.add_argument("--result_dir", type=str, help="path to performance results directory", default="results")
    parser.add_argument("--method", type=str, help="adaptation method")
    parser.add_argument("--train_epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--adapt_epochs", type=int, help="number of adaptation epochs", default=500)
    parser.add_argument("--adapt_lr", type=float, help="learning rate for adaptation optimizer", default=1e-3)
    parser.add_argument("--p_min", type=float, help="initial ratio of unlabeled data being pseudo-labeled", default=0.2)
    parser.add_argument("--p_max", type=float, help="final ratio of unlabeled data being pseudo-labeled", default=0.5)
    parser.add_argument("--p_inc", type=float, help="incremental value from p_min to p_max", default=0.05)
    parser.add_argument("--analyze_feat", help="whether save features", nargs='?', type=bool, const=1, default=0)
    parser.add_argument("--hidden_dim", type=int, help="GNN hidden layer dimension", default=64)
    parser.add_argument("--emb_dim", type=int, help="embedding dimension", default=64)
    parser.add_argument("--model_seed", type=int, help="random seed", default=42)
    parser.add_argument("--gpuID", type=int, help="which gpu to use", default=0)
    args = parser.parse_args()
    main(args)