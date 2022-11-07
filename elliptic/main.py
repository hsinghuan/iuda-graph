from copy import deepcopy
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from dataset import load_nc_dataset
from utils import set_model_seed, get_device
from model import TwoLayerGraphSAGE, MLPHead
from shared import train_stage_list, test_stage_list

def get_data(data_dir:str, sub_dataset:int):
    return load_nc_dataset(data_dir, 'elliptic', sub_dataset)


def train_epoch(encoder, mlp, optimizer, loader, loss_fn, device='cpu'):
    encoder.train()
    mlp.train()
    optimizer.zero_grad()

    total_train_loss = 0
    for data in loader:
        data = data.to(device)
        out, _ = mlp(encoder(data.x, data.edge_index))
        loss = loss_fn(out[data.mask], data.y[data.mask])
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    total_train_loss /= len(loader)
    return total_train_loss


@torch.no_grad()
def test_epoch(encoder, mlp, loader, loss_fn, device='cpu'):
    encoder.eval()
    mlp.eval()
    total_val_loss = 0
    total_f1 = 0
    for data in loader:
        data = data.to(device)
        out, _ = mlp(encoder(data.x, data.edge_index))
        loss = loss_fn(out[data.mask], data.y[data.mask])
        y_pred = torch.argmax(out, dim=1)
        f1 = f1_score(y_pred[data.mask].detach().cpu().numpy(), data.y[data.mask].detach().cpu().numpy())
        total_val_loss += loss.item()
        total_f1 += f1
    total_val_loss /= len(loader)
    total_f1 /= len(loader)
    return total_val_loss, total_f1

def train_source(split, device, args):
    feat_dim = 165
    encoder = TwoLayerGraphSAGE(feat_dim, args.hidden_dim, args.emb_dim).to(device)
    mlp = MLPHead(args.emb_dim, args.emb_dim // 4, 2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(mlp.parameters()), lr=1e-3)


    train_data = [get_data(args.data_dir, i) for i in range(split[0], split[1])]
    val_data = [get_data(args.data_dir, i) for i in range(split[1], split[2])]
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

    best_f1 = 0
    best_encoder = None
    best_mlp = None
    for e in range(1, args.train_epochs + 1):
        train_loss = train_epoch(encoder, mlp, optimizer, train_loader, loss_fn, device)
        val_loss, val_f1 = test_epoch(encoder, mlp, val_loader, loss_fn, device)
        print(
            f"Epoch:{e}/{args.train_epochs} Train Loss:{round(train_loss, 4)} Val Loss:{round(val_loss, 4)} Val F1:{round(val_f1, 4)}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_encoder = deepcopy(encoder)
            best_mlp = deepcopy(mlp)

    encoder = deepcopy(best_encoder)
    mlp = deepcopy(best_mlp)
    return encoder, mlp

def main(args):
    set_model_seed(args.model_seed)
    device = get_device(args.gpuID)
    encoder, mlp = train_source(train_stage_list[0], device, args)
    f1_list = []
    for test_stage in test_stage_list:
        test_loader = DataLoader(dataset=[get_data(args.data_dir, i) for i in range(test_stage[0], test_stage[1])])
        total_val_loss, total_f1 = test_epoch(encoder, mlp, test_loader, nn.CrossEntropyLoss(), device)
        f1_list.append(total_f1)
    print(f1_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to data directory")
    parser.add_argument("--log_dir", type=str, help="path to log directory", default="runs")
    parser.add_argument("--ckpt_dir", type=str, help="path to model checkpoint directory", default="checkpoints")
    parser.add_argument("--method", type=str, help="adaptation method")
    parser.add_argument("--train_epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--hidden_dim", type=int, help="GNN hidden layer dimension", default=128)
    parser.add_argument("--emb_dim", type=int, help="embedding dimension", default=128)
    parser.add_argument("--model_seed", type=int, help="random seed", default=42)
    parser.add_argument("--gpuID", type=int, help="which gpu to use", default=0)
    args = parser.parse_args()
    main(args)