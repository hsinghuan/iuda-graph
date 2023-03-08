import numpy as np
import os
from dataset import StochasticBlockModelBlobDataset
from utils import set_model_seed

def make_sbmb(block_change_data_dir:str):
    block_sizes = [600, 300, 100]

    start_edge_probs = np.array([[0.7, 0.2, 0.1],
                                 [0.2, 0.8, 0.3],
                                 [0.1, 0.3, 0.9]])
    end_edge_probs = np.array([[0.8, 0.4, 0.3],
                               [0.4, 0.6, 0.2],
                               [0.3, 0.2, 0.5]])

    feat_dim = 10
    centers = np.stack([np.random.normal(loc=0, size=feat_dim),
                        np.random.normal(loc=-1, size=feat_dim),
                        np.random.normal(loc=1, size=feat_dim), ])
    num_graphs = 20
    sbmb_dataset_list = [StochasticBlockModelBlobDataset(root=os.path.join(block_change_data_dir, str(i)), block_sizes=block_sizes,
                                                         edge_probs=start_edge_probs + (i + 1) * (
                                                                     end_edge_probs - start_edge_probs) / num_graphs,
                                                         num_channels=feat_dim, centers=centers) for i in
                         range(num_graphs)]
    sbmb_data_list = [sbmb_dataset[0] for sbmb_dataset in sbmb_dataset_list]
    return sbmb_data_list


def make_sbmb_ys(block_change_data_dir:str):
    start_block_sizes = np.array([600, 300, 100])
    end_block_sizes = np.array([500, 100, 400])

    start_edge_probs = np.array([[0.7, 0.2, 0.1],
                                 [0.2, 0.8, 0.3],
                                 [0.1, 0.3, 0.9]])
    end_edge_probs = np.array([[0.8, 0.4, 0.3],
                               [0.4, 0.6, 0.2],
                               [0.3, 0.2, 0.5]])

    feat_dim = 10
    centers = np.stack([np.random.normal(loc=0, size=feat_dim),
                        np.random.normal(loc=-1, size=feat_dim),
                        np.random.normal(loc=1, size=feat_dim), ])
    num_graphs = 20
    sbmb_dataset_list = []
    for i in range(num_graphs):
        edge_probs = start_edge_probs + (i + 1) * (end_edge_probs - start_edge_probs) / num_graphs
        block_sizes = np.rint(start_block_sizes + (i + 1) * (end_block_sizes - start_block_sizes) / num_graphs).tolist()
        sbmb_dataset_list.append(StochasticBlockModelBlobDataset(root=os.path.join(block_change_data_dir, str(i)),
                                                                 block_sizes=block_sizes,
                                                                 edge_probs=edge_probs,
                                                                 num_channels=feat_dim, centers=centers))

    sbmb_data_list = [sbmb_dataset[0] for sbmb_dataset in sbmb_dataset_list]
    return sbmb_data_list

def make_sbmf(feature_change_data_dir:str):
    block_sizes = [700, 300]
    edge_probs = np.array([[0.5, 0.4],
                           [0.4, 0.6]])
    feat_dim = 2
    num_graphs = 20
    sbmf_dataset_list = []
    for i in range(num_graphs):
        centers = np.stack([[np.cos(np.pi / 4 + i * (-np.pi / 4 - np.pi / 4) / num_graphs),
                             np.sin(np.pi / 4 + i * (-np.pi / 4 - np.pi / 4) / num_graphs)],
                            [np.cos(5 * np.pi / 4 + i * (3 * np.pi / 4 - 5 * np.pi / 4) / num_graphs),
                             np.sin(5 * np.pi / 4 + i * (3 * np.pi / 4 - 5 * np.pi / 4) / num_graphs)]])
        dataset = StochasticBlockModelBlobDataset(root=os.path.join(feature_change_data_dir, str(i)), block_sizes=block_sizes,
                                                  edge_probs=edge_probs, num_channels=feat_dim, centers=centers)
        sbmf_dataset_list.append(dataset)

    sbmf_data_list = [sbmf_dataset[0] for sbmf_dataset in sbmf_dataset_list]
    return sbmf_data_list


def make_sbmf_ys(feature_change_data_dir:str):
    start_block_sizes = np.array([700, 300])
    end_block_sizes = np.array([300, 700])
    edge_probs = np.array([[0.5, 0.4],
                           [0.4, 0.6]])
    feat_dim = 2
    num_graphs = 20
    sbmf_dataset_list = []
    for i in range(num_graphs):
        centers = np.stack([[np.cos(np.pi / 4 + i * (-np.pi / 4 - np.pi / 4) / num_graphs),
                             np.sin(np.pi / 4 + i * (-np.pi / 4 - np.pi / 4) / num_graphs)],
                            [np.cos(5 * np.pi / 4 + i * (3 * np.pi / 4 - 5 * np.pi / 4) / num_graphs),
                             np.sin(5 * np.pi / 4 + i * (3 * np.pi / 4 - 5 * np.pi / 4) / num_graphs)]])
        block_sizes = np.rint(start_block_sizes + (i + 1) * (end_block_sizes - start_block_sizes) / num_graphs).tolist()
        dataset = StochasticBlockModelBlobDataset(root=os.path.join(feature_change_data_dir, str(i)), block_sizes=block_sizes,
                                                  edge_probs=edge_probs, num_channels=feat_dim, centers=centers)
        sbmf_dataset_list.append(dataset)

    sbmf_data_list = [sbmf_dataset[0] for sbmf_dataset in sbmf_dataset_list]
    return sbmf_data_list


if __name__ == "__main__":
    set_model_seed(42)
    block_change_data_dir = "/home/hhchung/data/pyg-data/sbmb"
    feature_change_data_dir = "/home/hhchung/data/pyg-data/sbmf"
    sbmb_data_list = make_sbmb(block_change_data_dir)
    sbmf_data_list = make_sbmf(feature_change_data_dir)

    block_y_change_data_dir = "/home/hhchung/data/pyg-data/sbmb_ys"
    feature_y_change_data_dir = "/home/hhchung/data/pyg-data/sbmf_ys"
    sbmb_ys_data_list = make_sbmb_ys(block_y_change_data_dir)
    sbmf_ys_data_list = make_sbmf_ys(feature_y_change_data_dir)