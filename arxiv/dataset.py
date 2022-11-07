import numpy as np
import torch
from torch_geometric.data import Data

def take_second(element):
    return element[1]


def temp_partition_arxiv(data, year_bound=[-1, 2010, 2011], proportion=1.0):
    """
    [-1, 2010, 2011]: load graph up till 2011. Train 0~2010. Val: 2011.
    """
    assert len(year_bound) == 3
    node_years = data['node_year']
    n = node_years.shape[0]
    node_years = node_years.reshape(n)

    d = np.zeros(len(node_years))  # frequency of interaction of each node before year upper bound
    edges = data['edge_index']
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] < year_bound[2] and node_years[edges[1][i]] < year_bound[
            2]:  # if the edge happens before year upper bound
            d[edges[0][i]] += 1  # out node += 1
            d[edges[1][i]] += 1  # in node += 1

    nodes = []  # node id and frequency of interaction before year upper bound
    for i, year in enumerate(node_years):
        if year < year_bound[2]:
            nodes.append([i, d[i]])

    nodes.sort(key=take_second, reverse=True)

    nodes_new = nodes[:int(proportion * len(nodes))]  # take top popular nodes that happens before year upper bound

    # reproduce id
    result_edges = []
    result_features = []
    result_labels = []
    for node in nodes_new:
        result_features.append(data.x[node[0]])
    result_features = torch.stack(result_features)

    ids = {}
    for i, node in enumerate(nodes_new):
        ids[node[0]] = i  # previous node id to new node id

    for i in range(edges.shape[1]):
        if edges[0][i].item() in ids and edges[1][
            i].item() in ids:  # if in node and out node of an edge are both in result nodes, add the edge
            result_edges.append([ids[edges[0][i].item()], ids[edges[1][i].item()]])
    result_edges = torch.LongTensor(result_edges).transpose(1, 0)
    result_labels = data.y[[node[0] for node in nodes_new]]
    result_labels = result_labels.squeeze(dim=-1)  # to accomodate to GBT repo

    data_new = Data(x=result_features, edge_index=result_edges, y=result_labels)
    node_years_new = torch.tensor([node_years[node[0]] for node in nodes_new])
    # data_new.node_year = node_years_new
    data_new.train_mask = torch.logical_and(node_years_new >= year_bound[0], node_years_new < year_bound[1])
    data_new.val_mask = torch.logical_and(node_years_new >= year_bound[1], node_years_new < year_bound[2])
    return data_new