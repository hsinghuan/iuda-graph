"""
DropEdge and DropFeature from https://github.com/nerdslab/bgrl/blob/main/bgrl/transforms.py
"""
import random
import copy
import numpy as np
import torch
from torch_geometric.utils.dropout import dropout_edge, dropout_node, subgraph
from torch_geometric.utils import add_random_edge
from torch_geometric.transforms import Compose
from torch_geometric.utils.num_nodes import maybe_num_nodes

class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. <= p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. <= p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index

        edge_index, edge_mask = dropout_edge(edge_index, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index.to(data.edge_index.device)

        if 'edge_mask' in data:
            data.edge_mask[data.edge_mask == True] = edge_mask[:len(data.edge_mask)].to(data.edge_mask.device)
        else:
            data.edge_mask = edge_mask.to(data.x.device)
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

class DropNodes:
    def __init__(self, p):
        assert 0. <= p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        edge_index = data.edge_index
        edge_index, edge_mask, node_mask = dropout_node(edge_index, p=self.p)
        # pad the node mask if its shape is different from the original node numbers (due to dropedge and dropout node function that takes in edge index)
        num_node = data.x.shape[0]
        if len(node_mask) < num_node:
            pad = torch.ones(num_node, dtype=torch.bool)
            pad[:len(node_mask)] = node_mask
            node_mask = pad
        data.edge_index = edge_index.to(data.edge_index.device)

        if 'node_mask' in data: # if there is already a node mask (caused by subgraph), then the new node mask should be applied to the previously masked subset
            # print("before data node mask", data.node_mask)
            data.node_mask[:len(node_mask)] = torch.logical_and(data.node_mask[:len(node_mask)], node_mask.to(data.node_mask.device)) # apply to the first K nodes since the shape of node_mask is determined by the maximum node index in edge index
            # print("after edge index", edge_index)
            # print("after data node mask", data.node_mask)
        else:
            data.node_mask = node_mask.to(data.x.device)

        if 'edge_mask' in data:
            data.edge_mask[data.edge_mask==True] = edge_mask[:len(data.edge_mask)].to(data.edge_mask.device) # what if edge mask is longer than data.edge_mask since addedge happened before? place only the front part (original part) of the edge mask
        else:
            data.edge_mask = edge_mask.to(data.x.device)
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)

class AddEdges:
    def __init__(self, p):
        assert 0. <= p < 1., 'Add edge probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        edge_index = data.edge_index
        edge_index, added_edges = add_random_edge(edge_index, p=self.p)
        data.edge_index = edge_index.to(data.edge_index.device)
        # add an edge_mask where the add_edges entries are 0 # what happens if we add edge and then drop edge, or vice versa?
        # edge_mask = torch.ones(data.edge_index.size()[1], dtype=torch.bool)
        # edge_mask[-added_edges.size()[1]:] = False
        # data.edge_mask = edge_mask
        if 'edge_mask' not in data:
            # if edge mask is not in data, place a mask on the original edge index, otherwise don't change the edge mask
            data.edge_mask = torch.ones(data.edge_index.size()[1] - added_edges.size()[1], dtype=torch.bool).to(data.x.device)
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)

class Subgraph:
    """
    Modified from https://github.com/Shen-Lab/GraphCL_Automated/blob/master/unsupervised_TU/aug.py
    """
    def __init__(self, p):
        assert 0. <= p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        # what if node has been dropped before?
        node_num = data.x.size()[0] if 'node_mask' not in data else data.node_mask.sum().item()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * (1 - self.p))
        edge_index = data.edge_index.cpu().numpy()
        # select a node to start random walk with
        if 'node_mask' not in data:
            # select from any node
            idx_sub = [np.random.randint(node_num, size=1)[0]]
        else:
            # select from the node that has not been dropped
            idx_list = data.node_mask.nonzero().squeeze(1)
            idx_sub = [idx_list[torch.randperm(len(idx_list))[0]].item()]
        idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]]) # neighbors of the initial node

        count = 0
        while len(idx_sub) <= sub_num:
            count += 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]])) # add the neighbors of the newest included node

        idx_sub = torch.tensor(idx_sub)
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None
        edge_index, edge_attr, edge_mask = subgraph(idx_sub, edge_index, edge_attr, return_edge_mask=True)
        data.edge_index = edge_index.to(data.edge_index.device)
        if edge_attr is not None:
            data.edge_attr = edge_attr

        node_mask = torch.zeros(data.x.size()[0], dtype=torch.bool)
        node_mask[idx_sub] = True
        if 'node_mask' in data:
            data.node_mask[:len(node_mask)] = torch.logical_and(data.node_mask[:len(node_mask)], node_mask.to(data.node_mask.device)) # apply to the first K nodes since the shape of node_mask is determined by the maximum node index in edge index
        else:
            data.node_mask = node_mask.to(data.x.device)
        if 'edge_mask' in data:
            data.edge_mask[data.edge_mask==True] = edge_mask[:data.edge_mask.sum().item()].to(data.edge_mask.device)
        else:
            data.edge_mask = edge_mask.to(data.x.device)
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)


class WeakAugmentor():
    def __init__(self, dropedge_p=0.1, dropfeat_p=0.1):
        self.dropedges = DropEdges(dropedge_p)
        self.dropfeatures = DropFeatures(dropfeat_p)
    def __call__(self, data):
        return self.dropfeatures(self.dropedges(data))

class StrongAugmentor():
    """
    Select two out of the following augmentations
    1. DropNode
    2. SubGraph
    3. DropEdge
    4. DropFeat
    5. AddEdge
    6. DropPath
    """
    def __init__(self, dropnode_p=0.2, dropedge_p=0.2, dropfeat_p=0.2, addedge_p=0.2):
        self.pool = ['dropnodes', 'dropedges', 'dropfeatures', 'addedges']
        self.dropnodes = DropNodes(dropnode_p)
        self.dropedges = DropEdges(dropedge_p)
        self.dropfeatures = DropFeatures(dropfeat_p)
        self.addedges = AddEdges(addedge_p)
        # self.subgraph = Subgraph(subgraph_p)

    def __call__(self, data):
        # randomly select two augmentation strategies
        selected = random.sample(self.pool, k=2)
        transforms = list()
        # print(selected)
        for t in selected:
            transforms.append(getattr(self, t))
        augment_func = Compose(transforms)
        return augment_func(data)

    def test_dropnodes(self, data):
        return self.dropnodes(data)

    def test_dropedges(self, data):
        return self.dropedges(data)

    def test_dropfeatures(self, data):
        return self.dropfeatures(data)

    def test_addedges(self, data):
        return self.addedges(data)

    def test_subgraph(self, data):
        return self.subgraph(data)

    def test_subgraph_dropnodes(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.subgraph(data)
        print("After subgraph")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.dropnodes(data)
        print("After drop nodes")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        return data

    def test_dropnodes_subgraph(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.dropnodes(data)
        print("After drop nodes")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.subgraph(data)
        print("After subgraph")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        return data

    def test_dropnodes_dropedges(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.dropnodes(data)
        print("After drop nodes")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.dropedges(data)
        print("After drop edges")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        return data

    def test_dropedges_dropnodes(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.dropedges(data)
        print("After drop edges")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.dropnodes(data)
        print("After drop nodes")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        return data

    def test_dropedges_subgraph(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.dropedges(data)
        print("After drop edges")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.subgraph(data)
        print("After subgraph")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        return data

    def test_addedges_dropedges(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.addedges(data)
        print("After add edges")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.dropedges(data)
        print("After drop edges")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        return data

    def test_dropedges_addedges(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.dropedges(data)
        print("After drop edges")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.addedges(data)
        print("After add edges")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        return data

    def test_subgraph_dropfeatures(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.subgraph(data)
        print("After subgraph")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.dropfeatures(data)
        print("After drop features")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        return data

    def test_addedges_subgraph(self, data):
        print("Initial graph")
        print(data.x)
        print(data.edge_index)
        data = self.addedges(data)
        print("After add edges")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print(data.edge_mask)
        if 'node_mask' in data:
            print(data.node_mask)
        data = self.subgraph(data)
        print("After subgraph")
        print(data.x)
        print(data.edge_index)
        if 'edge_mask' in data:
            print("edge mask:", data.edge_mask)
        if 'node_mask' in data:
            print("node mask:", data.node_mask)
        return data

# from torch_geometric.data import Data
# from copy import deepcopy
# sa = StrongAugmentor()
# x = torch.randn(8,4)
# edge_index = torch.tensor([[0,0,1,1,2,2,2,3,3,4,4,5,5,6,6,6,7,7], [2,6,0,7,3,4,6,1,5,2,3,1,6,1,2,4,5,6]])
# data = Data(x=x, edge_index=edge_index)
# #
# data = sa.test_addedges_subgraph(deepcopy(data))
#
