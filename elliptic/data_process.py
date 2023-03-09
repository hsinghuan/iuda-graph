import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import json
import os
import pickle as pkl

def get_id(old_id, time_step, time_step_cnt, new_id):
    # returns the new id of old id (0, 1, ...)
    if new_id.get(old_id) is None: # new id dictionary doesn't contain old id
        if time_step_cnt.get(time_step) is None: # if doesn't include current time step, initialize as 0
            time_step_cnt[time_step] = 0
        new_id[old_id] = time_step_cnt[time_step] # the new id of the node is the current count of this time step
        time_step_cnt[time_step] += 1
    return new_id[old_id]


def load_elliptic(data_dir):
    feature_list = open(f'{data_dir}/elliptic_txs_features.csv').readlines()
    edge_list = open(f'{data_dir}/elliptic_txs_edgelist.csv').readlines()[1:]  # first line is useless
    label_list = open(f'{data_dir}/elliptic_txs_classes.csv').readlines()[1:]  # first line is useless

    time_step_cnt = {}
    new_id = {}
    time_steps = {}# key: node id. val: time step
    result = [] # each element is a len 3 list corresponding to each time step, 0:, 1: labels, 2: features

    for line in feature_list: # each line is a node
        features = line.strip().split(',')
        old_id = int(features[0]) # original node id
        time_step = int(features[1]) - 1  # 1-base to 0-base
        time_steps[old_id] = time_step 
        features = list(map(float, features[2: ]))

        while len(result) <= time_step:
            result.append([None, [], []]) # add place holders for current time step

        node_id = get_id(old_id, time_step, time_step_cnt, new_id)

        while len(result[time_step][2]) <= node_id:
            result[time_step][2].append(None) # add a place holder for the current node feature at time step

        result[time_step][2][node_id] = features

    for line in label_list:
        label = line.strip().split(',')
        old_id = int(label[0])
        label = label[1]
        
        # In original dataset, 1: illicit, 2: licit, unknown
        
        if label == 'unknown':
            label = -1
        else:
            label = 0 if int(label) == 2 else 1

        time_step = time_steps[old_id]
        node_id = get_id(old_id, time_step, time_step_cnt, new_id)

        while len(result[time_step][1]) <= node_id:
            result[time_step][1].append(None) # add a place holder for the current node label at time step
        result[time_step][1][node_id] = label

    for i in range(len(result)):
        result[i][0] = []
    #     result[i][0] = np.zeros((time_step_cnt[i], time_step_cnt[i]))

    for edge in edge_list:
        u, v = edge.strip().split(',')
        u = int(u)
        v = int(v)
        time_step = time_steps[u] # will time step of u be the same as v? In other words, are the graphs disconnected between time steps?
        # delete later
        if time_step != time_steps[v]:
            print("alert! different time step connected:", u, v)

        u = get_id(u, time_step, time_steps, new_id)
        v = get_id(v, time_step, time_steps, new_id)

        result[time_step][0].append([u, v]) # insert in to the edge list at time step

        # result[time_step][0][u][v] = 1

    # for i in range(len(result)):
    #     A = result[i][0]
    #     num = 0
    #     for k in range(A.shape[0]):
    #         num += ( np.sum(A[k] > 0) >= 2 )
    #     print(num, A.shape[0])
    file_path = os.path.join(data_dir, "processed")
    os.makedirs(file_path, exist_ok=True)

    for i in range(len(result)): 
        edge_list = result[i][0]
        src, targ = zip(*edge_list)
        A = scipy.sparse.csr_matrix((np.ones(len(src)), # values
                                     (np.array(src), np.array(targ))), # positions
                                    shape=(time_step_cnt[i], time_step_cnt[i])) # shape: # nodes * # nodes
        result[i][0] = A
        result[i][1] = np.array(result[i][1]).astype(np.int)
        result[i][2] = np.array(result[i][2]).astype(np.float32)

        with open(file_path + '/{}.pkl'.format(i), 'wb') as f:
            pkl.dump(result[i], f, pkl.HIGHEST_PROTOCOL)

    #     print(len(src), time_step_cnt[i])


    # print(result[0][1].shape)
    # print(result[0][2].shape)
    # print(len(result[0][1]), len(result[0][2]))

    return result

if __name__ == '__main__':
    data_dir = "path/to/elliptic_bitcoin_dataset"
    load_elliptic(data_dir)
