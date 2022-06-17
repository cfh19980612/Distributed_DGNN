import enum
from lib2to3.pytree import convert
import os
from statistics import mean
from time import time
import scipy
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import argparse

import dgl
import torch
import torch_geometric as pyg

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

current_path = os.getcwd()


def _generate_feats(adjs, time_steps):
    assert time_steps <= len(adjs), "Time steps is illegal"
    feats = [scipy.sparse.identity(adjs[time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[time_steps - 1].shape[0]]
    new_features = []

    # nomorlization
    for feat in feats:
        rowsum = np.array(feat.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.  # inf -> 0
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat).todense()
        # print('feat is a ', type(feat))
        new_features.append(feat)
        # new_features.append(feat_sp)
    return new_features

def load_graphs(args):
    r"""
    Load graphs with given the dataset name
    param:
        dataset_name: dataset's name
        platform: converse graph to which platform. dgl or pyg
        time_steps: the num of graphs for experiments
        features (bool): whether generate features with one-hot encoding
        graph_id: which graphs should be loaded
    """
    dataset_name = args['dataset']
    time_steps = args['time_steps']
    features = args['featureless']

    new_graphs = []
    # load networkx graphs data
    graph_path = current_path + '/{}/data/{}'.format(dataset_name, 'graphs.npz')
    if dataset_name == 'Enron':
        with open(graph_path, "rb") as f:
            graphs = pkl.load(f)
    else:
        graphs = np.load(graph_path, allow_pickle=True, encoding='latin1')['graph']
    
    time_steps = len(graphs)
    args['time_steps'] = len(graphs)
    # graphs = graphs[1:]

    # get num of nodes for each snapshot
    Nodes_info = []
    Edge_info = []
    for i in range(len(graphs)):
        Nodes_info.append(graphs[i].number_of_nodes())
        Edge_info.append(graphs[i].number_of_edges())
    args['nodes_info'] = Nodes_info
    args['edges_info'] = Edge_info
    print('Graphs average nodes: {}, average edges: {}'.format(mean(Nodes_info), mean(Edge_info)))

    adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), graphs))
    # print("Loaded {} graphs ".format(len(graphs)))

    if features:
        # save as sparse matrix
        feats_path = current_path + "/{}/data/eval_feats/".format(args['dataset'])
        try:
            # feats = np.load(feats_path, allow_pickle=True)
            num_feats = 0
            feats = []
            print("Loading node features!")
            for time in range(len(graphs)):
                path = feats_path+'no_{}.npz'.format(time)
                feat = sp.load_npz(path)
                feat_array = feat.toarray()
                if time == 0:
                    num_feats = feat_array.shape[1]
                feat_coo = feat.tocoo()

                values = feat_coo.data
                indices = np.vstack((feat_coo.row, feat_coo.col))

                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = feat_coo.shape

                feat_tensor_sp = torch.sparse.FloatTensor(i, v, torch.Size(shape))

                # feat_tensor_sp = torch.sparse.FloatTensor(torch.LongTensor([feat_coo.row.tolist(), feat_coo.col.tolist()]),
                #                  torch.LongTensor(feat_coo.data.astype(np.int32)))

                feats.append(torch.Tensor(feat_array))
                # feats.append(feat_tensor_sp)

        except IOError:
            print("Generating and saving node features ....")
            # method 1: compute the max degree over all graphs
            # max_deg, _ = _count_max_deg(graphs, adj_matrices)
            # feats = _generate_one_hot_feats(graphs, adj_matrices, max_deg)
            # method 2:
            feats = _generate_feats(adj_matrices, len(graphs))
            # print('saved feats, ',feats)

            folder_in = os.path.exists(feats_path)
            if not folder_in:
                os.makedirs(feats_path)
            pbar = tqdm(feats)
            for id,feat in enumerate(pbar):
                # print('feature shape is ', feat.shape)
                # feats_sp.append(sp.csr_matrix(feat))
                feat_sp = sp.csr_matrix(feat)
                path = feats_path+'no_{}.npz'.format(id)
                sp.save_npz(path, feat_sp)
                pbar.set_description('Compress feature and save:')

            num_feats = feats[0].shape[1]
            # to tensor_sp
            feats_tensor_sp = []
            for feat in feats:
                # feats_tensor_sp.append(torch.Tensor(feat).to_sparse())
                feats_tensor_sp.append(torch.Tensor(feat))
            # np.save(feats_path, feats)
            feats = feats_tensor_sp
    #normlized adj
    # adj_matrices = [_normalize_graph_gcn(adj) for adj in adj_matrices]

    return args, graphs, adj_matrices, feats, num_feats

def _build_pyg_graphs(features, adjs):
    pyg_graphs = []
    for feat, adj in zip(features, adjs):
        # x = torch.Tensor(feat).to_sparse()
        x = feat
        edge_index, edge_weight = pyg.utils.from_scipy_sparse_matrix(adj)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pyg_graphs.append(data)
    return pyg_graphs

def _build_dgl_graphs(graphs, features):
    dgl_graphs = []  # graph list
    for graph, feat in zip(graphs, features):
        # x = feat
        dgl_graph = dgl.from_networkx(graph)
        dgl_graph = dgl.add_self_loop(dgl_graph)
        # dgl_graph.ndata['feat'] = x
        dgl_graphs.append(dgl_graph)
    return dgl_graphs

def convert_graphs(graphs, adj, feats, framework):
    # converse nx-graphs to dgl-graph or pyg-graph
    if framework == 'dgl':
        new_graphs = _build_dgl_graphs(graphs, feats)
    elif framework == 'pyg':
        new_graphs = _build_pyg_graphs(feats, adj)

    return new_graphs


'''
gcn with dgl
'''
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test parameters')
    # parser.add_argument('--json-path', type=str, required=True,
    #                     help='the path of hyperparameter json file')
    # parser.add_argument('--test-type', type=str, required=True, choices=['local', 'dp', 'ddp'],
    #                     help='method for DGNN training')
    
    # for experimental configurations
    parser.add_argument('--featureless', type=bool, default= True,
                        help='generate feature with one-hot encoding')
    parser.add_argument('--time_steps', type=int, nargs='?', default=8,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--world_size', type=int, default=1,
                        help='method for DGNN training')
    parser.add_argument('--gate', type=bool, default=False,
                        help='method for DGNN training')
    parser.add_argument('--dataset', type=str, default='Enron',
                        help='method for DGNN training')
    parser.add_argument('--partition', type=str, nargs='?', default="Time",
                    help='How to partition the graph data')
    args = vars(parser.parse_args())


    _, graphs, adj_matrices, feats, _ = load_graphs(args)

    print('Converting graphs to specific framework!')
    graphs_new = convert_graphs(graphs, adj_matrices, feats, 'dgl')

    print('feature demension is ', feats[0].shape)
    model = GCN(in_feats = feats[0].shape[1], n_hidden=16, n_classes=10, n_layers=1, activation=F.relu, dropout=0.5)
    print('Initializing gcn model!')

    model.cuda()
    print('Starting to training!')

    num_epochs = 100
    GCN_time = [[] for j in range(len(graphs_new))]
    GCN_mem = [[] for j in range(len(graphs_new))]
    for epoch in range(num_epochs):
        pbar = tqdm(graphs_new)
        for index,graph in enumerate(pbar):
            time_current = time.time()
            out = model(graph.to('cuda:0'), feats[index].to('cuda:0'))
            time_cost = time.time() - time_current
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print('Graph {} completes with {} second and {} MB memory'.format(index, time_cost, gpu_mem_alloc))
            GCN_time[index].append(time_cost)
            GCN_mem.append(gpu_mem_alloc)
    
    Time_summary = [np.mean(GCN_time[i]) for i in range (len(graphs_new))]
    Mem_summary = [np.mean(GCN_mem[i]) for i in range (len(graphs_new))]
    print('Graph time cost: ', Time_summary)
    print('Graph memory cost: ', Mem_summary)

