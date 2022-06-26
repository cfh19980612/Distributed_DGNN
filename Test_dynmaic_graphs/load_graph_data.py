import os
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
def make_sparse_tensor(adj, tensor_type, torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def _count_max_deg(graphs, adjs):
    max_deg_out = []
    max_deg_in = []

    for (graph, adj) in zip(graphs, adjs):
        cur_out, cur_in = _get_degree_from_adj(adj,graph.number_of_nodes(),graph)
        # print(cur_out, cur_in)
        # max_deg_out.append(cur_out.max())
        # max_deg_in.append(cur_in.max())
        max_deg_out.append(max(cur_out))
        max_deg_in.append(max(cur_in))
    # exit()
    # max_deg_out = torch.stack(max_deg_out).max()
    # max_deg_in = torch.stack(max_deg_in).max()
    max_deg_out = max(max_deg_out)
    max_deg_in = max(max_deg_in)
    max_deg_out = int(max_deg_out) + 1
    max_deg_in = int(max_deg_in) + 1
    
    return max_deg_out, max_deg_in

def _get_degree_from_adj(adj, num_nodes, graph):
    # print(adj.todense())

    degs_out = list(dict(nx.degree(graph)).values())
    # print(degs_out)
    degs_in = degs_out
    return degs_out, degs_in

def _generate_one_hot_feats(graphs, adjs, max_degree):
    r'''
    generate the one-hot feats in a sparse tensors
    parameters: 
        adjs: a list of sparse adjacency_matrix
        max_degree: the maximum degree of total graphs
    '''
    new_feats = []
    feats_sp = []
    feats_torch_sp = []

    for (graph, adj) in zip(graphs, adjs):
        # print(adj)
        num_nodes = graph.number_of_nodes()
        degree_vec, _ = _get_degree_from_adj(adj, num_nodes, graph)
        feats_dict = {'idx': torch.cat([torch.arange(num_nodes).view(-1, 1), torch.tensor(degree_vec).view(-1, 1)], dim=1),
                      'vals': torch.ones(num_nodes)
        }
        feat = make_sparse_tensor(feats_dict, 'float', [num_nodes, max_degree])
        # torch sparse to scipy sparse for saving
        m_index = feat._indices().numpy()
        row = m_index[0]
        col = m_index[1]
        data = feat._values().numpy()

        feat_sp = sp.coo_matrix((data, (row, col)), shape=(feat.size()[0], feat.size()[1]))

        # print(feat)
        # new_feats.append(feat.to_dense().numpy())
        feats_sp.append(feat_sp)
        feats_torch_sp.append(feat)

    return feats_sp, feats_torch_sp

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

    # Num_average_edges = stat_age_difference(graphs)
    # print('Number of edges per age node', Num_average_edges)
    # return 0

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

                # feats.append(torch.Tensor(feat_array))
                feats.append(feat_tensor_sp)

        except IOError:
            print("Generating and saving node features ....")
            # method 1: compute the max degree over all graphs
            max_deg, _ = _count_max_deg(graphs, adj_matrices)
            feats_sp, feats_torch_sp = _generate_one_hot_feats(graphs, adj_matrices, max_deg)
            # method 2:
            # feats = _generate_feats(adj_matrices, len(graphs))
            # print('saved feats, ',feats)

            folder_in = os.path.exists(feats_path)
            if not folder_in:
                os.makedirs(feats_path)
            pbar = tqdm(feats_sp)
            
            # if method 2:
            for id,feat in enumerate(pbar):
                path = feats_path+'no_{}.npz'.format(id)
                sp.save_npz(path, feat)
                pbar.set_description('Compress feature and save:')
            feats = feats_torch_sp

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