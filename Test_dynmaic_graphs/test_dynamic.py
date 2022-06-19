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
        cur_out, cur_in = _get_degree_from_adj(adj,graph.number_of_nodes())
        # print(cur_out, cur_in)
        max_deg_out.append(cur_out.max())
        max_deg_in.append(cur_in.max())
    # exit()
    max_deg_out = torch.stack(max_deg_out).max()
    max_deg_in = torch.stack(max_deg_in).max()
    max_deg_out = int(max_deg_out) + 1
    max_deg_in = int(max_deg_in) + 1
    
    return max_deg_out, max_deg_in

def _get_degree_from_adj(adj, num_nodes):
    # print(adj.todense())
    adj_tensor = torch.tensor(adj.todense())
    sign_a = torch.sign(adj_tensor).int()
    # count
    degs_out = torch.count_nonzero(sign_a, dim=1).reshape(-1, 1)   

    adj_tensor = adj_tensor.t()
    sign_b = torch.sign(adj_tensor).int()
    # count
    degs_in = torch.count_nonzero(sign_b, dim=1).reshape(-1, 1)   
    # print(non_zero_a)


    # degs_out = torch.mul(adj_tensor, torch.ones(num_nodes,1,dtype = torch.LongTensor).cuda())
    # degs_in = adj_tensor.t().matmul(torch.ones(num_nodes,1,dtype = torch.int8).cuda())
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
        degree_vec, _ = _get_degree_from_adj(adj, num_nodes)
        feats_dict = {'idx': torch.cat([torch.arange(num_nodes).view(-1, 1), degree_vec.view(-1, 1)], dim=1),
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


            # if method 1
            # for id,feat in enumerate(pbar):
            #     # print('feature shape is ', feat.shape)
            #     # feats_sp.append(sp.csr_matrix(feat))
            #     feat_sp = sp.csr_matrix(feat)
            #     path = feats_path+'no_{}.npz'.format(id)
            #     sp.save_npz(path, feat_sp)
            #     pbar.set_description('Compress feature and save:')

            # num_feats = feats[0].shape[1]
            # # to tensor_sp
            # feats_tensor_sp = []
            # for feat in feats:
            #     feats_tensor_sp.append(torch.Tensor(feat).to_sparse())
            #     # feats_tensor_sp.append(torch.Tensor(feat))
            # # np.save(feats_path, feats)
            # feats = feats_tensor_sp


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


def stat_age_difference(graphs, feats, adj):
    Num_average_edges = [0 for i in range(len(graphs))]
    adj_dense_tensor =[torch.tensor(adj[i].todense()) for i in range(len(adj))]
    current_last_node = 0

    for i in range(len(graphs)):
        num_of_nodes = adj_dense_tensor[i].size(0) - current_last_node
        max_num_of_edges = torch.tensor([0 for k in range(num_of_nodes)])
        age = len(graphs) - i
        for j in range(len(graphs))[i:]:
            adj_temp = adj_dense_tensor[j][current_last_node:current_last_node + num_of_nodes]
            # count
            edges = torch.count_nonzero(adj_temp, dim=1).reshape(-1, 1).squeeze()
            mask = torch.gt(edges, max_num_of_edges)
            max_num_of_edges[mask] = edges[mask]

        Num_average_edges[i] += torch.mean(max_num_of_edges.float()).item()
        current_last_node += num_of_nodes

    return Num_average_edges


def stat_different(graphs, feats, adj):
    struc_different = [0 for i in range(len(graphs))]
    feat_different = [0 for i in range(len(graphs))]
    feat_different_Agg = [0 for i in range(len(graphs))]

    # # test the structure difference
    # for i in range (len(graphs) - 1):
    #     adj_A = torch.Tensor(adj[i].todense())
    #     adj_B = torch.Tensor(adj[i+1].todense()).to_sparse()
    #     # adj_B_sp = adj_B.to_sparse()
    #     # print(adj_B.size(0), adj_B_sp.size(0))

    #     # print('1 | ',i)
    #     padding_row = torch.zeros(adj_B.size(0) - adj_A.size(0), adj_A.size(1))
    #     adj_A_temp = torch.cat((adj_A, padding_row), dim=0)
    #     # print('2 | ',i)
    #     padding_col = torch.zeros(adj_A_temp.size(0), adj_B.size(1) - adj_A.size(1))
    #     adj_A_pad = torch.cat((adj_A_temp, padding_col), dim=1)
    #     # print('3 | ',i)

    #     adj_A_pad_sp = adj_A_pad.to_sparse()

    #     dif_matrix =  (adj_A_pad_sp - adj_B)
    #     # torch sparse to scipy sparse
    #     m_index = dif_matrix._indices().numpy()
    #     row = m_index[0]
    #     # print(row)
    #     change_node = np.unique(row)

    #     struc_different[i+1] += len(change_node)

    # # test the original feature difference
    # for i in range (len(graphs) - 1):
    #     feat_A = feats[i].to_dense()
    #     feat_B = feats[i+1].to_dense()
    #     padding = torch.zeros(feat_B.size(0) - feat_A.size(0), feat_A.size(1))
    #     feat_A_pad = torch.cat((feat_A, padding), dim=0)
    #     dif_matrix = feat_B - feat_A_pad
    #     sign_a = torch.sign(dif_matrix).int()
    #     # count
    #     difference = torch.count_nonzero(sign_a, dim=1).reshape(-1, 1)

    #     difference.squeeze()
    #     sign_b = torch.sign(difference).int()
    #     Num_of_changed_nodes = torch.count_nonzero(sign_b, dim=0)

    #     feat_different[i+1] += Num_of_changed_nodes.item()
    #     # print(Num_of_changed_nodes)

    # # test the feature difference after one-hop aggregation
    # for i in range (len(graphs) - 1):
    #     feat_A = feats[i].to_dense() 
    #     adj_A = torch.Tensor(adj[i].todense())
    #     feat_B = feats[i+1].to_dense()
    #     adj_B = torch.Tensor(adj[i+1].todense())
    #     # print(feat_A.size(), adj_A.size())
    #     feat_A_Agg = torch.mm(adj_A, feat_A)
    #     feat_B_Agg = torch.mm(adj_B, feat_B)
    #     padding = torch.zeros(feat_B.size(0) - feat_A.size(0), feat_A.size(1))
    #     feat_A_Agg_pad = torch.cat((feat_A, padding), dim=0)
        
    #     dif_matrix = feat_B_Agg - feat_A_Agg_pad
    #     sign_a = torch.sign(dif_matrix).int()
    #     difference = torch.count_nonzero(sign_a, dim=1).reshape(-1, 1)
    #     difference.squeeze()
    #     sign_b = torch.sign(difference).int()
    #     Num_of_changed_nodes = torch.count_nonzero(sign_b, dim=0)
    #     feat_different_Agg[i+1] += Num_of_changed_nodes.item()

    return struc_different, feat_different, feat_different_Agg
    

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
    # print(adj_matrices[0].todense())
    struc_different, feat_different, feat_different_Agg = stat_different(graphs_new, feats, adj_matrices)
    Num_average_edges = stat_age_difference(graphs_new, feats, adj_matrices)

    num_epochs = 100
    GCN_time = [[] for j in range(len(graphs_new))]
    GCN_mem = [[] for j in range(len(graphs_new))]
    for epoch in range(num_epochs):
        pbar = tqdm(graphs_new)
        model.eval()
        for index,graph in enumerate(pbar):
            time_current = time.time()
            out = model(graph.to('cuda:0'), feats[index].to_dense().to('cuda:0'))
            time_cost = time.time() - time_current
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            pbar.set_description('Epoch {} | Graph {} | {:.3f}s | {:.3f}MB'.format(epoch, index, time_cost, gpu_mem_alloc))
            GCN_time[index].append(time_cost)
            GCN_mem[index].append(gpu_mem_alloc)

            graph = graph.to('cpu')
            feats[index].to_dense().to('cpu')
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            
    
    Time_summary = [np.around(np.mean(GCN_time[i]), 3) for i in range (len(graphs_new))]
    Mem_summary = [np.around(np.mean(GCN_mem[i]), 3) for i in range (len(graphs_new))]

    print('Graph nodes information: ',args['nodes_info'])
    print('Graph edges information: ',args['edges_info'])
    print('Number of edges per age node', Num_average_edges)
    print('Number of nodes with changed features ', feat_different)
    print('Number of nodes with changed features after neighbor aggregation ', feat_different_Agg)
    print('Number of nodes with changed structure ', struc_different)
    print('Graph time cost: ', Time_summary)
    print('Graph memory cost: ', Mem_summary)

