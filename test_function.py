from os import remove
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import copy
import time
import pandas as pd

from data_process import load_graphs
from data_process import get_data_example
from data_process import load_dataset
from data_process import slice_graph
from data_process import convert_graphs

from data_distribution import *

from Model.DySAT import DySAT
from Model.MLP import Classifier
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from customized_ddp import DistributedGroupedDataParallel as LocalDDP
from utils import *

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class _My_DGNN(torch.nn.Module):
    def __init__(self, args, in_feats = None, total_workloads_GCN = None, total_workloads_RNN = None):
        super(_My_DGNN, self).__init__()
        if args['rank'] == 3:
            print('start to initialize dgnn')
        self.dgnn = DySAT(args, num_features = in_feats, workload_GCN=total_workloads_GCN, workload_RNN=total_workloads_RNN)
        if args['rank'] == 3:
            print('start to initialize classifier')
        self.classificer = Classifier(in_feature = self.dgnn.out_feats)

    # def set_comm(self):
    #     for p in self.dgnn.structural_attn.parameters():
    #         setattr(p, 'mp_comm', 'mp')
    #         setattr(p, 'dp_comm', 'dp')
    #     for p in self.dgnn.temporal_attn.parameters():
    #         setattr(p, 'dp_comm', 'dp')
    #     for p in self.classificer.parameters():
    #         setattr(p, 'dp_comm', 'dp')

    def forward(self, graphs, nids, gate = None):
        final_emb = self.dgnn(graphs, gate)
        # print(nids)
        # get embeddings of nodes in the last graph
        emb = final_emb[:, -1, :]

        # get target embeddings
        source_id = nids[:, 0]
        target_id = nids[:, 1]
        source_emb = emb[source_id]
        target_emb = emb[target_id]
        input_emb = source_emb.mul(target_emb)
        # print(input_emb)
        return self.classificer(input_emb)

Comm_backend='nccl'

def _gate(args):
    global_time_steps = args['timesteps']
    world_size = args['world_size']
    gate = torch.zeros(world_size, global_time_steps).bool()

    graphs_per_worker = global_time_steps/world_size

    for i in range (world_size):
        for j in range (global_time_steps):
            if j >= i*graphs_per_worker - 1 and j < (i+1)*graphs_per_worker:
                gate[i,j] = True
            else: gate[i,j] = False
    
    return gate

def _pre_comm_group(num_workers):
    group = []
    comm_method = Comm_backend
    group = [torch.distributed.new_group(
            ranks = [worker for worker in range(num_workers)],
            backend = comm_method,
            ) for i in range(num_workers)
            ]
    return group

def _pre_comm_group_gate(num_workers, timesteps, gate):
    comm_method = Comm_backend
    temporal_list = torch.tensor(range(timesteps))
    num_graph_per_worker = int(timesteps/num_workers)
    custom_group = []
    group_member = [[0,0]]
    for worker in range(num_workers):
        if worker == 0: # the first worker does not need to receive emb from others
            custom_group.append(None)
        else:
            req_temporals = temporal_list[gate[worker,:]]
            members = []
            for temp in req_temporals:
                belong_worker = temp.item()//num_graph_per_worker
                if belong_worker not in members: members.append(belong_worker)
            group_member.append(members)
            custom_group.append(torch.distributed.new_group(
                                    ranks = members,
                                    backend = comm_method,
                                    ))
    
    print('group_member: ', group_member)

    return group_member, custom_group

# TODO: complete the global forward
def run_dgnn_distributed(args):
    args['method'] = 'dist'
    args['connection'] = True
    args['gate'] = False
    device = args['device']
    rank = args['rank']
    world_size = args['world_size']
    if rank != world_size - 1:
        mp_group = args['mp_group'][rank]
    else: mp_group = None
    dp_group = args['dp_group']

    # torch.manual_seed(42 + rank)
    # torch.cuda.manual_seed(42 + rank)

    # TODO: Unevenly slice graphs
    # load graphs
    # total_graph, load_g, load_adj, load_feats, num_feats = slice_graph(*load_graphs(args))

    # get a complete DTDG
    _, graphs, load_adj, load_feats, num_feats = load_graphs(args)
    num_graph = len(graphs)
    adjs_list = []
    for i in range(args['timesteps']):
        # print(type(adj_matrices[i]))
        adj_coo = load_adj[i].tocoo()
        values = adj_coo.data
        indices = np.vstack((adj_coo.row, adj_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj_coo.shape

        adj_tensor_sp = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        adjs_list.append(adj_tensor_sp)
    args['adjs_list'] = adjs_list
    # Graph distribution: launch partition algorithm
    Num_nodes = args['nodes_info']
    nodes_list = [torch.tensor([j for j in range(Num_nodes[i])]) for i in range(args['timesteps'])]
    Degrees = [list(dict(nx.degree(graphs[t])).values()) for t in range(args['timesteps'])]
    if args['partition'] == 'node':
        if args['balance']:
            partition_obj = node_partition_balance(args, graphs, nodes_list, adjs_list, num_devices=args['world_size'])
        else:
            partition_obj = node_partition(args, nodes_list, adjs_list, num_devices=args['world_size'])
    elif args['partition'] == 'snapshot':
        if args['balance']:
            partition_obj = snapshot_partition_balance(args, graphs, nodes_list, adjs_list, num_devices=args['world_size'])
        else:
            partition_obj = snapshot_partition(args, nodes_list, adjs_list, num_devices=args['world_size'])
    elif args['partition'] == 'hybrid':
        if args['balance']:
            partition_obj = hybrid_partition_balance(args, graphs, nodes_list, adjs_list, num_devices=args['world_size'])
        else:
            partition_obj = hybrid_partition(args, graphs, nodes_list, adjs_list, num_devices=args['world_size'])
    
    total_workload_GCN, total_workload_RNN = partition_obj.workload_partition()
    local_workload_GCN = total_workload_GCN[rank]
    local_workload_RNN = total_workload_RNN[rank]
    # summary
    if rank == 0:
        print('Graphs average nodes: {}, average edges: {}'.format(np.mean(args['nodes_info']), np.mean(args['edges_info'])))
    GCN_nodes_compute = [torch.nonzero(local_workload_GCN[t] == True, as_tuple=False).view(-1) for t in range (args['timesteps'])]
    RNN_nodes_compute = [torch.nonzero(local_workload_RNN[t] == True, as_tuple=False).view(-1) for t in range (args['timesteps'])]
    GCN_nodes = torch.cat(GCN_nodes_compute, dim=0)
    RNN_nodes = torch.cat(RNN_nodes_compute, dim=0)
    print('Worker {} has structural workload {} nodes, temporal workloads {} nodes:'.format(rank, GCN_nodes.size(0), RNN_nodes.size(0)))

    # Graph distribution: distribute graphs according to the result of partition algorithm
    
    # gate = _gate(args)
    # # print(args['gate'])
    # # if args['gate']:
    # #     args['gated_group_member'], args['gated_group'] = _pre_comm_group_gate(args['world_size'], args['timesteps'], gate)
    # args['dist_group'] = _pre_comm_group(world_size)


    # # generate the num of graphs for each module in DGNN
    # args['structural_time_steps'] = num_graph
    # if args['connection']:
    #     if args['gate']:
    #         temporal_list = torch.tensor(range(args['timesteps']))
    #         args['temporal_time_steps'] = len(temporal_list[gate[rank,:]].numpy())
    #     else:
    #         args['temporal_time_steps'] = num_graph*(rank + 1)
    # else:
    #     args['temporal_time_steps'] = num_graph
    # print("Worer {} loads {}/{} graphs, where {} local graphs, and {} remote graphs. The dimension of node feature is {}".format(
    #     rank, num_graph, args['timesteps'],
    #     num_graph, args['temporal_time_steps'] - num_graph,
    #     num_feats))

    dataset = load_dataset(*get_data_example(graphs, args, num_graph))

    train_dataset = Data.TensorDataset(
                torch.tensor(dataset['train_data']), 
                torch.FloatTensor(dataset['train_labels'])
            )

    loader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = 1000,
        shuffle = True,
        num_workers=0,
    )

    # TODO: How to take the global forward?
    # convert graphs to dgl or pyg graphs
    graphs = convert_graphs(graphs, load_adj, load_feats, args['data_str'])
    print('Worker {} has already converted graphs'.format(rank, args['device']))

    model = _My_DGNN(args, in_feats=num_feats, total_workloads_GCN = total_workload_GCN, total_workloads_RNN = total_workload_RNN)
    print('Worker {} has already initialized DGNN model'.format(rank, args['device']))
    model = model.to(device)
    print('Worker {} has already put the model to device {}'.format(rank, args['device']))
    # gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
    # print('GPU memory uses {} after loaded the model!'.format(gpu_mem_alloc))

    # model.set_comm()
    # distributed ?
    # model = LocalDDP(copy.deepcopy(model), mp_group, dp_group, world_size)
    # model = DDP(model, process_group=dp_group, find_unused_parameters=True)

    # loss_func = nn.BCELoss()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # results info
    epochs_f1_score = []
    epochs_auc = []
    epochs_acc = []

    total_train_time = []
    total_gcn_time = []
    total_att_time = []
    total_comm_time = []
    total_comm_time_second = []

    log_loss = []
    log_acc = []
    
    # train
    for epoch in range (args['epochs']):
        Loss = []
        epoch_train_time = []
        epoch_comm_time = []
        epoch_gcn_time = []
        epoch_att_time = []
        epoch_mem = []
        epoch_time_start = time.time()
        args['comm_cost'] = 0
        args['gcn_time'] = 0
        args['att_time'] = 0
        args['comm_cost_second'] = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            model.train()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            # print('GPU memory uses {} after loaded the training data!'.format(gpu_mem_alloc))

            # graphs = [graph.to(device) for graph in graphs]
            train_start_time = time.time()
            out = model(graphs, batch_x)
            # print(out)
            loss = loss_func(out.squeeze(dim=-1), batch_y)
            Loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # print('epoch {} worker {} completes gradients computation!'.format(epoch, args['rank']))
            optimizer.step()
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            epoch_mem.append(gpu_mem_alloc)
            epoch_train_time.append(time.time() - train_start_time)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        # communication time
        if args['connection']:
            epoch_comm_time.append(args['comm_cost'])
            if epoch >= 5:
                total_comm_time.append(args['comm_cost'])
                total_comm_time_second.append(args['comm_cost_second'])
        else: epoch_comm_time.append(0)

        # individual module computation costs
        epoch_gcn_time.append(args['gcn_time'])
        epoch_att_time.append(args['att_time'])

        if epoch >= 5:
            total_train_time.append(np.sum(epoch_train_time))
            total_gcn_time.append(args['gcn_time'])
            total_att_time.append(args['att_time'])
        # print(out)
        # test
        if epoch % args['test_freq'] == 0 and rank != world_size - 1:
            graphs = [graph.to(device) for graph in graphs]
            test_result = model(graphs, torch.tensor(dataset['test_data']).to(device))

        elif epoch % args['test_freq'] == 0 and rank == world_size - 1:
            # model.eval()
            # graphs = [graph.to(device) for graph in graphs]
            test_result = model(graphs, torch.tensor(dataset['test_data']).to(device))
            prob_f1 = []
            prob_auc = []
            prob_f1.extend(np.argmax(test_result.detach().cpu().numpy(), axis = 1))
            prob_auc.extend(test_result[:, -1].detach().cpu().numpy())
            ACC = sum(prob_f1 == dataset['test_labels'])/len(dataset['test_labels'])
            F1_result = f1_score(dataset['test_labels'], prob_f1)
            AUC = roc_auc_score(dataset['test_labels'], prob_auc)
            epochs_f1_score.append(F1_result)
            epochs_auc.append(AUC)
            epochs_acc.append(ACC)
            log_loss.append(np.mean(Loss))
            log_acc.append(ACC)
            print("Epoch {:<3}, Loss = {:.3f}, F1 Score = {:.3f}, AUC = {:.3f}, ACC = {:.3f}, Time = " \
                "{:.5f}:[{:.5f}({:.3f}%)|{:.5f}({:.3f}%)|{:.5f}({:.3f}%)], Memory Usage = {:.2f}MB ({:.2f}%)".format(
                                                                epoch,
                                                                np.mean(Loss),
                                                                F1_result,
                                                                AUC,
                                                                ACC,
                                                                np.sum(epoch_train_time), 
                                                                np.sum(epoch_gcn_time),(np.sum(epoch_gcn_time)/np.sum(epoch_train_time))*100,
                                                                np.sum(epoch_att_time),(np.sum(epoch_att_time)/np.sum(epoch_train_time))*100,
                                                                np.sum(epoch_comm_time),(np.sum(epoch_comm_time)/np.sum(epoch_train_time))*100,
                                                                np.mean(epoch_mem), (np.mean(epoch_mem)/16160)*100
                                                                ))

    # print the training result info
    if rank == world_size - 1:
        best_f1_epoch = epochs_f1_score.index(max(epochs_f1_score))
        best_auc_epoch = epochs_auc.index(max(epochs_auc))
        best_acc_epoch = epochs_acc.index(max(epochs_acc))

        total_train_time.remove(max(total_train_time))
        total_gcn_time.remove(max(total_gcn_time))
        total_att_time.remove(max(total_att_time))
        total_comm_time.remove(max(total_comm_time))
        total_comm_time_second.remove(max(total_comm_time_second))

        print("Best f1 score epoch: {}, Best f1 score: {}".format(best_f1_epoch, max(epochs_f1_score)))
        print("Best auc epoch: {}, Best auc score: {}".format(best_auc_epoch, max(epochs_auc)))
        print("Best acc epoch: {}, Best acc score: {}".format(best_acc_epoch, max(epochs_acc)))
        print("Total training cost: {:.3f}, total GCN cost: {:.3f}, total ATT cost: {:.3f}, total communication cost: {:.3f}, total_communication_second cost: {:.3f}".format(
                                                                    sum(total_train_time), 
                                                                    sum(total_gcn_time),
                                                                    sum(total_att_time),
                                                                    sum(total_comm_time),
                                                                    sum(total_comm_time_second)))
        print('{:.3f},  {:.3f},  {:.3f},  {:.3f}, {:.3f}'.format(sum(total_train_time), sum(total_gcn_time), sum(total_att_time), sum(total_comm_time), sum(total_comm_time_second)))

        if args['save_log']:
            df_loss=pd.DataFrame(data=log_loss)
            df_loss.to_csv('./experiment_results/{}_{}_{}_{}_loss.csv'.format(args['dataset'], args['timesteps'], args['partition'], args['world_size']))
            df_acc=pd.DataFrame(data=log_acc)
            df_acc.to_csv('./experiment_results/{}_{}_{}_{}_acc.csv'.format(args['dataset'], args['timesteps'], args['partition'], args['world_size']))

def run_dgnn(args):
    r"""
    run dgnn with one process
    """

    args['method'] = 'local'
    args['connection'] = False
    device = args['device']
    # args['timesteps'] = 4

    # TODO: Unevenly slice graphs
    # load graphs
    _, load_g, load_adj, load_feats = slice_graph(*load_graphs(args))
    # print('features: ',load_feats)
    num_graph = len(load_g)
    print("Loaded {}/{} graphs".format(num_graph, args['timesteps']))
    args['structural_time_steps'] = num_graph
    args['temporal_time_steps'] = num_graph

    dataset = load_dataset(*get_data_example(load_g, args, num_graph))

    train_dataset = Data.TensorDataset(
                torch.tensor(dataset['train_data']), 
                torch.FloatTensor(dataset['train_labels'])
            )
    loader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = 1000,
        shuffle = True,
        num_workers=0,
    )

    # convert graphs to dgl or pyg graphs
    graphs = convert_graphs(load_g, load_adj, load_feats, args['data_str'])

    model = _My_DGNN(args, in_feats=load_feats[0].shape[1]).to(device)

    # loss_func = nn.BCELoss()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # results info
    epochs_f1_score = []
    epochs_auc = []
    epochs_acc = []
    log_loss = []
    log_acc = []
    total_train_time = 0
    # train
    for epoch in range (args['epochs']):
        Loss = []
        epoch_train_time = []
        epoch_gcn_time = []
        epoch_att_time = []
        epoch_comm_time = []
        epoch_mem = []
        epoch_time_start = time.time()
        args['comm_cost'] = 0
        args['gcn_time'] = 0
        args['att_time'] = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            model.train()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            graphs = [graph.to(device) for graph in graphs]
            train_start_time = time.time()
            out = model(graphs, batch_x)
            # print('outputs: ',out)
            # print('true_labels: ',batch_y)
            loss = loss_func(out.squeeze(dim=-1), batch_y)
            Loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            epoch_mem.append(gpu_mem_alloc)
            epoch_train_time.append(time.time() - train_start_time)

        # communication time
        epoch_comm_time.append(0)

        # individual module computation costs
        epoch_gcn_time.append(args['gcn_time'])
        epoch_att_time.append(args['att_time'])

        if epoch >= 5:
            total_train_time += np.sum(epoch_train_time)
        # print(out)
        # test
        if epoch % args['test_freq'] == 0:
            # test_source_id = test_data[:, 0]
            # test_target_id = test_data[:, 1]
            # model.eval()
            graphs = [graph.to(device) for graph in graphs]
            test_result = model(graphs, torch.tensor(dataset['test_data']).to(device))
            # print('outputs: ',test_result)
            # print('True labels: ',dataset['test_labels'])
            prob_f1 = []
            prob_auc = []
            prob_f1.extend(np.argmax(test_result.detach().cpu().numpy(), axis = 1))
            prob_auc.extend(test_result[:, -1].detach().cpu().numpy())
            ACC = sum(prob_f1 == dataset['test_labels'])/len(dataset['test_labels'])
            F1_result = f1_score(dataset['test_labels'], prob_f1)
            AUC = roc_auc_score(dataset['test_labels'], prob_auc)
            epochs_f1_score.append(F1_result)
            epochs_auc.append(AUC)
            epochs_acc.append(ACC)
            log_loss.append(np.mean(Loss))
            log_acc.append(ACC)
            print("Epoch {:<3}, Loss = {:.3f}, F1 Score = {:.3f}, AUC = {:.3f}, ACC = {:.3f}, Time = " \
                "{:.5f}:[{:.5f}({:.3f}%)|{:.5f}({:.3f}%)|{:.5f}({:.3f}%)], Memory Usage = {:.2f}MB ({:.2f}%)".format(
                                                                epoch,
                                                                np.mean(Loss),
                                                                F1_result,
                                                                AUC,
                                                                ACC,
                                                                np.sum(epoch_train_time), 
                                                                np.sum(epoch_gcn_time),(np.sum(epoch_gcn_time)/np.sum(epoch_train_time))*100,
                                                                np.sum(epoch_att_time),(np.sum(epoch_att_time)/np.sum(epoch_train_time))*100,
                                                                np.sum(epoch_comm_time),(np.sum(epoch_comm_time)/np.sum(epoch_train_time))*100,
                                                                np.mean(epoch_mem), (np.mean(epoch_mem)/16160)*100
                                                                ))

    # print the training result info
    best_f1_epoch = epochs_f1_score.index(max(epochs_f1_score))
    best_auc_epoch = epochs_auc.index(max(epochs_auc))
    best_acc_epoch = epochs_acc.index(max(epochs_acc))

    print("Best f1 score epoch: {}, Best f1 score: {}".format(best_f1_epoch, max(epochs_f1_score)))
    print("Best auc epoch: {}, Best auc score: {}".format(best_auc_epoch, max(epochs_auc)))
    print("Best acc epoch: {}, Best acc score: {}".format(best_acc_epoch, max(epochs_acc)))
    print("Total training cost: {:.3f}".format(total_train_time))

    if args['save_log']:
        df_loss=pd.DataFrame(data=log_loss)
        df_loss.to_csv('./experiment_results/{}_{}_{}_loss.csv'.format(args['dataset'], args['timesteps'], args['test_type']))
        df_acc=pd.DataFrame(data=log_acc)
        df_acc.to_csv('./experiment_results/{}_{}_{}_acc.csv'.format(args['dataset'], args['timesteps'], args['test_type']))








