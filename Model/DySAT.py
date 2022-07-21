import copy
import os
import sys
import copy
import torch
import time
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
# import torch.multiprocessing as mp
import multiprocessing as mp
from torch.nn.modules.loss import BCEWithLogitsLoss

# from Model.layers import StructuralAttentionLayer
# from Model.layers import TemporalAttentionLayer

from Model.layers_mb import GAT_Layer as StructuralAttentionLayer
from Model.layers_mb import ATT_layer as TemporalAttentionLayer

from utils import *

def _multi_process_gather(rank, dest, x_comm, world_size, Num_nodes_per_worker, gather_dict):
    comm_method = 'nccl'
    dist.init_process_group(backend="nccl")
    group = torch.distributed.new_group(
            ranks = [worker for worker in range(world_size)],
            backend = comm_method,
            )
            
    # group=mp_group[dest]
    # print('Hello!')
    if dest != world_size - 1:
        x_send = x_comm[dest*Num_nodes_per_worker:(dest+1)*Num_nodes_per_worker,:,:]
    else:
        x_send = x_comm[dest*Num_nodes_per_worker:,:,:]
    if rank == dest:
        # gather_dict = [torch.zeros_like(x_send).to(device) for j in range(world_size)]
        # comm_start = time.time()
        torch.distributed.gather(x_send, gather_list=gather_dict, dst=dest, group=group)
        # comm_time.append(time.time() - comm_start)
        # args['comm_cost'] += time.time() - comm_start
    else:
        torch.distributed.gather(x_send, gather_list=None, dst=dest, group=group)
    torch.distributed.destroy_process_group()
    if rank == dest:
        print(gather_dict)  

def _node_partition_comm_before(args, x):
    device = args['device']
    Total_nodes = args['nodes_info'][-1]
    world_size = args['world_size']
    rank = args['rank']
    Num_nodes_per_worker = int(Total_nodes//world_size)
    mp_group = args['dist_group']

    x_temp = x.clone().detach()
    zero_pad = torch.zeros(Total_nodes - x_temp.shape[0], x_temp.size(1), x_temp.size(2)).to(device)
    x_pad_temp = torch.cat((x_temp, zero_pad), dim=0)
    x_comm = x_pad_temp.detach()
    comm_time = []

    if rank != world_size -1:
        x_rec = x_comm[rank*Num_nodes_per_worker:(rank+1)*Num_nodes_per_worker,:,:]
    else:
        x_rec = x_comm[rank*Num_nodes_per_worker:,:,:]
    # gather_lists = [torch.zeros_like(x_rec).to(device) for j in range(world_size)]
    # # do not work!
    # workers = []
    # for i in range(world_size):
    #     p = mp.Process(target=_multi_process_gather, args=(rank, i, x_comm, world_size, Num_nodes_per_worker, gather_lists))
    #     # p = mp.Process(target=_multi_process_gather, args=(rank, i, rank, rank, world_sirankze, rank, rank))
    #     p.start()
    #     workers.append(p)
    # for p in workers:
    #     p.join()
    # print(gather_lists)
    ######################

    # for i in range (world_size):
    #     if i != world_size - 1:
    #         x_send = x_comm[i*Num_nodes_per_worker:(i+1)*Num_nodes_per_worker,:,:]
    #     else:
    #         x_send = x_comm[i*Num_nodes_per_worker:,:,:]
    #     if rank == i:
    #         gather_lists = [torch.zeros_like(x_send).to(device) for j in range(world_size)]
    #         comm_start = time.time()
    #         torch.distributed.gather(x_send, gather_list=gather_lists, dst=i, group=mp_group[i])
    #         comm_time.append(time.time() - comm_start)
    #         # args['comm_cost'] += time.time() - comm_start
    #     else:
    #         torch.distributed.gather(x_send, gather_list=None, dst=i, group=mp_group[i])
    ######################

    gather_lists = [torch.zeros_like(x_comm).to(device) for j in range(world_size)]
    # print('The communication tensor with size ', x_comm.size())
    comm_start = time.time()
    torch.distributed.all_gather(gather_lists, x_comm, group=mp_group[0])
    args['comm_cost'] += time.time() - comm_start
    # print(Total_nodes, Num_nodes_per_worker)
    final_temp = []
    for i in range(world_size):
        if rank != world_size - 1:
            final_temp.append(gather_lists[i][rank*Num_nodes_per_worker:(rank+1)*Num_nodes_per_worker,:,:])
        else:
            final_temp.append(gather_lists[i][rank*Num_nodes_per_worker:,:,:])
    # args['comm_cost'] += max(comm_time)

    final = torch.cat(final_temp, 1)
    # print('final size: ',x_comm.size(), gather_lists[0][rank*Num_nodes_per_worker:(rank+1)*Num_nodes_per_worker,:,:].size(), gather_lists[1].size(), final.size())
    return final

def _node_partition_comm_after(args, x):
    device = args['device']
    Total_nodes = args['nodes_info'][-1]
    world_size = args['world_size']
    rank = args['rank']
    mp_group = args['dist_group']
    time_steps = args["time_steps"]
    Num_nodes_per_worker = int(Total_nodes//world_size)
    Num_times_per_worker = int(time_steps//world_size)
    # print('input size: ',x.size())
    final_list = []
    comm_time = []

    Pad_total_node = Total_nodes - (world_size - 1)*Num_nodes_per_worker
    x_temp = x.clone().detach()
    zero_pad = torch.zeros(Pad_total_node - x_temp.shape[0], x_temp.size(1), x_temp.size(2)).to(device)
    x_pad_temp = torch.cat((x_temp, zero_pad), dim=0)
    x_comm = x_pad_temp.detach()
    comm_tensor = x.clone().detach()

    gather_lists = [torch.zeros_like(x_comm).to(device) for j in range(world_size)]
    # print('The communication tensor with size ', x_comm.size())
    comm_start = time.time()
    torch.distributed.all_gather(gather_lists, x_comm, group=mp_group[0])
    args['comm_cost_second'] += time.time() - comm_start
    
    final_temp = []
    for i in range(world_size):
        if i != world_size - 1:
            if rank != world_size - 1:
                final_temp.append(gather_lists[i][:Num_nodes_per_worker, rank*Num_times_per_worker:(rank+1)*Num_times_per_worker,:])
            else:
                final_temp.append(gather_lists[i][:Num_nodes_per_worker, rank*Num_times_per_worker:,:])
        else:
            if rank != world_size - 1:
                final_temp.append(gather_lists[i][:, rank*Num_times_per_worker:(rank+1)*Num_times_per_worker,:])
            else:
                final_temp.append(gather_lists[i][:, rank*Num_times_per_worker:,:])
    # args['comm_cost'] += max(comm_time)

    final = torch.cat(final_temp, 0)


    return final

# TODO: realize the masked communication
def _customized_embedding_comm(args, x, gate):
    r"""
    Gate: a [N, T] bool matrix, to identify the temporal dependecy for each snapshot
    """
    comm_method = 'gloo'
    # groups = args['mp_group']
    rank = args['rank']
    world_size = args['world_size']
    global_time_steps = args['time_steps']
    num_graph_per_worker = global_time_steps/world_size

    # re-construct the communication map according to the 'gate' matrix
    worker_list = torch.tensor(range(world_size))
    temporal_list = [worker_list[gate[:, i]].numpy() for i in range (global_time_steps)]
    # print('gate: ', gate)
    # print('temporal_list: ',temporal_list)
    mp_groups = [torch.distributed.new_group(
            ranks = temporal_list[i],
            backend = comm_method,
        ) for i in range (global_time_steps)
        ]

    comm_tensor = x.clone().detach()[:,0:1,:]
    result_list = []

    for i in range (global_time_steps):
        src = int(i//num_graph_per_worker)
        if rank < src:
            break
        if rank == src:
            comm_tensor = x.clone().detach()[:,int(i%num_graph_per_worker):int(i%num_graph_per_worker + 1),:]
        if len(temporal_list[i]) > 1 and rank in temporal_list[i]:
            # print('The {}-th graph need to be sent, sender {}, local rank {}'.format(i, int(src), rank))
            torch.distributed.broadcast(comm_tensor, src, group = mp_groups[i])
            if rank != src: # receive
                result_list.append(comm_tensor)
    
    if len(result_list) > 0:
        result_list.append(x)
        final = torch.cat(result_list, 1)
    else: final = x.clone()

    # print('rank: {} with fused tensor size {}'.format(rank, final.size()))

    return final

def _embedding_comm(args, x):
    # mp_group = args['mp_group']
    rank = args['rank']
    world_size = args['world_size']
    device = args['device']

    num_graph_per_worker = int(args['time_steps'] / world_size)
    result_list = []
    # for i in range (world_size - 1):
    #     if i > rank:
    #         break

    #     if i == rank:
    #         # Sender: copy the local data for sending
    #         comm_tensor = x.clone().detach()
    #     else:
    #         r'''
    #         Receiver: generate a empty tensor for receiving data, note that tensors generated from \
    #         different process have different sizes (due to different numbers of nodes)
    #         '''
    #         comm_tensor = torch.zeros(args['nodes_info'][num_graph_per_worker*(i + 1) - 1], num_graph_per_worker, x.shape[2]).to(device)
        
    #     # start to communciation
    #     comm_start = time.time()
    #     torch.distributed.broadcast(comm_tensor, i, group = mp_group[i])
    #     args['comm_cost'] += time.time() - comm_start
    
        # # gather the received tensors
        # if i != rank:
        #     result_list.append(comm_tensor)

    # if len(result_list) > 0:
    #     # step 1: pad tensor to the same size
    #     for i in range (len(result_list)):
    #         zero_pad = torch.zeros(x.shape[0] - args['nodes_info'][num_graph_per_worker*(i + 1) - 1], num_graph_per_worker, x.shape[2]).to(device)
    #         result_list[i] = torch.cat((result_list[i], zero_pad), dim=0).to(device)
    #     result_list.append(x)
    #     # step 2: gather all tensors
    #     final = torch.cat(result_list, 1)
    #     # print('rank: {} with fused tensor {}'.format(rank, final))

    # else: final = x.clone()

    Total_nodes = args['nodes_info'][-1]
    Num_nodes_per_worker = int(Total_nodes//world_size)
    mp_group = args['dist_group']

    x_temp = x.clone().detach()
    zero_pad = torch.zeros(Total_nodes - x_temp.shape[0], x_temp.size(1), x_temp.size(2)).to(device)
    x_pad_temp = torch.cat((x_temp, zero_pad), dim=0)
    x_comm = x_pad_temp.detach()
    comm_tensor = x.clone().detach()

    gather_lists = [torch.zeros_like(x_comm).to(device) for j in range(world_size)]
    # print('The communication tensor with size ', x_comm.size())
    comm_start = time.time()
    torch.distributed.all_gather(gather_lists, x_comm, group=mp_group[0], async_op=True)
    args['comm_cost'] += time.time() - comm_start

    final_temp = []
    for i in range(world_size):
        if i <= rank:
            final_temp.append(gather_lists[i])
    # args['comm_cost'] += max(comm_time)

    final = torch.cat(final_temp, 1)

    return final


def _simulate_comm_time(send_list, receive_list, node_size, bandwidth):
    # compute time
    receive_comm_time = [0 for i in range(num_devices)]
    send_comm_time = [0 for i in range(num_devices)]
    for device_id in range(num_devices):
        # receive
        total_nodes = 0
        for receive in receive_list[device_id]:
            if receive != torch.Size([]):
                total_nodes += receive.view(-1).size(0)
        receive_comm_time[device_id] += np.around(float(total_nodes*node_size)/bandwidth, 3)

        # send
        total_nodes = 0
        for send in send_list[device_id]:
            if send != torch.Size([]):
                total_nodes += send.view(-1).size(0)
        send_comm_time[device_id] += np.around(float(total_nodes*node_size)/bandwidth, 3)
    
    return receive_comm_time, send_comm_time

def _temporal_comm_nodes(rank, nodes_list, num_devices, workloads_GCN, workloads_RNN):
    '''
    Step 1: generate the required nodes list for each device
    Step 2: compare the required list with the RNN(GCN) workload list to compute the number of received nodes
    '''
    timesteps = len(nodes_list)
    Req = [[torch.full_like(nodes_list[time], False, dtype=torch.bool) for time in range(len(nodes_list))] for m in range(num_devices)]
    receive_list = []
    send_list = []

    # compute the required node list
    for time in range(timesteps):
        where_need_comp = torch.nonzero(workloads_RNN[rank][time] == True, as_tuple=False).view(-1)
        if where_need_comp!= torch.Size([]):
            for k in range(timesteps)[0:time+1]:
                idx = torch.tensor([i for i in range(Req[rank][k].size(0))])
                nodes_mask = workloads_RNN[rank][time][idx]
                where_need = torch.nonzero(nodes_mask == True, as_tuple=False).view(-1) # the previous nodes of the owned workload are needed
                # print(where_need)
                if (where_need.size(0) > 0):
                    Req[rank][k][where_need] = torch.ones(where_need.size(0), dtype=torch.bool)
    # remove already owned nodes
    for time in range(timesteps):
        where_have_nodes = torch.nonzero(workloads_GCN[rank][time] == True, as_tuple=False).view(-1)
        # print(where_have_nodes)
        if where_have_nodes!= torch.Size([]):
            # print(where_have_nodes)
            Req[rank][time][where_have_nodes] = torch.zeros(where_have_nodes.size(0), dtype=torch.bool)

    # print(Req)
    # Compute the number of nodes need to be sent and received
    for time in range(timesteps):
        # receive nodes list
        receive = torch.nonzero(Req[rank][time] == True, as_tuple=False).squeeze() # dimension = 2
        receive_list.append(receive.view(-1))

        # send nodes list
        need_send = torch.nonzero(workloads_RNN[rank][-1][:workloads_GCN[rank][time].size(0)] == False, as_tuple=False).view(-1)
        if need_send!= torch.Size([]):
            send_nodes = workloads_GCN[rank][time][need_send]
            send = torch.nonzero(send_nodes == True, as_tuple=False).view(-1)
            send_list.append(send)
        else: send_list.append([])

    return send_list, receive_list

def _structural_comm_nodes(adjs_list, local_workload_GCN):
    # compute need nodes
    receive_list = []
    send_list = []
    for time in range(len(local_workload_GCN)):
        adj = adjs_list[time].clone()
        local_node_mask = local_workload_GCN[time]
        remote_node_mask = ~local_node_mask
        edge_source = adj._indices()[0]
        edge_target = adj._indices()[1]

        # receive
        edge_source_local_mask = local_node_mask[edge_source] # check each source node whether it belongs to device_id
        need_receive_nodes = torch.unique(edge_target[edge_source_local_mask]) # get the target nodes with the source nodes belong to device_id
        receive_node_local = local_node_mask[need_receive_nodes] # check whether the received nodes in local?
        receive = torch.nonzero(receive_node_local == False, as_tuple=False).view(-1) # only receive nodes not in local (return indices)
        receive_list.append(receive)

        # send
        edge_source_remote_mask = remote_node_mask[edge_source] # check each source node whether it belongs to other devices
        need_send_nodes = torch.unique(edge_target[edge_source_remote_mask]) # get the target nodes with the source nodes belong to other devices
        send_node_local = local_node_mask[need_send_nodes] # check whether the send nodes in local?
        send = torch.nonzero(send_node_local == True, as_tuple=False).view(-1) # only send nodes in local
        send_list.append(send)
    
    return send_list, receive_list

def _structural_comm(args, features, workload_GCN, send_list, receive_list, node_size, bandwidth):
    '''
    Args:
        features: [N_i, F]
        workloads: list of bool matrix, workloads[rank][time] represents the partitioned time-th
        workloads on device rank
    Communicate node features snapshot by snapshot? only need neighborhoods
    '''
    rank = args['rank']
    world_size = args['world_size']
    device = args['device']
    dp_group = args['dp_group'] # [rank_0, ..., rank_N]

    # TODO: communicaiton component
    features_dense = features.to_dense()
    gather_lists = [torch.zeros_like(features_dense).to(device) for j in range(world_size)]
    # comm_start = time.time()
    torch.distributed.all_gather(gather_lists, features_dense, group=dp_group)
    # args['comm_cost'] += time.time() - comm_start

    # feature fusion
    final_feature = features_dense.clone().detach()
    for m in range(world_size):
        if receive_list != torch.Size([]) and receive_list.size(0) > 0:
            # print('info:', final_feature.size(), workload_GCN[m].size(), gather_lists[m].size())
            need_node = workload_GCN[m][receive_list]
            have_node = torch.nonzero(need_node == True, as_tuple=False).view(-1)
            # print('info:', final_feature.size(), have_node.size(), workload_GCN[m].size(), gather_lists[m].size())
            final_feature[have_node] = gather_lists[m][have_node]
    final_feature = final_feature.to_sparse()

    # simulated communication time
    receive_node = receive_list.size(0)
    receive_time = np.around(float(receive_node*node_size)/bandwidth, 3)
    send_node = send_list.size(0)
    send_time = np.around(float(send_node*node_size)/bandwidth, 3)
    comm_time = max(receive_time, send_time)
    args['str_comm'] += comm_time

    return comm_time, final_feature

def _temporal_comm(args, embedding, workload_GCN, send_list, receive_list, node_size, bandwidth):
    rank = args['rank']
    world_size = args['world_size']
    device = args['device']
    timesteps = args['timesteps']
    dp_group = args['dp_group'] # [rank_0, ..., rank_N]

    # TODO: communicaiton component
    gather_lists = [torch.zeros_like(embedding).to(device) for j in range(world_size)]
    # comm_start = time.time()
    torch.distributed.all_gather(gather_lists, embedding, group=dp_group)
    # args['comm_cost'] += time.time() - comm_start

    # embedding fusion
    final_embedding = embedding.clone().detach()
    for m in range(world_size):
        need_node = workload_GCN[m][receive_list]
        have_node = torch.nonzero(need_node == True, as_tuple=False).view(-1)
        final_embedding[have_node] = gather_lists[m][have_node]

    # simulated communication time
    receive_node = receive_list.size(0)
    receive_time = np.around(float(receive_node*node_size)/bandwidth, 3)
    send_node = send_list.size(0)
    send_time = np.around(float(send_node*node_size)/bandwidth, 3)
    comm_time = max(receive_time, send_time)

    args['tem_comm'] += comm_time

    return comm_time, final_embedding


class DySAT(nn.Module):
    def __init__(self, args, num_features, workload_GCN, workload_RNN):
        '''
        Args:
            args: hyperparameters
            num_features: input dimension
            time_steps: total timesteps in dataset
            sample_mask: sample different snapshot graphs
            method: adding time interval information methods
        '''
        super(DySAT, self).__init__()
        # structural_time_steps = args['structural_time_steps']
        # temporal_time_steps = args['temporal_time_steps']
        structural_time_steps = args['timesteps']
        temporal_time_steps = args['timesteps']
        args['window'] = -1
        self.args = args

        self.rank = args['rank']
        self.device = args['device']
        
        self.workload_GCN = workload_GCN
        self.workload_RNN = workload_RNN
        self.local_workload_GCN = workload_GCN[self.rank]
        self.local_workload_RNN = workload_RNN[self.rank]

        if args['window'] < 0:
            self.structural_time_steps = structural_time_steps # training graph per 'num_time_steps'
        else:
            self.structural_time_steps = min(structural_time_steps, args['window'] + 1)
        self.temporal_time_steps = temporal_time_steps
        self.num_features = num_features

        # network setting
        # self.structural_head_config = list(map(int, args.structural_head_config.split(","))) # num of heads per layer (structural layer)
        # self.structural_layer_config = list(map(int, args.structural_layer_config.split(","))) # embedding size (structural layer)
        # self.temporal_head_config = list(map(int, args.temporal_head_config.split(","))) # num of heads per layer (temporal layer)
        # self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(","))) # embedding size (temporal layer)
        # self.spatial_drop = args.spatial_drop
        # self.temporal_drop = args.temporal_drop

        self.structural_head_config = [8]
        self.structural_layer_config = [128]
        self.temporal_head_config = [8]
        self.temporal_layer_config = [128]
        self.spatial_drop = 0.1
        self.temporal_drop = 0.9
        self.out_feats = 128

        self.n_hidden = self.temporal_layer_config[-1]
        # self.method = method

        # construct layers
        self.structural_attn, self.temporal_attn = self.build_model()

        # loss function
        self.bceloss = BCEWithLogitsLoss()


    def forward(self, graphs, gate):
        # TODO: communicate the imtermediate embeddings after StructureAtt

        '''
        Input: workloads[rank]
        Compute the GCN embedding in a 'for' loop
            Step 1: communication for aggregating remote neighborhoods
            Step 2: feed the subgraph into the GAT(GCN) layer
        Compute the RNN embedding in a mini-batch manner [N, T, F]
            Step 1: communication for aggregating remote time-series information
            Step 2: feed the time-series input [N, T, F] into the ATT(RNN) layer
        '''
        GCN_emb_list = [torch.ones(self.args['nodes_info'][-1], self.structural_layer_config[-1])[:,None,:].to(self.device) for t in range(self.args['timesteps'])]
        RNN_emb_list = [torch.ones(self.args['nodes_info'][-1], self.temporal_layer_config[-1])[:,None,:].to(self.device) for t in range(self.args['timesteps'])]

        # structural attention forward
        # for t in range(self.args['timesteps']):
        #     features = graphs[t].ndata['feat']
        #     str_time, fusion_features = _structural_comm(self.args, features, self.workload_GCN[:][t], send_list[t], receive_list[t], self.num_features, bandwidth=float(1024*1024*8))
        #     graphs[t].ndata['feat'] = fusion_features

        structural_out = []
        for t in range(self.args['timesteps']):
            graphs[t] = graphs[t].to(self.device)
            node_local_idx = torch.nonzero(self.local_workload_GCN[t] == True, as_tuple=False).view(-1)
            send_list, receive_list = _structural_comm_nodes(self.args['adjs_list'], self.local_workload_GCN)
            features = graphs[t].ndata['feat']
            _, fusion_features = _structural_comm(self.args, features, [self.workload_GCN[m][t] for m in range(self.args['world_size'])], send_list[t], receive_list[t], self.num_features, bandwidth=float(1024*1024*8))
            graphs[t].ndata['feat'] = fusion_features
            if node_local_idx != torch.Size([]) and node_local_idx.size(0) > 0:
                if self.args['data_str'] == 'dgl':
                    node_idx = torch.cat((node_local_idx, receive_list[t]), dim=0)
                    subgraph = graphs[t].subgraph(node_idx.tolist())
                    out = self.structural_attn(subgraph)
                    GCN_emb_list[t][node_idx] = out[:,None,:]  # to [N, 1, F]
                    structural_out.append(out)
                else: return 0


        # temporal attention forward
        for t in range(self.args['timesteps']):
            send_list, receive_list = _temporal_comm_nodes(self.rank, self.args['nodes_list'], self.args['world_size'], self.workload_GCN, self.workload_RNN)
            _, fusion_embedding = _temporal_comm(self.args, GCN_emb_list[t], [self.workload_GCN[m][t] for m in range(self.args['world_size'])], send_list[t], receive_list[t], self.structural_layer_config[-1], bandwidth=float(1024*1024*8))
            GCN_emb_list[t] = fusion_embedding

        temporal_output = []
        temporal_input = torch.cat(GCN_emb_list, dim=1)
        for t in range(self.args['timesteps']):
            # send_list, receive_list = _temporal_comm_nodes(self.args['adjs_list'], self.local_workload_GCN)
            node_idx = torch.nonzero(self.local_workload_RNN[t] == True, as_tuple=False).view(-1)
            if node_idx != torch.Size([]) and node_idx.size(0) > 0:
                emb_input = temporal_input[node_idx,:t+1,:]
                # print(emb_input.size())
                out = self.temporal_attn(emb_input)[:,-1,:]
                RNN_emb_list[t][node_idx] = out[:,None,:]
                temporal_output.append(out)
        final_out = torch.cat(RNN_emb_list, 1)
        return final_out

        # Structural Attention forward
        structural_out = []
        gcn_time_start = time.time()
        for t in range(0, self.structural_time_steps):
            structural_out.append(self.structural_attn(graphs[t]))
            # torch.cuda.empty_cache()
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]
        self.args['gcn_time'] += time.time() - gcn_time_start

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]
        # self.args['gcn_time'] += time.time() - gcn_time_start

        # print('rank: {} with tensor size {}'.format(self.args['rank'], structural_outputs_padded.size()))


        # Temporal Attention forward
        if self.args['connection']:
            # comm_start = time.time()
            # # exchange node embeddings
            # if self.args['gate']:
            #     # fuse_structural_output = _customized_embedding_comm(self.args, structural_outputs_padded, gate)
            #     fuse_structural_output = _gated_emb_comm(self.args, structural_outputs_padded, gate)
            # print('start partition!')
            if self.args['partition'] == 'Time_Node':
                fuse_structural_output = _node_partition_comm_before(self.args, structural_outputs_padded)
            else:
                fuse_structural_output = _embedding_comm(self.args, structural_outputs_padded)
            # print('end partition!')
            # self.args['comm_cost'] += time.time() - comm_start
            # print('comm_cost in worker {} with time {}'.format(self.args['rank'], self.args['comm_cost']))
            # print('rank: {} with fused tensor size{}'.format(self.args['rank'], fuse_structural_output.size()))
            # print('worker {} has the attention input size: {}'.format(self.args['rank'], fuse_structural_output.size()))
            # print("attention input size: ", fuse_structural_output.size())

            # print('The attention tensor with size ', fuse_structural_output.size())
            pointer_temp = 0
            node_scale = 1000
            attention_output = []
            temporal_time_start = time.time()
            while(True):
                if pointer_temp + node_scale < fuse_structural_output.size(0):
                    attention_input = fuse_structural_output[pointer_temp:pointer_temp+node_scale,:,:]
                    attention_output.append(self.temporal_attn(attention_input))
                else:
                    attention_input = fuse_structural_output[pointer_temp:,:,:]
                    attention_output.append(self.temporal_attn(attention_input))
                    break
                pointer_temp += node_scale
                torch.cuda.empty_cache()
            self.args['att_time'] += time.time() - temporal_time_start
            temporal_out = torch.cat(attention_output, 0)

            # temporal_out = self.temporal_attn(fuse_structural_output)
            # self.args['att_time'] += time.time() - temporal_time_start

            if self.args['partition'] == 'Time_Node':
                temporal_out = _node_partition_comm_after(self.args, temporal_out)

        else: 
            temporal_time_start = time.time()
            print('The attention tensor with size ', structural_outputs_padded.size())
            # print('rank: {} with fused tensor size{}'.format(self.args['rank'], structural_outputs_padded.size()))
            temporal_out = self.temporal_attn(structural_outputs_padded)
            self.args['att_time'] += time.time() - temporal_time_start

        return temporal_out

    # construct model
    def build_model(self):
        input_dim = self.num_features
        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        # if self.args['rank'] == 3:
        #     print('start to initialize structural attention layer')
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(args=self.args,
                                             input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.args['residual'])
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]

        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        # if self.args['rank'] == 3:
        #     print('start to initialize temporal attention layer')
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(method=0,
                                           input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.temporal_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args['residual'],
                                           interval_ratio = self.args['interval_ratio'])
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers

if __name__ == '__main__':
    _customized_embedding_comm()