import copy
import os
import sys
import copy
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
# import torch.multiprocessing as mp
import multiprocessing as mp
from torch.nn.modules.loss import BCEWithLogitsLoss

from Model.layers import StructuralAttentionLayer
from Model.layers import TemporalAttentionLayer

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
    gather_lists = [torch.zeros_like(x_rec).to(device) for j in range(world_size)]
    # multiple processes to communicate
    # torch.multiprocessing.set_start_method('spawn')
    workers = []
    for i in range(world_size):
        p = mp.Process(target=_multi_process_gather, args=(rank, i, x_comm, world_size, Num_nodes_per_worker, gather_lists))
        # p = mp.Process(target=_multi_process_gather, args=(rank, i, rank, rank, world_sirankze, rank, rank))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
    # print(gather_lists)
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

    # args['comm_cost'] += max(comm_time)

    final = torch.cat(gather_lists, 1)
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

    final_list = []
    comm_time = []
    for i in range (world_size):
        if i != world_size - 1 and i != rank: # receiver
            comm_tensor = torch.zeros(Num_nodes_per_worker, x.size(1), x.size(2)).to(device)
        elif i == world_size - 1 and i != rank:
            comm_tensor = torch.zeros(Total_nodes - (world_size -1)*Num_nodes_per_worker, x.size(1), x.size(2)).to(device)
        else:
            comm_tensor = x.clone().detach()
            final_list.append(x)

        comm_start = time.time()
        torch.distributed.broadcast(comm_tensor, i, group = mp_group[i])
        comm_time.append(time.time() - comm_start)
        # args['comm_cost'] += time.time() - comm_start
        if i != rank:
            final_list.append(comm_tensor)
    
    args['comm_cost'] += max(comm_time)
    
    new_final_list = [emb[:,rank*Num_times_per_worker:(rank+1)*Num_times_per_worker,:] for emb in final_list]
    final = torch.cat(new_final_list, 0)
        

    return final

def _gated_emb_comm(args, x, gate):
    # gather()
    # mp_group = args['gated_group']
    # world_size = args['world_size']
    # global_time_steps = args['time_steps']
    # rank = args['rank']
    # device = args['device']
    # num_graph_per_worker = int(global_time_steps/world_size)
    # output = []
    # # print(x.size())

    # for worker in range(world_size):
    #     if worker == 0:
    #         continue
    #     current_process_worker = gate[worker, :]
    #     # print(current_process_worker)
    #     local_temp = current_process_worker[rank*num_graph_per_worker: (rank+1)*num_graph_per_worker]
    #     # print(local_temp)
    #     # comm_emb = x.clone().detach()[:,local_temp,:]
    #     # print(worker, rank, comm_emb.size(), comm_emb.dtype)
    #     # print(args['gated_group_member'][worker])
    #     if rank in args['gated_group_member'][worker]:
    #         if worker == rank:
    #             output = [torch.zeros((args['nodes_info'][rank*num_graph_per_worker - 1], 1, x.size(2))).to(device) for _ in range(len(args['gated_group_member'][worker]))]
    #             comm_emb = torch.zeros((args['nodes_info'][rank*num_graph_per_worker - 1], 1, x.size(2))).to(device)
    #             # print('worker {} will receive embeedings at current {} communication round!'.format(rank, worker))
    #             comm_start = time.time()
    #             torch.distributed.gather(comm_emb, gather_list=output, dst=worker, group=mp_group[worker])
    #             args['comm_cost'] += time.time() - comm_start
    #         else:
    #             comm_emb = x.clone().detach()[:,local_temp,:]
    #             # print(worker, rank, comm_emb.size(), comm_emb.dtype)
    #             # print('worker {} will send embeedings at current {} communication round!'.format(rank, worker))
    #             comm_start = time.time()
    #             torch.distributed.gather(comm_emb, gather_list=None, dst=worker, group=mp_group[worker])
    #             args['comm_cost'] += time.time() - comm_start
    # #     print('worker, ', worker, 'complete!')
    # # print('communication complete!')

    # if len(output) > 0:
    #     output.pop()
    #     # print(output)
    #     for i in range(len(output)):
    #         zero_pad = torch.zeros(x.shape[0] - output[i].size(0), output[i].size(1), x.shape[2]).to(device)
    #         output[i] = torch.cat((output[i], zero_pad), dim=0).to(device)
    #     output.append(x)
    #     final = torch.cat(output, 1)
    # else:
    #     final = x.clone()
    
    # return final
    return 0


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
    mp_group = args['mp_group']
    rank = args['rank']
    world_size = args['world_size']
    device = args['device']

    num_graph_per_worker = int(args['time_steps'] / world_size)
    result_list = []
    for i in range (world_size - 1):
        if i > rank:
            break

        if i == rank:
            # Sender: copy the local data for sending
            comm_tensor = x.clone().detach()
        else:
            r'''
            Receiver: generate a empty tensor for receiving data, note that tensors generated from \
            different process have different sizes (due to different numbers of nodes)
            '''
            comm_tensor = torch.zeros(args['nodes_info'][num_graph_per_worker*(i + 1) - 1], num_graph_per_worker, x.shape[2]).to(device)
        
        # start to communciation
        comm_start = time.time()
        torch.distributed.broadcast(comm_tensor, i, group = mp_group[i])
        args['comm_cost'] += time.time() - comm_start
        

        # gather the received tensors
        if i != rank:
            result_list.append(comm_tensor)

    if len(result_list) > 0:
        # step 1: pad tensor to the same size
        for i in range (len(result_list)):
            zero_pad = torch.zeros(x.shape[0] - args['nodes_info'][num_graph_per_worker*(i + 1) - 1], num_graph_per_worker, x.shape[2]).to(device)
            result_list[i] = torch.cat((result_list[i], zero_pad), dim=0).to(device)
        result_list.append(x)
        # step 2: gather all tensors
        final = torch.cat(result_list, 1)
        # print('rank: {} with fused tensor {}'.format(rank, final))
    else: final = x.clone()

    return final


class DySAT(nn.Module):
    def __init__(self, args, num_features):
        '''
        Args:
            args: hyperparameters
            num_features: input dimension
            time_steps: total timesteps in dataset
            sample_mask: sample different snapshot graphs
            method: adding time interval information methods
        '''
        super(DySAT, self).__init__()
        structural_time_steps = args['structural_time_steps']
        temporal_time_steps = args['temporal_time_steps']
        args['window'] = -1
        self.args = args
        

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

        # Structural Attention forward
        structural_out = []
        gcn_time_start = time.time()
        for t in range(0, self.structural_time_steps):
            structural_out.append(self.structural_attn(graphs[t]))
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
            temporal_time_start = time.time()
            temporal_out = self.temporal_attn(fuse_structural_output)
            self.args['att_time'] += time.time() - temporal_time_start
            if self.args['partition'] == 'Time_Node':
                temporal_out = _node_partition_comm_after(self.args, temporal_out)

        else: 
            temporal_time_start = time.time()
            # print('rank: {} with fused tensor size{}'.format(self.args['rank'], structural_outputs_padded.size()))
            temporal_out = self.temporal_attn(structural_outputs_padded)
            self.args['att_time'] += time.time() - temporal_time_start

        return temporal_out

    # construct model
    def build_model(self):
        input_dim = self.num_features
        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
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