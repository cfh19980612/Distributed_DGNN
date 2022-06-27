import scipy
import numpy as np
import scipy.sparse as sp
import argparse
import torch

from load_graph_data import load_graphs

# Simulation setting
# node_size = 10
# bandwidth = float(1000)
bandwidth_1MB = float(1024*1024*8)
bandwidth_10MB = float(10*1024*1024*8)
bandwidth_100MB = float(100*1024*1024*8)
bandwidth_GB = float(1024*1024*1024*8)


def generate_test_graph():
    num_snapshots = 3
    nodes_list = [torch.tensor(np.array([j for j in range(3+i*3)])) for i in range(num_snapshots)]
    adjs_list = [torch.ones(nodes_list[i].size(0), nodes_list[i].size(0)).to_sparse() for i in range(num_snapshots)]

    return nodes_list, adjs_list

def GCN_comm_nodes(nodes_list, adjs_list, num_devices, workloads_GCN):
    '''
    每个GPU只关心出口处和入口处的节点总数量 （big switch）
    Receive:
        STEP 1: 获取每个GPU需要接受的节点列表(mask方法)
        STEP 2: 计算接收节点的时间开销
    Send:
        STEP 1: 获取每个GPU需要向其他GPU发送的节点列表(mask方法)
        STEP 2: 计算发送节点的时间开销
    Total:
        Max(Receive, Send)
    '''
    receive_list = [[] for i in range(num_devices)]
    send_list = [[] for i in range(num_devices)]
    for time in range(len(nodes_list)):
        for device_id in range(num_devices):
            adj = adjs_list[time].clone()
            local_node_mask = workloads_GCN[device_id][time]
            remote_node_mask = ~workloads_GCN[device_id][time]
            edge_source = adj._indices()[0]
            edge_target = adj._indices()[1]

            # receive
            edge_source_local_mask = local_node_mask[edge_source] # check each source node whether it belongs to device_id
            need_receive_nodes = torch.unique(edge_target[edge_source_local_mask]) # get the target nodes with the source nodes belong to device_id
            receive_node_local = local_node_mask[need_receive_nodes] # check whether the received nodes in local?
            receive = torch.nonzero(receive_node_local == False, as_tuple=False).squeeze() # only the received nodes are not in local
            receive_list[device_id].append(receive.view(-1))

            # send
            edge_source_remote_mask = remote_node_mask[edge_source] # check each source node whether it belongs to other devices
            need_send_nodes = torch.unique(edge_target[edge_source_remote_mask]) # get the target nodes with the source nodes belong to other devices
            send_node_local = local_node_mask[need_send_nodes] # check whether the send nodes in local?
            send = torch.nonzero(send_node_local == True, as_tuple=False).squeeze() # only the send nodes are in local
            send_list[device_id].append(send.view(-1))
    
    return receive_list, send_list

def RNN_comm_nodes(nodes_list, num_devices, workloads_GCN, workloads_RNN):
    '''
    Compute the communication for RNN processing
    Receive:
        STEP 1: 比较GCN workload和RNN workload的区别
        STEP 2: RNN workload中为True，而GCN workload中为False的点即为要接收的点
    Send:
        STEP 1: 比较GCN workload和RNN workload的区别
        STEP 2: RNN workload中为False，而GCN workload中为True的点即为要发送的点
    '''
    receive_list = [[] for i in range(num_devices)]
    send_list = [[] for i in range(num_devices)]
    for device_id in range(num_devices):
        for time in range(len(nodes_list)):
            GCN_workload = workloads_GCN[device_id][time]
            RNN_workload = workloads_RNN[device_id][time]
            RNN_true_where = torch.nonzero(RNN_workload == True, as_tuple=False).squeeze()
            RNN_false_where = torch.nonzero(RNN_workload == False, as_tuple=False).squeeze()

            RNN_true_GCN_mask = GCN_workload[RNN_true_where]
            RNN_false_GCN_mask = GCN_workload[RNN_false_where]

            # receive: RNN true and GCN false
            receive = torch.nonzero(RNN_true_GCN_mask == False, as_tuple=False).squeeze()
            # send: RNN false and GCN true
            send = torch.nonzero(RNN_false_GCN_mask == True, as_tuple=False).squeeze()

            receive_list[device_id].append(receive.view(-1))
            send_list[device_id].append(send.view(-1))
    
    return receive_list, send_list

def Comm_time(num_devices, receive_list, send_list, node_size, bandwidth):
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

class node_partition():
    def __init__(self, args, nodes_list, adjs_list, num_devices):
        super(node_partition, self).__init__()
        '''
        //Parameter
            nodes_list: a list, in which each element is a [N_i,1] tensor to represent a node list of i-th snapshot
            adjs_list: a list, in which each element is a [N_i, N_i] tensor to represent a adjacency matrix of i-th snapshot
            num_devices: an integer constant to represent the number of devices
        '''
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.workload = [[] for i in range(num_devices)]

        self.workload_partition()

    def workload_partition(self):
        '''
        用bool来表示每个snapshot中的每个点属于哪一块GPU
        '''
        # for time, nodes in enumerate(self.nodes_list):
        #     num_of_nodes = nodes.size(0)
        #     nodes_per_device = num_of_nodes//self.num_devices
        #     for device_id in range(self.num_devices):
        #         work = torch.full_like(nodes, False, dtype=torch.bool)
        #         if device_id != self.num_devices - 1:
        #             work[nodes_per_device*device_id:nodes_per_device*(device_id+1)] = torch.ones(nodes_per_device, dtype=torch.bool)
        #         else:
        #             work[nodes_per_device*device_id:] = torch.ones(num_of_nodes - ((self.num_devices -1)*nodes_per_device), dtype=torch.bool)

        #         self.workload[device_id].append(work)
        
        num_nodes = self.nodes_list[-1].size(0)
        nodes_per_device = num_nodes // self.num_devices  # to guarantee the all temporal information of a same node will be in the same device
        node_partition_id = torch.tensor([0 for i in range(num_nodes)])
        for device_id in range(self.num_devices):
            if device_id != self.num_devices - 1:
                temp = torch.tensor([device_id for i in range(nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:(device_id + 1)*nodes_per_device] = temp
            else:
                temp = torch.tensor([device_id for i in range(num_nodes - (self.num_devices - 1)*nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:] = temp
        for device_id in range(self.num_devices):
            where_nodes = torch.nonzero(node_partition_id == device_id, as_tuple=False).squeeze()
            # print(where_nodes)
            nodes_local_mask = torch.full_like(self.nodes_list[-1], False, dtype=torch.bool)
            nodes_local_mask[where_nodes] = torch.ones(where_nodes.size(0), dtype=torch.bool)
            for (time, nodes) in enumerate(self.nodes_list):
                work = nodes_local_mask[nodes]
                self.workload[device_id].append(work)

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth):
        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workload)
        # print(GCN_receive_list)
        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)

        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_receive = [0 for i in range(self.num_devices)]
        RNN_send = [0 for i in range(self.num_devices)]

        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [0 for i in range(self.num_devices)]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)

        print('----------------------------------------------------------')
        print('Hybrid partition method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('RNN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with communication time: {} ( GCN: {} | RNN: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total communication time: {}'.format(max(GPU_total_time)))

class snapshot_partition():
    def __init__(self, args, nodes_list, adjs_list, num_devices):
        super(snapshot_partition, self).__init__()
        '''
        Snapshot partition [SC'21] first partitions the dynamic graphs via temporal dimention and then partition the workload via spatio dimention
        //Parameter
            nodes_list: a list, in which each element is a [N_i,1] tensor to represent a node list of i-th snapshot
            adjs_list: a list, in which each element is a [N_i, N_i] tensor to represent a adjacency matrix of i-th snapshot
            num_devices: an integer constant to represent the number of devices
        '''
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.workloads_GCN = [[] for i in range(num_devices)]
        self.workloads_RNN = [[] for i in range(num_devices)]

        self.workload_partition()
    
    def workload_partition(self):
        '''
        STEP 1: partition graphs via temporal dimension (for GCN process)
        STEP 2: partition graphs via spatio dimension (for RNN process)
        '''
        # temporal partition
        timesteps = len(self.nodes_list)
        time_per_device = timesteps // self.num_devices
        time_partition_id = torch.tensor([0 for i in range(timesteps)])
        for device_id in range(self.num_devices):
            if device_id != self.num_devices - 1:
                temp = torch.tensor([device_id for i in range(time_per_device)])
                time_partition_id[device_id*time_per_device:(device_id + 1)*time_per_device] = temp
            else:
                temp = torch.tensor([device_id for i in range(timesteps - (self.num_devices - 1)*time_per_device)])
                time_partition_id[device_id*time_per_device:] = temp
        for (time, nodes) in enumerate(self.nodes_list):
            for device_id in range(self.num_devices):
                if time_partition_id[time] == device_id:
                    work = torch.full_like(nodes, True, dtype=torch.bool)
                else:
                    work = torch.full_like(nodes, False, dtype=torch.bool)
                self.workloads_GCN[device_id].append(work)
        # print(self.workload_GCN[-1])

        # spatio partition
        num_nodes = self.nodes_list[-1].size(0)
        nodes_per_device = num_nodes // self.num_devices  # to guarantee the all temporal information of a same node will be in the same device
        node_partition_id = torch.tensor([0 for i in range(num_nodes)])
        for device_id in range(self.num_devices):
            if device_id != self.num_devices - 1:
                temp = torch.tensor([device_id for i in range(nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:(device_id + 1)*nodes_per_device] = temp
            else:
                temp = torch.tensor([device_id for i in range(num_nodes - (self.num_devices - 1)*nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:] = temp
        for device_id in range(self.num_devices):
            where_nodes = torch.nonzero(node_partition_id == device_id, as_tuple=False).squeeze()
            # print(where_nodes)
            nodes_local_mask = torch.full_like(self.nodes_list[-1], False, dtype=torch.bool)
            nodes_local_mask[where_nodes] = torch.ones(where_nodes.size(0), dtype=torch.bool)
            for (time, nodes) in enumerate(self.nodes_list):
                work = nodes_local_mask[nodes]
                self.workloads_RNN[device_id].append(work)

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth):
        RNN_receive_list, RNN_send_list = RNN_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_RNN)

        RNN_receive_comm_time, RNN_send_comm_time = Comm_time(self.num_devices, RNN_receive_list, RNN_send_list, RNN_node_size, bandwidth)

        GCN_receive = [0 for i in range(self.num_devices)]
        GCN_send = [0 for i in range(self.num_devices)]
        RNN_receive = [torch.cat(RNN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_send = [torch.cat(RNN_send_list[i], 0).size(0) for i in range(self.num_devices)]

        GCN_comm_time = [0 for i in range(self.num_devices)]
        RNN_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i]) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)

        print('----------------------------------------------------------')
        print('Hybrid partition method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('RNN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with communication time: {} ( GCN: {} | RNN: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total communication time: {}'.format(max(GPU_total_time)))

class hybrid_partition():
    def __init__(self, args, nodes_list, adjs_list, num_devices):
        super(hybrid_partition, self).__init__()
        '''
        Snapshot partition [SC'21] first partitions the dynamic graphs via temporal dimention and then partition the workload via spatio dimention
        //Parameter
            nodes_list: a list, in which each element is a [N_i,1] tensor to represent a node list of i-th snapshot
            adjs_list: a list, in which each element is a [N_i, N_i] tensor to represent a adjacency matrix of i-th snapshot
            num_devices: an integer constant to represent the number of devices
        '''
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.workloads_GCN = [[] for i in range(num_devices)]
        self.workloads_RNN = [[] for i in range(num_devices)]

        self.workload_partition()
    
    def workload_partition(self):
        '''
        先按点划分，因为点列表中，最前面的点拥有最长的时序
        再按时序划分，划分时注意每个时序图中已经被分配的节点
        '''
        partition_method = [0 for i in range(self.num_devices)] # 0: node partition; 1: snapshot partition
        for i in range(self.num_devices):
            if i >= (self.num_devices // 2):
                partition_method[i] = 1 # snapshot partition
        
        # STEP 1: the same RNN workloads
        num_nodes = self.nodes_list[-1].size(0)
        nodes_per_device = num_nodes // self.num_devices  # to guarantee the all temporal information of a same node will be in the same device
        node_partition_id = torch.tensor([0 for i in range(num_nodes)])
        for device_id in range(self.num_devices):
            if device_id != self.num_devices - 1:
                temp = torch.tensor([device_id for i in range(nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:(device_id + 1)*nodes_per_device] = temp
            else:
                temp = torch.tensor([device_id for i in range(num_nodes - (self.num_devices - 1)*nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:] = temp
        for device_id in range(self.num_devices):
            where_nodes = torch.nonzero(node_partition_id == device_id, as_tuple=False).squeeze()
            # print(where_nodes)
            nodes_local_mask = torch.full_like(self.nodes_list[-1], False, dtype=torch.bool)
            nodes_local_mask[where_nodes] = torch.ones(where_nodes.size(0), dtype=torch.bool)
            for (time, nodes) in enumerate(self.nodes_list):
                work = nodes_local_mask[nodes]
                self.workloads_RNN[device_id].append(work)
                # if partition_method[device_id] == 0:
                #     self.workloads_GCN[device_id].append(work)

        # print(self.workloads_RNN[0])
        # STEP 2: different partitions
        # num_nodes = self.nodes_list[-1].size(0)
        # nodes_per_device = num_nodes // 3
        for device_id in range(self.num_devices):
            if partition_method[device_id] == 0:
                if device_id != self.num_devices - 1:
                    temp = torch.tensor([device_id for i in range(nodes_per_device)])
                    node_partition_id[device_id*nodes_per_device:(device_id + 1)*nodes_per_device] = temp
                else:
                    temp = torch.tensor([device_id for i in range(num_nodes - (self.num_devices - 1)*nodes_per_device)])
                    node_partition_id[device_id*nodes_per_device:] = temp
        for device_id in range(self.num_devices):
            if partition_method[device_id] == 0:
                where_nodes = torch.nonzero(node_partition_id == device_id, as_tuple=False).squeeze()
                # print(where_nodes)
                nodes_local_mask = torch.full_like(self.nodes_list[-1], False, dtype=torch.bool)
                nodes_local_mask[where_nodes] = torch.ones(where_nodes.size(0), dtype=torch.bool)
                for (time, nodes) in enumerate(self.nodes_list):
                    work = nodes_local_mask[nodes]
                    self.workloads_GCN[device_id].append(work)

        # print(self.workloads_GCN[0])
        # STEP 3: snapshot partition
        # update graphs: if all nodes in some snapshots are partitioned already, these snapshot should not be partitioned again
        whether_partitioned = torch.zeros(len(self.nodes_list), dtype=torch.bool)
        partition_nodes_list = []
        for (time, node) in enumerate(self.nodes_list):
            partition_nodes = []
            for device_id in range(self.num_devices):
                if partition_method[device_id] == 0:
                    partition_nodes.append(torch.nonzero(self.workloads_GCN[device_id][time] == True, as_tuple=False).squeeze())
            already_partitioned = torch.cat(partition_nodes, dim=0)
            if node.size(0) == already_partitioned.size(0):
                whether_partitioned[time] = True
            partition_nodes_list.append(already_partitioned)
        need_partition_snapshot = torch.nonzero(whether_partitioned == False, as_tuple=False).squeeze()
        timesteps = need_partition_snapshot.size(0)
        partitioned_timesteps = len(self.nodes_list) - timesteps
        devices_for_snapshot_partition = self.num_devices - (self.num_devices//2)
        time_per_device = timesteps // devices_for_snapshot_partition

        snapshot_partition_id = torch.tensor([-1 for i in range(len(self.nodes_list))])
        # print(devices_for_snapshot_partition)
        # print(time_per_device)
        for device_id in range(devices_for_snapshot_partition):
            if device_id != devices_for_snapshot_partition - 1:
                temp = torch.tensor([device_id + self.num_devices//2 for i in range(time_per_device)])
                snapshot_partition_id[partitioned_timesteps + device_id*time_per_device:partitioned_timesteps + (device_id + 1)*time_per_device] = temp
            else:
                temp = torch.tensor([device_id + self.num_devices//2 for i in range(timesteps - (devices_for_snapshot_partition - 1)*time_per_device)])
                snapshot_partition_id[partitioned_timesteps + device_id*time_per_device:] = temp
        # print(snapshot_partition_id)
        for device_id in range(self.num_devices):
            if partition_method[device_id] == 1:
                for (time, nodes) in enumerate(self.nodes_list):
                    if whether_partitioned[time] == False and snapshot_partition_id[time] == device_id:
                        work = torch.full_like(nodes, True, dtype=torch.bool)
                        work[partition_nodes_list[time]] = torch.zeros(partition_nodes_list[time].size(0), dtype=torch.bool)
                        self.workloads_GCN[device_id].append(work)
                    else:
                        work = torch.full_like(nodes, False, dtype=torch.bool)
                        self.workloads_GCN[device_id].append(work)
        # print(self.workloads_GCN[-1])

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth):
        '''
        Both GCN communication time and RNN communication time are needed
        '''
        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workloads_GCN)
        RNN_receive_list, RNN_send_list = RNN_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_RNN)

        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)
        RNN_receive_comm_time, RNN_send_comm_time = Comm_time(self.num_devices, RNN_receive_list, RNN_send_list, RNN_node_size, bandwidth)

        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_receive = [torch.cat(RNN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_send = [torch.cat(RNN_send_list[i], 0).size(0) for i in range(self.num_devices)]

        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i]) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)

        print('----------------------------------------------------------')
        print('Hybrid partition method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('RNN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with communication time: {} ( GCN: {} | RNN: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total communication time: {}'.format(max(GPU_total_time)))

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

    # validation
    # nodes_list, adjs_list = generate_test_graph()

    _, graphs, adj_matrices, feats, _ = load_graphs(args)
    # print('Generate graphs!')
    graphs = graphs[68:]

    Num_nodes = args['nodes_info']
    time_steps = len(graphs)
    nodes_list = [torch.tensor([j for j in range(Num_nodes[i])]) for i in range(time_steps)]
    # print('Generate nodes list!')
    adjs_list = []
    for i in range(time_steps):
        # print(type(adj_matrices[i]))
        adj_coo = adj_matrices[i].tocoo()
        values = adj_coo.data
        indices = np.vstack((adj_coo.row, adj_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj_coo.shape

        adj_tensor_sp = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        adjs_list.append(adj_tensor_sp)
    
    print('Number of graphs: ', len(graphs))
    GCN_node_size = feats[0].size(0)*32
    RNN_node_size = 256*32

    node_partition_obj = node_partition(args, nodes_list, adjs_list, num_devices=args['world_size'])
    node_partition_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_1MB)

    snapshot_partition_obj = snapshot_partition(args, nodes_list, adjs_list, num_devices=args['world_size'])
    snapshot_partition_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_1MB)

    hybrid_partition_obj = hybrid_partition(args, nodes_list, adjs_list, num_devices=args['world_size'])
    hybrid_partition_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_1MB)
    # print(node_partition_obj.workload[0])

    # _, graphs, adj_matrices, feats, _ = load_graphs(args)
    # print('Graph nodes information: ',args['nodes_info'])
    # print('Graph edges information: ',args['edges_info'])
    # print('feature demension is ', feats[0].shape)

    # # graphs = graphs[:2]
    # print('Converting graphs to specific framework!')
    # graphs_new = convert_graphs(graphs, adj_matrices, feats, 'dgl')




    

