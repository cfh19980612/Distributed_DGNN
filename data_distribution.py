import scipy
import numpy as np
import scipy.sparse as sp
import argparse
import torch
import networkx as nx
import time

# public APIs
# compute the cross edges when schedyke workload p and q on m device
def Cross_edges(timesteps, adjs, nodes_list, Degrees, current_workload, workload, flag):
    num = 0
    if flag == 0:
        # graph-graph cross edges at a timestep
        # method 1: compute cross edges per node with sparse tensor (slow but memory efficient)
        time = workload[0]
        nodes = workload[1].tolist()
        adj = adjs[time].clone()
        edge_source = adj._indices()[0]
        edge_target = adj._indices()[1]
        idx_list = [torch.nonzero(edge_source == node, as_tuple=False).view(-1) for node in nodes]
        nodes_idx_list = [edge_target[idx] for idx in idx_list if idx.dim() != 0]
        if len(nodes_idx_list) > 0:
            nodes_idx = torch.cat((nodes_idx_list), dim=0)
            has_nodes = torch.nonzero(current_workload[time][nodes_idx] == True, as_tuple=False).view(-1)
            num += has_nodes.size(0)/sum(Degrees[time])
        # print(num)

    # node-graph cross edges at multiple timesteps
    else:
        time = workload[0]
        node_id = workload[1]
        adj = adjs[time].clone()
        edge_source = adj._indices()[0]
        edge_target = adj._indices()[1]
        # print(edge_source, edge_target)
        idx = torch.nonzero(edge_source == node_id, as_tuple=False).view(-1)
        # print(idx)
        nodes_idx = edge_target[idx]
        # print(nodes_idx)
        has_nodes = torch.nonzero(current_workload[time][nodes_idx] == True, as_tuple=False).view(-1)
        # print('all degrees: ',sum(Degrees[time]))
        num += has_nodes.size(0)/sum(Degrees[time])
    return num

# compute the cross nodes when schedule workload p on m device
def Cross_nodes(timesteps, nodes_list, current_workload, workload):
    num = 0
    same_nodes = []
    for time in range(timesteps):
        if nodes_list[time][-1] >= workload[-1]:
            same_nodes.append(current_workload[time][workload])
    if len(same_nodes) > 0:
        same_nodes_tensor = torch.cat((same_nodes), dim=0)
        has_nodes = torch.nonzero(same_nodes_tensor == True, as_tuple=False).view(-1)
        num += has_nodes.size(0)/(workload.size(0)*len(same_nodes))
    # print(num)
    return num

# partition workload in P and Q via a balanced method
def workload_balance(P_id, P_workload, P_snapshot, Q_id, Q_node_id, Q_workload, graphs, nodes_list, adjs_list, timesteps, num_devices):
    '''
    Schedule snapshot-level jobs first or schedule timeseries-level jobs first?
    '''
    Degrees = [list(dict(nx.degree(graphs[t])).values()) for t in range(timesteps)]
    workloads_GCN = [[torch.full_like(nodes_list[time], False, dtype=torch.bool) for time in range(timesteps)] for i in range(num_devices)]
    workloads_RNN = [[torch.full_like(nodes_list[time], False, dtype=torch.bool) for time in range(timesteps)] for i in range(num_devices)]

    Current_workload = [0 for i in range(num_devices)]
    Current_RNN_workload = [[0 for i in range(timesteps)]for m in range(num_devices)]
    # compute the average workload
    avg_workload = (sum(P_workload) + sum(Q_workload))/num_devices

    time_cost = 0
    for idx in range(len(Q_id)):
        Load = []
        Cross_edge = []
        for m in range(num_devices):
            Load.append(1 - float((Current_workload[m] + Q_workload[idx])/avg_workload))
            start = time.time()
            # Cross_edge.append(Cross_edges(timesteps, adjs_list, nodes_list, Degrees, workloads_GCN[m], (Q_id[idx], Q_node_id[idx]), flag=1))
            time_cost += time.time() - start
        # Cross_edge = [ce*args['beta'] for ce in Cross_edge]
        # result = np.sum([Load, Cross_edge], axis = 0).tolist()
        select_m = result.index(max(result))
        # select_m = Load.index(max(Load))
        # for m in range(num_devices):
        #     if m == select_m:
        for t in range(timesteps)[Q_id[idx]:]:
            # print(workloads_GCN[m][time])
            workloads_GCN[select_m][t][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
            workloads_RNN[select_m][t][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
            # Scheduled_workload[time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
        Current_workload[select_m] = Current_workload[select_m] + Q_workload[idx]
    # print('compute graph-graph cross edges time costs: ', time_cost_edges)
    # print('compute cross nodes time costs: ', time_cost_nodes)

    # print('compute node-graph cross edges time costs: ', time_cost)
    # print('GCN workload after scheduling timeseries-level jobs: ', self.workloads_GCN)

    time_cost_edges = 0
    time_cost_nodes = 0
    for idx in range(len(P_id)): # schedule snapshot-level job
        Load = []
        Cross_edge = []
        Cross_node = []
        for m in range(num_devices):
            Load.append(1 - float((Current_workload[m]+P_workload[idx])/avg_workload))
            # Cross_edge.append(Current_RNN_workload[m][P_id[idx]])
            start = time.time()
            Cross_edge.append(Cross_edges(timesteps, adjs_list, nodes_list, Degrees, workloads_GCN[m], (P_id[idx],P_snapshot[idx]), flag=0))
            time_cost_edges += time.time() - start
            start = time.time()
            Cross_node.append(Cross_nodes(timesteps, nodes_list, workloads_GCN[m], P_snapshot[idx]))
            time_cost_nodes+=  time.time() - start
        # print(Load, Cross_edge, Cross_node)
        result = np.sum([Load,Cross_node],axis=0).tolist()
        result = np.sum([result,Cross_edge],axis=0).tolist()

        # select_m = result.index(max(result))
        select_m = Load.index(max(Load))

        Node_start_idx = nodes_list[P_id[idx]].size(0) - P_workload[idx]
        workload = torch.full_like(P_snapshot[idx], True, dtype=torch.bool)
        workloads_GCN[select_m][P_id[idx]][P_snapshot[idx]] = workload
        workloads_RNN[select_m][P_id[idx]][P_snapshot[idx]] = workload
        Current_workload[select_m] = Current_workload[select_m]+P_workload[idx]
        Current_RNN_workload[select_m][P_id[idx]] += 1
        # print('GCN workload after scheduling snapshot-level jobs: ', self.workloads_GCN)
    return workloads_GCN, workloads_RNN

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
        self.workloads_GCN = [[] for i in range(num_devices)]
        self.workloads_RNN = [[] for i in range(num_devices)]

        self.workload_partition()

    def workload_partition(self):
        '''
        用bool来表示每个snapshot中的每个点属于哪一块GPU
        '''
        
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
                self.workloads_GCN[device_id].append(work)
                self.workloads_RNN[device_id].append(work)
        return self.workloads_GCN, self.workloads_RNN


class node_partition_balance():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(node_partition_balance, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)

        # runtime
        self.P_id, self.Q_id, self.Q_node_id, self.P_workload, self.P_snapshot, self.Q_workload = self.partition()

    def group(self):
        '''
        Step 1: partition snapshot into P set; partition nodes into Q set
        '''
        Total_workload = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        Total_workload_RNN = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        num_generations = [i+1 for i in range(self.timesteps)]
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
        Q_workload = []
        Degree = []
        for time in range(self.timesteps):
            if time == 0:
                    start = 0
            else:
                start = self.nodes_list[time - 1].size(0)
            end = self.nodes_list[time].size(0)
            workload = self.nodes_list[time][start:end]
            for node in workload.tolist():
                Q_id.append(time)
                # divided_nodes = self.nodes_list[time].size(0) - workload.size(0)
                Q_node_id.append(node)
                Q_workload.append(self.timesteps - time)
        return P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload
    
    def workload_partition(self):

        workloads_GCN, workloads_RNN = workload_balance(self.P_id, self.P_workload, self.P_snapshot, self.Q_id, self.Q_node_id, self.Q_workload, 
                                                            self.graphs, self.nodes_list, self.adjs_list, self.timesteps, self.num_devices)
        
        return workloads_GCN, workloads_RNN


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
        
        return self.workloads_GCN, self.workloads_RNN


class snapshot_partition_balance():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(snapshot_partition_balance, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_RNN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        # runtime
        self.P_id, self.Q_id, self.Q_node_id, self.P_workload, self.P_snapshot, self.Q_workload = self.group()

    def group(self):
        '''
        Step 1: partition snapshot into P set; partition nodes into Q set
        '''
        Total_workload = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        Total_workload_RNN = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        num_generations = [i+1 for i in range(self.timesteps)]
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
        Q_workload = []
        Degree = []
        for time in range(self.timesteps):
            if time == 0:
                    start = 0
            else:
                start = self.nodes_list[time - 1].size(0)
            end = self.nodes_list[time].size(0)

            workload_gcn = self.nodes_list[time]
            P_id.append(time)
            P_workload.append(workload_gcn.size(0))
            P_snapshot.append(workload_gcn)

            workload_rnn = self.nodes_list[time][start:end]
            for node in workload_rnn.tolist():
                Q_id.append(time)
                # divided_nodes = self.nodes_list[time].size(0) - workload.size(0)
                Q_node_id.append(node)
                Q_workload.append(self.timesteps - time)
        return P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload
    
    def workload_partition(self):
        Scheduled_workload = [torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)]
        Current_GCN_workload = [0 for i in range(self.num_devices)]
        Current_RNN_workload = [0 for i in range(self.num_devices)]
        # compute the average workload
        GCN_avg_workload = np.sum(self.P_workload)/self.num_devices
        RNN_avg_workload = np.sum(self.Q_workload)/self.num_devices

        for idx in range(len(self.P_id)): # schedule snapshot-level job to GCN workload
            Load = []
            Cross_edge = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_GCN_workload[m]+self.P_workload[idx])/GCN_avg_workload))
                # Cross_edge.append(Current_RNN_workload[m][P_id[idx]])
            select_m = Load.index(max(Load))
            workload = torch.full_like(self.P_snapshot[idx], True, dtype=torch.bool)
            self.workloads_GCN[select_m][self.P_id[idx]][self.P_snapshot[idx]] = workload

            Current_GCN_workload[select_m] = Current_GCN_workload[select_m]+self.P_workload[idx]
            # Current_RNN_workload[select_m][P_id[idx]] += 1

        # print('GCN workload after scheduling snapshot-level jobs: ', self.workloads_GCN)

        for idx in range(len(self.Q_id)):  # schedule node-level job to RNN workload
            Load = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_RNN_workload[m] + self.Q_workload[idx])/RNN_avg_workload))
            select_m = Load.index(max(Load))
            # for m in range(self.num_devices):
            #     if m == select_m:
            for time in range(self.timesteps)[self.Q_id[idx]:]:
                # print(self.workloads_GCN[m][time])
                self.workloads_RNN[select_m][time][self.Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                
                # Scheduled_workload[time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
            Current_RNN_workload[select_m] = Current_RNN_workload[select_m] + self.Q_workload[idx]
        # print('GCN workload after scheduling timeseries-level jobs: ', self.workloads_GCN)
        return self.workloads_GCN, self.workloads_RNN


class hybrid_partition():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(hybrid_partition, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_RNN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        self.Degrees = [list(dict(nx.degree(self.graphs[t])).values()) for t in range(self.timesteps)]

        # parameters
        # self.alpha = args['alpha']
        # self.alpha = 0.08
        # self.alpha = 0.01
        # self.alpha = 0.1
        self.alpha = 0.001

        # runtime
        self.P_id, self.Q_id, self.Q_node_id, self.P_workload, self.P_snapshot, self.Q_workload = self.group()
        # print('divide time cost: ', time.time() - start)
        # print('P_id: ',P_id)
        # print('Q_id: ',Q_id)
        # print('Q_node_id: ',Q_node_id)
        # print('P_workload: ',P_workload)
        # print('Q_workload: ',Q_workload)
        # print('P_snapshot: ',P_snapshot)


    def group(self):
        '''
        Step 1: compute the average degree of each snapshots
        Step 2: divide nodes into different job set according to the degree and time-series length
        '''
        Total_workload = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        Total_workload_temp = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        num_generations = [[j for j in range(i+1)] for i in range(self.timesteps)]
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
        Q_workload = []
        Degree = []
        for t in range(self.timesteps):
            for generation in num_generations[t]:
                # compute average degree of nodes in specific generation
                if generation == 0:
                    start = 0
                else:
                    start = self.nodes_list[generation - 1].size(0)
                end = self.nodes_list[generation].size(0)
                Degree_list = list(dict(nx.degree(self.graphs[t])).values())[start:end]
                avg_deg = np.mean(Degree_list)
                Degree.append(avg_deg)
                # print('alpha: ',self.alpha)
                # print('generation; ',generation)
                workload = self.nodes_list[t][start:end]
                if avg_deg > self.alpha*(self.timesteps - t): # GCN-sensitive job
                    P_id.append(t)
                    P_workload.append(workload.size(0))
                    P_snapshot.append(workload)
                else:
                    for node in workload.tolist():
                        Q_id.append(t)
                        # divided_nodes = self.nodes_list[time].size(0) - workload.size(0)
                        Q_node_id.append(node)
                        Q_workload.append(self.timesteps - t)
                    # update following snapshots
                    for k in range(self.timesteps)[t+1:]:
                        mask = torch.full_like(Total_workload[k], True, dtype=torch.bool)
                        mask[start:end] = torch.zeros(mask[start:end].size(0), dtype=torch.bool)
                        where = torch.nonzero(mask == True, as_tuple=False).view(-1)
                        Total_workload[k] = Total_workload[k][where]
                        num_generations[k] = num_generations[k][1:]


            # # compute average degree of the graphs
            # Degree_list = list(dict(nx.degree(self.graphs[time])).values())
            # avg_deg = np.mean(Degree_list)
            # Degree.append(avg_deg)

            # if avg_deg > self.alpha*(self.timesteps - time): # GCN-sensitive job
            #     P_id.append(time)
            #     P_workload.append(Total_workload[time].size(0))
            # else:                                            # RNN-sensitive job
            #     for node in range(Total_workload[time].size(0)):
            #         Q_id.append(time)
            #         divided_nodes = self.nodes_list[time].size(0) - Total_workload[time].size(0)
            #         Q_node_id.append(node + divided_nodes)
            #         Q_workload.append(self.timesteps - time)
            #     # update following snapshots
            #     for k in range(self.timesteps)[time+1:]:
            #         update_size = Total_workload[time].size(0)
            #         Total_workload[k] = Total_workload[k][update_size:]

        return P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload
    
    def workload_partition(self):
        '''
        Schedule snapshot-level jobs first or schedule timeseries-level jobs first?
        '''
        Current_workload = [0 for i in range(self.num_devices)]
        Current_RNN_workload = [[0 for i in range(self.timesteps)]for m in range(self.num_devices)]

        timeseries_per_device = len(self.Q_id) // self.num_devices
        for idx in range(len(self.Q_id)):

            select_m = idx // timeseries_per_device
            if select_m >= self.num_devices:
                select_m = self.num_devices - 1
            for t in range(self.timesteps)[self.Q_id[idx]:]:
                # print(self.workloads_GCN[m][time])
                self.workloads_GCN[select_m][t][self.Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                self.workloads_RNN[select_m][t][self.Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                # Scheduled_workload[time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
            Current_workload[select_m] = Current_workload[select_m] + self.Q_workload[idx]

        snapshot_per_device = len(self.P_id) // self.num_devices
        for idx in range(len(self.P_id)): # schedule snapshot-level job

            if snapshot_per_device != 0:
                select_m = idx // snapshot_per_device
            else:
                select_m = 0
            if select_m >= self.num_devices:
                select_m = self.num_devices - 1

            workload = torch.full_like(self.P_snapshot[idx], True, dtype=torch.bool)
            self.workloads_GCN[select_m][self.P_id[idx]][self.P_snapshot[idx]] = workload
            self.workloads_RNN[select_m][self.P_id[idx]][self.P_snapshot[idx]] = workload
            Current_workload[select_m] = Current_workload[select_m]+self.P_workload[idx]
            Current_RNN_workload[select_m][self.P_id[idx]] += 1
        
        return self.workloads_GCN, self.workloads_RNN


class hybrid_partition_balance():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(hybrid_partition_balance, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.Degrees = [list(dict(nx.degree(self.graphs[t])).values()) for t in range(self.timesteps)]

        # parameters
        # self.alpha = args['alpha']
        # self.alpha = 0.08
        # self.alpha = 0.01
        self.alpha = 0.001

        # runtime
        self.P_id, self.Q_id, self.Q_node_id, self.P_workload, self.P_snapshot, self.Q_workload = self.group()
        # print('divide time cost: ', time.time() - start)
        # print('P_id: ',P_id)
        # print('Q_id: ',Q_id)
        # print('Q_node_id: ',Q_node_id)
        # print('P_workload: ',P_workload)
        # print('Q_workload: ',Q_workload)
        # print('P_snapshot: ',P_snapshot)


    def group(self):
        '''
        Step 1: compute the average degree of each snapshots
        Step 2: divide nodes into different job set according to the degree and time-series length
        '''
        Total_workload = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        num_generations = [[j for j in range(i+1)] for i in range(self.timesteps)]
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
        Q_workload = []
        Degree = []
        for t in range(self.timesteps):
            for generation in num_generations[t]:
                # compute average degree of nodes in specific generation
                if generation == 0:
                    start = 0
                else:
                    start = self.nodes_list[generation - 1].size(0)
                end = self.nodes_list[generation].size(0)
                Degree_list = list(dict(nx.degree(self.graphs[t])).values())[start:end]
                avg_deg = np.mean(Degree_list)
                Degree.append(avg_deg)
                # print('alpha: ',self.alpha)
                # print('generation; ',generation)
                workload = self.nodes_list[t][start:end]
                if avg_deg > self.alpha*(self.timesteps - t): # GCN-sensitive job
                    P_id.append(t)
                    P_workload.append(workload.size(0))
                    P_snapshot.append(workload)
                else:
                    for node in workload.tolist():
                        Q_id.append(t)
                        # divided_nodes = self.nodes_list[time].size(0) - workload.size(0)
                        Q_node_id.append(node)
                        Q_workload.append(self.timesteps - t)
                    # update following snapshots
                    for k in range(self.timesteps)[t+1:]:
                        mask = torch.full_like(Total_workload[k], True, dtype=torch.bool)
                        mask[start:end] = torch.zeros(mask[start:end].size(0), dtype=torch.bool)
                        where = torch.nonzero(mask == True, as_tuple=False).view(-1)
                        Total_workload[k] = Total_workload[k][where]
                        num_generations[k] = num_generations[k][1:]

        return P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload
    
    def workload_partition(self):
        '''
        Schedule snapshot-level jobs first or schedule timeseries-level jobs first?
        '''
        workloads_GCN, workloads_RNN = workload_balance(self.P_id, self.P_workload, self.P_snapshot, self.Q_id, self.Q_node_id, self.Q_workload, 
                                                        self.graphs, self.nodes_list, self.adjs_list, self.timesteps, self.num_devices)
        return workloads_GCN, workloads_RNN
