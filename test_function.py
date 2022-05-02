import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import copy
import time

from data_process import load_graphs
from data_process import get_data_example
from data_process import load_dataset
from data_process import slice_graph
from data_process import convert_graphs

from Model.DySAT import DySAT
from Model.MLP import Classifier
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from customized_ddp import DistributedGroupedDataParallel as LocalDDP


class _My_DGNN(torch.nn.Module):
    def __init__(self, args, in_feats = None):
        super(_My_DGNN, self).__init__()
        self.dgnn = DySAT(args, num_features = in_feats)
        self.classificer = Classifier(in_feature = self.dgnn.out_feats)

    def set_comm(self):
        for p in self.dgnn.structural_attn.parameters():
            setattr(p, 'mp_comm', 'mp')
            setattr(p, 'dp_comm', 'dp')
        for p in self.dgnn.temporal_attn.parameters():
            setattr(p, 'dp_comm', 'dp')
        for p in self.classificer.parameters():
            setattr(p, 'dp_comm', 'dp')

    def forward(self, graphs, nids):
        final_emb = self.dgnn(graphs)
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


# TODO: complete the global forward
def run_dgnn_distributed(args):
    args['distributed'] = True
    device = args['device']
    rank = args['rank']
    world_size = args['world_size']
    if rank != world_size - 1:
        mp_group = args['mp_group'][rank]
    else: mp_group = None
    dp_group = args['dp_group']

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    # TODO: Unevenly slice graphs
    # load graphs
    load_g, load_adj, load_feats = slice_graph(*load_graphs(args))
    num_graph = len(load_g)
    args['structural_time_steps'] = num_graph
    args['temporal_time_steps'] = num_graph*(rank + 1)
    print("Loaded {}/{} graphs".format(num_graph, args['time_steps']))

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

    # TODO: How to take the global forward?
    # convert graphs to dgl or pyg graphs
    graphs = convert_graphs(load_g, load_adj, load_feats, args['data_str'])

    model = _My_DGNN(args, in_feats=load_feats[0].shape[1]).to(device)
    model.set_comm()
    # model = LocalDDP(copy.deepcopy(model), mp_group, dp_group, world_size)
    print('worker {} has already put the model to device {}'.format(rank, args['device']))
    model = DDP(model, process_group=dp_group)

    # loss_func = nn.BCELoss()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # results info
    epochs_f1_score = []
    epochs_auc = []
    epochs_acc = []
    total_train_time = 0
    total_comm_time = 0

    # train
    for epoch in range (args['epochs']):
        Loss = []
        epoch_train_time = []
        epoch_comm_time = []
        epoch_time_start = time.time()
        for step, (batch_x, batch_y) in enumerate(loader):
            model.train()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            train_start_time = time.time()
            graphs = [graph.to(device) for graph in graphs]
            out = model(graphs, batch_x)
            # print(out)
            loss = loss_func(out.squeeze(dim=-1), batch_y)
            Loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # print('epoch {} worker {} completes gradients computation!'.format(epoch, args['rank']))
            optimizer.step()
            epoch_train_time.append(time.time() - train_start_time)
            if args['distributed']:
                epoch_comm_time.append(args['comm_cost'])
                total_comm_time += args['comm_cost']
            else: epoch_comm_time.append(0)
        total_train_time += time.time() - epoch_time_start
        # print(out)
        # test
        if epoch % args['test_freq'] == 0 and rank != world_size - 1:
            graphs = [graph.to(device) for graph in graphs]
            test_result = model(graphs, torch.tensor(dataset['test_data']).to(device))

        elif epoch % args['test_freq'] == 0 and rank == world_size - 1:
            # model.eval()
            graphs = [graph.to(device) for graph in graphs]
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
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print("Epoch {:<3}, Loss = {:.3f}, F1 Score = {:.3f}, AUC = {:.3f}, ACC = {:.3f}, Time = {:.5f}|{:.5f}({:.3f}%), Memory Usage {:.2f}%".format(
                                                                epoch,
                                                                np.mean(Loss),
                                                                F1_result,
                                                                AUC,
                                                                ACC,
                                                                np.sum(epoch_train_time), np.sum(epoch_comm_time),
                                                                (np.sum(epoch_comm_time)/np.sum(epoch_train_time))*100,
                                                                gpu_mem_alloc/16160
                                                                ))

    # print the training result info
    if rank == world_size - 1:
        best_f1_epoch = epochs_f1_score.index(max(epochs_f1_score))
        best_auc_epoch = epochs_auc.index(max(epochs_auc))
        best_acc_epoch = epochs_acc.index(max(epochs_acc))

        print("Best f1 score epoch: {}, Best f1 score: {}".format(best_f1_epoch, max(epochs_f1_score)))
        print("Best auc epoch: {}, Best auc score: {}".format(best_auc_epoch, max(epochs_auc)))
        print("Best acc epoch: {}, Best acc score: {}".format(best_acc_epoch, max(epochs_acc)))
        print("Total training cost: {:.3f}, total communication cost: {:.3f}".format(total_train_time, total_comm_time))


def run_dgnn(args):
    r"""
    run dgnn with one process
    """
    args['distributed'] = False
    device = args['device']
    # args['time_steps'] = 4

    # TODO: Unevenly slice graphs
    # load graphs
    load_g, load_adj, load_feats = slice_graph(*load_graphs(args))
    num_graph = len(load_g)
    print("Loaded {}/{} graphs".format(num_graph, args['time_steps']))
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

    # train
    for epoch in range (args['epochs']):
        Loss = []
        for step, (batch_x, batch_y) in enumerate(loader):
            model.train()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            graphs = [graph.to(device) for graph in graphs]
            out = model(graphs, batch_x)
            # print(out)
            loss = loss_func(out.squeeze(dim=-1), batch_y)
            Loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(out)
        # test
        if epoch % args['test_freq'] == 0:
            # test_source_id = test_data[:, 0]
            # test_target_id = test_data[:, 1]
            # model.eval()
            graphs = [graph.to(device) for graph in graphs]
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
            print("Epoch {:<3}, Loss = {:.3f}, F1 Score {:.3f}, AUC {:.3f}, ACC {:.3f}".format(epoch,
                                                                np.mean(Loss),
                                                                F1_result,
                                                                AUC,
                                                                ACC))

    # print the training result info
    best_f1_epoch = epochs_f1_score.index(max(epochs_f1_score))
    best_auc_epoch = epochs_auc.index(max(epochs_auc))
    best_acc_epoch = epochs_acc.index(max(epochs_acc))

    print("Best f1 score epoch: {}, Best f1 score: {}".format(best_f1_epoch, max(epochs_f1_score)))
    print("Best auc epoch: {}, Best auc score: {}".format(best_auc_epoch, max(epochs_auc)))
    print("Best acc epoch: {}, Best acc score: {}".format(best_acc_epoch, max(epochs_acc)))








