import os
import sys
import argparse
import multiprocessing as mp
import torch

from test_function import run_dgnn_distributed, run_dgnn

comm_method = 'gloo' # currently use 'gloo' for CPU process communication

# TODO: implement the test with pytest framework

def _test_distributed(rank, args, real_dist):
    world_size = args['world_size']
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')

    if real_dist:
        device = torch.cuda.set_device(rank)
        args['device'] = device
    else:  
        device = torch.device("cuda:0")
        args['device'] = device

    # init the communication group
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    torch.distributed.init_process_group(backend = comm_method,
                                         init_method = dist_init_method,
                                         world_size = world_size,
                                         rank = rank,
                                        )

    # init mp group for intermediate data exchange
    args['mp_group'] = [
        torch.distributed.new_group(
            ranks = [i for i in range (worker + 1)
            ],
            backend = comm_method,
        )
        for worker in range (world_size)
    ]

    # init dp group for gradients synchronization
    args['dp_group'] = torch.distributed.new_group(
                        ranks = list(range(world_size)), 
                        backend = comm_method,
                        )   

    args['rank'] = rank
    # print(locals())
    run_dgnn_distributed(args)


def _test_local(args):
    args['device'] = torch.device("cuda:0")
    args['world_size'] = 1
    args['rank'] = 0
    run_dgnn(args)


def _test_dp(args):
    real_dist = False
    world_size = args['world_size']

    workers = []
    for rank in range(world_size):
        p = mp.Process(target=_test_distributed, args=(rank, args, real_dist))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
    

def _test_ddp(args):
    world_size = args['world_size']
    assert torch.cuda.device_count() >= world_size,\
        'No enough GPU!'
    real_dist = True

    workers = []
    for rank in range(world_size):
        p = mp.Process(target=_test_distributed, args=(rank, args, real_dist))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()


def _get_args():
    import json

    parser = argparse.ArgumentParser(description='Test parameters')
    parser.add_argument('--json-path', type=str, required=True,
                        help='the path of hyperparameter json file')
    parser.add_argument('--test-type', type=str, required=True, choices=['local', 'dp', 'ddp'],
                        help='method for DGNN training')
    args = vars(parser.parse_args())
    with open(args['json_path'],'r') as load_f:
        para = json.load(load_f)
    args.update(para)

    return args


if __name__ == "__main__":
    args = _get_args()
    # print(args)

    method = args['test_type']

    if method == 'local':
        _test_local(args)
    elif method == 'dp':
        _test_dp(args)
    else: _test_ddp(args)

