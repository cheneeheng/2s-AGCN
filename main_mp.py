#!/usr/bin/env python
# from __future__ import print_function

import os
import sys
import argparse
import yaml

import torch

from torch.utils import tensorboard

import torch.distributed as dist
import torch.multiprocessing as mp

from main_processor import *
from main_utils import *


# ------ Setting up the distributed environment -------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successfully
    # communicate across multiple process
    # involving multiple GPUs.


def cleanup():
    dist.destroy_process_group()


def train_model(rank, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, args.world_size)
    init_seed(args.seed)
    torch.cuda.device(rank)
    processor = Processor(args)
    processor.start()
    cleanup()


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WRONG ARG: {k}')
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()

    # this is responsible for spawning 'nprocs' number of processes of the
    # train_func function with the given
    # arguments as 'args'
    mp.spawn(train_model, args=(args,), nprocs=args.world_size, join=True)

    # # since this example shows a single process per GPU, the number of
    # # processes is simply replaced with the
    # # number of GPUs available for training.
    # n_gpus = torch.cuda.device_count()
    # run_train_model(train_model, n_gpus)
