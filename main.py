#!/usr/bin/env python
# from __future__ import print_function

import argparse
import json
import os
import random
import sys
import yaml

import torch
from torch.utils import tensorboard

import torch.distributed as dist
import torch.multiprocessing as mp

from utils.parser import *
from utils.processor import *
from utils.utils import *

# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8020'  # str(random.randint(8100, 8200))
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successfully
    # communicate across multiple process involving multiple GPUs.


def cleanup():
    dist.destroy_process_group()


def train_model(rank, args):
    print(f"Running DDP on rank {rank}.")
    torch.cuda.set_device(rank)
    setup(rank, args.world_size)
    init_seed(args.seed)
    processor = Processor(args, rank=rank)
    processor.start()
    cleanup()


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        if ".yaml" in p.config:
            with open(p.config, 'r') as f:
                default_arg = yaml.safe_load(f)
        elif ".json" in p.config:
            with open(p.config, 'r') as f:
                default_arg = json.load(f)
            default_arg = {k: v
                           for _, kv in default_arg.items()
                           for k, v in kv.items()}
        else:
            raise ValueError("Unknown config format...")

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WRONG ARG: {k}')
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()

    if args.ddp:
        # this is responsible for spawning 'nprocs' number of processes of the
        # train_func function with the given arguments as 'args'
        # # since this example shows a single process per GPU, the number of
        # # processes is simply replaced with the
        # # number of GPUs available for training.
        # n_gpus = torch.cuda.device_count()
        # run_train_model(train_model, n_gpus)
        mp.spawn(train_model, args=(args,), nprocs=args.world_size, join=True)

    else:
        init_seed(args.seed)
        processor = Processor(args)
        processor.start()
        print()
