#!/usr/bin/env python
# from __future__ import print_function

import os

import torch
from torch.utils import tensorboard

import torch.distributed as dist
import torch.multiprocessing as mp

from utils.parser import get_parser
from utils.parser import load_parser_args_from_config
from utils.processor import Processor
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
    args = load_parser_args_from_config(parser)

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
