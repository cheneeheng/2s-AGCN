#!/usr/bin/env python
# from __future__ import print_function

import argparse
import inspect
import json
import numpy as np
import os
import pickle
import shutil
import time
import yaml
from collections import OrderedDict
from tqdm import tqdm
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter

from feeders.loader import FeederDataLoader
from utils.loss import CategorialFocalLoss
from utils.loss import CosineLoss
from utils.loss import LabelSmoothingLoss
from utils.loss import MaximumMeanDiscrepancyLoss

from sam.sam.sam import SAM
from sam.sam.example.utility.bypass_bn import enable_running_stats
from sam.sam.example.utility.bypass_bn import disable_running_stats

from utils.utils import *


__all__ = ['Processor']


def get_vector_property(x):
    N, C = x.size()
    x1 = x.unsqueeze(0).expand(N, N, C)
    x2 = x.unsqueeze(1).expand(N, N, C)
    x1 = x1.reshape(N*N, C)
    x2 = x2.reshape(N*N, C)
    cos_sim = F.cosine_similarity(x1, x2, dim=1, eps=1e-6).view(N, N)
    cos_sim = torch.triu(cos_sim, diagonal=1).sum() * 2 / (N*(N-1))
    pdist = (LA.norm(x1-x2, ord=2, dim=1)).view(N, N)
    pdist = torch.triu(pdist, diagonal=1).sum() * 2 / (N*(N-1))
    return cos_sim, pdist


class Processor(object):
    """Processor for Skeleton-based Action Recognition """

    def __init__(self,
                 arg: argparse.Namespace,
                 rank: int = 0,
                 save_arg: bool = True):
        self.num_forward_inputs = 0

        self.rank = rank
        self.arg = arg
        if save_arg:
            self.save_arg()
        if self.rank == 0:
            self.create_folder_and_writer()
        self.global_step = 0
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.load_scheduler()

    def save_arg(self):
        # save arg
        if self.rank == 0:
            arg_dict = vars(self.arg)
            if not os.path.exists(self.arg.work_dir):
                os.makedirs(self.arg.work_dir)
            else:
                raise ValueError(f"{self.arg.work_dir} already exists ...")
            if ".yaml" in self.arg.config:
                with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
                    yaml.dump(arg_dict, f)
            elif ".json" in self.arg.config:
                with open(f'{self.arg.work_dir}/config.json', 'w') as f:
                    json.dump(arg_dict, f)
            else:
                raise ValueError("Unknown config format...")

    # --------------------------------------------------------------------------
    # UTILS
    # --------------------------------------------------------------------------

    def create_folder_and_writer(self):
        def dir_check(dir):
            try:
                os.makedirs(dir)
            except FileExistsError:
                if self.arg.weights is None:
                    msg = f'weights unspecified and {dir} already exist'
                    raise ValueError(msg)
        # weight
        if self.arg.phase == 'train':
            weight_dir = os.path.join(self.arg.work_dir, 'weight')
            dir_check(weight_dir)
        # scores
        if self.arg.save_score:
            score_dir = os.path.join(self.arg.work_dir, 'score')
            dir_check(score_dir)
        # right wrong predictions as txt
        if self.arg.phase == 'test':
            pred_dir = os.path.join(self.arg.work_dir, 'prediction')
            dir_check(pred_dir)
        # events
        if self.arg.phase == 'train':
            event_dir = os.path.join(self.arg.work_dir, 'event')
            if not self.arg.train_feeder_args['debug']:
                dir_check(event_dir)
                self.train_writer = SummaryWriter(
                    os.path.join(event_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(
                    os.path.join(event_dir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(
                    os.path.join(event_dir, 'train_debug'), 'train_debug')
                self.val_writer = self.train_writer

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, msg: str, print_time: bool = True):
        if self.rank == 0:
            if print_time:
                localtime = time.asctime(time.localtime(time.time()))
                msg = "[ " + localtime + ' ] ' + msg
            print(msg)
            if self.arg.print_log:
                with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                    print(msg, file=f)

    def record_time(self) -> float:
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self) -> float:
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def to_float_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().cuda(self.output_device, non_blocking=True)

    def to_long_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return x.long().cuda(self.output_device, non_blocking=True)

    def to_cuda(self,
                data: Tuple[tuple, torch.Tensor],
                label: Optional[torch.Tensor]
                ) -> Tuple[tuple, Optional[torch.Tensor]]:
        if isinstance(data, tuple):
            data = tuple(self.to_float_cuda(data[i])
                         for i in range(self.num_forward_inputs))
        elif isinstance(data, torch.Tensor):
            data = (self.to_float_cuda(data),)
        else:
            raise ValueError(f"Unknown data type : {type(data)}")
        if label is not None:
            label = self.to_long_cuda(label)
        return data, label

    def tensor_to_value(self, x: torch.Tensor) -> float:
        if self.arg.ddp:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x.data.item() / self.arg.world_size
        else:
            return x.data.item()

    # --------------------------------------------------------------------------
    # WRITERS
    # --------------------------------------------------------------------------

    def load_profilers(self):
        self.prof = profile(
            schedule=schedule(wait=1, warmup=1, active=5, repeat=1),  # noqa
            on_trace_ready=tensorboard_trace_handler(self.arg.work_dir + '/trace'),  # noqa
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.prof.start()

    def save_scores(self, epoch: int, loader_name: str, score: np.ndarray):
        sample_name = self.data_loader[loader_name].dataset.sample_name
        score_dict = dict(zip(sample_name, score))
        if self.arg.ddp:
            dist_tmp = [None for _ in range(self.arg.world_size)]
            dist.all_gather_object(dist_tmp, score_dict)
            score_dict = {k: v for d in dist_tmp for k, v in d.items()}
        score_path = os.path.join(self.arg.work_dir,
                                  'score',
                                  f'epoch{epoch + 1}_{loader_name}.pkl')
        with open(score_path, 'wb') as f:
            pickle.dump(score_dict, f)

    def writer(self, mode: str, **kwargs):
        for k, v in kwargs.items():
            self.train_writer.add_scalar(k, v, self.global_step)
        if mode == 'train':
            _lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', _lr, self.global_step)
            # self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)  # noqa

    # --------------------------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------------------------

    def save_weights(self, epoch: int):
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()]
                               for k, v in state_dict.items()])
        filename = f'{self.arg.model.split(".")[-1]}-{epoch}-{int(self.global_step)}.pt'  # noqa
        weight_path = os.path.join(self.arg.work_dir, 'weight', filename)
        torch.save(weights, weight_path)

    def load_weights(self):
        self.global_step = int(self.arg.weights[:-3].split('-')[-1])
        self.print_log(f'Load weights from {self.arg.weights}')
        if '.pkl' in self.arg.weights:
            with open(self.arg.weights, 'r') as f:
                weights = pickle.load(f)
        else:
            weights = torch.load(self.arg.weights)
        # dpp default scope naming is different
        if not self.arg.ddp:
            weights = OrderedDict(
                [[k.split('module.')[-1],
                    v.cuda(self.output_device)] for k, v in weights.items()])
        else:
            weights = OrderedDict(
                [[k if k[:7] == 'module.' else 'module.'+k,
                    v.cuda(self.output_device)] for k, v in weights.items()])
        # weights to ignore
        keys = list(weights.keys())
        for w in self.arg.ignore_weights:
            for key in keys:
                if w in key:
                    if weights.pop(key, None) is not None:
                        self.print_log(
                            f'Sucessfully Remove Weights: {key}')
                    else:
                        self.print_log(f'Can Not Remove Weights: {key}')
        # load weights
        try:
            self.model.load_state_dict(weights)
        except:  # noqa
            state = self.model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            print('Can not find these weights:')
            for d in diff:
                print('  ' + d)
            state.update(weights)
            self.model.load_state_dict(state)

    # --------------------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------------------

    def get_output_device(self):
        if self.arg.ddp:
            self.output_device = self.rank
        else:
            if type(self.arg.device) is list:
                output_device = self.arg.device[0]
            else:
                output_device = self.arg.device
            self.output_device = output_device

    def get_model(self):
        Model = import_class(self.arg.model)
        if self.rank == 0:
            # Saves a copy of the model file
            shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(self.output_device)
        self.num_forward_inputs = len(
            inspect.signature(self.model.forward).parameters)
        if self.arg.ddp:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.rank])

    def get_loss(self):
        # Main loss
        if self.arg.fl_gamma >= 0.0:
            self.loss = CategorialFocalLoss(
                classes=self.arg.model_args.get('num_class', None),
                smoothing=self.arg.label_smoothing,
                alpha=self.arg.fl_alpha,
                gamma=self.arg.fl_gamma
            ).cuda(self.output_device)
        elif self.arg.label_smoothing > 0.0:
            self.loss = LabelSmoothingLoss(
                classes=self.arg.model_args.get('num_class', None),
                smoothing=self.arg.label_smoothing
            ).cuda(self.output_device)
        else:
            self.loss = nn.CrossEntropyLoss().cuda(self.output_device)
        # Opt: NMD loss
        if self.arg.model_args.get('infogcn_noise_ratio', None) is not None:
            self.mmd_loss = MaximumMeanDiscrepancyLoss(
                classes=self.arg.model_args.get('num_class', None)
            ).cuda(self.output_device)
        else:
            self.mmd_loss = None
        # Opt: Cosine feature loss
        if self.arg.fsim_mode == 0:
            self.fsim_loss = None
        else:
            self.fsim_loss = CosineLoss(
                mode=self.arg.fsim_mode
            ).cuda(self.output_device)

    def load_model(self):
        self.get_output_device()
        self.get_model()
        self.get_loss()
        if self.arg.weights:
            self.load_weights()
        # non ddp multigpu
        if not self.arg.ddp:
            if isinstance(self.arg.device, list):
                if len(self.arg.device) > 1:
                    self.model = nn.DataParallel(
                        self.model,
                        device_ids=self.arg.device,
                        output_device=self.output_device
                    )

    # --------------------------------------------------------------------------
    # OPTIMIZER, SCHEDULER
    # --------------------------------------------------------------------------

    def adjust_learning_rate(self, epoch: int):
        opts1 = ['SGD', 'SAM_SGD', 'Adam', 'AdamW']
        opts2 = ['SGD-LLRD', 'AdamW-LLRD']
        if self.arg.optimizer in opts1:
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        elif self.arg.optimizer in opts2:
            for param_group in self.optimizer.param_groups:
                if epoch < self.arg.warm_up_epoch:
                    lr = param_group['base_lr'] * \
                        (epoch + 1) / self.arg.warm_up_epoch
                else:
                    lr = param_group['base_lr'] * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
                param_group['lr'] = lr
        else:
            raise ValueError()

    def get_params_lists(self):
        if 'LLRD' in self.arg.optimizer:
            params_dict = {}
            for (k, v) in self.model.named_parameters():
                if 'trans' not in k:
                    params_dict['-1'] = params_dict.get('-1', []) + [v]
                else:
                    _k = k.split('.')[1]
                    params_dict[_k] = params_dict.get(_k, []) + [v]
            params_dict = dict(sorted(params_dict.items(), reverse=True))
            params_list = []
            for idx, (k, v) in enumerate(params_dict.items()):
                if k == '-1':
                    lr = self.arg.base_lr
                    params_list.append({'params': v, 'lr': lr, 'base_lr': lr})
                else:
                    lr = self.arg.base_lr * self.arg.llrd_factor**idx
                    params_list.append({'params': v, 'lr': lr, 'base_lr': lr})
        else:
            params_list = self.model.parameters()
        return params_list

    def load_optimizer(self):
        params_list = self.get_params_lists()
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(params_list,
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=self.arg.nesterov,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'SGD-LLRD':
            self.optimizer = optim.SGD(params_list,
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=self.arg.nesterov,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(params_list,
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(params_list,
                                         lr=self.arg.base_lr,
                                         weight_decay=self.arg.weight_decay,
                                         eps=self.arg.eps)
        elif self.arg.optimizer == 'AdamW-LLRD':
            self.optimizer = optim.AdamW(params_list,
                                         lr=self.arg.base_lr,
                                         weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'SAM_SGD':
            self.optimizer = SAM(params_list,
                                 base_optimizer=optim.SGD,
                                 lr=self.arg.base_lr,
                                 momentum=0.9,
                                 nesterov=self.arg.nesterov,
                                 weight_decay=self.arg.weight_decay)
        else:
            raise ValueError("Unknown optimizer")

    def load_scheduler(self):
        if self.arg.scheduler == 'cycliclr':
            step_size_up = len(self.data_loader['train'])//2
            step_size_down = len(self.data_loader['train']) - step_size_up
            self.scheduler = (
                'BATCH',
                optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=self.arg.base_lr*1e-2,
                    max_lr=self.arg.base_lr,
                    step_size_up=step_size_up,
                    step_size_down=step_size_down,
                ))
        elif self.arg.scheduler == 'cycliclrtri2':
            step_size_up = len(self.data_loader['train'])//2
            step_size_down = len(self.data_loader['train']) - step_size_up
            self.scheduler = (
                'BATCH',
                optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=self.arg.base_lr*1e-2,
                    max_lr=self.arg.base_lr,
                    step_size_up=step_size_up,
                    step_size_down=step_size_down,
                    mode="triangular2"
                ))
        elif self.arg.scheduler == 'onecyclelr':
            self.scheduler = (
                'BATCH',
                optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.arg.base_lr,
                    steps_per_epoch=len(self.data_loader['train']),
                    epochs=self.arg.num_epoch,
                    pct_start=self.arg.warm_up_epoch/self.arg.num_epoch,
                    anneal_strategy=self.arg.anneal_strategy,
                    div_factor=self.arg.base_lr/self.arg.initial_lr,
                    final_div_factor=self.arg.base_lr/self.arg.final_lr,
                ))
        else:
            self.scheduler = (None, None)
        self.print_log(f'using warm up, epoch: {self.arg.warm_up_epoch}')

    # --------------------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------------------

    def load_data(self):
        kwargs = dict(
            world_size=self.arg.world_size,
            rank=self.rank,
            ddp=self.arg.ddp,
            num_worker=self.arg.num_worker,
            # worker_init_fn=init_seed,
        )
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            assert os.path.exists(self.arg.train_feeder_args['data_path'])
            assert os.path.exists(self.arg.train_feeder_args['label_path'])
            self.arg.train_dataloader_args['dataset'] = \
                self.arg.train_feeder_args['dataset']
            data_loader = FeederDataLoader(**self.arg.train_dataloader_args)
            self.data_loader['train'] = data_loader.get_loader(
                **kwargs,
                feeder=Feeder(**self.arg.train_feeder_args),
                shuffle_ds=True,
                shuffle_dl=not self.arg.ddp,
                batch_size=self.arg.batch_size,
                drop_last=True,
                collate_fn=data_loader.collate_fn_fix_train if self.arg.use_sgn_dataloader else None  # noqa
            )
        self.arg.test_dataloader_args['dataset'] = \
            self.arg.test_feeder_args['dataset']
        data_loader = FeederDataLoader(**self.arg.test_dataloader_args)
        if self.arg.test_dataloader_args['multi_test'] > 1:
            self.data_loader['val'] = data_loader.get_loader(
                **kwargs,
                feeder=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                collate_fn=data_loader.collate_fn_fix_test if self.arg.use_sgn_dataloader else None  # noqa
            )
        else:
            self.data_loader['val'] = data_loader.get_loader(
                **kwargs,
                feeder=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                collate_fn=data_loader.collate_fn_fix_val if self.arg.use_sgn_dataloader else None  # noqa
            )

    def get_data_iterator(self, epoch: int, loader_name: str):
        loader = self.data_loader[loader_name]
        if self.arg.ddp:
            loader.sampler.set_epoch(epoch)
        if self.rank == 0:
            return tqdm(self.data_loader[loader_name],
                        desc=f"Device {self.rank}",
                        position=self.rank)
        else:
            return self.data_loader[loader_name]

    # --------------------------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------------------------

    def forward_pass(self,
                     data: tuple,
                     label: Optional[torch.Tensor]
                     ) -> Tuple[tuple, Optional[torch.Tensor]]:
        # output, G, Z
        output_tuple = self.model(*data)
        output = output_tuple[0]

        if not self.model.training:
            freq = self.arg.test_dataloader_args['multi_test']
            if self.arg.use_sgn_dataloader and freq > 1:
                output = output.view((-1, freq, output.size(1))).mean(1)

        if isinstance(output, tuple):
            output, l1 = output
            l1 = l1.mean()
        else:
            l1 = 0

        loss_dict = {'loss': None}

        if label is None or self.loss is None:
            loss_dict['loss'] = None
        else:
            loss_dict['loss'] = self.loss(output, label) + l1

        if self.mmd_loss is not None:
            z = output_tuple[2]
            if not self.model.training:
                freq = self.arg.test_dataloader_args['multi_test']
                if self.arg.use_sgn_dataloader and freq > 1:
                    z = z.view((-1, freq, z.size(1))).mean(1)
            mmd_loss, l2_z_mean, z_mean = self.mmd_loss(
                z, self.model.z_prior, label)
            loss_dict['loss'] = (self.arg.mmd_lambda2 * mmd_loss +
                                 self.arg.mmd_lambda1 * l2_z_mean +
                                 loss_dict['loss'])
            cos_z, dis_z = get_vector_property(z_mean)
            cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
            loss_dict['l2_z_mean'] = l2_z_mean
            loss_dict['mmd_loss'] = mmd_loss
            loss_dict['cos_z'] = cos_z
            loss_dict['dis_z'] = dis_z
            loss_dict['cos_z_prior'] = cos_z_prior
            loss_dict['dis_z_prior'] = dis_z_prior

        if self.fsim_loss is not None:
            assert self.arg.fsim_alpha is not None
            assert len(self.arg.fsim_alpha) > 0
            x_tem_list = output_tuple[1]['x_tem_list']
            x_tem_list = [i for i in x_tem_list if i is not None]
            kernels = len(self.arg.model_args['multi_t'][-1])
            levels = (len(x_tem_list)//kernels) - 1
            fc_loss = 0
            for i in range(levels):
                for j in range(kernels):
                    fc_loss += \
                        self.arg.fsim_alpha[i*kernels+j] * \
                        self.fsim_loss(
                            x_tem_list[i*kernels+j],
                            x_tem_list[-kernels+j]
                        )
            loss_dict['loss'] = fc_loss + loss_dict['loss']
            loss_dict['fc_loss'] = fc_loss

        return output, loss_dict

    def train(self, epoch: int, save_model: bool = False):

        # 1. Timing
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        # 2. Model Train
        self.model.train()
        zero_grad_PA = False
        if self.arg.ddp:
            if self.arg.only_train_part:
                if epoch <= self.arg.only_train_epoch:
                    zero_grad_PA = True
        else:
            if self.arg.only_train_part:
                if epoch > self.arg.only_train_epoch:
                    print('only train part, require grad')
                    for key, value in self.model.named_parameters():
                        if 'PA' in key:
                            value.requires_grad = True
                            # print(key + '-require grad')
                else:
                    print('only train part, do not require grad')
                    for key, value in self.model.named_parameters():
                        if 'PA' in key:
                            value.requires_grad = False
                            # print(key + '-not require grad')

        # 3. Logging
        self.print_log(f'Training epoch: {epoch + 1}')
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)  # noqa
        if self.rank == 0:
            self.train_writer.add_scalar('epoch', epoch, self.global_step)

        loss_values = []
        acc_values = []
        mmd_loss_values = []
        l2_z_mean_values = []
        cos_z_values = []
        dis_z_values = []
        cos_z_prior_values = []
        dis_z_prior_values = []

        f_sim_values = []

        # 4. Loader
        process = self.get_data_iterator(epoch, 'train')

        # 5. Main loop
        if self.arg.profiler:
            self.load_profiler()
            prof_flag = True

        for batch_idx, (data, label, index) in enumerate(process):

            if self.arg.profiler:
                if batch_idx >= (1 + 1 + 5) * 1 and prof_flag:
                    prof_flag = False
                    self.prof.stop()

            self.global_step += 1

            # 5.1. get data
            data, label = self.to_cuda(data, label)

            timer['dataloader'] += self.split_time()

            # 5.2. forward + backward + optimize
            if self.arg.optimizer == 'SAM_SGD':
                # 1. first forward-backward pass
                enable_running_stats(self.model)
                output, loss_dict = self.forward_pass(data, label)
                loss = loss_dict['loss']
                with self.model.no_sync():
                    loss.backward()
                self.optimizer.first_step(zero_grad=True)
                # 2. second forward-backward pass
                # make sure to do a full forward pass
                disable_running_stats(self.model)
                output, loss_dict = self.forward_pass(data, label)
                loss = loss_dict['loss']
                loss.backward()
                self.optimizer.second_step(zero_grad=True)

            else:
                # forward
                output, loss_dict = self.forward_pass(data, label)
                loss = loss_dict['loss']
                # if batch_idx == 0 and epoch == 0:
                #     self.train_writer.add_graph(self.model, output)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if zero_grad_PA:
                    for name, param in self.model.named_parameters():
                        if 'PA' in name:
                            param.grad *= 0
                self.optimizer.step()

            # 5.3. scheduler if applicable.
            if self.scheduler[0] == 'BATCH':
                self.scheduler[1].step()

            timer['model'] += self.split_time()

            # 5.4. logging
            loss_value = self.tensor_to_value(loss)
            loss_values.append(loss_value)

            if self.mmd_loss is not None:
                l2_z_mean_value = self.tensor_to_value(loss_dict['l2_z_mean'])
                mmd_loss_value = self.tensor_to_value(loss_dict['mmd_loss'])
                cos_z_value = self.tensor_to_value(loss_dict['cos_z'])
                dis_z_value = self.tensor_to_value(loss_dict['dis_z'])
                cos_z_prior_value = self.tensor_to_value(loss_dict['cos_z_prior'])  # noqa
                dis_z_prior_value = self.tensor_to_value(loss_dict['dis_z_prior'])  # noqa
                l2_z_mean_values.append(l2_z_mean_value)
                mmd_loss_values.append(mmd_loss_value)
                cos_z_values.append(cos_z_value)
                dis_z_values.append(dis_z_value)
                cos_z_prior_values.append(cos_z_prior_value)
                dis_z_prior_values.append(dis_z_prior_value)

            if self.fsim_loss is not None:
                f_sim_value = self.tensor_to_value(loss_dict['fc_loss'])
                f_sim_values.append(f_sim_value)

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc = self.tensor_to_value(acc)
            acc_values.append(acc)

            if self.rank == 0:
                kwargs = {
                    'acc': acc,
                    'loss': loss_value
                }
                if self.mmd_loss is not None:
                    kwargs['l2_z_mean'] = l2_z_mean_value
                    kwargs['mmd_loss'] = mmd_loss_value
                    kwargs['cos_z'] = cos_z_value
                    kwargs['dis_z'] = dis_z_value
                    kwargs['cos_z_prior'] = cos_z_prior_value
                    kwargs['dis_z_prior'] = dis_z_prior_value
                self.writer(mode='train', **kwargs)

            timer['statistics'] += self.split_time()

            if self.arg.profiler:
                if prof_flag:
                    self.prof.step()

        # 6. Statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }
        self.print_log(f'\tMean training loss: {np.mean(loss_values):.4f}.')
        if self.mmd_loss is not None:
            self.print_log(f'\tMean l2_z_mean    : {np.mean(l2_z_mean_values):.4f}.')  # noqa
            self.print_log(f'\tMean mmd_loss     : {np.mean(mmd_loss_values):.4f}.')  # noqa
            self.print_log(f'\tMean cos_z        : {np.mean(cos_z_values):.4f}.')  # noqa
            self.print_log(f'\tMean dis_z        : {np.mean(dis_z_values):.4f}.')  # noqa
            self.print_log(f'\tMean cos_z_prior  : {np.mean(cos_z_prior_values):.4f}.')  # noqa
            self.print_log(f'\tMean dis_z_prior  : {np.mean(dis_z_prior_values):.4f}.')  # noqa
        if self.fsim_loss is not None:
            self.print_log(f'\tMean fc_loss : {np.mean(f_sim_values):.4f}.')  # noqa
        self.print_log(f'\tTime consumption  : '
                       f'[Data] {proportion["dataloader"]}, '
                       f'[Network] {proportion["model"]}')

        if save_model and self.rank == 0:
            self.save_weights(epoch + 1)

    # --------------------------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------------------------

    def eval(self,
             epoch: int,
             save_score: bool = False,
             loader_name: list = ['val'],
             wrong_file: Optional[str] = None,
             result_file: Optional[str] = None):

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        # 1. model eval
        self.model.eval()
        self.print_log(f'Eval epoch: {epoch + 1}')

        # 2. loop through data loaders
        for ln in loader_name:
            loss_values = []
            mmd_loss_values = []
            l2_z_mean_values = []
            cos_z_values = []
            dis_z_values = []
            cos_z_prior_values = []
            dis_z_prior_values = []

            score_frag = []
            step = 0
            process = self.get_data_iterator(epoch, 'val')

            # 3. main loop
            for batch_idx, (data, label, index) in enumerate(process):

                # 3. forward pass
                with torch.no_grad():
                    data, label = self.to_cuda(data, label)
                    output, loss_dict = self.forward_pass(data, label)
                    loss = loss_dict['loss']
                    if self.mmd_loss is not None:
                        l2_z_mean_value = self.tensor_to_value(loss_dict['l2_z_mean'])  # noqa
                        mmd_loss_value = self.tensor_to_value(loss_dict['mmd_loss'])  # noqa
                        cos_z_value = self.tensor_to_value(loss_dict['cos_z'])
                        dis_z_value = self.tensor_to_value(loss_dict['dis_z'])
                        cos_z_prior_value = self.tensor_to_value(loss_dict['cos_z_prior'])  # noqa
                        dis_z_prior_value = self.tensor_to_value(loss_dict['dis_z_prior'])  # noqa
                        l2_z_mean_values.append(l2_z_mean_value)
                        mmd_loss_values.append(mmd_loss_value)
                        cos_z_values.append(cos_z_value)
                        dis_z_values.append(dis_z_value)
                        cos_z_prior_values.append(cos_z_prior_value)
                        dis_z_prior_values.append(dis_z_prior_value)
                    score_frag.append(output.data.cpu().numpy())
                    loss_values.append(self.tensor_to_value(loss))
                    _, predict_label = torch.max(output.data, 1)

                step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(batch_idx) + ',' +
                                      str(x) + ',' + str(true[i]) + '\n')

            # 4. loss
            loss_value = np.mean(loss_values)
            if self.mmd_loss is not None:
                l2_z_mean_value = np.mean(l2_z_mean_values)
                mmd_loss_value = np.mean(mmd_loss_values)
                cos_z_value = np.mean(cos_z_values)
                dis_z_value = np.mean(dis_z_values)
                cos_z_prior_value = np.mean(cos_z_prior_values)
                dis_z_prior_value = np.mean(dis_z_prior_values)

            # 5. logits
            score = np.concatenate(score_frag)
            if self.arg.ddp:
                dist_tmp = [None for _ in range(self.arg.world_size)]
                dist.all_gather_object(dist_tmp, score)
                score = np.concatenate(dist_tmp)
                for idx, val in enumerate(dist_tmp):
                    score[idx::self.arg.world_size] = val

            # 6. accuracy
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            # 7. logging
            if self.rank == 0:
                if self.arg.phase == 'train':
                    kwargs = {
                        'acc': accuracy,
                        'loss': loss_value
                    }
                    if self.mmd_loss is not None:
                        kwargs['l2_z_mean'] = l2_z_mean_value
                        kwargs['mmd_loss'] = mmd_loss_value
                        kwargs['cos_z'] = cos_z_value
                        kwargs['dis_z'] = dis_z_value
                        kwargs['cos_z_prior'] = cos_z_prior_value
                        kwargs['dis_z_prior'] = dis_z_prior_value
                    self.writer(mode='val', **kwargs)

            # self.print_log(f'Model   : {self.arg.work_dir}')
            self.print_log(
                f'\tMean {ln} '
                f'loss of {len(self.data_loader[ln])} '
                f'batches: {loss_value:.4f}'
            )
            self.print_log(f'\tAccuracy   : {accuracy:.4f}')
            if self.mmd_loss is not None:
                self.print_log(f'\tl2_z_mean  : {np.mean(l2_z_mean_values):.4f}.')  # noqa
                self.print_log(f'\tmmd_loss   : {np.mean(mmd_loss_values):.4f}.')  # noqa
                self.print_log(f'\tcos_z      : {np.mean(cos_z_values):.4f}.')
                self.print_log(f'\tdis_z      : {np.mean(dis_z_values):.4f}.')
                self.print_log(f'\tcos_z_prior: {np.mean(cos_z_prior_values):.4f}.')  # noqa
                self.print_log(f'\tdis_z_prior: {np.mean(dis_z_prior_values):.4f}.')  # noqa

            for k in self.arg.show_topk:
                top_k = 100 * self.data_loader[ln].dataset.top_k(score, k)
                self.print_log(f'\tTop{k}: {top_k:.2f}%')

            if save_score and self.rank == 0:
                self.save_scores(epoch, ln, score)

            self.print_log('-'*51)

    # --------------------------------------------------------------------------
    # MAIN
    # --------------------------------------------------------------------------

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n')
            if self.rank == 0:
                print_arg(self.arg)
            self.global_step = self.arg.start_epoch * \
                len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = \
                    ((epoch + 1) % self.arg.save_interval == 0) or \
                    ((epoch + 1) == self.arg.num_epoch)
                if self.scheduler[0] is None:
                    self.adjust_learning_rate(epoch)
                self.train(epoch, save_model=save_model)
                if self.scheduler[0] == 'EPOCH':
                    self.scheduler[1].step()
                if ((epoch + 1) % self.arg.eval_interval == 0) or \
                        ((epoch + 1) == self.arg.num_epoch):
                    self.eval(epoch, save_score=self.arg.save_score,
                              loader_name=['val'])
            self.print_log(f'Best Accuracy: {self.best_acc*100:.2f}%')
            self.print_log(f'Best Epoch   : {self.best_acc_epoch}')
            self.print_log(f'Model Name   : {self.arg.work_dir}')
            self.print_log('Done.\n')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'prediction', 'wrong.txt')
                rf = os.path.join(self.arg.work_dir, 'prediction', 'right.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log(f'Model  : {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')
            self.eval(epoch=0, save_score=self.arg.save_score,
                      loader_name=['val'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
