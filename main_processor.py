#!/usr/bin/env python
# from __future__ import print_function

import inspect
import numpy as np
import os
import pickle
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from collections import OrderedDict

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from sam.sam.sam import SAM
from sam.sam.example.utility.bypass_bn import enable_running_stats
from sam.sam.example.utility.bypass_bn import disable_running_stats

from main_utils import *


__all__ = ['Processor']


class Processor():
    """Processor for Skeleton-based Action Recognition """

    def __init__(self, arg, rank=0, save_arg=True):
        self.rank = rank
        self.arg = arg
        if save_arg:
            self.save_arg()
        if self.rank == 0:
            if arg.phase == 'train':
                if not arg.train_feeder_args['debug']:
                    if os.path.isdir(arg.model_saved_name):
                        if self.arg.weights is None:
                            raise ValueError(f'log_dir: {arg.model_saved_name} already exist')  # noqa
                        # print('log_dir: ', arg.model_saved_name, 'already exist')  # noqa
                        # answer = input('delete it? y/n:')
                        # if answer == 'y':
                        #     shutil.rmtree(arg.model_saved_name)
                        #     print('Dir removed: ', arg.model_saved_name)
                        #     input('Refresh the website of tensorboard by pressing any keys')  # noqa
                        # else:
                        #     print('Dir not removed: ', arg.model_saved_name)
                    self.train_writer = SummaryWriter(
                        os.path.join(arg.model_saved_name, 'train'), 'train')
                    self.val_writer = SummaryWriter(
                        os.path.join(arg.model_saved_name, 'val'), 'val')
                else:
                    self.train_writer = self.val_writer = SummaryWriter(
                        os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.best_acc = 0

    def save_arg(self):
        # save arg
        if self.rank == 0:
            arg_dict = vars(self.arg)
            if not os.path.exists(self.arg.work_dir):
                os.makedirs(self.arg.work_dir)
            with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
                yaml.dump(arg_dict, f)

    # **************************************************************************
    # LOADERS
    # **************************************************************************

    def load_model(self):
        if self.arg.ddp:
            self.output_device = self.rank
        else:
            output_device = self.arg.device[0] if type(
                self.arg.device) is list else self.arg.device
            self.output_device = output_device
        Model = import_class(self.arg.model)
        if self.rank == 0:
            shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        # print(Model)
        self.model = Model(**self.arg.model_args).cuda(self.output_device)
        if self.arg.ddp:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.rank])
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(self.output_device)

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log(f'Load weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            if not self.arg.ddp:
                weights = OrderedDict(
                    [[k.split('module.')[-1],
                      v.cuda(self.output_device)] for k, v in weights.items()])
            else:
                weights = OrderedDict(
                    [[k if k[:7] == 'module.' else 'module.'+k,
                      v.cuda(self.output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log(
                                f'Sucessfully Remove Weights: {key}')
                        else:
                            self.print_log(f'Can Not Remove Weights: {key}')

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

        if not self.arg.ddp:
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    self.model = nn.DataParallel(
                        self.model,
                        device_ids=self.arg.device,
                        output_device=self.output_device)

    def load_optimizer(self):
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

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
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
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(),
                                         lr=self.arg.base_lr,
                                         weight_decay=self.arg.weight_decay,
                                         eps=self.arg.eps)
        elif self.arg.optimizer == 'AdamW-LLRD':
            self.optimizer = optim.AdamW(params_list,
                                         lr=self.arg.base_lr,
                                         weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'SAM_SGD':
            self.optimizer = SAM(self.model.parameters(),
                                 base_optimizer=optim.SGD,
                                 lr=self.arg.base_lr,
                                 momentum=0.9,
                                 nesterov=self.arg.nesterov,
                                 weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

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

        # lr_scheduler_post = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=self.arg.step, gamma=0.1)
        # self.lr_scheduler = GradualWarmupScheduler(
        #     self.optimizer,
        #     total_epoch=self.arg.warm_up_epoch,
        #     after_scheduler=lr_scheduler_post)

        self.print_log(f'using warm up, epoch: {self.arg.warm_up_epoch}')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            assert os.path.exists(self.arg.train_feeder_args['data_path'])
            assert os.path.exists(self.arg.train_feeder_args['label_path'])
            training_set = Feeder(**self.arg.train_feeder_args)
            train_data_sampler = DistributedSampler(
                dataset=training_set,
                num_replicas=self.arg.world_size,
                rank=self.rank,
                shuffle=True
            ) if self.arg.ddp else None
            self.data_loader['train'] = DataLoader(
                dataset=training_set,
                batch_size=self.arg.batch_size,
                shuffle=(train_data_sampler is None),
                sampler=train_data_sampler,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed
            )
        testing_set = Feeder(**self.arg.test_feeder_args)
        test_data_sampler = DistributedSampler(
            dataset=testing_set,
            num_replicas=self.arg.world_size,
            rank=self.rank,
            shuffle=False,
        ) if self.arg.ddp else None
        self.data_loader['test'] = DataLoader(
            dataset=testing_set,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            sampler=test_data_sampler,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed
        )

    # **************************************************************************
    # UTILS
    # **************************************************************************

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if self.rank == 0:
            if print_time:
                localtime = time.asctime(time.localtime(time.time()))
                str = "[ " + localtime + ' ] ' + str
            print(str)
            if self.arg.print_log:
                with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                    print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def adjust_learning_rate(self, epoch):
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

    # **************************************************************************
    # TRAIN AND EVAL
    # **************************************************************************

    def train(self, epoch, save_model=False):

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
        loss_value = []

        # 4. Loader
        loader = self.data_loader['train']
        if self.arg.ddp:
            loader.sampler.set_epoch(epoch)

        if self.rank == 0:
            process = tqdm(loader,
                           desc=f"Device {self.rank}",
                           position=self.rank)
        else:
            process = loader

        # 5. Main loop
        for batch_idx, (data, label, index) in enumerate(process):

            self.global_step += 1

            # 5.1. get data
            data = data.float().cuda(self.output_device, non_blocking=True)
            label = label.long().cuda(self.output_device, non_blocking=True)
            timer['dataloader'] += self.split_time()

            # 5.2. forward + backward + optimize
            if self.arg.optimizer == 'SAM_SGD':
                # 1. first forward-backward pass
                enable_running_stats(self.model)
                output, _ = self.model(data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                loss = self.loss(output, label) + l1
                with self.model.no_sync():
                    loss.backward()
                self.optimizer.first_step(zero_grad=True)
                # 2. second forward-backward pass
                # make sure to do a full forward pass
                disable_running_stats(self.model)
                output, _ = self.model(data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                loss = self.loss(output, label) + l1
                loss.backward()
                self.optimizer.second_step(zero_grad=True)

            else:
                # forward
                output, _ = self.model(data)
                # if batch_idx == 0 and epoch == 0:
                #     self.train_writer.add_graph(self.model, output)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                loss = self.loss(output, label) + l1

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if zero_grad_PA:
                    for name, param in self.model.named_parameters():
                        if 'PA' in name:
                            param.grad *= 0
                self.optimizer.step()

            # 5.3. scheduler if applicable.
            if self.scheduler[0] == 'BATCH':
                self.scheduler[1].step()

            # 5.4. logging
            timer['model'] += self.split_time()

            if self.arg.ddp:
                # dist_tmp = [None for _ in range(self.arg.world_size)]
                # dist.all_gather_object(dist_tmp, _loss)
                # _loss = np.mean(dist_tmp)
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                _loss = loss.data.item() / self.arg.world_size
            else:
                _loss = loss.data.item()
            loss_value.append(_loss)

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            if self.arg.ddp:
                # dist_tmp = [None for _ in range(self.arg.world_size)]
                # dist.all_gather_object(dist_tmp, acc)
                # acc = np.mean(dist_tmp)
                dist.all_reduce(acc, op=dist.ReduceOp.SUM)
                acc = acc.data.item() / self.arg.world_size
            else:
                acc = acc.data.item()

            if self.rank == 0:
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', _loss, self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)
                # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)  # noqa

            # 5.5. Statistics
            if self.rank == 0:
                _lr = self.optimizer.param_groups[0]['lr']
                self.train_writer.add_scalar('lr', _lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # 6. Statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }
        self.print_log(
            f'\tMean training loss: {np.mean(loss_value):.4f}.'.format())
        self.print_log(
            f'\tTime consumption: '
            f'[Data]{proportion["dataloader"]}, '
            f'[Network]{proportion["model"]}')

        if save_model and self.rank == 0:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' +
                       str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'],
             wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log(f'Eval epoch: {epoch + 1}')
        for ln in loader_name:
            loss_value = []
            score_frag = []
            step = 0
            if self.rank == 0:
                process = tqdm(self.data_loader[ln],
                               desc=f"Device {self.rank}",
                               position=self.rank)
            else:
                process = self.data_loader[ln]
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output, _ = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' +
                                      str(x) + ',' + str(true[i]) + '\n')

            loss = np.mean(loss_value)
            if self.arg.ddp:
                dist_tmp = [None for _ in range(self.arg.world_size)]
                dist.all_gather_object(dist_tmp, loss)
                loss = np.mean(dist_tmp)

            score = np.concatenate(score_frag)
            if self.arg.ddp:
                dist_tmp = [None for _ in range(self.arg.world_size)]
                dist.all_gather_object(dist_tmp, score)
                score = np.concatenate(dist_tmp)
                for idx, val in enumerate(dist_tmp):
                    score[idx::len(dist_tmp)] = val

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            if accuracy > self.best_acc:
                self.best_acc = accuracy
            self.print_log(f'Accuracy: {accuracy:.4f}')
            self.print_log(f'Model: {self.arg.model_saved_name}')

            if self.rank == 0:
                if self.arg.phase == 'train':
                    self.val_writer.add_scalar('loss', loss, self.global_step)
                    self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                    self.val_writer.add_scalar(
                        'acc', accuracy, self.global_step)

            self.print_log(f'\tMean {ln} '
                           f'loss of {len(self.data_loader[ln])} '
                           f'batches: {np.mean(loss_value):.4f}')

            for k in self.arg.show_topk:
                top_k = 100 * self.data_loader[ln].dataset.top_k(score, k)
                self.print_log(f'\tTop{k}: {top_k:.2f}%')

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            if self.arg.ddp:
                dist_tmp = [None for _ in range(self.arg.world_size)]
                dist.all_gather_object(dist_tmp, score_dict)
                score_dict = {k: v for d in dist_tmp for k, v in d.items()}
            if save_score and self.rank == 0:
                s_path = f'{self.arg.work_dir}/epoch{epoch + 1}_{ln}_score.pkl'
                with open(s_path, 'wb') as f:
                    pickle.dump(score_dict, f)

            self.print_log('-'*51)

    # **************************************************************************
    # MAIN
    # **************************************************************************

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
                # if ((epoch + 1) % self.arg.eval_interval == 0) or \
                #         ((epoch + 1) == self.arg.num_epoch):
                self.eval(epoch, save_score=self.arg.save_score,
                          loader_name=['test'])
            self.print_log(f'Best Accuracy: {self.best_acc}\n')
            self.print_log(f'Model Name: {self.arg.model_saved_name}\n')
            self.print_log('Done.\n')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')
            self.eval(epoch=0, save_score=self.arg.save_score,
                      loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
