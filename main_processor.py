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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sam.sam.sam import SAM
from sam.sam.example.utility.bypass_bn import enable_running_stats
from sam.sam.example.utility.bypass_bn import disable_running_stats

from main_utils import *


__all__ = ['Processor']


class DistributedWrapper():
    def __init__(self, dist: bool) -> None:
        self.dist = dist
        if dist:
            torch.distributed.init_process_group(
                backend='nccl', init_method='env://')
            print(
                f"[{os.getpid()}] rank = {torch.distributed.get_rank()}, "
                + f"world_size = {torch.distributed.get_world_size()}"
            )

    def only_main(self):
        if self.dist:
            if torch.distributed.get_rank() == 0:
                return True
            else:
                False
        else:
            return True


class Processor():
    """Processor for Skeleton-based Action Recognition """

    def __init__(self, arg, save_arg=True):
        self.arg = arg
        self.dist = DistributedWrapper()
        if save_arg:
            self.save_arg()
        if arg.phase == 'train' and self.dist.only_main():
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    if self.arg.weights is None:
                        raise ValueError(
                            f'log_dir: {arg.model_saved_name} already exist')
                    # print('log_dir: ', arg.model_saved_name, 'already exist')
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
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def save_arg(self):
        # save arg
        if self.dist.only_main():
            arg_dict = vars(self.arg)
            if not os.path.exists(self.arg.work_dir):
                os.makedirs(self.arg.work_dir)
            with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
                yaml.dump(arg_dict, f)

    def load_model(self):
        # output_device = self.arg.device[0] if type(
        #     self.arg.device) is list else self.arg.device
        self.output_device = torch.distributed.get_rank()
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(self.output_device)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(self.output_device)

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log(f'Load weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
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

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                # torch.distributed.init_process_group(
                #     backend='nccl', world_size=len(self.arg.device))
                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=self.arg.device
                )
                # self.model = nn.DataParallel(
                #     self.model,
                #     device_ids=self.arg.device,
                #     output_device=self.output_device
                # )

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=self.arg.nesterov,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'SAM_SGD':
            self.optimizer = SAM(params=self.model.parameters(),
                                 base_optimizer=optim.SGD,
                                 lr=self.arg.base_lr,
                                 momentum=0.9,
                                 nesterov=self.arg.nesterov,
                                 weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

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
            dataset = Feeder(**self.arg.train_feeder_args)
            kwargs = {
                'dataset': dataset,
                'sampler': self.sampler,
                'batch_size': self.arg.batch_size,
                'num_workers': self.arg.num_worker,
                'drop_last': True,
                'worker_init_fn': init_seed
            }
            if self.dist.dist:
                self.sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset=dataset,
                    shuffle=True,
                    seed=self.arg.seed,
                    drop_last=True,
                )
                kwargs['sampler'] = self.sampler
            else:
                kwargs['shuffle'] = True
                kwargs['sampler'] = None
            self.data_loader['train'] = torch.utils.data.DataLoader(**kwargs)

        else:
            dataset = Feeder(**self.arg.test_feeder_args)
            kwargs = {
                'dataset': dataset,
                'sampler': self.sampler,
                'batch_size': self.arg.batch_size,
                'num_workers': self.arg.num_worker,
                'drop_last': False,
                'worker_init_fn': init_seed
            }
            if self.dist.dist:
                self.sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset=dataset,
                    shuffle=False,
                    seed=self.arg.seed,
                    drop_last=False,
                )
                kwargs['sampler'] = self.sampler
            else:
                kwargs['shuffle'] = False
                kwargs['sampler'] = None
            self.data_loader['test'] = torch.utils.data.DataLoader(**kwargs)

    def adjust_learning_rate(self, epoch):
        opts = ['SGD', 'SAM_SGD', 'Adam']
        if self.arg.optimizer in opts:
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        if self.dist.only_main():
            localtime = time.asctime(time.localtime(time.time()))
            self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if self.dist.only_main():
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

    def train(self, epoch, save_model=False):
        self.model.train()
        torch.distributed.barrier()
        self.print_log(f'Training epoch: {epoch + 1}')
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)  # noqa
        loss_value = []
        if self.dist.only_main():
            self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
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
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = data.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

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
                self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())

            if self.dist.only_main():
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar(
                    'loss', loss.data.item(), self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)
                # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)  # noqa

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            if self.dist.only_main():
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }
        if self.dist.only_main():
            self.print_log(
                f'\tMean training loss: {np.mean(loss_value):.4f}.'.format())
            self.print_log(
                f'\tTime consumption: '
                f'[Data]{proportion["dataloader"]}, '
                f'[Network]{proportion["model"]}')

        if save_model and self.dist.only_main():
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' +
                       str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'],
             wrong_file=None, result_file=None):
        if wrong_file is not None and self.dist.only_main():
            f_w = open(wrong_file, 'w')
        if result_file is not None and self.dist.only_main():
            f_r = open(result_file, 'w')
        self.model.eval()
        torch.distributed.barrier()
        self.print_log(f'Eval epoch: {epoch + 1}')
        for ln in loader_name:
            loss_value = []
            score_frag = []
            step = 0
            process = tqdm(self.data_loader[ln])
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

                if self.dist.only_main():
                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' +
                                          str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
            self.print_log(f'Accuracy: {accuracy}')
            self.print_log(f'Model: {self.arg.model_saved_name}')

            if self.arg.phase == 'train' and self.dist.only_main():
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} '
                           f'loss of {len(self.data_loader[ln])} '
                           f'batches: {np.mean(loss_value):.4f}')
            for k in self.arg.show_topk:
                top_k = 100 * self.data_loader[ln].dataset.top_k(score, k)
                self.print_log(f'\tTop{k}: {top_k:.2f}%')

            if save_score and self.dist.only_main():
                s_path = f'{self.arg.work_dir}/epoch{epoch + 1}_{ln}_score.pkl'
                with open(s_path, 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
            self.global_step = self.arg.start_epoch * \
                len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = \
                    ((epoch + 1) % self.arg.save_interval == 0) or \
                    ((epoch + 1) == self.arg.num_epoch)
                self.sampler.set_epoch(epoch)
                self.train(epoch, save_model=save_model)
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

    def end(self):
        # Tear down the process group
        torch.distributed.destroy_process_group()
