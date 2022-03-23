import numpy as np
import os
import pickle
import time

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from feeders import tools
from feeders.sgn_dataloader import NTUDataLoaders, _transform

from main_utils import init_seed

from utils.visualization import visualize_3dskeleton_in_matplotlib


class Feeder(Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 dataset='NTU60-CV',
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 normalization=False,
                 random_zaxis_flip=False,
                 random_xaxis_scale=False,
                 random_yaxis_scale=False,
                 random_subsample=None,
                 random_rotation=False,
                 stretch=False,
                 debug=False,
                 use_mmap=True):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the
        input sequence
        :param random_shift: If true, randomly pad zeros at the begining or
        end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save
        the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.dataset = dataset
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.random_zaxis_flip = random_zaxis_flip
        self.random_xaxis_scale = random_xaxis_scale
        self.random_yaxis_scale = random_yaxis_scale
        self.random_subsample = random_subsample
        self.random_rotation = random_rotation
        self.stretch = stretch
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except UnicodeDecodeError:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(
                    f, encoding='latin1')
        else:
            raise ValueError("label data cannot be opened...")

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(
            axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(
            axis=0, keepdims=False)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    # def __iter__(self):
    #     return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.stretch:
            data_numpy = tools.stretch_to_maximum_length(data_numpy)
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.random_zaxis_flip:
            data_numpy = tools.random_zaxis_flip(data_numpy)
        if self.random_xaxis_scale:
            data_numpy = tools.random_xaxis_scale(data_numpy)
        if self.random_yaxis_scale:
            data_numpy = tools.random_yaxis_scale(data_numpy)
        if self.random_subsample is not None:
            assert self.random_subsample > 0 and self.random_subsample < 300
            data_numpy = tools.random_subsample(
                data_numpy, self.random_subsample)
        if self.random_rotation:
            if 'NTU60' in self.dataset:
                if 'CS' in self.dataset:
                    theta = 0.3
                elif 'CV' in self.dataset:
                    theta = 0.5
            elif 'NTU120' in self.dataset:
                theta = 0.3
            data_numpy = tools.random_rotation(data_numpy, theta)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


class FeederDataLoader(NTUDataLoaders):
    def __init__(self, dataset='NTU60-CV', aug=1, seg=30, multi_test=5):
        if 'CS' in dataset:
            case = 0
        elif 'CV' in dataset:
            case = 1
        else:
            case = -1
        super(FeederDataLoader, self).__init__(
            dataset, case, aug, seg, multi_test)

    def get_loader(self,
                   feeder: Dataset,
                   world_size: int = 1,
                   rank: int = 0,
                   ddp: bool = False,
                   shuffle_ds: bool = False,
                   shuffle_dl: bool = False,
                   batch_size: int = 1,
                   num_worker: int = 1,
                   drop_last: bool = False,
                   worker_init_fn=None,
                   collate_fn: str = None
                   ):
        data_sampler = DistributedSampler(
            dataset=feeder,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle_ds
        ) if ddp else None
        data_loader = DataLoader(
            dataset=feeder,
            batch_size=batch_size,
            shuffle=shuffle_dl,
            sampler=data_sampler,
            num_workers=num_worker,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )
        return data_loader


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)
        visualize_3dskeleton_in_matplotlib(data, graph, is_3d)


if __name__ == '__main__':

    # import os
    # os.environ['DISPLAY'] = 'localhost:10.0'

    data_path = "./data/data/ntu_nopad/xview/val_data_joint.npy"
    label_path = "./data/data/ntu_nopad/xview/val_label.pkl"
    # graph = 'graph.ntu_rgb_d.Graph'
    # test(data_path, label_path, vid='S004C001P003R001A032', graph=graph,
    #      is_3d=True)

    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)

    # s = time.time()
    # collate_fn = None
    # loader = FeederDataLoader(dataset='NTU60-CV').get_loader(
    #     feeder=Feeder(data_path, label_path,
    #                   dataset='NTU60-CV', random_rotation=True),
    #     batch_size=1,
    #     num_worker=1,
    #     drop_last=False,
    #     collate_fn=collate_fn,
    #     worker_init_fn=init_seed
    # )
    # for batch_idx, (data, label, idx) in enumerate(loader):
    #     # print("-"*80)
    #     break
    #     if batch_idx > 1000:
    #         print(data.shape)
    #         break
    # print("No-collate", (time.time()-s) / 1000)

    # s = time.time()
    # collate_fn = FeederDataLoader(
    #     dataset='NTU60-CV', seg=300).collate_fn_fix_train
    # loader = FeederDataLoader(dataset='NTU60-CV').get_loader(
    #     feeder=Feeder(data_path, label_path,
    #                   dataset='NTU60-CV', random_rotation=False),
    #     batch_size=1,
    #     num_worker=1,
    #     drop_last=False,
    #     collate_fn=collate_fn,
    #     worker_init_fn=init_seed
    # )
    # for batch_idx, (data, label, idx) in enumerate(loader):
    #     # print("-"*80)
    #     break
    #     if batch_idx > 1000:
    #         print(data.shape)
    #         break
    # print("---Collate", (time.time()-s) / 1000)
    # SGN style slower
