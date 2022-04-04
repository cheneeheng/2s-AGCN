# From : https://github.com/microsoft/SGN/blob/master/data.py

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

import numpy as np
from typing import Tuple, Optional
from functools import partial

from feeders import tools


COLLATE_OUT_TYPE = Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None]


class NTUDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y, dtype='int')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.x[index], int(self.y[index])]


class NTUDataLoaders(object):
    def __init__(self,
                 dataset: str = 'NTU',
                 case: int = 0,
                 aug: int = 1,
                 seg: int = 30,
                 multi_test: int = 5,
                 motion_sampler: int = 0):
        self.dataset = dataset
        self.case = case
        self.aug = aug
        self.seg = seg
        self.train_set, self.val_set, self.test_set = None, None, None
        self.multi_test = multi_test
        self.motion_sampler = motion_sampler

    def get_train_loader(self, batch_size, num_workers):
        if self.aug == 0:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val,
                              pin_memory=False, drop_last=True)
        elif self.aug == 1:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_train,
                              pin_memory=True, drop_last=True)

    def get_val_loader(self, batch_size, num_workers):
        if 'kinetics' in self.dataset or 'NTU' in self.dataset:
            return DataLoader(self.val_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val,
                              pin_memory=True, drop_last=True)
        else:
            return DataLoader(self.val_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val,
                              pin_memory=True, drop_last=True)

    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=self.collate_fn_fix_test,
                          pin_memory=True, drop_last=True)

    def get_train_size(self):
        return len(self.train_Y)

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def collate_fn_fix(self,
                       batch: list,
                       sampling_frequency: int,
                       sort_data: bool,
                       ) -> COLLATE_OUT_TYPE:
        """Puts each data field into a tensor with outer dimension batch size.

        Args:
            batch (list): A mini-batch list of data tuples from dataset.
            sampling_frequency (int): sub-sampling frequency.
            sort_data (bool): Whether to sort data based on valid length.

        Returns:
            COLLATE_OUT_TYPE: tuple of data, label, None
        """
        x, y, _ = zip(*batch)
        x = [i.transpose(1, 3, 2, 0).reshape(x[0].shape[1], -1) for i in x]

        if 'kinetics' in self.dataset:
            x = np.array(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.reshape(-1, x.shape[1] * x.shape[2], x.shape[3]*x.shape[4])
            x = list(x)

        x, s, y = self.to_fix_length(x, y, sampling_frequency)

        if sort_data:
            # sort sequence by valid length in descending order
            lens = np.array([x_.shape[0] for x_ in x], dtype=np.int)
            idx = lens.argsort()[::-1]
            y = np.array(y)[idx]
        else:
            idx = range(len(x))

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        s = torch.stack([torch.from_numpy(s[i]) for i in idx], 0)
        y = torch.LongTensor(y)
        return (x, s), y, None

    def collate_fn_fix_train(self, batch: list) -> COLLATE_OUT_TYPE:
        """Puts each data field into a tensor with outer dimension batch size
        for training.

        Args:
            batch (list): A mini-batch list of data tuples from dataset.

        Returns:
            COLLATE_OUT_TYPE: tuple of data, label, None
        """
        (x, x1), y, _ = self.collate_fn_fix(batch,
                                            sampling_frequency=1,
                                            sort_data=True)
        # data augmentation
        if 'NTU60' in self.dataset:
            if self.case == 0:
                theta = 0.3
            elif self.case == 1:
                theta = 0.5
        elif 'NTU120' in self.dataset:
            theta = 0.3
        else:
            raise ValueError("unknown dataset name")
        x = tools.torch_transform(x, theta)
        return (x, x1), y, None

    def collate_fn_fix_val(self, batch: list) -> COLLATE_OUT_TYPE:
        """Puts each data field into a tensor with outer dimension batch size
        for validation.

        Args:
            batch (list): A mini-batch list of data tuples from dataset.

        Returns:
            COLLATE_OUT_TYPE: tuple of data, label, None
        """
        return self.collate_fn_fix(batch,
                                   sampling_frequency=1,
                                   sort_data=False)

    def collate_fn_fix_test(self, batch: list) -> COLLATE_OUT_TYPE:
        """Puts each data field into a tensor with outer dimension batch size
        for testing.

        Args:
            batch (list): A mini-batch list of data tuples from dataset.

        Returns:
            COLLATE_OUT_TYPE: tuple of data, label, None
        """
        return self.collate_fn_fix(batch,
                                   sampling_frequency=self.multi_test,
                                   sort_data=False)

    def to_fix_length(self,
                      skeleton_seqs: list,
                      labels: Optional[list],
                      sampling_frequency: int = 1
                      ) -> Tuple[list, list, Optional[list]]:
        # skeleton_seqs: n,t,mvc
        # labels: n
        new_skeleton_seqs = []
        subject_seqs = []
        for _, skeleton_seq in enumerate(skeleton_seqs):
            zero_row = []
            for i in range(len(skeleton_seq)):
                if (skeleton_seq[i, :] == np.zeros((1, skeleton_seq.shape[-1]))).all():  # noqa
                    zero_row.append(i)
            skeleton_seq = np.delete(skeleton_seq, zero_row, axis=0)
            skeleton_seq, subject_seq = self.turn_two_to_one(skeleton_seq)
            skeleton_seq, subject_seq = self.pad_sequence(
                skeleton_seq=skeleton_seq,
                subject_seq=subject_seq
            )
            skeleton_seqs, subject_seqs = self.subsample_sequence(
                skeleton_seq=skeleton_seq,
                subject_seq=subject_seq,
                skeleton_seqs=new_skeleton_seqs,
                subject_seqs=subject_seqs,
                sampling_frequency=sampling_frequency
            )
        return new_skeleton_seqs, subject_seqs, labels

    def subsample_sequence(self,
                           skeleton_seq: np.ndarray,
                           subject_seq: np.ndarray,
                           skeleton_seqs: list,
                           subject_seqs: list,
                           sampling_frequency: int = 1) -> Tuple[list, list]:

        if self.motion_sampler == 1:
            x = np.linalg.norm(skeleton_seq, axis=1)  # T
            seg_sum = (sum((x[:-1] + x[1:])) / 2) / self.seg
            trapz = (np.cumsum(x[:-1]) + np.cumsum(x[1:])) / 2
            intervals = np.zeros(self.seg+1, dtype=int)  # T+1
            intervals[1:-1] = np.unique(trapz // seg_sum, return_index=True)[1]
            intervals[-1] = len(x)-1
            intervals_range = intervals[1:] - intervals[:-1]  # T
            random_intervals_range_fn = partial(np.random.randint,
                                                low=0,
                                                high=intervals_range)
            intervals = intervals[:-1]  # T

        else:
            avg_range = skeleton_seq.shape[0] // self.seg
            intervals = np.multiply(list(range(self.seg)), avg_range)
            random_intervals_range_fn = partial(np.random.randint,
                                                low=avg_range,
                                                size=self.seg)

        assert sampling_frequency >= 1
        for _ in range(sampling_frequency):
            offsets = intervals + random_intervals_range_fn()
            assert len(offsets) == self.seg, f"{len(offsets)} =/= {self.seg}"
            skeleton_seqs.append(skeleton_seq[offsets])
            subject_seqs.append(subject_seq[offsets])
        return skeleton_seqs, subject_seqs

    def pad_sequence(self,
                     skeleton_seq: np.ndarray,
                     subject_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if 'SYSU' in self.dataset:
            skeleton_seq = skeleton_seq[::2, :]
            subject_seq = subject_seq[::2, :]
        skeleton_seq = self.pad_to_segment_length(skeleton_seq)
        subject_seq = self.pad_to_segment_length(subject_seq)
        return skeleton_seq, subject_seq

    def pad_to_segment_length(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < self.seg:
            shape = (self.seg - x.shape[0], x.shape[1])
            pad = np.zeros(shape, dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
        return x

    def turn_two_to_one(self,
                        skeleton_seq: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        new_skeleton_seq = []
        subject_seq = []
        for _, skel in enumerate(skeleton_seq):
            idx = skel.shape[-1] // 2
            if (skel[0:idx] == np.zeros((1, idx))).all():
                new_skeleton_seq.append(skel[idx:])
                subject_seq.append([1.0])
            elif (skel[idx:] == np.zeros((1, idx))).all():
                new_skeleton_seq.append(skel[0:idx])
                subject_seq.append([0.0])
            else:
                new_skeleton_seq.append(skel[0:idx])
                new_skeleton_seq.append(skel[idx:])
                subject_seq.append([0.0])
                subject_seq.append([1.0])
        assert len(new_skeleton_seq) == len(subject_seq)
        return np.array(new_skeleton_seq), np.array(subject_seq)


class FeederDataLoader(NTUDataLoaders):
    def __init__(self,
                 dataset: str = 'NTU60-CV',
                 aug: int = 1,
                 seg: int = 30,
                 multi_test: int = 5,
                 motion_sampler: int = 0):
        if 'CS' in dataset:
            case = 0
        elif 'CV' in dataset:
            case = 1
        else:
            case = -1
        super(FeederDataLoader, self).__init__(
            dataset, case, aug, seg, multi_test, motion_sampler)

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
                   ) -> DataLoader:
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
