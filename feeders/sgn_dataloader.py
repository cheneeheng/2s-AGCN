# From : https://github.com/microsoft/SGN/blob/master/data.py

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from typing import Tuple, Callable


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
    def __init__(self, dataset='NTU', case=0, aug=1, seg=30, multi_test=5):
        self.dataset = dataset
        self.case = case
        self.aug = aug
        self.seg = seg
        # self.create_datasets()
        # self.train_set = NTUDataset(self.train_X, self.train_Y)
        # self.val_set = NTUDataset(self.val_X, self.val_Y)
        # self.test_set = NTUDataset(self.test_X, self.test_Y)
        self.train_set, self.val_set, self.test_set = None, None, None
        self.multi_test = multi_test

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
        x = _transform(x, theta)
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
                      labels: list,
                      sampling_frequency: int = 1) -> Tuple[list, list, list]:
        # skeleton_seqs: n,t,mvc
        # labels: n
        new_skeleton_seqs = []
        subject_seqs = []
        for _, skeleton_seq in enumerate(skeleton_seqs):
            zero_row = []
            for i in range(len(skeleton_seq)):
                if (skeleton_seq[i, :] == np.zeros((1, 150))).all():
                    zero_row.append(i)
            skeleton_seq = np.delete(skeleton_seq, zero_row, axis=0)
            skeleton_seq, subject_seq = turn_two_to_one(skeleton_seq)
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
        ave_duration = skeleton_seq.shape[0] // self.seg
        assert sampling_frequency >= 1
        for _ in range(sampling_frequency):
            offsets = np.multiply(list(range(self.seg)), ave_duration)
            offsets += np.random.randint(ave_duration, size=self.seg)
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


def turn_two_to_one(skeleton_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_skeleton_seq = []
    subject_seq = []
    for _, skel in enumerate(skeleton_seq):
        if (skel[0:75] == np.zeros((1, 75))).all():
            new_skeleton_seq.append(skel[75:])
            subject_seq.append([1.0])
        elif (skel[75:] == np.zeros((1, 75))).all():
            new_skeleton_seq.append(skel[0:75])
            subject_seq.append([0.0])
        else:
            new_skeleton_seq.append(skel[0:75])
            new_skeleton_seq.append(skel[75:])
            subject_seq.append([0.0])
            subject_seq.append([1.0])
    assert len(new_skeleton_seq) == len(subject_seq)
    return np.array(new_skeleton_seq), np.array(subject_seq)


def _rot(rot: torch.Tensor) -> torch.Tensor:
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:, :, 0:1], sin_r[:, :, 0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:, :, 0:1], cos_r[:, :, 0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=2)

    ry1 = torch.stack((cos_r[:, :, 1:2], zeros, -sin_r[:, :, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, :, 1:2], zeros, cos_r[:, :, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=2)

    rz1 = torch.stack((cos_r[:, :, 2:3], sin_r[:, :, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, :, 2:3], cos_r[:, :, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=2)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def _transform(x: torch.Tensor, theta: float) -> torch.Tensor:
    x = x.contiguous().view(x.size()[:2] + (-1, 3))
    rot = x.new(x.size()[0], 3).uniform_(-theta, theta)
    # rot = np.float32(np.random.uniform(-theta, theta, (1, 3)))
    # rot = torch.from_numpy(rot)
    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)
    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x
