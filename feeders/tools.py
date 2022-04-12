import numpy as np
import random
from scipy import interpolate
from typing import Tuple

import torch


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_axis_scale(data_numpy, candidate, channel):
    C, T, V, M = data_numpy.shape
    S = np.random.choice(candidate, 1)
    distance = \
        data_numpy[channel, :, :, 1] - data_numpy[channel, :, :, 0]  # T,V
    data_numpy[channel, :, :, 1] = \
        data_numpy[channel, :, :, 0] + (distance * S)
    return data_numpy


def random_xaxis_scale(data_numpy, candidate=None):
    candidate = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    return random_axis_scale(data_numpy, candidate, 0)


def random_yaxis_scale(data_numpy, candidate=None):
    candidate = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    return random_axis_scale(data_numpy, candidate, 1)


# def random_flip(data_numpy, channel):
#     C, T, V, M = data_numpy.shape
#     flip_idx = random.sample(range(0, T), T//2)
#     data_numpy[channel, flip_idx, :, :] = -data_numpy[channel, flip_idx, :, :]
#     return data_numpy

def random_flip(data_numpy, channel):
    C, T, V, M = data_numpy.shape
    flip_prob = random.random()
    if flip_prob > 0.5:
        data_numpy[channel, :, :, :] = -data_numpy[channel, :, :, :]
    return data_numpy


def random_xaxis_flip(data_numpy):
    return random_flip(data_numpy, channel=0)


def random_yaxis_flip(data_numpy):
    return random_flip(data_numpy, channel=1)


def random_zaxis_flip(data_numpy):
    return random_flip(data_numpy, channel=2)


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(
            S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(
            T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(
            T_y[i], T_y[i + 1], node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def _rot(rot):
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    zeros = np.zeros(rot.shape[:2] + (1,))
    ones = np.ones(rot.shape[:2] + (1,))

    r1 = np.stack((ones, zeros, zeros), axis=-1)
    rx2 = np.stack((zeros, cos_r[:, :, 0:1], sin_r[:, :, 0:1]), axis=-1)
    rx3 = np.stack((zeros, -sin_r[:, :, 0:1], cos_r[:, :, 0:1]), axis=-1)
    rx = np.concatenate((r1, rx2, rx3), axis=2)

    ry1 = np.stack((cos_r[:, :, 1:2], zeros, -sin_r[:, :, 1:2]), axis=-1)
    r2 = np.stack((zeros, ones, zeros), axis=-1)
    ry3 = np.stack((sin_r[:, :, 1:2], zeros, cos_r[:, :, 1:2]), axis=-1)
    ry = np.concatenate((ry1, r2, ry3), axis=2)

    rz1 = np.stack((cos_r[:, :, 2:3], sin_r[:, :, 2:3], zeros), axis=-1)
    r3 = np.stack((zeros, zeros, ones), axis=-1)
    rz2 = np.stack((-sin_r[:, :, 2:3], cos_r[:, :, 2:3], zeros), axis=-1)
    rz = np.concatenate((rz1, rz2, r3), axis=2)

    # rot = rz.matmul(ry).matmul(rx)
    rot = np.matmul(np.matmul(rz, ry), rx)
    return rot


# https://github.com/microsoft/SGN/blob/master/data.py
def random_rotation(data_numpy, theta=0.5):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    # data_numpy: N,T'(m is merged into it), vc = n,t,vc
    data_numpy = data_numpy.transpose(3, 1, 2, 0)
    rot = np.random.uniform(-theta, theta, (1, 3))
    rot = np.tile(rot, (1, T))
    rot = rot.reshape((-1, T, 3))
    rot = _rot(rot)
    data_numpy = np.transpose(data_numpy, (0, 1, 3, 2))
    data_numpy = np.matmul(rot, data_numpy)
    data_numpy = np.transpose(data_numpy, (2, 1, 3, 0))
    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M 偏移其中一段
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[:: -1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias: bias + size, :, :] = data_numpy[:, begin: end, :, :]

    return data_shift


# https://github.com/microsoft/SGN/blob/master/data/ntu/seq_transformation.py
def random_subsample(data_numpy, freq):
    C, T, V, M = data_numpy.shape
    segment_len = T // freq
    segments = np.multiply(list(range(freq)), segment_len)
    offsets = segments + np.random.randint(segment_len, size=freq)
    data_subsampled = data_numpy[:, offsets, :, :]
    return data_subsampled


def stretch_to_maximum_length(data_numpy):
    C, T, V, M = data_numpy.shape
    t_last = T - np.where(np.flip(data_numpy.sum((0, 2, 3))) != 0.0)[0][0]
    unpadded_data = data_numpy[:, : t_last, :, :]  # c,t,v,m
    unpadded_data = np.transpose(unpadded_data, (0, 2, 3, 1))  # c,v,m,t
    unpadded_data = unpadded_data.reshape(C*V*M, -1)
    f = interpolate.interp1d(np.arange(0, t_last), unpadded_data)
    stretched_data = f(np.linspace(0, t_last-1, T))
    stretched_data = stretched_data.reshape(C, V, M, T)
    stretched_data = np.transpose(stretched_data, (0, 3, 1, 2))
    return stretched_data


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0: T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0: 2, 0: T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0: 2, 1: T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = \
            data_numpy[:, t, :, forward_map[t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


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


def torch_transform(data_torch: torch.Tensor, theta: float) -> torch.Tensor:
    data_torch = data_torch.contiguous().view(data_torch.size()[:2] + (-1, 3))
    rot = data_torch.new(data_torch.size()[0], 3).uniform_(-theta, theta)
    # rot = np.float32(np.random.uniform(-theta, theta, (1, 3)))
    # rot = torch.from_numpy(rot)
    rot = rot.repeat(1, data_torch.size()[1])
    rot = rot.contiguous().view((-1, data_torch.size()[1], 3))
    rot = _rot(rot)
    data_torch = torch.transpose(data_torch, 2, 3)
    data_torch = torch.matmul(rot, data_torch)
    data_torch = torch.transpose(data_torch, 2, 3)
    data_torch = data_torch.contiguous().view(data_torch.size()[:2] + (-1,))
    return data_torch


def cumulative_auc(data_numpy: np.ndarray, norm: bool = False) -> np.ndarray:
    # cumulative area under the curve using trapezoid rule
    assert data_numpy.ndim == 2 or data_numpy.ndim == 1
    if norm:
        data_numpy = np.linalg.norm(data_numpy, axis=1)
    return (np.cumsum(data_numpy[:-1]) + np.cumsum(data_numpy[1:])) / 2


def split_idx_using_auc(data_numpy: np.ndarray,
                        num_segments: int) -> Tuple[np.ndarray, float]:
    N, C = data_numpy.shape
    assert data_numpy.ndim == 2 and N >= num_segments

    if N == num_segments:
        return list(range(num_segments+1))

    # trapezoid [N-1]
    cum_auc = cumulative_auc(data_numpy, norm=True)
    seg_area = cum_auc[-1] / num_segments
    # segment lower bounds [N-1]
    seg_idx, seg_lbs = np.unique((cum_auc / seg_area).astype(int),
                                 return_index=True)

    if len(seg_idx) > num_segments + 1:
        raise ValueError("seg_lbs_range length is more than seg")
    elif len(seg_idx) < num_segments + 1:
        c = 0
        for i in range(num_segments+1):
            if i not in seg_idx:
                added = False
                for j in range(len(seg_lbs)):
                    # shift left
                    if i-j > 0:
                        if seg_lbs[i-j] - seg_lbs[i-j-1] > 1:
                            seg_lbs = np.concatenate(
                                [seg_lbs[:i-j],
                                 [seg_lbs[i-j]-1],
                                 seg_lbs[i-j:]]
                            )
                            added = True
                            break
                    # shift right
                    if i+j+1 < len(seg_lbs):
                        if seg_lbs[i+j+1] - seg_lbs[i+j] > 1:
                            seg_lbs = np.concatenate(
                                [seg_lbs[:i+j+1],
                                 [seg_lbs[i+j+1]-1],
                                 seg_lbs[i+j+1:]]
                            )
                            added = True
                            break
                if not added:
                    seg_lbs = np.concatenate(
                        [seg_lbs[:i],
                         [seg_lbs[i]],
                         seg_lbs[i:]+1]
                    )
                c += 1

    # lower bound is valid
    seg_lbs[1:-1] += 1
    seg_lbs[-1] = N
    return seg_lbs, cum_auc[-1]


# ##############################################################################
# Tests
# ##############################################################################
# dummy = np.array([
#     [[[1, 2], [2, 3]], [[2, 3], [3, 4]]],
#     [[[2, 3], [3, 4]], [[3, 4], [4, 5]]],
#     [[[3, 4], [4, 5]], [[4, 5], [5, 6]]]
# ])  # c,t,v,m
# dummy = np.concatenate([dummy, dummy*0, dummy*0], axis=1)
# dummy1 = stretch_to_maximum_length(dummy)
# print(np.array_str(dummy, precision=1, suppress_small=True))
# print(np.array_str(dummy1, precision=1, suppress_small=True))

# xarr = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ])
# xarr = np.array([1, 1, 1, 1, 11, 2, 33, 2, 1, 2])
# xarr = np.array([1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 1])
# xarr = np.array([1, 1, 1, 1, 111, 2, 33, 2, 1, 2, 1, 100, 33, 2, 1, 1])
# xarr = np.array([1, 1, 1, 1, 111, 2, 33, 2, 1, 2, 1, 100, 33, 2, 1, 1,
#                  1, 1, 111, 2, 33, 2, 1, 2, 1, 100, 33, 2, 1, 1])
# xarr = np.expand_dims(xarr, -1)
# print("shape :", xarr.shape)
# splits = split_idx_using_auc(xarr, 10)
# print("split idx :", splits, len(splits))
# # print("split diff :", (splits[1:]-splits[:-1]).sum())
