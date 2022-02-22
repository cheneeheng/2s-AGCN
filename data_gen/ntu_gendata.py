import argparse
import numpy as np
import os
import pickle
import random
from scipy import interpolate

from tqdm import tqdm

from data_gen.preprocess import pre_normalization

from utils.multiprocessing import parallel_processing

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300


def stretch_to_maximum_length(data_numpy):
    C, T, V, M = data_numpy.shape
    unpadded_data = data_numpy  # c,t,v,m
    unpadded_data = np.transpose(unpadded_data, (0, 2, 3, 1))  # c,v,m,t
    unpadded_data = unpadded_data.reshape(C*V*M, -1)
    f = interpolate.interp1d(np.arange(0, T), unpadded_data)
    stretched_data = f(np.linspace(0, T-1, max_frame))
    stretched_data = stretched_data.reshape(C, V, M, max_frame)
    stretched_data = np.transpose(stretched_data, (0, 3, 1, 2))
    return stretched_data


def randomize(data, seed=None):
    if seed is not None:
        random.seed(seed)
    random.shuffle(data)


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + \
            s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None,
            benchmark='xview', part='eval', stretch=False, seed=None):

    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    sample_name = []
    sample_label = []
    filenames = os.listdir(data_path)
    randomize(filenames, seed)
    for filename in tqdm(filenames):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame,
                  num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s),
                        max_body=max_body_kinect, num_joint=num_joint)
        if stretch:
            fp[i, :, :, :, :] = stretch_to_maximum_length(data)
        else:
            fp[i, :, 0:data.shape[1], :, :] = data

    fp = pre_normalization(fp, pad=False)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


def tmp_fn(data_path, seeds, pid=0):
    for i in tqdm(seeds):
        filenames = os.listdir(data_path)
        randomize(filenames, i)
        if filenames[0] == 'S001C003P004R002A038.skeleton':
            print(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data-path',
        default='./data/data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument(
        '--ignored-sample-path',
        default='./data/data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument(
        '--out-folder',
        default='./data/data/ntu_stretched/')
    parser.add_argument(
        '--benchmark',
        default=['xsub', 'xview'],
        nargs='+',
        help='which Top K accuracy will be shown')
    parser.add_argument(
        '--split',
        default=['train', 'val'],
        nargs='+',
        help='which Top K accuracy will be shown')

    parser.add_argument(
        '--stretch',
        default=True)

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed used to select file during data generation'
    )

    arg = parser.parse_args()
    benchmark = arg.benchmark
    part = arg.split

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)

            # parallel_processing(
            # tmp_fn, 6, [i for i in range(10000)], arg.data_path)
            # exit()

            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p,
                stretch=arg.stretch,
                seed=arg.seed,)
