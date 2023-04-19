import os
import numpy as np
import argparse
import pickle
from tqdm import tqdm

from data_gen.preprocess import pre_normalization
from data_gen.ntu_gendata import read_xyz, randomize


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 15
num_joint_ntu = 25
max_frame = 300

# openpose : ntu
joint_mapping = {
    0: 4,
    1: 21,
    2: 9,
    3: 10,
    4: 11,
    5: 5,
    6: 6,
    7: 7,
    8: 1,
    9: 17,
    10: 18,
    11: 19,
    12: 13,
    13: 14,
    14: 15,
}

# original : new labels
label_mapping = {
    1: 0,
    2: 0,
    8: 1,
    9: 2,
    27: 3,
    31: 4,
    43: 5,
    56: 6,
    59: 7,
    60: 8
}


def gendata(data_path, out_path, ignored_sample_path=None,
            benchmark='xview', part='eval', seed=None,
            label_mapping=False):

    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    sample_name = []
    sample_label = []
    filenames = sorted(os.listdir(data_path))
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

        if label_mapping:
            if action_class not in label_mapping.keys():
                continue

            action_class = label_mapping[action_class]

        else:
            action_class = action_class - 1

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
            sample_label.append(action_class)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, sample_label), f)

    # N, C, T, V, M
    fp = np.zeros((len(sample_label),
                   3,
                   max_frame,
                   num_joint,
                   max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        # C, T, V, M
        data = read_xyz(os.path.join(data_path, s),
                        max_body=max_body_kinect,
                        num_joint=num_joint_ntu)
        for new_id, old_id in joint_mapping.items():
            fp[i, :, :data.shape[1], new_id, :] = data[:, :, old_id-1, :]

    fp = pre_normalization(fp, zaxis=[8, 1], xaxis=[2, 5], verbose=True)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


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
        default='./data/data/openpose_b25_j15_ntu/')
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
        '--seed',
        type=int,
        default=1,
        help='random seed used to select file during data generation')
    parser.add_argument('--label_mapping', type=bool, default=False)
    args = parser.parse_args()

    for b in args.benchmark:
        for p in args.split:
            out_path = os.path.join(args.out_folder, b)
            os.makedirs(out_path, exist_ok=True)
            print(b, p)
            gendata(
                args.data_path,
                out_path,
                args.ignored_sample_path,
                benchmark=b,
                part=p,
                seed=args.seed,
                label_mapping=args.label_mapping)
