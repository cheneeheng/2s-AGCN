import os
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_gen.preprocess import pre_normalization
from data_gen.ntu_gendata import read_xyz, randomize


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
# num_joint = 15
num_joint_ntu = 25
max_frame = 300

# openpose : ntu
joint_mapping_15 = {
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

# openpose : ntu
joint_mapping_11 = {
    0: 4,
    1: 21,
    2: 9,
    3: 10,
    4: 5,
    5: 6,
    6: 1,
    7: 17,
    8: 18,
    9: 13,
    10: 14,
}

# original : new labels
label_mapping_9l = {
    1: 0,  # drink
    2: 0,  # eat
    8: 1,  # sit down
    9: 2,  # standup
    27: 3,  # jump
    31: 4,  # pointing
    43: 5,  # falling
    56: 6,  # giving to another person
    59: 7,  # waling towards each other
    60: 8  # walking apart
}

# original : new labels
label_mapping_5l = {
    1: 0,  # drink
    2: 0,  # eat
    8: 1,  # sit down
    9: 2,  # stand up
    26: 3,  # hop
    27: 3,  # jump
    43: 4,  # falling
}

# original : new labels
label_mapping_4l = {
    1: 0,  # drink
    2: 0,  # eat
    8: 1,  # sit down
    9: 2,  # stand up
    43: 3,  # falling
}


def gendata(data_path, out_path, ignored_sample_path=None,
            benchmark='xview', part='eval', seed=None,
            custom_label='', num_joints=15):

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

        if custom_label == '9l':
            if action_class not in label_mapping_9l.keys():
                continue
            action_class = label_mapping_9l[action_class]
        elif custom_label == '5l':
            if action_class not in label_mapping_5l.keys():
                continue
            action_class = label_mapping_5l[action_class]
        elif custom_label == '4l':
            if action_class not in label_mapping_4l.keys():
                continue
            action_class = label_mapping_4l[action_class]
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
                   num_joints,
                   max_body_true), dtype=np.float32)

    if num_joints == 15:
        joint_mapping = joint_mapping_15
    elif num_joints == 11:
        joint_mapping = joint_mapping_11
    else:
        raise ValueError("Unknown number of joints...")

    for i, s in enumerate(tqdm(sample_name)):
        # C, T, V, M
        data = read_xyz(os.path.join(data_path, s),
                        max_body=max_body_kinect,
                        num_joint=num_joint_ntu)
        for new_id, old_id in joint_mapping.items():
            fp[i, :, :data.shape[1], new_id, :] = data[:, :, old_id-1, :]

    # np.save(
    #     f'output/data_{custom_label}_{benchmark}_{part}_j{num_joints}.npy',
    #     fp)

    if num_joints == 15:
        fp = pre_normalization(fp, zaxis=[8, 1], xaxis=[2, 5], verbose=True)
    elif num_joints == 11:
        fp = pre_normalization(fp, zaxis=[6, 1], xaxis=[2, 4], verbose=True)
    np.save(
        f'output/data_{custom_label}_{benchmark}_{part}_j{num_joints}.npy',
        fp)

    # fp = np.load(
    #     f'output/data_{custom_label}_{benchmark}_{part}_j{num_joints}.npy')

    # num_labels = len(np.unique(sample_label))
    # x, y, z = [], [], []
    # for i in range(num_labels):
    #     x += [fp[np.array(sample_label) == i][:, 0].reshape(-1)]
    #     y += [fp[np.array(sample_label) == i][:, 1].reshape(-1)]
    #     z += [fp[np.array(sample_label) == i][:, 2].reshape(-1)]
    # for i in range(num_labels):
    #     x[i] = x[i][z[i] > 1.0]
    #     y[i] = y[i][z[i] > 1.0]
    #     z[i] = z[i][z[i] > 1.0]

    # fig, axes = plt.subplots(nrows=2, ncols=num_labels//2)
    # axes = axes.flatten()
    # for i in range(num_labels):
    #     axes[i].hist(x[i][x[i] != 0.0], bins=100, alpha=0.5, label='x')
    #     axes[i].hist(y[i][y[i] != 0.0], bins=100, alpha=0.5, label='y')
    #     axes[i].hist(z[i][z[i] != 0.0], bins=100, alpha=0.5, label='z')
    #     axes[i].legend(loc='upper right')
    #     # axes[i].set_xlim([-2, 5])
    #     # axes[i].set_ylim([0, 500])
    # plt.tight_layout()

    # fig, axes = plt.subplots(nrows=1, ncols=3)
    # axes = axes.flatten()
    # for i in range(num_labels):
    #     axes[0].hist(x[i][x[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    #     axes[1].hist(y[i][y[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    #     axes[2].hist(z[i][z[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    # for i in range(3):
    #     axes[i].legend(loc='upper right')
    #     # axes[i].set_xlim([-2, 5])
    #     # axes[i].set_ylim([0, 500])
    # plt.tight_layout()

    # fp = pre_normalization(fp, zaxis=[8, 1], xaxis=[2, 5], verbose=True)

    # x, y, z = [], [], []
    # for i in range(num_labels):
    #     x += [fp[np.array(sample_label) == i][0].reshape(-1)]
    #     y += [fp[np.array(sample_label) == i][1].reshape(-1)]
    #     z += [fp[np.array(sample_label) == i][2].reshape(-1)]

    # fig, axes = plt.subplots(nrows=2, ncols=num_labels//2)
    # axes = axes.flatten()
    # for i in range(num_labels):
    #     axes[i].hist(x[i][x[i] != 0.0], bins=100, alpha=0.5, label='x')
    #     axes[i].hist(y[i][y[i] != 0.0], bins=100, alpha=0.5, label='y')
    #     axes[i].hist(z[i][z[i] != 0.0], bins=100, alpha=0.5, label='z')
    #     axes[i].legend(loc='upper right')
    #     axes[i].set_xlim([-5, 5])
    #     axes[i].set_ylim([0, 500])
    # plt.tight_layout()

    # fig, axes = plt.subplots(nrows=1, ncols=3)
    # axes = axes.flatten()
    # for i in range(num_labels):
    #     axes[0].hist(x[i][x[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    #     axes[1].hist(y[i][y[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    #     axes[2].hist(z[i][z[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    # for i in range(3):
    #     axes[i].legend(loc='upper right')
    #     axes[i].set_xlim([-2, 2])
    #     axes[i].set_ylim([0, 800])
    # plt.tight_layout()

    # plt.show()

    print("Finished")


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
        default='./data/data/openpose_b25_j11_5l_ntu_delme/')
    parser.add_argument(
        '--benchmark',
        default=['xview', 'xsub'],
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
    # parser.add_argument('--custom_label', default="")
    # parser.add_argument('--custom_label', default="9l")
    # parser.add_argument('--custom_label', default="5l")
    parser.add_argument('--custom_label', default="4l")
    parser.add_argument('--num_joints', type=int, default=15)
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
                custom_label=args.custom_label,
                num_joints=args.num_joints)

    plt.show()
