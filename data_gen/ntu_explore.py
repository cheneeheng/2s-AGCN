import argparse
import json
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_gen.preprocess import pre_normalization
from data_gen.ntu_gendata import read_xyz
from infer.ntu60.visualize_ntu_skel import visualize_3dskeleton_in_matplotlib


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

with open('data/data/nturgbd_raw/index_to_name.json') as f:
    MAPPING = json.load(f)


def gendata(data_path, ignored_sample_path=None,
            benchmark='xview', part='eval'):

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

    # fp = np.zeros((len(sample_label), 3, max_frame,
    #               num_joint, max_body_true), dtype=np.float32)

    fig = plt.figure(figsize=(16, 8))

    for i, s in enumerate(tqdm(sample_name)):
        # if sample_label[i]+1 not in [1, 8, 9, 27, 31, 43, 56, 59, 60]:
        if sample_label[i]+1 not in [8, 9]:
            continue
        # c,t,v,m
        data = read_xyz(os.path.join(data_path, s),
                        max_body=max_body_kinect, num_joint=num_joint)
        # fp[i, :, 0:data.shape[1], :, :] = data
        visualize_3dskeleton_in_matplotlib(
            data=np.expand_dims(data, axis=0),
            graph='graph.ntu_rgb_d.Graph',
            is_3d=True,
            speed=1e-8,
            text_per_t=[MAPPING[str(sample_label[i]+1)]
                        for _ in range(max_frame)],
            fig=fig
        )
        fig.clear()

    # fp = pre_normalization(fp, pad=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data-path',
        default='./data/data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument(
        '--ignored-sample-path',
        default='./data/data/nturgbd_raw/samples_with_missing_skeletons.txt')
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
    args = parser.parse_args()

    for b in args.benchmark:
        for p in args.split:
            print(b, p)
            gendata(
                args.data_path,
                args.ignored_sample_path,
                benchmark=b,
                part=p)
