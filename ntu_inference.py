import argparse
import json
import os
import numpy as np
import torch
import yaml
import time

from data_gen.ntu_gendata import (
    read_skeleton_filter,
    get_nonzero_std,
    training_cameras,
    training_subjects,
    max_frame,
    max_body_true,
    max_body_kinect,
    num_joint
)
from data_gen.preprocess import pre_normalization

from main import get_parser, import_class, init_seed

from inference import DataPreprocessor, prepare_model, arg_parser


def get_datasplit_and_labels(
        data_path: str,
        ignored_sample_path: str = None,
        benchmark: str = 'xview',
        part: str = 'eval'):

    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
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

    return sample_name, sample_label


def read_xyz(file, max_body=4, num_joint=25):
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data  # M, T, V, C


if __name__ == '__main__':
    init_seed(0)

    parser = get_parser()
    parser.add_argument(
        '--data_path',
        default='./data/data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument(
        '--ignored_sample_path',
        default='./data/data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument(
        '--model_path',
        default='./data/model/211116110001/')
    parser.add_argument(
        '--model_config',
        default='./data/model/211116110001/config.yaml')
    parser.add_argument(
        '--out_folder',
        default='./data/data/ntu/')
    arg = arg_parser(parser)

    # with open(os.path.join(arg.model_path, 'index_to_name.json'), 'r') as f:
    #     MAPPING = {int(i): j for i, j in json.load(f).items()}

    with open('./data/data/nturgbd_raw/index_to_name.json', 'r') as f:
        MAPPING = {int(i): j for i, j in json.load(f).items()}

    # Data processor -----------------------------------------------------------
    DataProc = DataPreprocessor(num_joint, max_frame)

    # Prepare model ------------------------------------------------------------
    AAGCN = prepare_model(arg)

    # Loop data ----------------------------------------------------------------
    for b in ['xsub', 'xview']:
        # for p in ['train', 'val']:
        for p in ['val']:
            print(f"Benchmark : {b}")
            print(f"Datasplit : {p}")

            # 0. Get all the relevant data and labels. -------------------------
            # Get the list of filenames and the labels from the dataset.
            sample_names, sample_labels = get_datasplit_and_labels(
                arg.data_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)

            # Loop through the recorded sequences.
            # Each sequence is in a file.
            for sample_name, sample_label in zip(sample_names, sample_labels):
                print(f"Processing : {sample_name}")

                # M, T, V, C
                data = read_xyz(os.path.join(arg.data_path, sample_name),
                                max_body=max_body_kinect,
                                num_joint=num_joint)

                DataProc.clear_data_array()

                start = time.time()

                # 1. Read raw frames. ------------------------------------------
                # Loop through the sequence.
                # Each sequence will be gradually added into the data processor.
                # This mimics the real setting where the frame is
                # continously fed into the system.
                for i in range(data.shape[1]):

                    # 2. Batch frames to fixed length.
                    DataProc.append_data(data[:, i:i+1, :, :])

                    # draw_skeleton(
                    #     data=np.transpose(data[:, i:i+1, :, :], [3, 1, 2, 0]),
                    #     action=MAPPING[sample_label+1]
                    # )

                    # 3. Normalization.
                    input_data = DataProc.select_skeletons_and_normalize_data(
                        max_body_true)

                    # N, C, T, V, M
                    input_data = np.transpose(input_data, [0, 4, 2, 3, 1])

                    # 4. Inference.
                    with torch.no_grad():
                        output = AAGCN(torch.Tensor(input_data))
                        _, predict_label = torch.max(output, 1)

                    if i % 10 == 0:
                        t = f"{(time.time() - start)/10:04.2f}s"
                        pred = f"{predict_label.item()}"
                        targ = f"{sample_label}"
                        print(f"{i:03d} :: {targ} :: {pred} :: {t}")
                        start = time.time()

                    # print(output.numpy().tolist())
                    # print(f"{sample_label} :: {predict_label.item()}")

                    # 5. View sequence + predicted action + GT action.
