import json
import os
import numpy as np
import torch
import time

from tqdm import tqdm
from typing import Tuple

from data_gen.ntu_gendata import (
    read_skeleton_filter,
    training_cameras,
    training_subjects,
    max_frame,
    max_body_true,
    max_body_kinect,
    num_joint
)
from infer.data_preprocess import DataPreprocessor
from infer.inference import prepare_model, arg_parser, append_data_and_predict

from utils.utils import get_parser, init_seed


def get_datasplit_and_labels(
        data_path: str,
        ignored_sample_path: str = None,
        benchmark: str = 'xview',
        part: str = 'eval') -> Tuple[list, list]:

    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    sample_name = []
    sample_label = []
    for filename in tqdm(sorted(os.listdir(data_path))):
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


def evaluate_sequence(data: np.ndarray,
                      preprocessor: DataPreprocessor,
                      model: torch.nn.Module,
                      sample_label: int = 0) -> Tuple[list, list]:

    logits_list = []
    prediction_list = []

    preprocessor.reset_data()

    start = time.time()

    # 1. Read raw frames. ------------------------------------------------------
    # Loop through the sequence.
    # Each sequence will be gradually added into the data processor.
    # This mimics the real setting where the frame is
    # continously fed into the system.
    for i in range(data.shape[1]):

        data_i = data[:, i:i+1, :, :]

        # draw_skeleton(
        #     data=np.transpose(data_i, [3, 1, 2, 0]),
        #     action=MAPPING[sample_label+1]
        # )

        # 2. Batch frames to fixed length. -------------------------------------
        # 3. Normalization. ----------------------------------------------------
        # 4. Inference. --------------------------------------------------------
        logits, prediction, = append_data_and_predict(
            data_i, preprocessor, model, max_body_true)

        if i % 10 == 0:
            t = f"{(time.time() - start)/10:04.2f}s"
            pred = f"{prediction}"
            targ = f"{sample_label}"
            print(f"{i:03d} :: {targ} :: {pred} :: {t}")
            start = time.time()

        # print(output.numpy().tolist())
        # print(f"{sample_label} :: {predict_label.item()}")

        # 5. View sequence + predicted action + GT action. ---------------------
        logits_list.append(logits)
        prediction_list.append(prediction)

    return logits_list, prediction_list


def evaluate_recordings(sample_names: list,
                        sample_labels: list,
                        preprocessor: DataPreprocessor,
                        model: torch.nn.Module,) -> None:
    """ Loop through the recorded sequences. Each sequence is in a file.

    Args:
        sample_names (list): path names, each name is a sequence.
        sample_labels (list): labels
    """
    # Loop through the recorded sequences.
    # Each sequence is in a file.
    for sample_name, sample_label in zip(sample_names, sample_labels):
        print(f"Processing : {sample_name}")

        # M, T, V, C
        data = read_xyz(file=sample_name,
                        max_body=max_body_kinect,
                        num_joint=num_joint)

        logits_list, prediction_list = evaluate_sequence(
            data, preprocessor, model, sample_label)


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
    AAGCN = AAGCN.cuda(0)

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
            sample_names = [os.path.join(arg.data_path, sample_name)
                            for sample_name in sample_names]

            evaluate_recordings(sample_names, sample_labels, DataProc, AAGCN)
