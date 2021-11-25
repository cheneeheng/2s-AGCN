import argparse
import json
import os
import numpy as np
import torch
import yaml
import time
from datetime import datetime

from data_gen.ntu_gendata import (
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

from visualize_ntu_skel import draw_skeleton


def read_xyz(file, max_body=4, num_joint=25):
    skel_data = np.loadtxt(file, delimiter=',')
    data = np.zeros((max_body, 1, num_joint, 3))
    for m, body_joint in enumerate(skel_data):
        for j in range(0, len(body_joint), 3):
            if m < max_body and j//3 < num_joint:
                data[m, 0, j//3, :] = [body_joint[j],
                                       body_joint[j+1],
                                       body_joint[j+2]]
            else:
                pass
    return data  # M, T, V, C


class DataPreprocessor(object):

    def __init__(self, num_joint=25, max_seq_length=300) -> None:
        super().__init__()
        self.num_joint = num_joint
        self.max_seq_length = max_seq_length
        self.data = None
        self.data_counter = 0
        self.clear_data_array()

    def clear_data_array(self) -> None:
        """
        Creates an empty/zero array of size (M,T,V,C).
        We assume that the input data can have up to 4 possible skeletons ids.
        """
        self.data = np.zeros((4,
                              self.max_seq_length,
                              self.num_joint,
                              3),
                             dtype=np.float32)
        self.data_counter = 0

    def append_data(self, data: np.ndarray) -> None:
        """Append data.

        Args:
            data (np.ndarray): (M, 1, V, C)
        """
        if self.data_counter < self.max_seq_length:
            self.data[:, self.data_counter:self.data_counter+1, :, :] = data
            self.data_counter += 1
        else:
            self.data[:, 1:, :, :] = self.data[:, 0:-2, :, :]
            self.data[:, -2:, :, :] = data

    def select_skeletons(self, num_skels: int = 2) -> np.ndarray:
        """Select the `num_skels` most active skeletons. """
        # select two max energy body
        energy = np.array([get_nonzero_std(x) for x in self.data])
        index = energy.argsort()[::-1][0:num_skels]
        return self.data[index]  # m', T, V, C

    def normalize_data(self, data: np.ndarray) -> None:
        if data.ndim < 4 or data.ndim > 5:
            raise ValueError("Dimension not supported...")
        if data.ndim == 4:
            data = np.expand_dims(data, axis=0)
        data = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M
        data = pre_normalization(data)
        data = np.transpose(data, [0, 4, 2, 3, 1])  # N, M, T, V, C
        return data

    def select_skeletons_and_normalize_data(self,
                                            num_skels: int = 2) -> np.ndarray:
        data = self.select_skeletons(num_skels=num_skels)
        return self.normalize_data(data)


def prepare_model(arg):
    Model = import_class(arg.model)
    AAGCN = Model(**arg.model_args)
    AAGCN.eval()
    weight_file = [i for i in os.listdir(arg.model_path) if '.pt' in i]
    weight_file = os.path.join(arg.model_path, weight_file[0])
    weights = torch.load(weight_file)
    AAGCN.load_state_dict(weights)
    return AAGCN


def arg_parser(parser):
    p = parser.parse_args()
    with open(os.path.join(p.model_path, 'config.yaml'), 'r') as f:
        default_arg = yaml.safe_load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print(f'WRONG ARG: {k}')
            assert (k in key)
    parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    return arg


if __name__ == '__main__':

    init_seed(0)

    parser = get_parser()
    parser.add_argument(
        '--inference_interval',
        default=30)
    parser.add_argument(
        '--data_path',
        default='/data/openpose/skeleton/')
    parser.add_argument(
        '--model_path',
        default='/data/2s-agcn/model/211116110001/')
    parser.add_argument(
        '--out_folder',
        default='/data/2s-agcn/prediction/211116110001/')

    # load arg form config file ------------------------------------------------
    arg = arg_parser(parser)
    with open(os.path.join(arg.model_path, 'index_to_name.json'), 'r') as f:
        MAPPING = {int(i): j for i, j in json.load(f).items()}

    output_file = os.path.join(
        arg.out_folder, datetime.now().strftime("%y%m%d%H%M%S") + '.txt')

    skel_dir = os.path.join(arg.data_path,
                            sorted(os.listdir(arg.data_path))[-1])

    # Data processor -----------------------------------------------------------
    DataProc = DataPreprocessor(num_joint, max_frame)

    # Prepare model ------------------------------------------------------------
    AAGCN = prepare_model(arg)

    # MAIN LOOP ----------------------------------------------------------------
    start = time.time()
    skel_path_mem = None
    infer_flag = False

    while True:

        # infer if
        # a. more than interval.
        # b. a valid skeleton is available.
        if time.time() - start <= arg.inference_interval:
            continue
        else:
            if infer_flag:
                start = time.time()
                infer_flag = False

        skel_path = os.path.join(skel_dir, sorted(os.listdir(skel_dir))[-1])

        if skel_path == skel_path_mem:
            continue
        else:
            skel_path_mem = skel_path
            infer_flag = True

        # 1. Read raw frames. --------------------------------------------------
        # M, T, V, C
        data = read_xyz(skel_path,
                        max_body=max_body_kinect,
                        num_joint=num_joint)

        # 2. Batch frames to fixed length. -------------------------------------
        DataProc.append_data(data)

        # 3. Normalization.
        input_data = DataProc.select_skeletons_and_normalize_data(
            max_body_true)

        # N, C, T, V, M
        input_data = np.transpose(input_data, [0, 4, 2, 3, 1])

        # 4. Inference.
        with torch.no_grad():
            output = AAGCN(torch.Tensor(input_data))
            _, predict_label = torch.max(output, 1)

        with open(output_file, 'a+') as f:
            output_str = ",".join([str(pred) for pred in output.tolist()])
            print(f'{predict_label.item()},{output_str}', file=f)
