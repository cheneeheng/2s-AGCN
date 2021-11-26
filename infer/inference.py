import json
import numpy as np
import os
import time
import torch
import yaml

from datetime import datetime
from typing import Tuple

from data_gen.ntu_gendata import (
    max_frame,
    max_body_true,
    max_body_kinect,
    num_joint
)

from data_preprocess import DataPreprocessor
from main import get_parser, import_class, init_seed


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


def prepare_model(arg):
    Model = import_class(arg.model)
    AAGCN = Model(**arg.model_args)
    AAGCN.eval()
    weight_file = [i for i in os.listdir(arg.model_path) if '.pt' in i]
    weight_file = os.path.join(arg.model_path, weight_file[0])
    weights = torch.load(weight_file)
    AAGCN.load_state_dict(weights)
    return AAGCN


def append_data_and_predict(data: np.ndarray,
                            preprocessor: DataPreprocessor,
                            model: torch.nn.Module,
                            num_skels: int) -> Tuple[list, int]:
    """
    Args:
        data (np.ndarray): (M, 1, V, C)
    """
    # 1. Batch frames to fixed length.
    preprocessor.append_data(data)

    # 2. Normalization.
    input_data = preprocessor.select_skeletons_and_normalize_data(num_skels)
    input_data = np.transpose(input_data, [0, 4, 2, 3, 1])  # N, C, T, V, M

    # 3. Inference.
    with torch.no_grad():
        if next(model.parameters()).is_cuda:
            output = model(torch.Tensor(input_data).cuda(0))
            _, predict_label = torch.max(output, 1)
            output = output.data.cpu()
            predict_label = predict_label.data.cpu()

        else:
            output = model(torch.Tensor(input_data))
            _, predict_label = torch.max(output, 1)

    return output.tolist(), predict_label.item()


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

    os.makedirs(arg.out_folder, exist_ok=True)
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
        # 3. Normalization. ----------------------------------------------------
        # 4. Inference. --------------------------------------------------------
        logits, prediction, = append_data_and_predict(
            data, DataProc, AAGCN, max_body_true)

        with open(output_file, 'a+') as f:
            output_str = ",".join([str(logit) for logit in logits])
            print(f'{prediction},{output_str}', file=f)
