import json
import numpy as np
import os
import time
import torch
import yaml
import pickle

from datetime import datetime
from functools import partial
from typing import Tuple

from data_gen.ntu_gendata import (
    max_frame,
    max_body_true,
    max_body_kinect,
    num_joint
)
from infer.data_preprocess import DataPreprocessor
from data_gen.preprocess import pre_normalization
from data_gen.ntu_gendata import read_xyz as reader
from main_utils import get_parser, import_class, init_seed


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def arg_parser(parser):
    p = parser.parse_args()
    with open(os.path.join(p.weight_path, 'config.yaml'), 'r') as f:
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
                data[m, 0, j//3, :] = [-body_joint[j],
                                       -body_joint[j+2],
                                       -body_joint[j+1]]
            else:
                pass
    data = np.swapaxes(data, 0, 3)/1000.0   # M, T, V, C > C, T, V, M
    return data


def predict(preprocessor: DataPreprocessor,
            model: torch.nn.Module,
            num_skels: int) -> Tuple[list, int]:
    """
    Args:
        data (np.ndarray): (M, 1, V, C)
    """
    # 2. Normalization.
    input_data = preprocessor.select_skeletons_and_normalize_data(num_skels)
    input_data = np.transpose(input_data, [0, 4, 2, 3, 1])  # N, C, T, V, M

    # 3. Inference.
    with torch.no_grad():
        if next(model.parameters()).is_cuda:
            output, _ = model(torch.Tensor(input_data).cuda(0))
            _, predict_label = torch.max(output, 1)
            output = output.data.cpu()
            predict_label = predict_label.data.cpu()

        else:
            output, _ = model(torch.Tensor(input_data))
            _, predict_label = torch.max(output, 1)

    return output.tolist(), predict_label.item()


if __name__ == '__main__':

    init_seed(0)

    parser = get_parser()
    # parser.add_argument('--max-person', type=int, default=2)
    parser.add_argument('--max-frame', type=int, default=max_frame)
    parser.add_argument('--max-num-skeleton-true', type=int, default=2)  # noqa
    parser.add_argument('--max-num-skeleton', type=int, default=4)  # noqa
    parser.add_argument('--num-joint', type=int, default=15)

    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--timing', type=bool, default=False)
    parser.add_argument('--interval', type=int, default=0)

    # parser.add_argument(
    #     '--data-path',
    #     type=str,
    #     default='/data/openpose/skeleton/')
    parser.add_argument(
        '--model-path',
        type=str,
        # default='/data/2s-agcn/model/ntu_15j/')
        default='./data/model/ntu_15j/')
    parser.add_argument(
        '--weight-path',
        type=str,
        # default='./data/model/ntu_25j/'
        # default='/data/2s-agcn/model/ntu_15j/xview/211130150001/'
        # default='/data/2s-agcn/model/ntu_15j/xview/220314100001/'
        # default='/data/2s-agcn/model/ntu_15j/xsub/220314090001/'
        default='./data/model/ntu_15j/xview/220314100001/'
    )
    parser.add_argument(
        '--out-folder',
        type=str,
        default='/data/2s-agcn/prediction/ntu_15j/')

    # load arg form config file ------------------------------------------------
    arg = arg_parser(parser)
    with open(os.path.join(arg.model_path, 'index_to_name.json'), 'r') as f:
        MAPPING = {int(i): j for i, j in json.load(f).items()}

    # LOAD DATA ----------------------------------------------------------------
    joint_path = './data/data_tmp/220324153743'
    joint_files = [os.path.join(joint_path, i)
                   for i in sorted(os.listdir(joint_path))]
    data = []
    data_raw = []
    data_path = []
    for joint_file in joint_files:
        data_i = read_xyz(joint_file, max_body=4, num_joint=15)  # C, T, V, M
        data_raw.append(np.array(data_i))
        data_i = np.expand_dims(data_i, axis=0)
        # data_i = pre_normalization(data_i, zaxis2=[8, 1], xaxis=[2, 5],
        #                            verbose=False, tqdm=False)
        data_i = np.swapaxes(data_i[0], 0, 3)  # C, T, V, M > M, T, V, C
        data.append(data_i)
        data_path.append(joint_file)
    data = np.concatenate(data, 1)[:, 0:, :, :]  # C, T, V, M

    # Data processor -----------------------------------------------------------
    preprocess_fn = partial(pre_normalization,
                            zaxis=[8, 1],
                            xaxis=[2, 5],
                            verbose=False,
                            tqdm=False)

    DataProc = DataPreprocessor(num_joint=arg.num_joint,
                                max_seq_length=100,
                                # max_seq_length=arg.max_frame,
                                max_person=2,  # arg.max_num_skeleton,
                                moving_avg=5,
                                preprocess_fn=preprocess_fn)

    # Prepare model ------------------------------------------------------------
    Model = import_class(arg.model)(**arg.model_args)
    Model.eval()
    weight_file = [i for i in os.listdir(arg.weight_path) if '.pt' in i]
    weight_file = os.path.join(arg.weight_path, weight_file[0])
    weights = torch.load(weight_file)
    Model.load_state_dict(weights)
    if arg.gpu:
        Model = Model.cuda(0)
    print("Model loaded...")

    # MAIN LOOP ----------------------------------------------------------------
    start = time.time()
    skel_path_mem = None
    infer_flag = False

    last_skel_file = None

    c = 0

    with open(f'infer/openpose_b25_j15/result_{joint_path.split("/")[-1]}_ma5_100pads.txt', 'w') as f:  # noqa

        print("Start loop...")
        while True:

            # 1. Read raw frames. ----------------------------------------------
            # M, T, V, C
            DataProc.append_data(data[:1, c:c+1, :, :])

            # 2. Batch frames to fixed length. ---------------------------------
            # 3. Normalization. ------------------------------------------------
            # 4. Inference. ----------------------------------------------------

            # Normalization.
            input_data = DataProc.select_skeletons_and_normalize_data(
                arg.max_num_skeleton_true)
            input_data = np.transpose(
                input_data, [0, 4, 2, 3, 1])  # N, C, T, V, M
            # zeros = np.zeros((*input_data.shape[0:2],
            #                   arg.max_frame,
            #                   *input_data.shape[3:]))
            # zeros[:, :, :100, :, :] = input_data
            # input_data = zeros
            input_data = np.concatenate(
                [input_data, input_data, input_data], axis=2)

            # Inference.
            with torch.no_grad():

                if next(Model.parameters()).is_cuda:
                    output, _ = Model(torch.Tensor(input_data).cuda(0))
                    _, predict_label = torch.max(output, 1)
                    output = output.data.cpu()
                    predict_label = predict_label.data.cpu()

                else:
                    output, _ = Model(torch.Tensor(input_data))
                    _, predict_label = torch.max(output, 1)

            logits, pred = output.tolist(), predict_label.item()

            logits = softmax(logits[0])
            pred_i = np.argmax(logits)
            assert pred == pred_i

            if c % 1 == 0:
                print(MAPPING[pred+1])
                print(MAPPING[pred+1], file=f)
                # print(f'frame : {c}  ::  '
                #       f'pred  : {pred+1}  ::  '
                #       f'logit : {logits[pred]:.4f}  ::  '
                #       f'pred  : {MAPPING[pred+1]}  ::  '
                #       )

            c += 1
            if c > data.shape[1]:
                break
