import argparse
import json
import numpy as np
import os
import time
import yaml

from collections import OrderedDict
from functools import partial
from typing import Tuple

import torch
from torch.nn import functional as F

from data_gen.preprocess import pre_normalization
from feeders.loader import NTUDataLoaders
from infer.data_preprocess import DataPreprocessor
from utils.parser import get_parser as get_default_parser
from utils.utils import import_class


def parse_arg(parser: argparse.ArgumentParser) -> argparse.Namespace:
    [p, _] = parser.parse_known_args()
    with open(os.path.join(p.weight_path, 'config.yaml'), 'r') as f:
        default_arg = yaml.safe_load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print(f'WRONG ARG: {k}')
            assert (k in key)
    parser.set_defaults(**default_arg)
    return parser.parse_known_args()


def read_xyz(file: str, max_body: int = 4, num_joint: int = 25) -> np.ndarray:
    skel_data = np.loadtxt(file, delimiter=',')
    data = np.zeros((max_body, 1, num_joint, 3))
    for m, body_joint in enumerate(skel_data):
        for j in range(0, len(body_joint), 3):
            if m < max_body and j//3 < num_joint:
                # x subject right, y to camera, z up
                data[m, 0, j//3, :] = [body_joint[j],
                                       body_joint[j+1],
                                       body_joint[j+2]]
            else:
                pass
    return data  # M, T, V, C


def filter_logits(logits: list) -> Tuple[list, list]:
    # {
    #     "8": "sitting down",
    #     "9": "standing up (from sitting position)",
    #     "10": "clapping",
    #     "23": "hand waving",
    #     "26": "hopping (one foot jumping)",
    #     "27": "jump up",
    #     "35": "nod head/bow",
    #     "36": "shake head",
    #     "43": "falling",
    #     "56": "giving something to other person",
    #     "58": "handshaking",
    #     "59": "walking towards each other",
    #     "60": "walking apart from each other"
    # }
    ids = [7, 8, 9, 22, 25, 27, 34, 35, 42, 55, 57, 58, 59]
    sort_idx = np.argsort(-np.array(logits)).tolist()
    sort_idx = [i for i in sort_idx if i in ids]
    new_logits = [logits[i] for i in sort_idx]
    return sort_idx, new_logits


def get_parser() -> argparse.ArgumentParser:
    parser = get_default_parser()
    # parser.add_argument('--max-person', type=int, default=2)
    parser.add_argument('--max-frame', type=int, default=100)
    parser.add_argument('--max-num-skeleton-true', type=int, default=2)  # noqa
    parser.add_argument('--max-num-skeleton', type=int, default=4)  # noqa
    parser.add_argument('--num-joint', type=int, default=15)

    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--timing', type=bool, default=False)
    parser.add_argument('--interval', type=int, default=0)

    parser.add_argument(
        '--data-path',
        type=str,
        default='/data/openpose/skeleton/'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='/data/2s-agcn/model/ntu_15j/'
    )
    parser.add_argument(
        '--weight-path',
        type=str,
        # default='./data/model/ntu_25j/'
        # default='/data/2s-agcn/model/ntu_15j/xview/220405153001/'  # ntu15j sgn
        # default='/data/2s-agcn/model/ntu_15j/xview/220327213001_1337/'
        default='/data/2s-agcn/model/ntu_15j/xview/211130150001/'
        # default='/data/2s-agcn/model/ntu_15j/xview/220314100001/'
        # default='/data/2s-agcn/model/ntu_15j/xsub/220314090001/'
        # default='/data/2s-agcn/model/ntu_15j/xview/220405153001/'
    )
    parser.add_argument(
        '--out-folder',
        type=str,
        default='/data/2s-agcn/prediction/ntu_15j/'
    )
    return parser


def init_file_and_folders(arg: argparse.Namespace):
    # action id mapper
    with open(os.path.join(arg.model_path, 'index_to_name.json'), 'r') as f:
        MAPPING = {int(i): j for i, j in json.load(f).items()}
    # raw skeleton dir
    skel_fol = sorted(os.listdir(arg.data_path))[-1]
    skel_dir = os.path.join(arg.data_path, skel_fol)
    # action output dir
    output_dir = os.path.join(arg.out_folder, skel_fol)
    os.makedirs(output_dir, exist_ok=True)
    return MAPPING, skel_dir, output_dir


def init_preprocessor(arg: argparse.Namespace):
    if 'sgn' in arg.model:
        preprocess_fn = NTUDataLoaders(dataset='NTU60',
                                       case=0,
                                       aug=0,
                                       seg=20,
                                       multi_test=5).to_fix_length
        preprocess_fn = partial(preprocess_fn,
                                labels=None, sampling_frequency=5)
    else:
        preprocess_fn = partial(pre_normalization, zaxis2=[8, 1], xaxis=[2, 5],
                                verbose=False, tqdm=False)
    return DataPreprocessor(num_joint=arg.num_joint,
                            max_seq_length=arg.max_frame,
                            max_person=arg.max_num_skeleton,
                            moving_avg=5,
                            preprocess_fn=preprocess_fn)


def init_model(arg: argparse.Namespace):
    Model = import_class(arg.model)(**arg.model_args)
    Model.eval()
    weight_file = [i for i in os.listdir(arg.weight_path) if '.pt' in i]
    weight_file = os.path.join(arg.weight_path, weight_file[0])
    weights = torch.load(weight_file)
    # temporary hack
#    if 'sgn' in arg.model:
#        weights = OrderedDict([[k.replace('joint_', 'pos_'), v]
#                               for k, v in weights.items()])
#        weights = OrderedDict([[k.replace('dif_', 'vel_'), v]
#                               for k, v in weights.items()])
#        weights = OrderedDict([[k.replace('cnn.cnn2', 'cnn.cnn2.cnn'), v]
#                               for k, v in weights.items()])
    Model.load_state_dict(weights)
    if arg.gpu:
        Model = Model.cuda(0)
    print("Model loaded...")
    return Model


def model_inference(arg: argparse.Namespace,
                    model: torch.nn.Module,
                    input_data: np.ndarray
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():

        if arg.gpu:
            output, _ = model(torch.from_numpy(input_data).cuda(0))
            if 'sgn' in arg.model:
                output = output.view((-1, 5, output.size(1)))
                output = output.mean(1)
            output = F.softmax(output, 1)
            _, predict_label = torch.max(output, 1)
            output = output.data.cpu()
            predict_label = predict_label.data.cpu()

        else:
            output, _ = Model(torch.from_numpy(input_data))
            if 'sgn' in arg.model:
                output = output.view((-1, 5, output.size(1)))
                output = output.mean(1)
            output = F.softmax(output, 1)
            _, predict_label = torch.max(output, 1)

    return output, predict_label


if __name__ == '__main__':

    # init_seed(0)

    [args, _] = parse_arg(get_parser())

    # prepare file and folders -------------------------------------------------
    MAPPING, skel_dir, output_dir = init_file_and_folders(args)

    # Data processor -----------------------------------------------------------
    DataProc = init_preprocessor(args)

    # Prepare model ------------------------------------------------------------
    Model = init_model(args)

    # MAIN LOOP ----------------------------------------------------------------
    start = time.time()
    skel_path_mem = None
    infer_flag = False

    last_skel_file = None

    print("Start loop...")
    while True:

        # infer if
        # a. more than interval.
        # b. a valid skeleton is available.
        if time.time() - start <= int(args.interval):
            continue
        else:
            if infer_flag:
                start = time.time()
                infer_flag = False

        skel_files = sorted(os.listdir(skel_dir))[-args.max_frame:]
        if last_skel_file is not None:
            try:
                skel_files = skel_files[skel_files.index(last_skel_file)+1:]
            except ValueError:
                skel_files = skel_files[:]
        if len(skel_files) != 0:
            last_skel_file = skel_files[-1]

        infer_flag = True

        if args.timing:
            start_time = time.time()

        # 1. Read raw frames. --------------------------------------------------
        # M, T, V, C
        for skel_file in skel_files:
            data = read_xyz(os.path.join(skel_dir, skel_file),
                            max_body=args.max_num_skeleton,
                            num_joint=args.num_joint)
            # Batch frames to fixed length.
            DataProc.append_data(data[:2, :, :, :])

        # 2. Batch frames to fixed length. -------------------------------------
        # 3. Normalization. ----------------------------------------------------
        # 4. Inference. --------------------------------------------------------

        # Normalization.
        input_data = DataProc.select_skeletons_and_normalize_data(
            args.max_num_skeleton_true,
            sgn='sgn' in args.model
        )

        if 'aagcn' in args.model:
            # N, C, T, V, M
            input_data = np.concatenate(
                [input_data, input_data, input_data], axis=2)

        # Inference.
        output, predict_label = model_inference(args, Model, input_data)

        logits, preds = output.tolist(), predict_label.item()

        sort_idx, new_logits = filter_logits(logits)

        output_file = os.path.join(output_dir, skel_file)
        with open(output_file, 'a+') as f:
            output_str1 = ",".join([str(i) for i in preds])
            output_str2 = ",".join([str(i) for i in new_logits])
            output_str = f'{output_str1};{output_str2}\n'
            output_str = output_str.replace('[', '').replace(']', '')
            f.write(output_str)
            print(f"{sort_idx[0]: >2}, {new_logits[0]*100:>5.2f}")

        if args.timing:
            end_time = time.time() - start_time
            print(f"Processed : {skel_file} in {end_time:.4f}s")
        else:
            print(f"Processed : {skel_file}")
