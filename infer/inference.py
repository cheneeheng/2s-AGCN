import json
import numpy as np
import os
import time
import torch
import yaml

from collections import OrderedDict
from functools import partial

from data_gen.preprocess import pre_normalization
from infer.data_preprocess import DataPreprocessor
from main_utils import get_parser, import_class, init_seed


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
                # ntu
                # data[m, 0, j//3, :] = [body_joint[j],
                #                        body_joint[j+1],
                #                        body_joint[j+2]]
                # openpose
                data[m, 0, j//3, :] = [-body_joint[j],
                                       -body_joint[j+2],
                                       -body_joint[j+1]]
            else:
                pass
    # openpose
    data /= 1000.0
    return data  # M, T, V, C


if __name__ == '__main__':

    init_seed(0)

    parser = get_parser()
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
        default='/data/openpose/skeleton/')
    parser.add_argument(
        '--model-path',
        type=str,
        default='/data/2s-agcn/model/ntu_15j/')
    parser.add_argument(
        '--weight-path',
        type=str,
        # default='./data/model/ntu_25j/'
        default='/data/2s-agcn/model/ntu_15j/xview/211130150001/'
        # default='/data/2s-agcn/model/ntu_15j/xview/220314100001/'
        # default='/data/2s-agcn/model/ntu_15j/xsub/220314090001/'
    )
    parser.add_argument(
        '--out-folder',
        type=str,
        default='/data/2s-agcn/prediction/ntu_15j/')

    # load arg form config file ------------------------------------------------
    arg = arg_parser(parser)
    with open(os.path.join(arg.model_path, 'index_to_name.json'), 'r') as f:
        MAPPING = {int(i): j for i, j in json.load(f).items()}

    skel_fol = sorted(os.listdir(arg.data_path))[-1]
    skel_dir = os.path.join(arg.data_path, skel_fol)

    output_dir = os.path.join(arg.out_folder, skel_fol)
    os.makedirs(output_dir, exist_ok=True)

    # Data processor -----------------------------------------------------------
    preprocess_fn = partial(pre_normalization, zaxis2=[8, 1], xaxis=[2, 5],
                            verbose=False, tqdm=False)

    DataProc = DataPreprocessor(num_joint=arg.num_joint,
                                max_seq_length=arg.max_frame,
                                max_person=arg.max_num_skeleton,
                                moving_avg=5,
                                preprocess_fn=preprocess_fn)

    # Prepare model ------------------------------------------------------------
    Model = import_class(arg.model)(**arg.model_args)
    Model.eval()
    weight_file = [i for i in os.listdir(arg.weight_path) if '.pt' in i]
    weight_file = os.path.join(arg.weight_path, weight_file[0])
    weights = torch.load(weight_file)
    # temporary hack
    if 'sgn' in arg.model:
        weights = OrderedDict([[k.replace('joint_', 'pos_'), v]
                               for k, v in weights.items()])
        weights = OrderedDict([[k.replace('dif_', 'vel_'), v]
                               for k, v in weights.items()])
        weights = OrderedDict([[k.replace('cnn.cnn2', 'cnn.cnn2.cnn'), v]
                               for k, v in weights.items()])
    Model.load_state_dict(weights)
    if arg.gpu:
        Model = Model.cuda(0)
    print("Model loaded...")

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
        if time.time() - start <= int(arg.interval):
            continue
        else:
            if infer_flag:
                start = time.time()
                infer_flag = False

        skel_files = sorted(os.listdir(skel_dir))[-arg.max_frame:]
        if last_skel_file is not None:
            try:
                skel_files = skel_files[skel_files.index(last_skel_file)+1:]
            except ValueError:
                skel_files = skel_files[:]
        if len(skel_files) != 0:
            last_skel_file = skel_files[-1]

        infer_flag = True

        if arg.timing:
            start_time = time.time()

        # 1. Read raw frames. --------------------------------------------------
        # M, T, V, C
        for skel_file in skel_files:
            data = read_xyz(os.path.join(skel_dir, skel_file),
                            max_body=arg.max_num_skeleton,
                            num_joint=arg.num_joint)
            # Batch frames to fixed length.
            DataProc.append_data(data[:2, :, :, :])

        # 2. Batch frames to fixed length. -------------------------------------
        # 3. Normalization. ----------------------------------------------------
        # 4. Inference. --------------------------------------------------------

        # Normalization.
        input_data = DataProc.select_skeletons_and_normalize_data(
            arg.max_num_skeleton_true)
        input_data = np.transpose(input_data, [0, 4, 2, 3, 1])  # N, C, T, V, M
        input_data = np.concatenate(
            [input_data, input_data, input_data], axis=2)

        # Inference.
        with torch.no_grad():
            if next(Model.parameters()).is_cuda:
                output, _ = Model(torch.from_numpy(input_data).cuda(0))
                _, predict_label = torch.max(output, 1)
                output = output.data.cpu()
                predict_label = predict_label.data.cpu()

            else:
                output, _ = Model(torch.from_numpy(input_data))
                _, predict_label = torch.max(output, 1)

        logits, pred = output.tolist(), predict_label.item()

        output_file = os.path.join(output_dir, skel_file)
        with open(output_file, 'a+') as f:
            output_str = ",".join([str(logit) for logit in logits])
            output_str = f'{pred},{output_str}\n'
            f.write(output_str.replace('[', '').replace(']', ''))
            print(pred)

        if arg.timing:
            end_time = time.time() - start_time
            print(f"Processed : {skel_file} in {end_time:.4f}s")
        else:
            print(f"Processed : {skel_file}")
