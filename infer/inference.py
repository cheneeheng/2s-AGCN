import json
import numpy as np
import os
import time
import yaml
import random

from functools import partial
from typing import Tuple

import torch
from torch.nn import functional as F

from data_gen.preprocess import pre_normalization
from feeders.loader import NTUDataLoaders
from infer.data_preprocess import DataPreprocessorV2
from utils.parser import get_parser as get_default_parser
from utils.utils import import_class


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


def init_seed(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


if __name__ == '__main__':

    init_seed(1)

    # Parse args ---------------------------------------------------------------
    parser = get_default_parser()
    parser.add_argument('--max-frame', type=int, default=300)
    parser.add_argument('--max-num-skeleton-true', type=int, default=2)  # noqa
    parser.add_argument('--max-num-skeleton', type=int, default=4)  # noqa
    parser.add_argument('--num-joint', type=int, default=15)

    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--timing', type=bool, default=False)
    parser.add_argument('--interval', type=int, default=0)
    parser.add_argument('--moving-avg', type=int, default=1)

    parser.add_argument('--aagcn-normalize', type=bool, default=True)
    parser.add_argument('--sgn-preprocess', type=bool, default=True)
    parser.add_argument('--multi-test', type=int, default=5)

    parser.add_argument(
        '--data-path',
        type=str,
        # default='data/data_tmp/S003C001P018R001A009_15j'
        # default='data/data_tmp/S003C001P018R001A027_15j'
        # default='data/data_tmp/S003C001P018R001A031_15j'
        # default='data/data_tmp/S003C001P002R001A031_15j'
        default='data/data_tmp/S003C001P018R001A043_15j'
        # default='data/data_tmp/S003C001P018R001A056_15j'
    )
    parser.add_argument(
        '--label-mapping-file',
        type=str,
        # default='data/model/ntu_15j/index_to_name.json'
        default='data/model/ntu_15j_9l/index_to_name.json'
    )
    parser.add_argument(
        '--weight-file',
        type=str,
        # default='data/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-116-68208.pt'  # noqa
        default='data/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-110-10670.pt'  # noqa
        # default='data/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-29400.pt'  # noqa
        # default='data/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-4850.pt'  # noqa
    )
    parser.add_argument(
        '--config-file',
        type=str,
        # default='data/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml'  # noqa
        default='data/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml'  # noqa
        # default='data/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/config.yaml'  # noqa
        # default='data/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/config.yaml'  # noqa
    )
    parser.add_argument(
        '--out-folder',
        type=str,
        # default='data/data_tmp/inference_predictions_aagcn_preprocess_sgn_model'  # noqa
        # default='data/data_tmp/inference_predictions_aagcn_joint'
        # default='data/data_tmp/inference_predictions_aagcn_preprocess_sgn_model_9l'  # noqa
        # default='data/data_tmp/inference_predictions_aagcn_joint_9l'
        default='data/data_tmp/delme'
    )

    # PARSE CONFIGS FROM YAML
    [p, _] = parser.parse_known_args()
    with open(p.config_file, 'r') as f:
        default_arg = yaml.safe_load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print(f'WRONG ARG: {k}')
            assert (k in key)
    parser.set_defaults(**default_arg)

    [args, _] = parser.parse_known_args()

    # prepare file and folders -------------------------------------------------
    # action id mapper
    with open(args.label_mapping_file, 'r') as f:
        MAPPING = {int(i): j for i, j in json.load(f).items()}
    # raw skeleton dir
    skel_dir = args.data_path
    # action output dir
    output_dir = args.out_folder
    os.makedirs(output_dir, exist_ok=True)

    # Data processor -----------------------------------------------------------
    fn = NTUDataLoaders(dataset='NTU60',
                        seg=20,
                        multi_test=args.multi_test).to_fix_length
    sgn_preprocess_fn = partial(fn,
                                labels=None,
                                sampling_frequency=args.multi_test)
    aagcn_normalize_fn = partial(pre_normalization,
                                 zaxis=[8, 1],
                                 xaxis=[2, 5],
                                 verbose=False,
                                 tqdm=False)
    DataProc = DataPreprocessorV2(num_joint=args.num_joint,
                                  max_seq_length=args.max_frame,
                                  max_person=args.max_num_skeleton,
                                  moving_avg=args.moving_avg,
                                  aagcn_normalize_fn=aagcn_normalize_fn,
                                  sgn_preprocess_fn=sgn_preprocess_fn)

    # Prepare model ------------------------------------------------------------
    Model = import_class(args.model)(**args.model_args)
    Model.eval()
    Model.load_state_dict(torch.load(args.weight_file))
    if args.gpu:
        Model = Model.cuda(0)
    if int(torch.__version__.split('.')[0]) == 2:
        Model = torch.compile(Model)
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
        for idx, skel_file in enumerate(skel_files):
            skel_data = np.loadtxt(os.path.join(skel_dir, skel_file),
                                   delimiter=',')
            data = np.zeros((args.max_num_skeleton, 1, args.num_joint, 3))
            for m, body_joint in enumerate(skel_data):
                for j in range(0, len(body_joint), 3):
                    if m < args.max_num_skeleton and j//3 < args.num_joint:
                        # x subject right, y to camera, z up
                        data[m, 0, j//3, :] = [body_joint[j],
                                               body_joint[j+1],
                                               body_joint[j+2]]
                    else:
                        pass
            # data  # M, T, V, C
            DataProc.append_data(data[:args.max_num_skeleton_true, :, :, :])

            # 2. Batch frames to fixed length. ---------------------------------
            # 3. Normalization. ------------------------------------------------
            input_data = DataProc.select_skeletons_and_normalize_data(
                args.max_num_skeleton_true,
                aagcn_normalize=args.aagcn_normalize,
                sgn_preprocess=args.sgn_preprocess
            )

            # 4. Inference. ----------------------------------------------------
            with torch.no_grad():
                torch_input = torch.from_numpy(input_data)
                if args.gpu:
                    torch_input = torch_input.cuda(0)
                output, _ = Model(torch_input)
                if 'sgn' in args.model:
                    output = output.view((-1, args.multi_test, output.size(1)))
                    output = output.mean(1)
                output = F.softmax(output, 1)
                _, predict_label = torch.max(output, 1)
                if args.gpu:
                    output = output.data.cpu()
                    predict_label = predict_label.data.cpu()

            logits, preds = output.tolist(), predict_label.item()

            if args.timing and idx > 10:
                end_time = time.time() - start_time
                start_time = time.time()
                print(f"{skel_file} , {end_time:.4f}s , "
                      f"{preds+1} , {logits[0][preds]:.2f} , "
                      f"{MAPPING[preds+1]}")
            else:
                print(f"{skel_file} , "
                      f"{preds+1} , {logits[0][preds]:.2f} , "
                      f"{MAPPING[preds+1]}")

        break
