# TAKEN FROM 2s-agcn infer/inference.py

import argparse
import json
import numpy as np
import os
import time

from functools import partial
from typing import Tuple

import torch
from torch.nn import functional as F

from data_gen.preprocess import pre_normalization
from feeders.loader import NTUDataLoaders
from infer.data_preprocess import DataPreprocessorV2
from utils.parser import get_parser as get_default_parser
from utils.parser import load_parser_args_from_config
from utils.utils import import_class
from utils.utils import init_seed


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


class ActionRecognition:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Data processor -------------------------------------------------------
        fn = NTUDataLoaders(dataset='NTU60',
                            seg=self.args.model_args['seg'],
                            multi_test=self.args.multi_test).to_fix_length
        self.sgn_preprocess_fn = partial(
            fn,
            labels=None,
            sampling_frequency=self.args.multi_test
        )
        self.aagcn_normalize_fn = partial(
            pre_normalization,
            zaxis=[8, 1],
            xaxis=[2, 5],
            verbose=False,
            tqdm=False
        )
        self.DataProc = DataPreprocessorV2(
            num_joint=self.args.num_joint,
            max_seq_length=self.args.max_frame,
            max_person=self.args.max_num_skeleton,
            moving_avg=self.args.moving_avg,
            aagcn_normalize_fn=self.aagcn_normalize_fn,
            sgn_preprocess_fn=self.sgn_preprocess_fn
        )
        # Prepare model --------------------------------------------------------
        self.Model = import_class(self.args.model)(**self.args.model_args)
        self.Model.eval()
        self.Model.load_state_dict(torch.load(self.args.weights))
        if self.args.gpu:
            self.Model = self.Model.cuda(0)
        # if int(torch.__version__.split('.')[0]) == 2:
        #     self.Model = torch.compile(self.Model)
        print("Model loaded...")

    def append_data(self, data: np.ndarray):
        # data  # M, 1, V, C
        assert data.shape[1] == 1
        self.DataProc.append_data(data)

    def predict(self):
        # Batch frames to fixed length. ----------------------------------------
        # Normalization. -------------------------------------------------------
        input_data = self.DataProc.select_skeletons_and_normalize_data(
            self.args.max_num_skeleton_true,
            aagcn_normalize=self.args.aagcn_normalize,
            sgn_preprocess=self.args.sgn_preprocess
        )
        # Inference. -----------------------------------------------------------
        with torch.no_grad():
            torch_input = torch.from_numpy(input_data)
            if self.args.gpu:
                torch_input = torch_input.cuda(0)
            output, _ = self.Model(torch_input)
            if 'sgn' in self.args.model:
                output = output.view(
                    (-1, self.args.multi_test, output.size(1)))
                output = output.mean(1)
            output = F.softmax(output, 1)
            _, predict_label = torch.max(output, 1)
            if self.args.gpu:
                output = output.data.cpu()
                predict_label = predict_label.data.cpu()
        logits, prediction = output.tolist(), predict_label.item()
        return logits[0], prediction


if __name__ == '__main__':

    init_seed(1)

    # Parse args ---------------------------------------------------------------
    # data_path = "data/data_tmp/S003C001P018R001A009_15j"
    # data_path = "data/data_tmp/S003C001P018R001A027_15j"
    # data_path = "data/data_tmp/S003C001P018R001A031_15j"
    # data_path = "data/data_tmp/S003C001P002R001A031_15j"
    data_path = "data/data_tmp/S003C001P018R001A043_15j"
    # data_path = "data/data_tmp/S003C001P018R001A056_15j"
    # label_mapping_file = "data/model/ntu_15j/index_to_name.json"
    label_mapping_file = "data/model/ntu_15j_9l/index_to_name.json"
    # weights = "data/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-116-68208.pt"  # noqa
    weights = "data/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-110-10670.pt"  # noqa
    # weights = "data/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-29400.pt"  # noqa
    # weights = "data/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-4850.pt"  # noqa
    # config = "data/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml"  # noqa
    config = "data/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml"  # noqa
    # config = "data/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/config.yaml"  # noqa
    # config = "data/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/config.yaml"  # noqa
    parser = get_default_parser()
    parser.set_defaults(**{'config': config})
    args = load_parser_args_from_config(parser)
    args.max_frame = 300
    args.max_num_skeleton_true = 2
    args.max_num_skeleton = 4
    args.num_joint = 15
    args.gpu = True
    args.timing = False
    args.interval = 0
    args.moving_avg = 1
    args.aagcn_normalize = True
    args.sgn_preprocess = True
    args.multi_test = 5
    args.out_folder = "data/data_tmp/delme"
    args.data_path = data_path
    args.weights = weights
    args.label_mapping_file = label_mapping_file

    # prepare file and folders -------------------------------------------------
    # action id mapper
    with open(args.label_mapping_file, 'r') as f:
        MAPPING = {int(i): j for i, j in json.load(f).items()}
    # raw skeleton dir
    skel_dir = args.data_path
    # action output dir
    output_dir = args.out_folder
    os.makedirs(output_dir, exist_ok=True)

    # Setup action recognition -------------------------------------------------
    AR = ActionRecognition(args)

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

            # data  # M, 1, V, C
            AR.append_data(data)

            # 2. Batch frames to fixed length. ---------------------------------
            # 3. Normalization. ------------------------------------------------
            # 4. Inference. ----------------------------------------------------
            logits, preds = AR.predict()

            if args.timing and idx > 10:
                end_time = time.time() - start_time
                start_time = time.time()
                print(f"{skel_file} , {end_time:.4f}s , "
                      f"{preds+1} , {logits[preds]:.2f} , "
                      f"{MAPPING[preds+1]}")
            else:
                print(f"{skel_file} , "
                      f"{preds+1} , {logits[preds]:.2f} , "
                      f"{MAPPING[preds+1]}")

        break
