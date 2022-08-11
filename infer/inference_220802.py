import argparse
import json
import numpy as np
import os
import time
import yaml
import pickle
import matplotlib.pyplot as plt

from PIL import Image
from functools import partial
from typing import Tuple

import torch
from torch.nn import functional as F

from data_gen.preprocess import pre_normalization
from feeders.loader import NTUDataLoaders
from infer.data_preprocess import DataPreprocessor
from utils.parser import get_parser as get_default_parser
from utils.utils import import_class, init_seed


# def filter_logits(logits: list) -> Tuple[list, list]:
#     # {
#     #     "8": "sitting down",
#     #     "9": "standing up (from sitting position)",
#     #     "10": "clapping",
#     #     "23": "hand waving",
#     #     "26": "hopping (one foot jumping)",
#     #     "27": "jump up",
#     #     "35": "nod head/bow",
#     #     "36": "shake head",
#     #     "43": "falling",
#     #     "56": "giving something to other person",
#     #     "58": "handshaking",
#     #     "59": "walking towards each other",
#     #     "60": "walking apart from each other"
#     # }
#     ids = [7, 8, 9, 22, 25, 27, 34, 35, 42, 55, 57, 58, 59]
#     sort_idx = np.argsort(-np.array(logits)).tolist()
#     sort_idx = [i for i in sort_idx if i in ids]
#     new_logits = [logits[i] for i in sort_idx]
#     return sort_idx, new_logits


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

    J15_MODEL_PATH = '/data/2s-agcn/model/ntu_15j'
    J25_MODEL_PATH = '/data/2s-agcn/model/ntu_25j'
    parser.add_argument(
        '--model-path',
        type=str,
        default='/code/2s-AGCN/data/model/ntu_25j'  # noqa
    )
    parser.add_argument(
        '--weight-path',
        type=str,
        # AAGCN
        # default=J15_MODEL_PATH + '/xview/211130150001'
        # default=J15_MODEL_PATH + '/xview/220314100001'
        # default=J15_MODEL_PATH + '/xsub/220314090001'
        # SGN
        # default=J15_MODEL_PATH + '/xview/220410210001'  # v5
        # default=J15_MODEL_PATH + '/xview/220405153001'  # v4
        # default=J15_MODEL_PATH + '/xview/220327213001_1337'  # v2
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220520150001_gcnfpn0_multit333_pregcntemsem'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220601220001_gcnfpn7_multit357357357shared'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220601220001_gcnfpn7_multit357'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220520150001_rerun_orisgn'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220804140001_sgnori_smpemb'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220810140001_sgnori_nogcnres'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220804140001_sgnori_tmode3'  # noqa
        default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220809140001_sgnori_tmode3_1layer'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v12/220714183001_bs128_sgd_lr1e1_steps90110'  # noqa
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
    skel_fol = [i for i in os.listdir(arg.data_path)
                if os.path.isdir(os.path.join(arg.data_path, i))][0]
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
    weight_path = os.path.join(arg.weight_path, 'weight')
    weight_file = [i for i in os.listdir(weight_path) if '.pt' in i]
    weight_file = os.path.join(weight_path, weight_file[-1])
    weights = torch.load(weight_file)
    # temporary hack
#    if 'sgn' in arg.model:
#        weights = OrderedDict([[k.replace('joint_', 'pos_'), v]
#                               for k, v in weights.items()])
#        weights = OrderedDict([[k.replace('dif_', 'vel_'), v]
#                               for k, v in weights.items()])
#        weights = OrderedDict([[k.replace('cnn.cnn2', 'cnn.cnn2.cnn'), v]
#                               for k, v in weights.items()])
    weights['sgcn.gcn_g1.alpha'] = torch.ones(1)
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
            output = model(torch.from_numpy(input_data).cuda(0))
            output, output_dict = output[0], output[1]
            if 'sgn' in arg.model:
                output = output.view((-1, 5, output.size(1)))
                output = output.mean(1)
            output = F.softmax(output, 1)
            _, predict_label = torch.max(output, 1)
            output = output.data.cpu()
            predict_label = predict_label.data.cpu()
        else:
            output = Model(torch.from_numpy(input_data))
            output, output_dict = output[0], output[1]
            if 'sgn' in arg.model:
                output = output.view((-1, 5, output.size(1)))
                output = output.mean(1)
            output = F.softmax(output, 1)
            _, predict_label = torch.max(output, 1)
    return output, predict_label, output_dict


if __name__ == '__main__':

    # init_seed(0)

    [args, _] = parse_arg(get_parser())
    args.data_path = '/code/2s-AGCN/data/data/nturgbd_raw'
    args.out_folder = '/code/2s-AGCN/output/test_inference'

    init_seed(args.seed)

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

    with open('/code/2s-AGCN/data/data/ntu_sgn/processed_data/NTU_CV_test_180.pkl', 'rb') as f:  # noqa
        data1 = pickle.load(f)
    with open('/code/2s-AGCN/data/data/ntu_sgn/processed_data/NTU_CV_test_label_180.pkl', 'rb') as f:  # noqa
        data2 = pickle.load(f)

    fig1, axes1 = plt.subplots(5, 1)
    fig1.tight_layout()
    fig2, axes2 = plt.subplots(5, 1)
    fig2.tight_layout()
    fig3, axes3 = plt.subplots(5, 1)
    fig3.tight_layout()
    fig4, axes4 = plt.subplots(5, 18)
    fig4.tight_layout()

    freq = 2
    c = 60

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

        infer_flag = True

        if args.timing:
            start_time = time.time()

        # M, T, V, C
        # n,c,t,v,m
        input_data = data1[c].reshape(1, 300, 2, 25, 3)
        input_data = input_data.reshape((1, 300, -1))
        data, _, _, _ = DataProc.preprocess_fn(input_data)  # N,'T, MVC
        input_data = np.array(data, dtype=data[0].dtype)  # N, 'T, MVC

        # Inference.
        output, predict_label, output_dict = \
            model_inference(args, Model, input_data)
        g_spa = output_dict['g_spa']
        tem_a = output_dict['attn_tem_list']
        x_spa_list = output_dict['x_spa_list']
        featuremap_spa_list = output_dict['featuremap_spa_list']

        logits, preds = output[0].tolist(), predict_label.item()

        print(f"Label : {data2[c]:3d} , Pred : {preds:3d} , Logit : {logits[preds]*100:>5.2f}")  # noqa

        # PLOTTING -------------------------------------------------------------
        fig1.suptitle(MAPPING[preds+1] + " : " + MAPPING[data2[c]+1])
        fig4.suptitle(MAPPING[preds+1] + " : " + MAPPING[data2[c]+1])

        # G spatial --------------------------------------------------
        img = []
        for j in range(5):
            img_j = []
            for i in range(0, 20, freq):
                img_i = g_spa[0][j][i].data.cpu().numpy()
                img_j.append(img_i)
            img.append(np.concatenate(img_j, axis=1))
            axes1[j].imshow(img[-1])
            axes1[j].xaxis.set_ticks(np.arange(0, 25*(20//freq), 25))
            axes1[j].yaxis.set_ticks(np.arange(0, 25, 5))

        # A in temporal branch for t_mode=3 ---------------------------
        if tem_a[0] is not None:
            img = []
            for j in range(5):
                img_j = []
                for i in range(len(tem_a[0])):
                    # last 0 cause spa_maxpool
                    img_i = tem_a[0][i][j][0].data.cpu().numpy()
                    img_j.append(img_i)
                img.append(np.concatenate(img_j, axis=1))
                axes2[j].imshow(img[-1])
                axes2[j].xaxis.set_ticks(np.arange(0, 20*len(tem_a[0]), 20))
                axes2[j].yaxis.set_ticks(np.arange(0, 20, 5))

        # featuremap after each SGCN ----------------------------------
        featuremap = np.linalg.norm(x_spa_list[-1], axis=1)  # N,V,T
        img = []
        for j in range(5):
            img_j = featuremap[j]
            axes3[j].imshow(img_j)
            axes3[j].xaxis.set_ticks(np.arange(0, 20+1, 5))
            axes3[j].yaxis.set_ticks(np.arange(0, 25+1, 5))

        # featuremaps (l2 in channel dimension) in each SGCN -----------
        def vmin_vmax_sgcn_fm(j):
            # input x
            b1 = np.linalg.norm(featuremap_spa_list[j]['x'], axis=1)
            # G
            b2 = np.linalg.norm(g_spa[0], axis=1)  # in T dimension
            # G @ x
            b3 = np.linalg.norm(featuremap_spa_list[j]['x3'], axis=1)
            # W1 * (G @ x)
            b4 = np.linalg.norm(featuremap_spa_list[j]['x4'], axis=1)
            # W2 * x
            if isinstance(featuremap_spa_list[j]['x5'], torch.Tensor):
                b5 = np.linalg.norm(featuremap_spa_list[j]['x5'], axis=1)
            # output x
            b6 = np.linalg.norm(featuremap_spa_list[j]['x9'], axis=1)

            if isinstance(featuremap_spa_list[j]['x5'], torch.Tensor):
                vmin = min([b1.min(), b3.min(), b4.min(), b5.min(), b6.min()])  # noqa
                vmax = max([b1.max(), b3.max(), b4.max(), b5.max(), b6.max()])  # noqa
            else:
                vmin = min([b1.min(), b3.min(), b4.min(), b6.min()])
                vmax = max([b1.max(), b3.max(), b4.max(), b6.max()])

            return vmin, vmax

        vmin1, vmax1 = vmin_vmax_sgcn_fm(0)
        vmin2, vmax2 = vmin_vmax_sgcn_fm(1)
        vmin3, vmax3 = vmin_vmax_sgcn_fm(2)
        vmin = min([vmin1, vmin2, vmin3])
        vmax = max([vmax1, vmax2, vmax3])

        def plot_sgcn_fm(j, ax):
            # input x
            b1 = np.linalg.norm(featuremap_spa_list[j]['x'], axis=1)
            # G
            b2 = np.linalg.norm(g_spa[0], axis=1)  # in T dimension
            # G @ x
            b3 = np.linalg.norm(featuremap_spa_list[j]['x3'], axis=1)
            # W1 * (G @ x)
            b4 = np.linalg.norm(featuremap_spa_list[j]['x4'], axis=1)
            # W2 * x
            if isinstance(featuremap_spa_list[j]['x5'], torch.Tensor):
                b5 = np.linalg.norm(featuremap_spa_list[j]['x5'], axis=1)
            # output x
            b6 = np.linalg.norm(featuremap_spa_list[j]['x9'], axis=1)

            for i in range(5):
                ax[i, j*6+0].imshow(b1[i], vmin=vmin, vmax=vmax)
            for i in range(5):
                ax[i, j*6+1].imshow(b2[i])
            for i in range(5):
                ax[i, j*6+2].imshow(b3[i], vmin=vmin, vmax=vmax)
            for i in range(5):
                ax[i, j*6+3].imshow(b4[i], vmin=vmin, vmax=vmax)
            # W2 * x
            if isinstance(featuremap_spa_list[j]['x5'], torch.Tensor):
                for i in range(5):
                    ax[i, j*6+4].imshow(b5[i], vmin=vmin, vmax=vmax)
            for i in range(5):
                ax[i, j*6+5].imshow(b6[i], vmin=vmin, vmax=vmax)

        plot_sgcn_fm(0, axes4)
        plot_sgcn_fm(1, axes4)
        plot_sgcn_fm(2, axes4)

        plt.show(block=False)

        # out_dir = f"output/{args.weight_path.split('/')[-2]}/{args.weight_path.split('/')[-1]}"  # noqa
        # os.makedirs(out_dir, exist_ok=True)
        # fig1.savefig(f"{out_dir}/{data2[c]:03d}_fig1")  # noqa
        # fig2.savefig(f"{out_dir}/{data2[c]:03d}_fig2")  # noqa
        # fig3.savefig(f"{out_dir}/{data2[c]:03d}_fig3")  # noqa
        # fig4.savefig(f"{out_dir}/{data2[c]:03d}_fig4")  # noqa
        # fig5.savefig(f"{out_dir}/{data2[c]:03d}_fig5")  # noqa
        # fig6.savefig(f"{out_dir}/{data2[c]:03d}_fig6")  # noqa

        c += 1