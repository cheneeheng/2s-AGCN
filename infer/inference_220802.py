import argparse
import json
import numpy as np
import os
import time
import yaml
import pickle

from PIL import Image
from functools import partial
from typing import Tuple, List, Sequence

import torch
from torch.nn import functional as F

from data_gen.preprocess import pre_normalization
from feeders.loader import NTUDataLoaders
from infer.data_preprocess import DataPreprocessor
from utils.parser import get_parser as get_default_parser
from utils.utils import import_class, init_seed

from utils.loss import CosineLoss

import matplotlib
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# https://stackoverflow.com/questions/67278053

# NTU60
rightarm = np.array([24, 12, 11, 10, 9, 21]) - 1
leftarm = np.array([22, 8, 7, 6, 5, 21]) - 1
righthand = np.array([25, 12]) - 1
lefthand = np.array([23, 8]) - 1
rightleg = np.array([19, 18, 17, 1]) - 1
leftleg = np.array([15, 14, 13, 1]) - 1
rightfeet = np.array([20, 19]) - 1
leftfeet = np.array([16, 15]) - 1
body = np.array([4, 3, 21, 2, 1]) - 1  # body


def get_chains(dots: np.ndarray,   # shape == (n_dots, 3)
               ):
    return (dots[rightarm.tolist()],
            dots[leftarm.tolist()],
            dots[righthand.tolist()],
            dots[lefthand.tolist()],
            dots[rightleg.tolist()],
            dots[leftleg.tolist()],
            dots[rightfeet.tolist()],
            dots[leftfeet.tolist()],
            dots[body.tolist()])


def subplot_nodes(dots: np.ndarray, ax):
    return ax.scatter3D(*dots.T, s=1, c=dots[:, -1])


def subplot_bones(chains: Tuple[np.ndarray, ...], ax):
    return [ax.plot(*chain.T) for chain in chains]


def plot_skeletons(skeletons: Sequence[np.ndarray], fig):
    # fig = plt.figure()
    for i, dots in enumerate(skeletons, start=1):
        chains = get_chains(dots)
        ax = fig.add_subplot(5, 20, i, projection='3d')
        subplot_nodes(dots, ax)
        subplot_bones(chains, ax)
    # plt.show()


# def test():
#     """Plot random poses of simplest skeleton"""
#     skeletons = np.random.standard_normal(size=(10, 11, 3))
#     chains_ixs = ([0, 1, 2, 3, 4],  # hand_l, elbow_l, chest, elbow_r, hand_r
#                   [5, 2, 6],        # pelvis, chest, head
#                   [7, 8, 5, 9, 10])  # foot_l, knee_l, pelvis, knee_r, foot_r
#     plot_skeletons(skeletons, chains_ixs)

# ------------------------------------------------------------------------------

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
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220810140001_sgnori_sgcnattn1'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220810140001_sgnori_sgcnattn2'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220810140001_sgnori_sgcnattn3'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220812150001_sgnori_sgcnattn10'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220816140001_sgnori_inpos11_invel11'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220819140001_sgnori_inch2'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220804140001_sgnori_tmode3'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220809140001_sgnori_tmode3_1layer'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v11/220901130001_sgnori_flgamma1'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v12/220714183001_bs128_sgd_lr1e1_steps90110'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220824167001'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220824167001_tmode3'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220825170001_alpha05'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220825170001_gt2_alpha01'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220826113001_gt2_alpha05_sigmoid'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220829113001_gt2_varalpha_sigmoid'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220831100001_gt2_fpn10_sigmoid'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220831170001_gt2_fpn10_sigmoid_allvaralpha'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220902150001_gt4_varalpha_sigmoid'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220902150001_gt4_varalpha_sigmoid_flgamma1'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220915120001_gt4_varalpha_sigmoid_multit357'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220906163001_gt5_varalpha'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220906163001_gt5_varalpha_allactnorm'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220908210001_gt6_varalpha_fsim1alpha1'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220908210001_gt6_varalpha_fsim2alpha1'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220909150001_gt6_varalpha_fsim1alpha1'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220909150001_gt6_varalpha_fsim2alpha1'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220920150001_tmode4_k7'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220920140001_gt4_varalpha_tmode4'  # noqa
        default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v13/220922140001_gt4_varalpha_sigmoid_tmode4_k3'  # noqa
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
                    input_data: np.ndarray,
                    sampling_freq: int = 5
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if arg.gpu:
            output = model(torch.from_numpy(input_data).cuda(0))
            output, output_dict = output[0], output[1]
            if 'sgn' in arg.model:
                output = output.view((-1, sampling_freq, output.size(1)))
                output = output.mean(1)
            output = F.softmax(output, 1)
            _, predict_label = torch.max(output, 1)
            output = output.data.cpu()
            predict_label = predict_label.data.cpu()
        else:
            output = Model(torch.from_numpy(input_data))
            output, output_dict = output[0], output[1]
            if 'sgn' in arg.model:
                output = output.view((-1, sampling_freq, output.size(1)))
                output = output.mean(1)
            output = F.softmax(output, 1)
            _, predict_label = torch.max(output, 1)
    return output, predict_label, output_dict


if __name__ == '__main__':

    init_seed(1337)

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

    _data_dir = '/code/2s-AGCN/data/data/ntu_sgn/processed_data'
    with open(_data_dir + '/NTU_CV_test_180.pkl', 'rb') as f:
        data1 = pickle.load(f)
    with open(_data_dir + '/NTU_CV_test_label_180.pkl', 'rb') as f:
        data2 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_test.pkl', 'rb') as f:
    #     data1 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_test_label.pkl', 'rb') as f:
    #     data2 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_train_180.pkl', 'rb') as f:
    #     data1 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_train_label_180.pkl', 'rb') as f:
    #     data2 = pickle.load(f)

    # with open(_data_dir + '/NTU_CV_train_180.pkl', 'wb') as f:
    #     pickle.dump(data1[:180, :, :], f)
    # with open(_data_dir + '/NTU_CV_train_label_180.pkl', 'wb') as f:
    #     pickle.dump(data2[:180], f)

    enable = {i: False for i in range(0, 10, 1)}

    # skeleton -----------
    fig0 = []
    for _ in range(1):
        fig00 = plt.figure(figsize=(16, 6))
        fig00.tight_layout()
        fig0.append(fig00)
    enable[0] = True
    # G spatial -----------
    fig1, axes1 = plt.subplots(5, 1, figsize=(16, 6))
    fig1.tight_layout()
    enable[1] = True
    # # A in temporal branch for t_mode=3 -----------
    # fig2, axes2 = plt.subplots(5, 1, figsize=(7, 7))
    # fig2.tight_layout()
    # enable[2] = True
    # featuremap after each SGCN -----------
    fig3, axes3 = plt.subplots(5, 1, figsize=(3, 7))
    fig3.tight_layout()
    enable[3] = True
    # # featuremaps (l2 in channel dimension) in each SGCN -----------
    # fig4, axes4 = plt.subplots(5, 18, figsize=(16, 7))
    # fig4.tight_layout()
    # enable[4] = True
    # GT spatial -----------
    # fig5, axes5 = plt.subplots(5, 4, figsize=(3, 7))
    # fig5.tight_layout()
    # enable[5] = True
    # # featuremap after each SGCN2 -----------
    # fig6, axes6 = plt.subplots(5, 1, figsize=(3, 7))
    # fig6.tight_layout()
    # enable[6] = True
    # # featuremaps (l2 in channel dimension) in each SGCN2 -----------
    # fig7, axes7 = plt.subplots(5, 18, figsize=(16, 7))
    # fig7.tight_layout()
    # enable[7] = True
    # # input data distribution -----------
    # fig8, axes8 = plt.subplots(1, 3, figsize=(6, 3))
    # fig8.tight_layout()
    # enable[8] = True
    # x tem list -----------
    fig9, axes9 = plt.subplots(5, 4, figsize=(3, 7))
    fig9.tight_layout()
    enable[9] = True

    freq = 1
    SAMP_FREQ = 5

    print("Start loop...")
    for c in range(0, 180, 1):
        # for c in [10, 11, 70, 71, 130, 131]:

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

        # for SAMP_FREQ in range(1, 100):

        # M, T, V, C
        # n,c,t,v,m
        input_data = data1[c].reshape(1, 300, 2, 25, 3)
        input_data = input_data.reshape((1, 300, -1))
        data, _, _, _ = DataProc.preprocess_fn(
            input_data, sampling_frequency=SAMP_FREQ)  # N,'T, MVC
        input_data = np.array(data, dtype=data[0].dtype)  # N, 'T, MVC

        # Inference.
        output, predict_label, output_dict = \
            model_inference(args, Model, input_data,
                            sampling_freq=SAMP_FREQ)
        g_spa = output_dict['g_spa']
        tem_a = output_dict['attn_tem_list']
        x_spa_list = output_dict['x_spa_list']
        featuremap_spa_list = output_dict['featuremap_spa_list']
        x_spa_list2 = output_dict.get('x_spa_list2')
        featuremap_spa_list2 = output_dict.get('featuremap_spa_list2')
        x_tem_list = output_dict.get('x_tem_list')

        logits, preds = output[0].tolist(), predict_label.item()

        # if (logits[preds]*100 < 50) or data2[c] != preds:
        print(f"Label : {data2[c]:3d} , Pred : {preds:3d} , Logit : {logits[preds]*100:>5.2f}, SAMP_FREQ : {SAMP_FREQ}, {c}")  # noqa

        if data2[c] == preds:
            # if data2[c] != 53:
            continue

        # PLOTTING ----------------------------------------------------------

        if enable[0]:
            d = input_data.reshape((-1, 25, 3))
            # d = input_data[0].reshape((-1, 25, 3))
            for fig00 in fig0:
                fig00.clear()
            dd = np.stack([d[:, :, 2], d[:, :, 0], d[:, :, 1]], axis=-1)
            plot_skeletons(dd, fig0[0])
            # dd = np.stack([d[:, :, 2], d[:, :, 1], d[:, :, 0]], axis=-1)
            # plot_skeletons(dd, fig0[1])
            # dd = np.stack([d[:, :, 0], d[:, :, 1], d[:, :, 2]], axis=-1)
            # plot_skeletons(dd, fig0[0])
            # dd = np.stack([d[:, :, 0], d[:, :, 2], d[:, :, 1]], axis=-1)
            # plot_skeletons(dd, fig0[1])
            # dd = np.stack([d[:, :, 1], d[:, :, 0], d[:, :, 2]], axis=-1)
            # plot_skeletons(dd, fig0[2])
            # dd = np.stack([d[:, :, 2], d[:, :, 0], d[:, :, 1]], axis=-1)
            # plot_skeletons(dd, fig0[3])
            # dd = np.stack([d[:, :, 1], d[:, :, 2], d[:, :, 0]], axis=-1)
            # plot_skeletons(dd, fig0[4])
            # dd = np.stack([d[:, :, 2], d[:, :, 1], d[:, :, 0]], axis=-1)
            # plot_skeletons(dd, fig0[5])

        # G spatial --------------------------------------------------
        if enable[1]:
            fig1.suptitle(MAPPING[preds+1] + " : " + MAPPING[data2[c]+1] +
                          f" : {logits[preds]*100:>5.2f}")
            fig1.subplots_adjust(top=0.9)
            img = []
            for j in range(SAMP_FREQ):
                img_j = []
                for i in range(0, 20, freq):
                    if isinstance(g_spa[0], tuple):
                        img_i = g_spa[0][0][j][i].data.cpu().numpy()
                    else:
                        img_i = g_spa[0][j][i].data.cpu().numpy()
                    img_j.append(img_i)
                img.append(np.concatenate(img_j, axis=1))
                axes1[j].imshow(img[-1])
                axes1[j].xaxis.set_ticks(np.arange(0, 25*(20//freq), 25))
                axes1[j].yaxis.set_ticks(np.arange(0, 25, SAMP_FREQ))

        # A in temporal branch for t_mode=3 ---------------------------
        if enable[2]:
            if tem_a[0] is not None:
                img = []
                for j in range(SAMP_FREQ):
                    img_j = []
                    for i in range(len(tem_a[0])):
                        # last 0 cause spa_maxpool
                        img_i = tem_a[0][i][j][0].data.cpu().numpy()
                        img_j.append(img_i)
                    img.append(np.concatenate(img_j, axis=1))
                    axes2[j].imshow(img[-1])
                    axes2[j].xaxis.set_ticks(
                        np.arange(0, 20*len(tem_a[0]), 20))
                    axes2[j].yaxis.set_ticks(np.arange(0, 20, SAMP_FREQ))

        # featuremap after each SGCN ----------------------------------
        if enable[3]:
            featuremap = np.linalg.norm(x_spa_list[-1], axis=1)  # N,V,T
            img = []
            for j in range(SAMP_FREQ):
                img_j = featuremap[j]
                axes3[j].imshow(img_j)
                axes3[j].xaxis.set_ticks(np.arange(0, 20+1, SAMP_FREQ))
                axes3[j].yaxis.set_ticks(np.arange(0, 25+1, SAMP_FREQ))

        if enable[6]:
            if x_spa_list2 is not None:
                featuremap = np.linalg.norm(x_spa_list2[-1], axis=1)  # N,V,T
                img = []
                for j in range(SAMP_FREQ):
                    img_j = featuremap[j]
                    axes6[j].imshow(img_j)
                    axes6[j].xaxis.set_ticks(np.arange(0, 20+1, SAMP_FREQ))
                    axes6[j].yaxis.set_ticks(np.arange(0, 25+1, SAMP_FREQ))

        # featuremaps (l2 in channel dimension) in each SGCN -----------
        # def vmin_vmax_sgcn_fm(j):
        #     # input x
        #     b1 = np.linalg.norm(featuremap_spa_list[j]['x'], axis=1)
        #     vmin, vmax = b1.min(), b1.max()
        #     # G
        #     if isinstance(g_spa[0], tuple):
        #         b2 = np.linalg.norm(g_spa[0][0], axis=1)  # in T dimension
        #     else:
        #         b2 = np.linalg.norm(g_spa[0], axis=1)  # in T dimension
        #     vmin, vmax = min(vmin, b2.min()), max(vmax, b2.max())
        #     # G @ x
        #     if featuremap_spa_list[j]['x3'] is not None:
        #         b3 = np.linalg.norm(featuremap_spa_list[j]['x3'], axis=1)
        #         vmin, vmax = min(vmin, b3.min()), max(vmax, b3.max())
        #     # W1 * (G @ x)
        #     if featuremap_spa_list[j]['x4'] is not None:
        #         b4 = np.linalg.norm(featuremap_spa_list[j]['x4'], axis=1)
        #         vmin, vmax = min(vmin, b4.min()), max(vmax, b4.max())
        #     # W2 * x
        #     if isinstance(featuremap_spa_list[j]['x5'], torch.Tensor):
        #         b5 = np.linalg.norm(featuremap_spa_list[j]['x5'], axis=1)
        #         vmin, vmax = min(vmin, b5.min()), max(vmax, b5.max())
        #     # output x
        #     b6 = np.linalg.norm(featuremap_spa_list[j]['x9'], axis=1)
        #     vmin, vmax = min(vmin, b6.min()), max(vmax, b6.max())

        #     return vmin, vmax

        # vmin1, vmax1 = vmin_vmax_sgcn_fm(0)
        # vmin2, vmax2 = vmin_vmax_sgcn_fm(1)
        # vmin3, vmax3 = vmin_vmax_sgcn_fm(2)
        # vmin = min([vmin1, vmin2, vmin3])
        # vmax = max([vmax1, vmax2, vmax3])
        vmin = None
        vmax = None

        def plot_sgcn_fm(j, ax, fm_list):
            # input x
            b1 = np.linalg.norm(fm_list[j]['x'], axis=1)
            # G
            if isinstance(g_spa[0], tuple):
                b2 = np.linalg.norm(g_spa[0][0], axis=1)  # in T dimension
            else:
                b2 = np.linalg.norm(g_spa[0], axis=1)  # in T dimension
            # G @ x
            if fm_list[j]['x3'] is not None:
                b3 = np.linalg.norm(fm_list[j]['x3'], axis=1)
            # W1 * (G @ x)
            if fm_list[j]['x4'] is not None:
                b4 = np.linalg.norm(fm_list[j]['x4'], axis=1)
            # W2 * x
            if isinstance(fm_list[j]['x5'], torch.Tensor):
                b5 = np.linalg.norm(fm_list[j]['x5'], axis=1)
            # output x
            b6 = np.linalg.norm(fm_list[j]['x9'], axis=1)

            for i in range(SAMP_FREQ):
                ax[i, j*6+0].imshow(b1[i], vmin=vmin, vmax=vmax)
            for i in range(SAMP_FREQ):
                ax[i, j*6+1].imshow(b2[i])
            if fm_list[j]['x3'] is not None:
                for i in range(SAMP_FREQ):
                    ax[i, j*6+2].imshow(b3[i], vmin=vmin, vmax=vmax)
            if fm_list[j]['x4'] is not None:
                for i in range(SAMP_FREQ):
                    ax[i, j*6+3].imshow(b4[i], vmin=vmin, vmax=vmax)
            # W2 * x
            if isinstance(fm_list[j]['x5'], torch.Tensor):
                for i in range(SAMP_FREQ):
                    ax[i, j*6+4].imshow(b5[i], vmin=vmin, vmax=vmax)
            for i in range(SAMP_FREQ):
                ax[i, j*6+5].imshow(b6[i], vmin=vmin, vmax=vmax)

        if enable[4]:
            fig4.suptitle(MAPPING[preds+1] + " : " + MAPPING[data2[c]+1])
            fig4.subplots_adjust(top=0.9)
            plot_sgcn_fm(0, axes4, featuremap_spa_list)
            plot_sgcn_fm(1, axes4, featuremap_spa_list)
            plot_sgcn_fm(2, axes4, featuremap_spa_list)
            for i in range(3):
                axes4[0, 6*i+0].set_title("x")
                axes4[0, 6*i+1].set_title("g")
                axes4[0, 6*i+2].set_title("g @ x")
                axes4[0, 6*i+3].set_title("g @ x @ W1")
                axes4[0, 6*i+4].set_title("x @ W2")
                axes4[0, 6*i+5].set_title("x out")

        if enable[7]:
            if featuremap_spa_list2 is not None:
                fig7.suptitle(MAPPING[preds+1] + " : " + MAPPING[data2[c]+1])
                plot_sgcn_fm(0, axes7, featuremap_spa_list2)
                plot_sgcn_fm(1, axes7, featuremap_spa_list2)
                plot_sgcn_fm(2, axes7, featuremap_spa_list2)
                for i in range(3):
                    axes7[0, 6*i+0].set_title("x")
                    axes7[0, 6*i+1].set_title("g")
                    axes7[0, 6*i+2].set_title("g @ x")
                    axes7[0, 6*i+3].set_title("g @ x @ W1")
                    axes7[0, 6*i+4].set_title("x @ W2")
                    axes7[0, 6*i+5].set_title("x out")

        # GT spatial --------------------------------------------------
        if enable[5]:
            if isinstance(g_spa[0], tuple):
                for i in range(SAMP_FREQ):
                    img_i = g_spa[0][1][i].data.cpu().numpy()
                    img_i = img_i.swapaxes(0, -1)
                    axes5[i, 0].imshow(img_i)

        # x tem list --------------------------------------------------
        if enable[9]:
            if x_tem_list is not None:
                vmin = [100000 for _ in range(SAMP_FREQ)]
                vmax = [0 for _ in range(SAMP_FREQ)]
                for i in range(SAMP_FREQ):
                    for j in range(len(x_tem_list)):
                        img_ji = x_tem_list[j][i].data.cpu().numpy()
                        if img_ji.shape[-1] != 1:
                            img_ji = img_ji.swapaxes(0, -1)
                        if img_ji.shape[-1] > 3:
                            img_ji = np.linalg.norm(img_ji, axis=-1)
                            vmin[i] = min(vmin[i], img_ji.min())
                            vmax[i] = max(vmax[i], img_ji.max())
                for i in range(SAMP_FREQ):
                    for j in range(len(x_tem_list)):
                        img_ji = x_tem_list[j][i].data.cpu().numpy()
                        if img_ji.shape[-1] != 1:
                            img_ji = img_ji.swapaxes(0, -1)
                        if img_ji.shape[-1] > 3:
                            img_ji = np.linalg.norm(img_ji, axis=-1)
                        axes9[i, j].imshow(img_ji, vmin=vmin[i], vmax=vmax[i])

        if enable[8]:
            axes8[0].cla()
            axes8[1].cla()
            axes8[2].cla()
            axes8[0].hist(input_data.reshape(-1, 75)
                          [:, 0::3].reshape(-1), bins=10)
            axes8[1].hist(input_data.reshape(-1, 75)
                          [:, 1::3].reshape(-1), bins=10)
            axes8[2].hist(input_data.reshape(-1, 75)
                          [:, 2::3].reshape(-1), bins=10)

        plt.show(block=False)

        # print(CosineLoss(1)(x_tem_list[-2], x_tem_list[-1]))

        # out_dir = f"output/{args.weight_path.split('/')[-2]}/{args.weight_path.split('/')[-1]}"  # noqa
        # os.makedirs(out_dir, exist_ok=True)
        # fig4.savefig(f"{out_dir}/{data2[c]:03d}_1_fig4")  # noqa
        # fig7.savefig(f"{out_dir}/{data2[c]:03d}_2_fig7")  # noqa
        # fig1.savefig(f"{out_dir}/{data2[c]:03d}_3_fig1")  # noqa
        # fig5.savefig(f"{out_dir}/{data2[c]:03d}_4_fig5")  # noqa

        output, predict_label, output_dict = None, None, None
        g_spa = None
        tem_a = None
        x_spa_list = None
        featuremap_spa_list = None
        x_spa_list2 = None
        featuremap_spa_list2 = None
        # x_tem_list = None

        top5 = sorted(range(len(logits)), key=lambda i: logits[i])[-5:]
        top5.reverse()
        top5 = [(round(logits[i], 4), MAPPING[i+1]) for i in top5]
        print(top5)

        a = 1
