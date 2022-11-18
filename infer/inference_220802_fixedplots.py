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
from infer.inference import read_xyz
from infer.inference import parse_arg
from infer.inference import init_file_and_folders
from infer.inference import init_preprocessor
from infer.plot_skeleton import plot_skeletons

import matplotlib
import matplotlib.pyplot as plt

from model.layers import tensor_list_mean


_PATH = '/code/2s-AGCN/data/data/ntu_result/xview/sgn'


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
        default=_PATH + '/sgn_v14/221017160001_gt0_1gcn_gcnffn1_tmode3_1layer_8heads_16dim_256ffn_noshartedg_drop01_rerun'  # noqa
        # default=_PATH + '/sgn_v14/221017160001_gt0_1gcn_gcnffn1_tmode3_1layer_16heads_16dim_256ffn_noshartedg_drop01'  # noqa
        # default=_PATH + '/sgn_v14/221020170001_gt0_1gcn_gcnffn1_tmode3_1layer_8heads_16dim_512ffn_noshartedg_drop01'  # noqa
        # default=_PATH + '/sgn_v14/221104130001_gt0_1gcn_gcnffn1_tmode3_1layer_8heads_16dim_256ffn_noshartedg_drop01_bs128_sgd_lr1e1_steps90110'  # noqa
        # default=_PATH + '/sgn_v14/221108160001_gt0_1gcn_gcnffn1_tmode3_1layer_8heads_16dim_256ffn_noshartedg_drop01_galpham1'  # noqa
        # default=_PATH + '/sgn_v14/221108160001_gt0_1gcn_gcnffn1_tmode3_1layer_8heads_16dim_256ffn_noshartedg_drop01_galpham2'  # noqa
        # default=_PATH + '/sgn_v14/221108160001_gt0_1gcn_gcnffn1_tmode3_1layer_8heads_16dim_256ffn_noshartedg_drop01_smp3_tmp3b'  # noqa
    )
    parser.add_argument(
        '--out-folder',
        type=str,
        default='/data/2s-agcn/prediction/ntu_15j/'
    )
    return parser


def init_model(arg: argparse.Namespace):
    Model = import_class(arg.model)(**arg.model_args)
    Model.eval()
    weight_path = os.path.join(arg.weight_path, 'weight')
    weight_file = [i for i in os.listdir(weight_path) if '.pt' in i]
    weight_file = os.path.join(weight_path, weight_file[-1])
    weights = torch.load(weight_file)
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
    # with open(_data_dir + '/NTU_CV_test_180.pkl', 'rb') as f:
    #     data1 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_test_label_180.pkl', 'rb') as f:
    #     data2 = pickle.load(f)
    with open(_data_dir + '/NTU_CV_test.pkl', 'rb') as f:
        data1 = pickle.load(f)
    with open(_data_dir + '/NTU_CV_test_label.pkl', 'rb') as f:
        data2 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_train_180.pkl', 'rb') as f:
    #     data1 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_train_label_180.pkl', 'rb') as f:
    #     data2 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_train_180.pkl', 'wb') as f:
    #     pickle.dump(data1[:180, :, :], f)
    # with open(_data_dir + '/NTU_CV_train_label_180.pkl', 'wb') as f:
    #     pickle.dump(data2[:180], f)

    enable = {i: False for i in range(0, 20, 1)}

    # skeleton -----------
    fig0 = []
    for _ in range(1):
        fig00 = plt.figure(figsize=(14, 6))
        fig00.tight_layout()
        fig0.append(fig00)
    enable[0] = True
    # G spatial -----------
    fig1, axes1 = plt.subplots(5, 1, figsize=(14, 6))
    fig1.tight_layout()
    enable[1] = True
    # A in temporal branch for t_mode=3 -----------
    fig2, axes2 = plt.subplots(5, 1, figsize=(7, 6))
    fig2.tight_layout()
    enable[2] = True
    # featuremap after each SGCN -----------
    fig3, axes3 = plt.subplots(5, 1, figsize=(2, 6))
    fig3.tight_layout()
    enable[3] = True
    # # input data distribution --------------------------------------------
    # fig8, axes8 = plt.subplots(1, 3, figsize=(6, 3))
    # fig8.tight_layout()
    # enable[8] = True
    # x tem list -------------------------------------------------------
    fig9, axes9 = plt.subplots(5, 1, figsize=(2, 6))
    fig9.tight_layout()
    enable[9] = True
    # x tmp list -------------------------------------------------------
    fig7, axes7 = plt.subplots(5, 1, figsize=(2, 6))
    fig7.tight_layout()
    enable[7] = True
    # tem emb -------------------------------------------------------
    fig10, axes10 = plt.subplots(5, 1, figsize=(2, 6))
    fig10.tight_layout()
    enable[10] = True
    # spa emb -------------------------------------------------------
    fig14, axes14 = plt.subplots(5, 1, figsize=(2, 6))
    fig14.tight_layout()
    enable[14] = True
    # pos emb -------------------------------------------------------
    fig17, axes17 = plt.subplots(5, 1, figsize=(2, 6))
    fig17.tight_layout()
    enable[17] = True
    # vel emb -------------------------------------------------------
    fig18, axes18 = plt.subplots(5, 1, figsize=(2, 6))
    fig18.tight_layout()
    enable[18] = True

    fig19, axes19 = plt.subplots(6, 1, figsize=(2, 7))
    fig19.tight_layout()
    enable[19] = True

    freq = 1
    SAMP_FREQ = 5

    print("Start loop...")
    for c in range(0, len(data1), 1):
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
        raw_data = data1[c].reshape(1, 300, 2, 25, 3)
        raw_data = raw_data.reshape((1, 300, -1))

        loop = True

        # while loop:
        n_data, _, _, _ = DataProc.preprocess_fn(
            raw_data, sampling_frequency=SAMP_FREQ)  # N,'T, MVC
        input_data = np.array(n_data, dtype=n_data[0].dtype)  # N, 'T, MVC

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
        x_smp_list = output_dict.get('x_smp_list')
        tem_emb = output_dict.get('tem_emb')
        spa_emb = output_dict.get('spa_emb')
        pos_emb = output_dict.get('pos_emb')
        vel_emb = output_dict.get('vel_emb')

        logits, preds = output[0].tolist(), predict_label.item()

        # if (logits[preds]*100 < 50) or data2[c] != preds:
        print(f"Label : {data2[c]:3d} , Pred : {preds:3d} , Logit : {logits[preds]*100:>5.2f}, SAMP_FREQ : {SAMP_FREQ}, {c}")  # noqa

        # if logits[preds] > 0.3:
        #     loop = False

        # if data2[c] < 22:
        #     continue
        if data2[c] == preds:
            continue
        # if data2[c] != 47:
        #     continue
        # if logits[preds]*100 > 80.0:
        #     continue

        # PLOTTING ----------------------------------------------------------

        if enable[0]:
            d = input_data.reshape((-1, 25, 3))
            # d = input_data[0].reshape((-1, 25, 3))
            for fig00 in fig0:
                fig00.clear()
            dd = np.stack([d[:, :, 2], d[:, :, 0], d[:, :, 1]], axis=-1)
            plot_skeletons(dd, fig0[0])

        # G spatial --------------------------------------------------
        if enable[1]:
            for idx, fg, ax in [(0, fig1, axes1), ]:
                try:
                    fg.suptitle(MAPPING[preds+1] + " : " + MAPPING[data2[c]+1] +
                                f" : {logits[preds]*100:>5.2f}")
                    fg.subplots_adjust(top=0.9)
                    for j in range(SAMP_FREQ):
                        img_j = []
                        for i in range(0, 20, freq):
                            if isinstance(g_spa[idx], tuple):
                                img_i = g_spa[idx][0][j][i].data.cpu().numpy()
                            else:
                                img_i = g_spa[idx][j][i].data.cpu().numpy()
                            img_j.append(img_i)
                        img = np.concatenate(img_j, axis=1)
                        ax[j].imshow(img)
                        ax[j].xaxis.set_ticks(np.arange(0, 25*(20//freq), 25))
                        ax[j].yaxis.set_ticks(np.arange(0, 25, SAMP_FREQ))
                except:
                    pass

        # A in temporal branch for t_mode=3 ---------------------------
        if enable[2]:
            if tem_a[0] is not None:
                img = []
                for j in range(SAMP_FREQ):
                    # img_j = []
                    # for i in range(len(tem_a[0])):
                    #     # last 0 cause spa_maxpool
                    #     img_i = tem_a[0][i][j][0].data.cpu().numpy()
                    #     img_j.append(img_i)
                    # img.append(np.concatenate(img_j, axis=1))
                    # axes2[j].imshow(img[-1])
                    # axes2[j].xaxis.set_ticks(
                    #     np.arange(0, 20*len(tem_a[0]), SAMP_FREQ))
                    # axes2[j].yaxis.set_ticks(np.arange(0, 20, SAMP_FREQ))
                    for i in range(len(tem_a[0])):
                        # last 0 cause spa_maxpool
                        img_i = tem_a[0][i][j].data.cpu().numpy()
                        img_i = img_i.swapaxes(0, 1)
                        img_i = img_i.reshape(img_i.shape[0], -1)
                        axes2[j].imshow(img_i)
                        axes2[j].xaxis.set_ticks(
                            np.arange(0, img_i.shape[-1], 10))
                        axes2[j].yaxis.set_ticks(
                            np.arange(0, img_i.shape[0], 10))

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
                        # axes9[i, j].imshow(img_ji, vmin=vmin[i], vmax=vmax[i])
                        axes9[i].imshow(img_ji)

        if enable[7]:
            if x_smp_list is not None:
                vmin = [100000 for _ in range(SAMP_FREQ)]
                vmax = [0 for _ in range(SAMP_FREQ)]
                for i in range(SAMP_FREQ):
                    for j in range(len(x_smp_list)):
                        img_ji = x_smp_list[j][i].data.cpu().numpy()
                        if img_ji.shape[-1] != 1:
                            img_ji = img_ji.swapaxes(0, -1)
                        if img_ji.shape[-1] > 3:
                            img_ji = np.linalg.norm(img_ji, axis=-1)
                            vmin[i] = min(vmin[i], img_ji.min())
                            vmax[i] = max(vmax[i], img_ji.max())
                for i in range(SAMP_FREQ):
                    for j in range(len(x_smp_list)):
                        img_ji = x_smp_list[j][i].data.cpu().numpy()
                        if img_ji.shape[-1] != 1:
                            img_ji = img_ji.swapaxes(0, -1)
                        if img_ji.shape[-1] > 3:
                            img_ji = np.linalg.norm(img_ji, axis=-1)
                        # axes7[i, j].imshow(img_ji, vmin=vmin[i], vmax=vmax[i])
                        axes7[i].imshow(img_ji)

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

        if enable[10]:
            for j in range(SAMP_FREQ):
                img_j = np.linalg.norm(tem_emb[j], axis=0)
                axes10[j].imshow(np.expand_dims(img_j, axis=-1))
                axes10[j].xaxis.set_ticks(np.arange(0, 20+1, SAMP_FREQ))
                axes10[j].yaxis.set_ticks(np.arange(0, 25+1, SAMP_FREQ))

        if enable[14]:
            for j in range(SAMP_FREQ):
                img_j = np.linalg.norm(spa_emb[j], axis=0)
                axes14[j].imshow(np.expand_dims(img_j, axis=-1))
                axes14[j].xaxis.set_ticks(np.arange(0, 20+1, SAMP_FREQ))
                axes14[j].yaxis.set_ticks(np.arange(0, 25+1, SAMP_FREQ))

        if enable[17]:
            for j in range(SAMP_FREQ):
                img_j = np.linalg.norm(pos_emb[j], axis=0)
                axes17[j].imshow(np.expand_dims(img_j, axis=-1))
                axes17[j].xaxis.set_ticks(np.arange(0, 20+1, SAMP_FREQ))
                axes17[j].yaxis.set_ticks(np.arange(0, 25+1, SAMP_FREQ))

        if enable[18]:
            for j in range(SAMP_FREQ):
                img_j = np.linalg.norm(vel_emb[j], axis=0)
                axes18[j].imshow(np.expand_dims(img_j, axis=-1))
                axes18[j].xaxis.set_ticks(np.arange(0, 20+1, SAMP_FREQ))
                axes18[j].yaxis.set_ticks(np.arange(0, 25+1, SAMP_FREQ))

        if enable[19]:
            img_j = np.linalg.norm(featuremap_spa_list[0]['x0'][0], axis=0)
            axes19[0].imshow(np.expand_dims(img_j, axis=-1))
            img_j = np.linalg.norm(featuremap_spa_list[0]['x4'][0], axis=0)
            axes19[1].imshow(np.expand_dims(img_j, axis=-1))
            img_j = np.linalg.norm(featuremap_spa_list[0]['x5'][0], axis=0)
            axes19[2].imshow(np.expand_dims(img_j, axis=-1))
            img_j = np.linalg.norm(featuremap_spa_list[0]['x6'][0], axis=0)
            axes19[3].imshow(np.expand_dims(img_j, axis=-1))
            img_j = np.linalg.norm(featuremap_spa_list[0]['x7'][0], axis=0)
            axes19[4].imshow(np.expand_dims(img_j, axis=-1))
            img_j = np.linalg.norm(featuremap_spa_list[0]['x9'][0], axis=0)
            axes19[5].imshow(np.expand_dims(img_j, axis=-1))

        plt.show(block=False)

        # print(CosineLoss(1)(x_tem_list[-2], x_tem_list[-1]))

        # out_dir = f"output/{args.weight_path.split('/')[-2]}/{args.weight_path.split('/')[-1]}"  # noqa
        # os.makedirs(out_dir, exist_ok=True)
        # fig4.savefig(f"{out_dir}/{data2[c]:03d}_1_fig4")  # noqa
        # fig7.savefig(f"{out_dir}/{data2[c]:03d}_2_fig7")  # noqa
        # fig1.savefig(f"{out_dir}/{data2[c]:03d}_3_fig1")  # noqa
        # fig5.savefig(f"{out_dir}/{data2[c]:03d}_4_fig5")  # noqa

        top5 = sorted(range(len(logits)), key=lambda i: logits[i])[-5:]
        top5.reverse()
        top5 = [(round(logits[i], 4), MAPPING[i+1]) for i in top5]
        print(top5)

        output, predict_label, output_dict = None, None, None
        g_spa = None
        tem_a = None
        x_spa_list = None
        featuremap_spa_list = None
        x_spa_list2 = None
        featuremap_spa_list2 = None
        # x_tem_list = None
        tem_emb = None

        a = 1
