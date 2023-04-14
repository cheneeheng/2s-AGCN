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
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/2210121800'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/2210121800_mhadimmult1'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/2210121800_mhadimmult1_3heads'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/2210121800_2layers'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/2210121800_mhadimmult1_2layers'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/2210121800_mhadimmult1_3heads_2layers'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/2210121800_mhadimmult1_9_1heads'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/2210141000_mhadimmult1_12_12heads'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/221122163001_8head_32hdim_256out_8head_64hdim_512out'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/221122163001_8head_16hdim_256out_8head_32hdim_512out'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/221124163001_8head_16hdim_256ffn_256out_8head_32hdim_512ffn_512out'  # noqa
        default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/221124163001_8head_16hdim_128ffn_256out_8head_32hdim_256ffn_512out_qkv'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/221128113001_2l_8head_16hdim_128ffn_256out_2l_8head_32hdim_256ffn_512out_qkv'  # noqa
        # default='/code/2s-AGCN/data/data/ntu_result/xview/sgn/sgn_v15/221130163001_8head_16hdim_128ffn_256out_8head_32hdim_256ffn_512out_qkv_1001posvel'  # noqa
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
                    input_data: np.ndarray,
                    sampling_freq: int = 5
                    ) -> Tuple[torch.Tensor, torch.Tensor]:

    with torch.no_grad():
        if arg.gpu:
            output = model(torch.from_numpy(input_data).cuda(0))
        else:
            output = model(torch.from_numpy(input_data))
        output, output_dict = output[0], output[1]
        if 'sgn' in arg.model:
            output = output.view((-1, sampling_freq, output.size(1)))
            for i in range(sampling_freq):
                output_i = F.softmax(output[:, i, :], -1)
                _, predict_label_i = torch.max(output_i, 1)
                print(f"logit : {output_i[0, predict_label_i]}, label:{predict_label_i}")  # noqa
            output = output.mean(1)
        output = F.softmax(output, 1)
        _, predict_label = torch.max(output, 1)
        if arg.gpu:
            output = output.data.cpu()
            predict_label = predict_label.data.cpu()
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
    # with open(_data_dir + '/NTU_CV_train.pkl', 'rb') as f:
    #     data1 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_train_label.pkl', 'rb') as f:
    #     data2 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_train_180.pkl', 'rb') as f:
    #     data1 = pickle.load(f)
    # with open(_data_dir + '/NTU_CV_train_label_180.pkl', 'rb') as f:
    #     data2 = pickle.load(f)

    # with open(_data_dir + '/NTU_CV_train_180.pkl', 'wb') as f:
    #     pickle.dump(data1[:180, :, :], f)
    # with open(_data_dir + '/NTU_CV_train_label_180.pkl', 'wb') as f:
    #     pickle.dump(data2[:180], f)

    enable = {i: False for i in range(0, 100, 1)}

    # skeleton -----------
    fig0 = []
    for _ in range(1):
        fig00 = plt.figure(figsize=(16, 6))
        fig00.tight_layout()
        fig0.append(fig00)
    enable[0] = True
    # A in spatial and temporal ------------
    fig2, axes2 = {}, {}
    for i in range(1):
        fig2[i], axes2[i] = plt.subplots(1, 1, figsize=(16, 6))
        fig2[i].tight_layout()
    enable[2] = True
    fig12, axes12 = plt.subplots(5, 1, figsize=(3, 7))
    fig12.tight_layout()
    enable[7] = True
    fig3, axes3 = {}, {}
    for i in range(1):
        fig3[i], axes3[i] = plt.subplots(5, 1, figsize=(3, 7))
        fig3[i].tight_layout()
    enable[3] = True
    fig13, axes13 = plt.subplots(5, 1, figsize=(3, 7))
    fig13.tight_layout()
    enable[7] = True
    # spa emb -----------
    fig4, axes4 = plt.subplots(5, 1, figsize=(3, 7))
    fig4.tight_layout()
    enable[4] = True
    # tem emb -----------
    fig5, axes5 = plt.subplots(5, 1, figsize=(3, 7))
    fig5.tight_layout()
    enable[5] = True
    # spa fm -----------
    fig6, axes6 = plt.subplots(5, 1, figsize=(3, 7))
    fig6.tight_layout()
    enable[6] = True
    # tem fm -----------
    fig7, axes7 = plt.subplots(5, 1, figsize=(3, 7))
    fig7.tight_layout()
    enable[7] = True
    # # attn J@T -----------
    # fig8, axes8 = plt.subplots(5, 1, figsize=(3, 7))
    # fig8.tight_layout()
    # enable[8] = True
    # attn_list = []

    freq = 1
    SAMP_FREQ = 5
    SEGMENTS = 20
    JOINTS = 25

    print("Start loop...")
    for c in range(0, data1.shape[0], 1):
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

        tem_emb = output_dict['tem_emb']
        spa_emb = output_dict['spa_emb']
        spatial_attn_list = output_dict['spatial_attn_list']
        temporal_attn_list = output_dict['temporal_attn_list']
        spatial_featuremap = output_dict.get('spatial_featuremap')
        temporal_featuremap = output_dict.get('temporal_featuremap')

        logits, preds = output[0].tolist(), predict_label.item()

        # if (logits[preds]*100 < 50) or data2[c] != preds:
        print(f"Label : {data2[c]:3d} , Pred : {preds:3d} , Logit : {logits[preds]*100:>5.2f}, SAMP_FREQ : {SAMP_FREQ}, {c}")  # noqa

        # if data2[c] < 22:
        if data2[c] == preds:
            # if data2[c] != 53:
            # if logits[preds]*100 > 80.0:
            continue

        # PLOTTING ----------------------------------------------------------

        # skeleton
        if enable[0]:
            d = input_data.reshape((-1, 25, 3))
            # d = input_data[0].reshape((-1, 25, 3))
            for fig00 in fig0:
                fig00.clear()
            dd = np.stack([d[:, :, 2], d[:, :, 0], d[:, :, 1]], axis=-1)
            plot_skeletons(dd, fig0[0])

        # spatial A ---------------------------------------------------------
        if enable[2]:
            for idx, sa in enumerate(spatial_attn_list):
                fig2[idx].suptitle(MAPPING[preds+1] +
                                   " : " +
                                   MAPPING[data2[c]+1] +
                                   f" : {logits[preds]*100:>5.2f}")
                fig2[idx].subplots_adjust(top=0.9)

                img_i = sa.data.cpu().numpy()  # NT,H,V,V
                img_i = img_i.reshape(SAMP_FREQ, SEGMENTS, img_i.shape[1],
                                      img_i.shape[2], img_i.shape[3])
                for j in range(1):
                    img_j = []
                    for k in range(0, SEGMENTS, freq):
                        img_k = img_i[j][k]  # h,v,v
                        img_k = img_k.reshape(-1, img_k.shape[-1])
                        img_j.append(img_k)
                    img = np.concatenate(img_j, axis=1)
                    axes2[idx].imshow(img)
                    axes2[idx].xaxis.set_ticks(
                        np.arange(0, JOINTS*(SEGMENTS//freq), JOINTS))
                    axes2[idx].yaxis.set_ticks(
                        np.arange(0, JOINTS, SAMP_FREQ))
                    # axes2[idx][j].imshow(img)
                    # axes2[idx][j].xaxis.set_ticks(
                    #     np.arange(0, JOINTS*(SEGMENTS//freq), JOINTS))
                    # axes2[idx][j].yaxis.set_ticks(
                    #     np.arange(0, JOINTS, SAMP_FREQ))

                for j in range(SAMP_FREQ):
                    img_j = img_i[j]
                    img_j = np.sum(img_j, axis=1)
                    img_j = img_j.swapaxes(0, 1)
                    img_j = img_j.reshape(img_j.shape[0], -1)
                    axes12[j].imshow(np.expand_dims(img_j, axis=-1))
                    axes12[j].xaxis.set_ticks(
                        np.arange(0, JOINTS*(SEGMENTS//freq), JOINTS))
                    axes12[j].yaxis.set_ticks(
                        np.arange(0, SEGMENTS, SAMP_FREQ))

        # temporal A ---------------------------------------------------------
        if enable[3]:
            for idx, ta in enumerate(temporal_attn_list):
                fig3[idx].suptitle(MAPPING[preds+1] +
                                   " : " +
                                   MAPPING[data2[c]+1] +
                                   f" : {logits[preds]*100:>5.2f}")
                fig3[idx].subplots_adjust(top=0.9)

                for j in range(SAMP_FREQ):
                    # last 0 cause spa_maxpool
                    img_i = ta[j].data.cpu().numpy()
                    img_i = img_i.swapaxes(0, 1)
                    img_i = img_i.reshape(img_i.shape[0], -1)
                    axes3[idx][j].imshow(img_i)
                    axes3[idx][j].xaxis.set_ticks(
                        np.arange(0, SEGMENTS, SAMP_FREQ))
                    axes3[idx][j].yaxis.set_ticks(
                        np.arange(0, SEGMENTS, SAMP_FREQ))

                    img_i = ta[j].data.cpu().numpy()
                    img_i = np.sum(img_i, axis=0)
                    axes13[j].imshow(np.expand_dims(img_i, axis=-1))
                    axes13[j].xaxis.set_ticks(
                        np.arange(0, SEGMENTS, SAMP_FREQ))
                    axes13[j].yaxis.set_ticks(
                        np.arange(0, SEGMENTS, SAMP_FREQ))

        # spatial enb ---------------------------------------------------------
        if enable[4]:
            for j in range(SAMP_FREQ):
                img_j = np.linalg.norm(spa_emb[j], axis=0)
                axes4[j].imshow(np.expand_dims(img_j, axis=-1))
                axes4[j].xaxis.set_ticks(np.arange(0, SEGMENTS+1, SAMP_FREQ))
                axes4[j].yaxis.set_ticks(np.arange(0, JOINTS+1, SAMP_FREQ))

        # temporal emb ---------------------------------------------------------
        if enable[5]:
            for j in range(SAMP_FREQ):
                img_j = np.linalg.norm(tem_emb[j], axis=0)
                axes5[j].imshow(np.expand_dims(img_j, axis=-1))
                axes5[j].xaxis.set_ticks(np.arange(0, SEGMENTS+1, SAMP_FREQ))
                axes5[j].yaxis.set_ticks(np.arange(0, JOINTS+1, SAMP_FREQ))

        # spatial fm ---------------------------------------------------------
        if enable[6]:
            for j in range(SAMP_FREQ):
                img_j = np.linalg.norm(spatial_featuremap[j], axis=0)
                axes6[j].imshow(np.expand_dims(img_j, axis=-1))
                axes6[j].xaxis.set_ticks(np.arange(0, SEGMENTS+1, SAMP_FREQ))
                axes6[j].yaxis.set_ticks(np.arange(0, JOINTS+1, SAMP_FREQ))

        # temporal fm ---------------------------------------------------------
        if enable[7]:
            for j in range(SAMP_FREQ):
                img_j = np.linalg.norm(temporal_featuremap[j], axis=0)
                axes7[j].imshow(np.expand_dims(img_j, axis=-1))
                axes7[j].xaxis.set_ticks(np.arange(0, SEGMENTS+1, SAMP_FREQ))

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
        tem_emb = None

        top5 = sorted(range(len(logits)), key=lambda i: logits[i])[-5:]
        top5.reverse()
        top5 = [(round(logits[i], 4), MAPPING[i+1]) for i in top5]
        print(top5)

        a = 1
