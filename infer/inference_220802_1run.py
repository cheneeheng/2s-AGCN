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
        # default=_PATH + '/sgn_v14/220923140001_tmode5_1357'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_1layer'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_1layer_sgcnattn1'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_1layer_3heads'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_1layer_6heads'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_1layer_9heads'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_1layer_3heads_1024dhead'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_3heads'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_3layers'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_4layers'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_2layer_6heads'  # noqa
        # default=_PATH + '/sgn_v14/220928170001_tmode3_2layer_9heads'  # noqa
        # default=_PATH + '/sgn_v14/221006150001_tmode3_absposenc'  # noqa
        # default=_PATH + '/sgn_v14/221006150001_tmode3_cosposenc'  # noqa
        # default=_PATH + '/sgn_v14/221006150001_tmode3_3heads_absposenc'  # noqa
        # default=_PATH + '/sgn_v14/221006150001_tmode3_6heads_absposenc'  # noqa
        # default=_PATH + '/sgn_v14/221006150001_tmode3_9heads_absposenc'  # noqa
        # default=_PATH + '/sgn_v14/221010140001_tmode3_3heads_256ffn_nosemfr'  # noqa
        # default=_PATH + '/sgn_v14/221010140001_tmode3_3heads_256ffn'  # noqa
        # default=_PATH + '/sgn_v14/221010140001_tmode3_256ffn'  # noqa
        # default=_PATH + '/sgn_v14/221010140001_tmode3_3heads_256ffn_absposenc'  # noqa
        # default=_PATH + '/sgn_v14/221010140001_tmode3_3heads_256ffn_gt0'  # noqa
        # default=_PATH + '/sgn_v14/221010140001_tmode3_3heads_256ffn_noshartedg_gt0'  # noqa
        # default=_PATH + '/sgn_v14/221010140001_tmode3_3heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_tmode3_1layer_3heads_256ffn'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gcn4layers_tmode3_1layer_3heads_256ffn'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_tmode3_1layer_3heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gcn4layers_tmode3_1layer_3heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gcn5layers_tmode3_1layer_3heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gcnffn1_tmode3_1layer_3heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gt0_gcnffn1_tmode3_1layer_3heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_1gcn_gcnffn1_tmode3_1layer_3heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_1gcn_gcnffn1_tmode3_1layer_9heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_1gcn_gcnffn1_tmode3_1layer_15heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gt0_1gcn_gcnffn1_tmode3_1layer_3heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gt0_1gcn_gcnffn1_tmode3_1layer_9heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gt0_1gcn_gcnffn1_tmode3_1layer_12heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221011110001_gt0_1gcn_gcnffn1_tmode3_1layer_15heads_256ffn_noshartedg'  # noqa
        # default=_PATH + '/sgn_v14/221013160001_gt0_1gcn_gcnffn1_tmode3_1layer_9heads_256ffn_noshartedg_smp3_semjointsmp1'  # noqa
        # default=_PATH + '/sgn_v14/221013160001_gt0_1gcn_gcnffn1_tmode3_1layer_9heads_256ffn_noshartedg_smp4_semjointsmp1'  # noqa
        # default=_PATH + '/sgn_v14/221017160001_gt0_1gcn_gcnffn1_tmode3_1layer_8heads_32dim_256ffn_noshartedg_drop01'  # noqa
        # default=_PATH + '/sgn_v14/221017160001_gt0_1gcn_gcnffn1_tmode3_1layer_16heads_32dim_256ffn_noshartedg_drop01'  # noqa
        default=_PATH + '/sgn_v14/221017160001_gt0_1gcn_gcnffn1_tmode3_1layer_8heads_16dim_256ffn_noshartedg_drop01_rerun'  # noqa
        # default=_PATH + '/sgn_v14/221017160001_gt0_1gcn_gcnffn1_tmode3_1layer_16heads_16dim_256ffn_noshartedg_drop01'  # noqa
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
    freq = 1
    SAMP_FREQ = 5

    TRIAL = 0
    samples = 10

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

    enable = {}

    # skeleton -----------
    fig0 = []
    for _ in range(1):
        fig00 = plt.figure()
        fig00.tight_layout()
        fig0.append(fig00)
    enable['skeleton'] = True
    skeleton_data = np.zeros((0, 25, 3))
    # G spatial -----------
    fig1, axes1 = plt.subplots(1, 1)
    fig1.tight_layout()
    enable['spatial_G'] = True
    spatial_g_data = []
    # A in temporal branch for t_mode=3 -----------
    fig2, axes2 = plt.subplots(1, 1)
    fig2.tight_layout()
    enable['temporal_A'] = True
    temporal_a_data = []
    # featuremap after each SGCN -----------
    fig3, axes3 = plt.subplots(1, 1)
    fig3.tight_layout()
    enable['gcn_fm'] = True
    gcn_fm_data = []
    # x tem list -----------
    fig9, axes9 = plt.subplots(1, 1)
    fig9.tight_layout()
    enable['temporal_x'] = True
    temporal_x_data = []
    # tem emb -----------
    fig10, axes10 = plt.subplots(1, 1)
    fig10.tight_layout()
    enable['temporal_emb'] = True
    temporal_emb_data = []
    # spa emb -----------
    fig14, axes14 = plt.subplots(1, 1)
    fig14.tight_layout()
    enable['spatial_emb'] = True
    spatial_emb_data = []

    pred_data = []

    print("Start loop...")
    for c in range(20, len(data1), 1):
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
        tem_emb = output_dict.get('tem_emb')
        spa_emb = output_dict.get('spa_emb')

        logits, preds = output[0].tolist(), predict_label.item()

        # if (logits[preds]*100 < 50) or data2[c] != preds:
        print(f"Label : {data2[c]:3d} , Pred : {preds:3d} , Logit : {logits[preds]*100:>5.2f}, SAMP_FREQ : {SAMP_FREQ}, {c}")  # noqa

        top5 = sorted(range(len(logits)), key=lambda i: logits[i])[-5:]
        top5.reverse()
        top5 = [(round(logits[i], 4), MAPPING[i+1]) for i in top5]
        print(top5)

        pred_data.append(MAPPING[preds+1])

        # if data2[c] < 22:
        #     # if data2[c] == preds:
        #     # if data2[c] != 53:
        #     # if logits[preds]*100 > 80.0:
        #     continue

        j = TRIAL

        x = input_data.reshape((-1, 25, 3))
        skeleton_data = np.concatenate(
            [skeleton_data, x[j*20:(j+1)*20]], axis=0)

        x = []
        j = TRIAL
        for i in range(0, 20, freq):
            if isinstance(g_spa[0], tuple):
                img_i = g_spa[0][0][j][i].data.cpu().numpy()
            else:
                img_i = g_spa[0][j][i].data.cpu().numpy()
            x.append(img_i)
        spatial_g_data.append(np.concatenate(x, axis=1))

        x = []
        # last 0 cause spa_maxpool
        img_i = tem_a[0][0][j].data.cpu().numpy()
        img_i = img_i.swapaxes(0, 1)
        img_i = img_i.reshape(img_i.shape[0], -1)
        temporal_a_data.append(img_i)

        featuremap = np.linalg.norm(x_spa_list[-1], axis=1)[j]  # V,T
        gcn_fm_data.append(featuremap)

        i = 0
        img_ji = x_tem_list[i][j].data.cpu().numpy()
        if img_ji.shape[-1] != 1:
            img_ji = img_ji.swapaxes(0, -1)
        if img_ji.shape[-1] > 3:
            img_ji = np.linalg.norm(img_ji, axis=-1)
        temporal_x_data.append(img_ji.swapaxes(0, -1))

        img_j = np.linalg.norm(tem_emb[j], axis=0)
        temporal_emb_data.append(np.expand_dims(img_j, axis=-1))

        img_j = np.linalg.norm(spa_emb[j], axis=0)
        spatial_emb_data.append(np.expand_dims(img_j, axis=-1))

        if (c+1) % samples == 0:
            break

    # PLOTTING ----------------------------------------------------------

    # SKELETON ---------------------------------------------------
    if 'skeleton' in enable:
        for fig00 in fig0:
            fig00.clear()
        dd = np.stack([skeleton_data[:, :, 2],
                       skeleton_data[:, :, 0],
                       skeleton_data[:, :, 1]], axis=-1)
        plot_skeletons(dd, fig0[0], samples=samples)

    # G spatial --------------------------------------------------
    if 'spatial_G' in enable:
        axes1.imshow(np.concatenate(spatial_g_data, axis=0))
        axes1.xaxis.set_ticks(np.arange(0, 25*20+1, 25))
        # axes1.yaxis.set_ticks(np.arange(0, samples*25+1, 25))
        axes1.set_yticks(np.arange(len(pred_data))*25, labels=pred_data)
        fig1.tight_layout()

    # A in temporal branch for t_mode=3 ---------------------------
    if 'temporal_A' in enable:
        axes2.imshow(np.concatenate(temporal_a_data, axis=0))
        axes2.xaxis.set_ticks(np.arange(0, 200+1, 20))
        # axes2.yaxis.set_ticks(np.arange(0, samples*20+1, 20))
        axes2.set_yticks(np.arange(len(pred_data))*20+10, labels=pred_data)
        fig2.tight_layout()

    # featuremap after each SGCN ----------------------------------
    if 'gcn_fm' in enable:
        axes3.imshow(np.concatenate(gcn_fm_data, axis=1))
        # axes3.xaxis.set_ticks(np.arange(0, samples*20+1, 20))
        axes3.yaxis.set_ticks(np.arange(0, 25+1, 5))
        axes3.set_xticks(np.arange(len(pred_data))*20+10, labels=pred_data)
        fig3.tight_layout()

    # x tem list --------------------------------------------------
    if 'temporal_x' in enable:
        axes9.imshow(np.concatenate(temporal_x_data, axis=0))
        axes9.xaxis.set_ticks(np.arange(0, 20+1, 5))
        # axes9.yaxis.set_ticks(np.arange(0, samples+1, 5))
        axes9.set_yticks(np.arange(len(pred_data)), labels=pred_data)
        fig9.tight_layout()

    if 'temporal_emb' in enable:
        axes10.imshow(temporal_emb_data[0])
        axes10.xaxis.set_ticks(np.arange(0, 20+1, 5))
        axes10.yaxis.set_ticks(np.arange(0, 25+1, 5))

    if 'spatial_emb' in enable:
        axes14.imshow(spatial_emb_data[0])
        axes14.xaxis.set_ticks(np.arange(0, 20+1, 5))
        axes14.yaxis.set_ticks(np.arange(0, 25+1, 5))

    plt.show(block=False)

    # print(CosineLoss(1)(x_tem_list[-2], x_tem_list[-1]))

    # out_dir = f"output/{args.weight_path.split('/')[-2]}/{args.weight_path.split('/')[-1]}"  # noqa
    # os.makedirs(out_dir, exist_ok=True)
    # fig4.savefig(f"{out_dir}/{data2[c]:03d}_1_fig4")  # noqa
    # fig7.savefig(f"{out_dir}/{data2[c]:03d}_2_fig7")  # noqa
    # fig1.savefig(f"{out_dir}/{data2[c]:03d}_3_fig1")  # noqa
    # fig5.savefig(f"{out_dir}/{data2[c]:03d}_4_fig5")  # noqa

    a = 1
