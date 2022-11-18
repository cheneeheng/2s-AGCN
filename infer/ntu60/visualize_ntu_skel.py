"""
Based on :
https://programmer.ink/think/how-to-realize-ntu-rgb-d-skeleton-visualization-gracefully-with-matplotlib.html
"""

from sklearn.cluster import KMeans

import numpy as np
import os

from matplotlib import pyplot as plt
from plotly import graph_objects as go
from time import sleep

from utils.visualization import visualize_3dskeleton_in_matplotlib
from data_gen.preprocess import pre_normalization


JOINT_COLOR = [
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 85),
    # (255, 0, 170),
    (255, 0, 255),
    (170, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (85, 255, 0),
    (255, 85, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 170, 0),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (0, 0, 255),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 0),
    # (255, 255, 0),
    (255, 191, 0),
    (255, 255, 0),
]


def _plot_skel(p, f, m=0):

    # Determine which nodes are connected as bones according
    #  to NTU skeleton structure
    # Note that the sequence number starts from 0 and needs to be minus 1
    rightarm = np.array([24, 12, 11, 10, 9, 21]) - 1
    leftarm = np.array([22, 8, 7, 6, 5, 21]) - 1
    righthand = np.array([25, 12]) - 1
    lefthand = np.array([23, 8]) - 1
    rightleg = np.array([19, 18, 17, 1]) - 1
    leftleg = np.array([15, 14, 13, 1]) - 1
    rightfeet = np.array([20, 19]) - 1
    leftfeet = np.array([16, 15]) - 1
    body = np.array([4, 3, 21, 2, 1]) - 1  # body

    return [
        # Bones
        go.Scatter3d(
            x=p[0, f, body, m],
            y=p[1, f, body, m],
            z=p[2, f, body, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[8]}',
            ),
            marker=dict(size=5),
        ),
        go.Scatter3d(
            x=p[0, f, rightarm, m],
            y=p[1, f, rightarm, m],
            z=p[2, f, rightarm, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[3]}',
            ),
            marker=dict(size=5),
        ),
        go.Scatter3d(
            x=p[0, f, leftarm, m],
            y=p[1, f, leftarm, m],
            z=p[2, f, leftarm, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[leftarm[0]]}',
            ),
            marker=dict(size=5),
        ),
        go.Scatter3d(
            x=p[0, f, righthand, m],
            y=p[1, f, righthand, m],
            z=p[2, f, righthand, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[3]}',
            ),
            marker=dict(size=5),
        ),
        go.Scatter3d(
            x=p[0, f, lefthand, m],
            y=p[1, f, lefthand, m],
            z=p[2, f, lefthand, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[lefthand[0]]}',
            ),
            marker=dict(size=5),
        ),
        go.Scatter3d(
            x=p[0, f, rightleg, m],
            y=p[1, f, rightleg, m],
            z=p[2, f, rightleg, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[rightleg[0]]}',
            ),
            marker=dict(size=5),
        ),
        go.Scatter3d(
            x=p[0, f, leftleg, m],
            y=p[1, f, leftleg, m],
            z=p[2, f, leftleg, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[leftleg[0]]}',
            ),
            marker=dict(size=5),
        ),
        go.Scatter3d(
            x=p[0, f, rightfeet, m],
            y=p[1, f, rightfeet, m],
            z=p[2, f, rightfeet, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[rightfeet[0]]}',
            ),
            marker=dict(size=5),
        ),
        go.Scatter3d(
            x=p[0, f, leftfeet, m],
            y=p[1, f, leftfeet, m],
            z=p[2, f, leftfeet, m],
            mode='lines+markers',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[leftfeet[0]]}',
            ),
            marker=dict(size=5),
        ),
    ]


def draw_single_skeleton(data, f, pause_sec=10, action=""):

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=3.0)
    )

    fig = go.Figure(data=_plot_skel(data, f))

    fig.update_layout(scene_camera=camera,
                      autosize=False,
                      showlegend=False,
                      width=600,
                      height=600,
                      margin=dict(l=0, r=0, b=0, t=0),
                      yaxis=dict(range=[-1, 1]),
                      xaxis=dict(range=[-1, 1]))

    return fig


def draw_skeleton(data, pause_sec=10, action=""):

    data[0, :, :, :] *= -1
    data[2, :, :, :] *= -1

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=2.25)
    )

    f = 0
    fig = go.Figure(_plot_skel(data, f))
    fig.update_layout(scene_camera=camera,
                      showlegend=False,
                      margin=dict(l=0, r=0, b=0, t=0),
                      yaxis=dict(range=[-1, 1]),
                      xaxis=dict(range=[-1, 1]))
    fig.show()

    for f in range(1, 50):
        fig.update(data=_plot_skel(data, f))
        fig.update_layout(scene_camera=camera,
                          showlegend=False,
                          margin=dict(l=0, r=0, b=0, t=0),
                          yaxis=dict(range=[-1, 1]),
                          xaxis=dict(range=[-1, 1]))
        fig.show()

        sleep(pause_sec)


def draw_skeleton_offline(data, pause_sec=10, action=""):

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=3.0, z=0)
    )
    fig = go.Figure(
        data=_plot_skel(data, 0),
        layout=go.Layout(
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None])])]),
        frames=[go.Frame(data=_plot_skel(data, k))
                for k in range(data.shape[1])]
    )
    fig.update_layout(scene_camera=camera,
                      showlegend=False,
                      margin=dict(l=0, r=0, b=0, t=0),
                      yaxis=dict(range=[-1, 1]),
                      xaxis=dict(range=[-1, 1]))
    fig.show(renderer='notebook_connected')


if __name__ == '__main__':

    fig, axs = plt.subplots(2)

    base_path = './data/data/nturgbd_raw/nturgb+d_skeletons'
    paths = [os.path.join(base_path, f) for f in sorted(os.listdir(base_path))]
    paths = [i for i in paths if 'A012' in i]

    failed1_path = './data/data/ntu_sgn/denoised_data/denoised_failed_1.log'
    paths = np.loadtxt(failed1_path, dtype=str)[300:]

    for file_name in paths:

        # if 'P001' not in file_name or 'S008' not in file_name:
        #     continue
        # if 'S001C001P002R002A012' not in file_name:
        #     continue

        print(file_name)
        file_name = os.path.join(base_path, file_name + '.skeleton')

        # file_name = r'./data/data/nturgbd_raw/nturgb+d_skeletons/S001C001P001R001A012.skeleton'  # noqa
        max_V = 25  # Number of nodes
        max_M = 2  # Number of skeletons
        with open(file_name, 'r') as fr:
            frame_num = int(fr.readline())
            data = np.zeros((3, frame_num, 25, 2))
            for frame in range(frame_num):
                person_num = int(fr.readline())
                for person in range(person_num):
                    fr.readline()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        v = fr.readline().split(' ')
                        if joint < max_V and person < max_M:
                            data[0, frame, joint, person] = float(
                                v[0])  # A coordinate of a joint
                            data[1, frame, joint, person] = float(v[1])
                            data[2, frame, joint, person] = float(v[2])
        data = pre_normalization(np.expand_dims(data, 0),
                                 xaxis=None,
                                 zaxis=None,
                                 center=False,
                                 verbose=False,
                                 tqdm=False)[0]

        print(file_name)
        print('read data done!')
        print(data.shape)  # C, T, V, M

        graph = 'graph.ntu_rgb_d.Graph'
        # data = data.reshape((1,) + data.shape)[:, :, :, :, 0:1]
        data = data.reshape((1,) + data.shape)[:, :, :, :, :]

        import pickle
        _data_dir = '/code/2s-AGCN/data/data/ntu_sgn/processed_data'
        with open(_data_dir + '/NTU_CV_test.pkl', 'rb') as f:
            data1 = pickle.load(f)
        with open(_data_dir + '/NTU_CV_test_label.pkl', 'rb') as f:
            data2 = pickle.load(f)
        data = (data1[194].reshape(1, 300, 2, 25, 3)).swapaxes(1, -1).swapaxes(2, -1)  # noqa
        data = (data1[329].reshape(1, 300, 2, 25, 3)).swapaxes(1, -1).swapaxes(2, -1)  # noqa
        data = (data1[330].reshape(1, 300, 2, 25, 3)).swapaxes(1, -1).swapaxes(2, -1)  # noqa
        data = (data1[332].reshape(1, 300, 2, 25, 3)).swapaxes(1, -1).swapaxes(2, -1)  # noqa
        data = (data1[333].reshape(1, 300, 2, 25, 3)).swapaxes(1, -1).swapaxes(2, -1)  # noqa
        data = (data1[335].reshape(1, 300, 2, 25, 3)).swapaxes(1, -1).swapaxes(2, -1)  # noqa
        data = (data1[366].reshape(1, 300, 2, 25, 3)).swapaxes(1, -1).swapaxes(2, -1)  # noqa
        data = (data1[403].reshape(1, 300, 2, 25, 3)).swapaxes(1, -1).swapaxes(2, -1)  # noqa

        visualize_3dskeleton_in_matplotlib(data, graph=graph, is_3d=True, speed=0.1)  # noqa

        print(1)

        # # T, VC
        # data_i = data.transpose([0, 2, 3, 1, 4]).reshape((data.shape[2], 25*3))
        # # T-1, VC
        # data_j = data_i[1:] - data_i[:-1]
        # # T-1 (l2 normed values)
        # data_k = np.linalg.norm(data_j, axis=1)
        # # T-1 (l2 normed and shifted to mid value range)
        # data_l = abs(data_k - (data_k.max() - data_k.min())/2)
        # # T-1, VC
        # data_m = np.expand_dims(np.cumsum(data_l), -1)
        # data_n = np.expand_dims(np.cumsum(data_k), -1)
        # kmeans1 = KMeans(n_clusters=20, random_state=0).fit(data_m)
        # kmeans2 = KMeans(n_clusters=20, random_state=0).fit(data_n)
        # center1 = kmeans1.labels_
        # center2 = kmeans2.labels_

        # print(center1, center2)

        # c = 0
        # i_mem = -1
        # idx1 = []
        # label_map1 = {i: -1 for i in range(20)}
        # for idx, i in enumerate(center1):
        #     if i != i_mem:
        #         label_map1[i] = c
        #         i_mem = i
        #         c += 1
        #         idx1.append(idx)

        # c = 0
        # i_mem = -1
        # idx2 = []
        # label_map2 = {i: -1 for i in range(20)}
        # for idx, i in enumerate(center2):
        #     if i != i_mem:
        #         label_map2[i] = c
        #         i_mem = i
        #         c += 1
        #         idx2.append(idx)

        # _center1 = center1*0 - 1
        # for k, v in label_map1.items():
        #     _center1[center1 == k] = v
        # center1 = _center1

        # _center2 = center2*0 - 1
        # for k, v in label_map2.items():
        #     _center2[center2 == k] = v
        # center2 = _center2

        # axs[0].set_title('Normed + Shifted')
        # axs[0].plot(data_l, 'b.-')
        # for i in range(len(idx1)-1):
        #     axs[0].axvspan(idx1[i], idx1[i+1],
        #                    facecolor='b' if i % 2 == 0 else 'g',
        #                    alpha=0.3)
        # axs[0].plot(center1/20, 'r^-')
        # axs[1].set_title('Normed')
        # axs[1].plot(data_k, 'b.-')
        # for i in range(len(idx2)-1):
        #     axs[1].axvspan(idx2[i], idx2[i+1],
        #                    facecolor='b' if i % 2 == 0 else 'g',
        #                    alpha=0.3)
        # axs[1].plot(center2/20, 'r^-')
        # plt.show(block=False)
        # plt.pause(200)
        # axs[0].cla()
        # axs[1].cla()

    # data_i = data.transpose([0, 2, 3, 1, 4]).reshape((data.shape[2], 25*3))
    # data_j = data_i[1:] - data_i[:-1]
    # data_k = np.linspace(0, 1, num=data_j.shape[0])
    # data_k *= (np.max(data_j) - np.min(data_j))
    # data_j = (data_j - np.min(data_j, axis=1, keepdims=True)) / \
    #     (np.max(data_j, axis=1, keepdims=True) -
    #      np.min(data_j, axis=1, keepdims=True))
    # data_j = np.linalg.norm(data_j, axis=1, keepdims=True)
    # data_j = np.concatenate([data_j, np.expand_dims(data_k, axis=-1)], axis=1)
    # # data_j = np.concatenate([data_i[1:], data_j], axis=1)
    # # data_j = np.concatenate([np.clip(1/data_j, 1e-8, 1e8), data_j], axis=1)
    # kmeans = KMeans(n_clusters=20, random_state=0).fit(data_j)
    # print(kmeans.labels_)

    # .reshape((1,71,25,3,1)).transpose([0,3,1,2,4]).shape

    # data = np.concatenate([data for _ in range(100)], 2)
    # visualize_3dskeleton_in_matplotlib(data, graph=graph, is_3d=True)

    # draw_skeleton(data)

    # import plotly.graph_objects as go
    # import numpy as np
    # import plotly.io as pio
    # print(pio.renderers)
    # pio.renderers.default = 'browser'

    # # Helix equation
    # t = np.linspace(0, 10, 50)
    # x, y, z = np.cos(t), np.sin(t), t

    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
    #                                    mode='markers')])
    # fig.update_layout(
    #     title={
    #         'text': "Plot Title",
    #         'y': 0.9,
    #         'x': 0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'
    #     },
    #     font=dict(
    #         family="Courier New, monospace",
    #         size=25,
    #         color="DarkGreen"
    #     )
    # )
    # fig.show()
