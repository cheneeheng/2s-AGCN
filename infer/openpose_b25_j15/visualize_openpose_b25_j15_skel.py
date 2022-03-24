"""
Based on :
https://programmer.ink/think/how-to-realize-ntu-rgb-d-skeleton-visualization-gracefully-with-matplotlib.html
"""

import typing as tp
import dash
from dash import dcc
from dash import html
import numpy as np

from dash.dependencies import Input, Output

import os
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from plotly import graph_objects as go
from time import sleep

from data_gen.preprocess import pre_normalization
from utils.visualization import visualize_3dskeleton_in_matplotlib


def read_xyz(file, max_body=4, num_joint=25):
    skel_data = np.loadtxt(file, delimiter=',')
    data = np.zeros((max_body, 1, num_joint, 3))
    for m, body_joint in enumerate(skel_data):
        for j in range(0, len(body_joint), 3):
            if m < max_body and j//3 < num_joint:
                data[m, 0, j//3, :] = [-body_joint[j],
                                       -body_joint[j+2],
                                       -body_joint[j+1]]
            else:
                pass
    data = np.swapaxes(data, 0, 3)/1000.0   # M, T, V, C > C, T, V, M
    return data


if __name__ == '__main__':
    # data : C,T,V,M
    print("START")

    joint_path = './data/data_tmp/220324153743'
    joint_files = [os.path.join(joint_path, i)
                   for i in sorted(os.listdir(joint_path))]
    data = []
    data_raw = []
    data_path = []
    for joint_file in joint_files:
        data_i = read_xyz(joint_file, max_body=4, num_joint=25)  # C, T, V, M
        data_raw.append(np.array(data_i))
        data_i = np.expand_dims(data_i, axis=0)
        # data_i = pre_normalization(
        #     data_i,
        #     zaxis=[8, 1],
        #     xaxis=[2, 5],
        #     verbose=False,
        #     tqdm=False
        # )
        data_i = data_i[0]
        data_i = np.stack(data[-3:] + [data_i], axis=0)
        data_i = np.mean(data_i, axis=0)
        data.append(data_i)
        data_path.append(joint_file)
    data = np.concatenate(data, 1)

    # np.save('./data/data_tmp/220317182701.npy', data)
    # data = np.load('./data/data_tmp/220317182701.npy')

    data = data[:, 100:, :, :2]  # c,t,v,m
    data = pre_normalization(
        np.expand_dims(data, axis=0),
        zaxis=[8, 1],
        xaxis=[2, 5],
        # xaxis=None,
        # zaxis=None,
        verbose=False,
        tqdm=False
    )[0]

    # data = np.load(
    #     "./data/data/openpose_b25_j15_ntu/xview/val_data_joint_100.npy")
    # label_file = r'/code/2s-AGCN/data/data/openpose_b25_j15_ntu/xview/val_label.pkl'  # noqa
    # with open(label_file, 'rb') as f:
    #     sample_name, label = pickle.load(f, encoding='latin1')
    # print(np.array(label).max())

    # idx = 50
    # data = data[idx]
    # label = label[idx]
    # print(label)

    # data_mem = data.copy()
    # for i in range(data.shape[1]):
    #     data_i = pre_normalization(
    #         np.expand_dims(data_mem[:, i:i+300, :, :], axis=0),
    #         zaxis=[8, 1],
    #         xaxis=[2, 5],
    #         verbose=False,
    #         tqdm=False
    #     )
    #     print(f"Idx : {i}")

    with open(f'infer/openpose_b25_j15/result_{joint_path.split("/")[-1]}_ma5_100pads.txt', "r") as f:  # noqa
        text = f.readlines()

    graph = 'graph.openpose_b25_j15.Graph'
    data = data.reshape((1,) + data.shape)
    visualize_3dskeleton_in_matplotlib(
        data, graph=graph, is_3d=True, speed=0.01, text_per_t=text)


# JOINT_COLOR = [
#     (255, 0, 0),
#     (255, 0, 0),
#     (255, 0, 85),
#     # (255, 0, 170),
#     (255, 0, 255),
#     (170, 255, 0),
#     (170, 255, 0),
#     (85, 255, 0),
#     (85, 255, 0),
#     (255, 85, 0),
#     (255, 85, 0),
#     (255, 170, 0),
#     (255, 170, 0),
#     (0, 170, 255),
#     (0, 85, 255),
#     (0, 0, 255),
#     (0, 0, 255),
#     (0, 255, 85),
#     (0, 255, 170),
#     (0, 255, 255),
#     (0, 255, 255),
#     (255, 0, 0),
#     (0, 255, 0),
#     (0, 255, 0),
#     # (255, 255, 0),
#     (255, 191, 0),
#     (255, 255, 0),
# ]


# def _plot_skel(p, f, m=0):

#     # Determine which nodes are connected as bones according
#     #  to NTU skeleton structure
#     # Note that the sequence number starts from 0 and needs to be minus 1
#     rightarm = np.array([4, 3, 2, 1])
#     leftarm = np.array([7, 6, 5, 1])
#     rightleg = np.array([11, 10, 9, 8])
#     leftleg = np.array([14, 13, 12, 8])
#     body = np.array([0, 1, 8])    # body
#     # print(f"{p[0, f, 0, m]}, {p[1, f, 0, m]}, {p[2, f, 0, m]}")

#     return [
#         # Bones
#         go.Scatter3d(
#             x=p[0, f, body, m],
#             y=p[1, f, body, m],
#             z=p[2, f, body, m],
#             mode='lines+markers',
#             line=dict(
#                 width=5,
#                 color=f'rgb{JOINT_COLOR[8]}',
#             ),
#             marker=dict(size=4),
#         ),
#         go.Scatter3d(
#             x=p[0, f, rightarm, m],
#             y=p[1, f, rightarm, m],
#             z=p[2, f, rightarm, m],
#             mode='lines+markers',
#             line=dict(
#                 width=5,
#                 color=f'rgb{JOINT_COLOR[3]}',
#             ),
#             marker=dict(size=4),
#         ),
#         go.Scatter3d(
#             x=p[0, f, leftarm, m],
#             y=p[1, f, leftarm, m],
#             z=p[2, f, leftarm, m],
#             mode='lines+markers',
#             line=dict(
#                 width=5,
#                 color=f'rgb{JOINT_COLOR[leftarm[0]]}',
#             ),
#             marker=dict(size=4),
#         ),
#         go.Scatter3d(
#             x=p[0, f, rightleg, m],
#             y=p[1, f, rightleg, m],
#             z=p[2, f, rightleg, m],
#             mode='lines+markers',
#             line=dict(
#                 width=5,
#                 color=f'rgb{JOINT_COLOR[rightleg[0]]}',
#             ),
#             marker=dict(size=4),
#         ),
#         go.Scatter3d(
#             x=p[0, f, leftleg, m],
#             y=p[1, f, leftleg, m],
#             z=p[2, f, leftleg, m],
#             mode='lines+markers',
#             line=dict(
#                 width=5,
#                 color=f'rgb{JOINT_COLOR[leftleg[0]]}',
#             ),
#             marker=dict(size=4),
#         ),
#     ]


# def draw_single_skeleton(data, f, pause_sec=10, action=""):

#     camera = dict(
#         up=dict(x=0, y=0, z=1),
#         center=dict(x=0, y=0, z=0),
#         eye=dict(x=0, y=0, z=3.0)
#     )

#     fig = go.Figure(data=_plot_skel(data, f))

#     fig.update_layout(scene_camera=camera,
#                       autosize=False,
#                       showlegend=False,
#                       width=600,
#                       height=600,
#                       margin=dict(l=0, r=0, b=0, t=0),
#                       yaxis=dict(range=[-1, 1]),
#                       xaxis=dict(range=[-1, 1]))

#     return fig


# def draw_skeleton(data, pause_sec=10, action=""):

#     data[0, :, :, :] *= -1
#     data[2, :, :, :] *= -1

#     camera = dict(
#         up=dict(x=0, y=1, z=0),
#         center=dict(x=0, y=0, z=0),
#         eye=dict(x=1.25, y=1.25, z=2.25)
#     )

#     f = 0
#     fig = go.Figure(_plot_skel(data, f))
#     fig.update_layout(scene_camera=camera,
#                       showlegend=False,
#                       margin=dict(l=0, r=0, b=0, t=0),
#                       yaxis=dict(range=[-1, 1]),
#                       xaxis=dict(range=[-1, 1]))
#     fig.show()

#     for f in range(1, 50):
#         print("New Frame")
#         _data = _plot_skel(data, f)
#         fig.update(data=_data)
#         # fig.update_layout(scene_camera=camera,
#         #                   showlegend=False,
#         #                   margin=dict(l=0, r=0, b=0, t=0),
#         #                   yaxis=dict(range=[-1, 1]),
#         #                   xaxis=dict(range=[-1, 1]))
#         fig.show()
#         sleep(pause_sec)


# def draw_skeleton_offline(data, pause_sec=10, action=""):

#     camera = dict(
#         up=dict(x=0, y=0, z=1),
#         center=dict(x=0, y=0, z=0),
#         eye=dict(x=0, y=3.0, z=0)
#     )
#     fig = go.Figure(
#         data=_plot_skel(data, 0),
#         layout=go.Layout(
#             updatemenus=[dict(type="buttons",
#                               buttons=[dict(label="Play",
#                                             method="animate",
#                                             args=[None])])]),
#         frames=[go.Frame(data=_plot_skel(data, k))
#                 for k in range(data.shape[1])]
#     )
#     fig.update_layout(scene_camera=camera,
#                       showlegend=False,
#                       margin=dict(l=0, r=0, b=0, t=0),
#                       yaxis=dict(range=[-1, 1]),
#                       xaxis=dict(range=[-1, 1]))
#     fig.show(renderer='notebook_connected')


# # # joint_file = r'/code/2s-AGCN/data/data/openpose_b25_j15_ntu/xview/val_data_joint.npy'  # noqa
# # joint_file = r'/code/2s-AGCN/data/data/openpose_b25_j15_ntu/xview/val_data_joint_100.npy'  # noqa
# # label_file = r'/code/2s-AGCN/data/data/openpose_b25_j15_ntu/xview/val_label.pkl'  # noqa

# # # np.where(np.array(label)==22)[0][0]
# # # data: N C T V M
# # with open(label_file, 'rb') as f:
# #     sample_name, label = pickle.load(f, encoding='latin1')
# # data = np.load(joint_file)
# # # data = np.load(joint_file, mmap_mode='r')

# joint_path = './data/data_tmp/220317182701'
# joint_files = [os.path.join(joint_path, i)
#                for i in sorted(os.listdir(joint_path))]
# data = []
# data_raw = []
# data_path = []
# for joint_file in joint_files:
#     data_i = read_xyz(joint_file, max_body=4, num_joint=25)  # C, T, V, M
#     data_raw.append(np.array(data_i))
#     data_i = np.expand_dims(data_i, axis=0)
#     data.append(data_i[0])
#     data_path.append(joint_file)
# data = np.concatenate(data, 1)
# # np.save('./data/data_tmp/220317182701.npy', data)

# # data = np.load('./data/data_tmp/220317182701.npy')
# data = np.expand_dims(data, axis=0)[:, :, 1700:, :, 0:1]
# data = pre_normalization(data,
#                          zaxis=[8, 1],
#                          #  zaxis=None,
#                          #  zaxis2=None,
#                          #  xaxis=None,
#                          xaxis=[2, 5],
#                          verbose=False,
#                          tqdm=False)
# # data = pre_normalization(data,
# #                          #  zaxis=None,
# #                          zaxis2=[8, 1],
# #                          xaxis=[2, 5],
# #                          verbose=False,
# #                          tqdm=False)
# # data = pre_normalization(data,
# #                          xaxis=[2, 5],
# #                          zaxis=[8, 1],
# #                          verbose=False,
# #                          tqdm=False)
# data = data[0]

# app = dash.Dash(__name__, update_title=None)
# app.layout = html.Div(
#     [dcc.Input(id='text', type="text", value=0),
#      dcc.Graph(id='graph'),
#      dcc.Interval(id="interval", interval=0.5 * 1000)])


# @ app.callback(Output('graph', 'figure'),
#                [Input('interval', 'n_intervals'),
#                Input('text', 'value')])
# def update_data(n_intervals, aid):

#     idx = np.where(np.array(label) == int(aid))[0][0]

#     if n_intervals is None:
#         return [None]

#     camera = dict(
#         up=dict(x=0, y=0, z=1),
#         center=dict(x=0, y=0, z=0),
#         eye=dict(x=1.25, y=0.75, z=0.25)
#         # eye=dict(x=0, y=0, z=0.)
#     )

#     # data_i = read_xyz(
#     #     joint_files[n_intervals % len(joint_files)],
#     #     max_body=4, num_joint=25)  # C, T, V, M
#     # data_i = np.expand_dims(data_i, axis=0)
#     # data_i = pre_normalization(data_i, zaxis=[8, 1], xaxis=[2, 5],
#     #                            verbose=False, tqdm=False)[0]
#     # fig = go.Figure(data=_plot_skel(data_i, 0, 0))
#     fig = go.Figure(_plot_skel(data, (n_intervals*3) % (data.shape[1]//3)))
#     # print(n_intervals % data.shape[1],
#     #       joint_files[n_intervals % data.shape[1]])

#     # fig = go.Figure(_plot_skel(data[idx][:, ::3, :, :], n_intervals % 100))

#     fig.update_layout(scene_camera=camera,
#                       showlegend=False,
#                       margin=dict(l=0, r=0, b=0, t=0),
#                       autosize=False,
#                       paper_bgcolor='black',
#                       width=600,
#                       height=600,
#                       xaxis_fixedrange=True,
#                       yaxis_fixedrange=True,
#                       scene=dict(
#                           #   aspectratio=dict(x=1, y=1.5, z=5),
#                           aspectratio=dict(x=1, y=1, z=1),
#                           yaxis=dict(range=[-1, 5],
#                                      #  showticklabels=False,
#                                      #  showgrid=False,
#                                      #  zeroline=False,
#                                      #  visible=False,
#                                      ),
#                           xaxis=dict(range=[-1, 1],
#                                      #  showticklabels=False,
#                                      #  showgrid=False,
#                                      #  zeroline=False,
#                                      #  visible=False,
#                                      ),
#                           zaxis=dict(range=[-1.5, 1],
#                                      #  showticklabels=False,
#                                      #  showgrid=False,
#                                      #  zeroline=False,
#                                      #  visible=False,
#                                      )
#                       ))
#     print('Action :', label[idx],
#           (n_intervals*3) % (data.shape[1]//3), data.shape[1]//3)

#     return fig


# if __name__ == '__main__':
#     # data : C,T,V,M
#     app.run_server()
