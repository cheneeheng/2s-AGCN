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

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from matplotlib.animation import FuncAnimation

from plotly import graph_objects as go
from time import sleep


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
    rightarm = np.array([4, 3, 2, 1])
    leftarm = np.array([7, 6, 5, 1])
    rightleg = np.array([11, 10, 9, 8])
    leftleg = np.array([14, 13, 12, 8])
    body = np.array([0, 1, 8])    # body
    print(f"{p[0, f, 0, m]}, {p[1, f, 0, m]}, {p[2, f, 0, m]}")

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
            marker=dict(size=4),
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
            marker=dict(size=4),
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
            marker=dict(size=4),
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
            marker=dict(size=4),
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
            marker=dict(size=4),
        ),
    ]


def draw_single_skeleton(data, f, pause_sec=10, action=""):

    camera = dict(
        up=dict(x=0, y=0, z=1),
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
        print("New Frame")
        _data = _plot_skel(data, f)
        fig.update(data=_data)
        # fig.update_layout(scene_camera=camera,
        #                   showlegend=False,
        #                   margin=dict(l=0, r=0, b=0, t=0),
        #                   yaxis=dict(range=[-1, 1]),
        #                   xaxis=dict(range=[-1, 1]))
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


joint_file = r'/code/2s-AGCN/data/data/openpose_b25_j15_ntu/xview/val_data_joint.npy'  # noqa
# joint_file = r'/code/2s-AGCN/data/data/openpose_b25_j15_ntu/xview/val_data_joint_100.npy'  # noqa
label_file = r'/code/2s-AGCN/data/data/openpose_b25_j15_ntu/xview/val_label.pkl'  # noqa

# np.where(np.array(label)==22)[0][0]
# data: N C T V M
with open(label_file, 'rb') as f:
    sample_name, label = pickle.load(f, encoding='latin1')
data = np.load(joint_file)
# data = np.load(joint_file, mmap_mode='r')

app = dash.Dash(__name__, update_title=None)
app.layout = html.Div(
    [dcc.Input(id='text', type="text", value=0),
     dcc.Graph(id='graph'),
     dcc.Interval(id="interval", interval=0.5 * 1000)])


@app.callback(Output('graph', 'figure'),
              [Input('interval', 'n_intervals'),
               Input('text', 'value')])
def update_data(n_intervals, aid):

    idx = np.where(np.array(label) == int(aid))[0][0]

    if n_intervals is None:
        return [None]

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=0.75, z=0.25)
        # eye=dict(x=0, y=0, z=0.)
    )

    fig = go.Figure(_plot_skel(data[idx][:, ::3, :, :], n_intervals % 100))
    fig.update_layout(scene_camera=camera,
                      showlegend=False,
                      margin=dict(l=0, r=0, b=0, t=0),
                      autosize=False,
                      paper_bgcolor='black',
                      width=600,
                      height=600,
                      xaxis_fixedrange=True,
                      yaxis_fixedrange=True,
                      scene=dict(
                          #   aspectratio=dict(x=1, y=1.5, z=5),
                          aspectratio=dict(x=1, y=1, z=1),
                          yaxis=dict(range=[-1, 5],
                                     #  showticklabels=False,
                                     #  showgrid=False,
                                     #  zeroline=False,
                                     #  visible=False,
                                     ),
                          xaxis=dict(range=[-1, 1],
                                     #  showticklabels=False,
                                     #  showgrid=False,
                                     #  zeroline=False,
                                     #  visible=False,
                                     ),
                          zaxis=dict(range=[-1.5, 1],
                                     #  showticklabels=False,
                                     #  showgrid=False,
                                     #  zeroline=False,
                                     #  visible=False,
                                     )
                      ))
    print('Action :', label[idx], n_intervals)

    return fig


if __name__ == '__main__':
    app.run_server()
