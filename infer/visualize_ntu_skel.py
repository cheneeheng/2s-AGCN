"""
Based on :
https://programmer.ink/think/how-to-realize-ntu-rgb-d-skeleton-visualization-gracefully-with-matplotlib.html
"""

import numpy as np

from plotly import graph_objects as go
from time import sleep


JOINT_COLOR = [
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 85),
    (255, 0, 170),
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
    (255, 255, 0),
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
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[body[0]]}',
            ),
        ),
        go.Scatter3d(
            x=p[0, f, rightarm, m],
            y=p[1, f, rightarm, m],
            z=p[2, f, rightarm, m],
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[rightarm[0]]}',
            ),
        ),
        go.Scatter3d(
            x=p[0, f, leftarm, m],
            y=p[1, f, leftarm, m],
            z=p[2, f, leftarm, m],
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[leftarm[0]]}',
            ),
        ),
        go.Scatter3d(
            x=p[0, f, righthand, m],
            y=p[1, f, righthand, m],
            z=p[2, f, righthand, m],
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[righthand[0]]}',
            ),
        ),
        go.Scatter3d(
            x=p[0, f, lefthand, m],
            y=p[1, f, lefthand, m],
            z=p[2, f, lefthand, m],
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[lefthand[0]]}',
            ),
        ),
        go.Scatter3d(
            x=p[0, f, rightleg, m],
            y=p[1, f, rightleg, m],
            z=p[2, f, rightleg, m],
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[rightleg[0]]}',
            ),
        ),
        go.Scatter3d(
            x=p[0, f, leftleg, m],
            y=p[1, f, leftleg, m],
            z=p[2, f, leftleg, m],
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[leftleg[0]]}',
            ),
        ),
        go.Scatter3d(
            x=p[0, f, rightfeet, m],
            y=p[1, f, rightfeet, m],
            z=p[2, f, rightfeet, m],
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[rightfeet[0]]}',
            ),
        ),
        go.Scatter3d(
            x=p[0, f, leftfeet, m],
            y=p[1, f, leftfeet, m],
            z=p[2, f, leftfeet, m],
            mode='lines',
            line=dict(
                width=5,
                color=f'rgb{JOINT_COLOR[leftfeet[0]]}',
            ),
        ),
        # Joints
        go.Scatter3d(
            x=p[0, f, :, m],
            y=p[1, f, :, m],
            z=p[2, f, :, m],
            mode='markers',
            marker=dict(
                size=5,
                color=[f'rgb{c}' for c in JOINT_COLOR],
                # colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )
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


if __name__ == '__main__':

    file_name = r'/workspaces/2s-AGCN/data/data/nturgbd_raw/nturgb+d_skeletons/S001C001P001R001A009.skeleton'  # noqa
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

    print('read data done!')
    print(data.shape)  # C, T, V, M
    draw_skeleton(data)

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
