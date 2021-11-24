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


def draw_skeleton(data, pause_sec=10, action=""):

    data[0, :, :, :] *= -1
    data[2, :, :, :] *= -1

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

    def _plot_skel(p, i):
        return [
            # Bones
            go.Scatter3d(
                x=p[0, i, body, 0],
                y=p[1, i, body, 0],
                z=p[2, i, body, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[body[0]]}',
                ),
            ),
            go.Scatter3d(
                x=p[0, i, rightarm, 0],
                y=p[1, i, rightarm, 0],
                z=p[2, i, rightarm, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[rightarm[0]]}',
                ),
            ),
            go.Scatter3d(
                x=p[0, i, leftarm, 0],
                y=p[1, i, leftarm, 0],
                z=p[2, i, leftarm, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[leftarm[0]]}',
                ),
            ),
            go.Scatter3d(
                x=p[0, i, righthand, 0],
                y=p[1, i, righthand, 0],
                z=p[2, i, righthand, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[righthand[0]]}',
                ),
            ),
            go.Scatter3d(
                x=p[0, i, lefthand, 0],
                y=p[1, i, lefthand, 0],
                z=p[2, i, lefthand, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[lefthand[0]]}',
                ),
            ),
            go.Scatter3d(
                x=p[0, i, rightleg, 0],
                y=p[1, i, rightleg, 0],
                z=p[2, i, rightleg, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[rightleg[0]]}',
                ),
            ),
            go.Scatter3d(
                x=p[0, i, leftleg, 0],
                y=p[1, i, leftleg, 0],
                z=p[2, i, leftleg, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[leftleg[0]]}',
                ),
            ),
            go.Scatter3d(
                x=p[0, i, rightfeet, 0],
                y=p[1, i, rightfeet, 0],
                z=p[2, i, rightfeet, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[rightfeet[0]]}',
                ),
            ),
            go.Scatter3d(
                x=p[0, i, leftfeet, 0],
                y=p[1, i, leftfeet, 0],
                z=p[2, i, leftfeet, 0],
                mode='lines',
                line=dict(
                    width=5,
                    color=f'rgb{JOINT_COLOR[leftfeet[0]]}',
                ),
            ),
            # Joints
            go.Scatter3d(
                x=p[0, i, :, 0],
                y=p[1, i, :, 0],
                z=p[2, i, :, 0],
                mode='markers',
                marker=dict(
                    size=5,
                    color=[f'rgb{c}' for c in JOINT_COLOR],
                    # colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )
            ),
        ]

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=2.25)
    )

    i = 0
    fig = go.Figure(_plot_skel(data, i))
    fig.update_layout(scene_camera=camera,
                      showlegend=False,
                      margin=dict(l=0, r=0, b=0, t=0),
                      yaxis=dict(range=[-1, 1]),
                      xaxis=dict(range=[-1, 1]))
    fig.show()

    for i in range(1, 50):
        fig.update(data=_plot_skel(data, i))
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
