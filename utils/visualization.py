import numpy as np
import matplotlib.figure as figure
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional


def import_class(name: str):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def visualize_3dskeleton_in_matplotlib(data: np.ndarray,
                                       graph: Optional[str] = None,
                                       is_3d: bool = False,
                                       speed: float = 0.01,
                                       text_per_t: Optional[List[str]] = None,
                                       fig: Optional[figure.Figure] = None,
                                       mode: str = 'ntu'):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    assert mode in ['ntu', 'openpose']

    # for batch_idx, (data, label) in enumerate(loader):
    N, C, T, V, M = data.shape

    plt.ion()
    if fig is None:
        fig = plt.figure()
        close_fig = True
    else:
        close_fig = False
    if len(fig.axes) == 0:
        if is_3d:
            # from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]

    if graph is None:
        p_type = ['b.', 'g.', 'r.', 'c.',
                  'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
        pose = [
            ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0]
            for m in range(M)
        ]
        ax.axis([-1, 1, -1, 1])
        for t in range(T):
            for m in range(M):
                pose[m].set_xdata(data[0, 0, t, :, m])
                pose[m].set_ydata(data[0, 1, t, :, m])
            fig.canvas.draw()
            plt.pause(speed)
    else:
        p_type = ['b-', 'g-', 'r-', 'c-',
                  'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
        import sys
        from os import path
        sys.path.append(
            path.dirname(
                path.dirname(
                    path.dirname(
                        path.abspath(__file__)
                    )
                )
            )
        )
        G = import_class(graph)()
        edge = G.inward
        pose = []
        for m in range(M):
            a = []
            for i in range(len(edge)):
                if mode == 'ntu':
                    if i in [8, 9, 10]:
                        c = 'r-'
                    elif i in [4, 5, 6]:
                        c = 'm-'
                    elif i in [16, 17, 18]:
                        c = 'c-'
                    elif i in [12, 13, 14]:
                        c = 'k-'
                    else:
                        c = p_type[m]
                elif mode == 'openpose':
                    if i in [2, 3]:
                        c = 'r-'
                    elif i in [5, 6]:
                        c = 'm-'
                    elif i in [9, 10]:
                        c = 'c-'
                    elif i in [12, 13]:
                        c = 'k-'
                    else:
                        c = p_type[m]
                n = 3 if is_3d else 2
                a.append(ax.plot(np.zeros(n), np.zeros(n), c)[0])
            pose.append(a)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.view_init(elev=90, azim=-90)
        ax.view_init(elev=-90, azim=0, roll=90)
        # ax.view_init(elev=-60, azim=5, roll=90)
        # ax.view_init(elev=0, azim=0, roll=90)
        # ax.view_init(elev=-30, azim=10, roll=90)
        # ax.set_xlim(-1, 4)
        # ax.set_ylim(-2, 3)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        if is_3d:
            ax.set_zlim3d(1, 5)
        for t in range(T):
            visualize_3dskeleton_in_matplotlib_step(
                data, t, pose, edge, is_3d, speed, text_per_t, fig)
    if close_fig:
        plt.close()
    return pose, edge, fig


def visualize_3dskeleton_in_matplotlib_step(
    data: np.ndarray,
    t: int,
    pose: np.ndarray,
    edge: List[Tuple[int, int]],
    is_3d: bool = False,
    speed: float = 0.01,
    text_per_t: Optional[List[str]] = None,
    fig: Optional[figure.Figure] = None,
):
    N, C, T, V, M = data.shape
    for m in range(M):
        for i, (v1, v2) in enumerate(edge):
            x1 = data[0, :2, t, v1, m]
            x2 = data[0, :2, t, v2, m]
            if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                if is_3d:
                    pose[m][i].set_3d_properties(
                        data[0, 2, t, [v1, v2], m])
    if text_per_t is not None:
        fig.suptitle(
            f'Frame : {t} >>> Action : {text_per_t[t]}', fontsize=16)
    else:
        fig.suptitle(f'Frame : {t}', fontsize=16)
    fig.canvas.draw()
    plt.pause(speed)
