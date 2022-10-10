import numpy as np
import os

from typing import Tuple, List, Sequence

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
        ax.axis('off')
        subplot_nodes(dots, ax)
        subplot_bones(chains, ax)
    # plt.show()
