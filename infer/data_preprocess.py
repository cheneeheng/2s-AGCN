import numpy as np

from data_gen.ntu_gendata import get_nonzero_std
from data_gen.preprocess import pre_normalization


class DataPreprocessor(object):

    def __init__(self, num_joint=25, max_seq_length=300) -> None:
        super().__init__()
        self.num_joint = num_joint
        self.max_seq_length = max_seq_length
        self.data = None
        self.data_counter = 0
        self.clear_data_array()

    def clear_data_array(self) -> None:
        """
        Creates an empty/zero array of size (M,T,V,C).
        We assume that the input data can have up to 4 possible skeletons ids.
        """
        self.data = np.zeros((4,
                              self.max_seq_length,
                              self.num_joint,
                              3),
                             dtype=np.float32)
        self.data_counter = 0

    def append_data(self, data: np.ndarray) -> None:
        """Append data.

        Args:
            data (np.ndarray): (M, 1, V, C)
        """
        if self.data_counter < self.max_seq_length:
            self.data[:, self.data_counter:self.data_counter+1, :, :] = data
            self.data_counter += 1
        else:
            self.data[:, 0:-1, :, :] = self.data[:, 1:, :, :]
            self.data[:, -1:, :, :] = data

    def select_skeletons(self, num_skels: int = 2) -> np.ndarray:
        """Select the `num_skels` most active skeletons. """
        # select two max energy body
        energy = np.array([get_nonzero_std(x) for x in self.data])
        index = energy.argsort()[::-1][0:num_skels]
        return self.data[index]  # m', T, V, C

    def normalize_data(self, data: np.ndarray, verbose: bool = False) -> None:
        if data.ndim < 4 or data.ndim > 5:
            raise ValueError("Dimension not supported...")
        if data.ndim == 4:
            data = np.expand_dims(data, axis=0)
        data = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M
        data = pre_normalization(data, verbose=verbose, tqdm=False)
        data = np.transpose(data, [0, 4, 2, 3, 1])  # N, M, T, V, C
        return data

    def select_skeletons_and_normalize_data(self,
                                            num_skels: int = 2) -> np.ndarray:
        data = self.select_skeletons(num_skels=num_skels)
        return self.normalize_data(data)
