import numpy as np

from data_gen.ntu_gendata import get_nonzero_std


class DataPreprocessor(object):

    def __init__(self,
                 num_joint: int = 25,
                 max_seq_length: int = 300,
                 max_person: int = 4,
                 moving_avg: int = 1,
                 preprocess_fn=None) -> None:
        super().__init__()
        self.num_joint = num_joint
        self.max_seq_length = max_seq_length
        self.max_person = max_person
        self.data = None
        self.counter = 0  # for pre max_seq_length moving_avg calculation.
        self.moving_avg = moving_avg
        self.reset_data()
        if preprocess_fn is None:
            self.preprocess_fn = lambda x: x
        else:
            self.preprocess_fn = preprocess_fn

    def reset_data(self) -> None:
        """
        Creates an empty/zero array of size (M,T,V,C). We assume that
        the input data can have up to `max_person` possible skeletons ids.
        """
        self.data = np.zeros((self.max_person,
                              self.max_seq_length,
                              self.num_joint,
                              3),
                             dtype=np.float32)
        self.counter = 0

    def append_data(self, data: np.ndarray) -> None:
        """Append data.

        Args:
            data (np.ndarray): (M, 1, V, C)
        """
        M, T, V, C = data.shape
        if self.counter < self.max_seq_length:
            self.data[:M, self.counter:self.counter+1, :V, :C] = data
            self.counter += 1
            if self.moving_avg > 1 and self.counter > self.moving_avg - 1:
                data_i = np.mean(self.data[:, self.counter-self.moving_avg:self.counter, :, :], axis=1, keepdims=True)  # noqa
                self.data[:, self.counter-1:self.counter, :, :] = data_i
        else:
            self.data[:, :-1, :, :] = self.data[:, 1:, :, :]
            self.data[:M, -1:, :V, :C] = data
            if self.moving_avg > 1:
                data_i = np.mean(self.data[:, -self.moving_avg:, :, :], axis=1, keepdims=True)  # noqa
                self.data[:, -1:, :, :] = data_i

    def select_skeletons(self, num_skels: int = 2) -> np.ndarray:
        """Select the `num_skels` most active skeletons. """
        # select two max energy body
        energy = np.array([get_nonzero_std(x) for x in self.data])
        index = energy.argsort()[::-1][0:num_skels]
        return self.data[index]  # m', T, V, C

    def select_skeletons_and_normalize_data(self,
                                            num_skels: int = 2,
                                            sgn: bool = False) -> np.ndarray:
        data = self.select_skeletons(num_skels=num_skels)  # M, T, V, C
        if sgn:
            data = np.transpose(data, [1, 0, 2, 3])  # T, M, V, C
            data = data.reshape((data.shape[0], -1))  # T, MVC
            data = [data]  # N,T,MVC
            data, _, _ = self.preprocess_fn(data)  # N,'T, MVC
            return np.array(data, dtype=data[0].dtype)  # N, 'T, MVC
        else:
            if data.ndim < 4 or data.ndim > 5:
                raise ValueError("Dimension not supported...")
            if data.ndim == 4:
                data = np.expand_dims(data, axis=0)
            data = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M
            return self.preprocess_fn(data)  # N, C, T, V, M


class DataPreprocessorV2(DataPreprocessor):

    def __init__(self,
                 num_joint: int = 25,
                 max_seq_length: int = 300,
                 max_person: int = 4,
                 moving_avg: int = 1,
                 aagcn_normalize_fn=None,
                 sgn_preprocess_fn=None) -> None:
        super().__init__(num_joint, max_seq_length, max_person, moving_avg)
        if aagcn_normalize_fn is None:
            self.aagcn_normalize_fn = lambda x: x
        else:
            self.aagcn_normalize_fn = aagcn_normalize_fn
        if sgn_preprocess_fn is None:
            self.sgn_preprocess_fn = lambda x: (x, None)
        else:
            self.sgn_preprocess_fn = sgn_preprocess_fn

    def select_skeletons_and_normalize_data(
            self,
            num_skels: int = 2,
            aagcn_normalize: bool = False,
            sgn_preprocess: bool = False) -> np.ndarray:
        data = self.select_skeletons(num_skels=num_skels)  # M, T, V, C

        data = np.expand_dims(data, axis=0)  # N, M, T, V, C

        if aagcn_normalize:
            data = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M
            data = self.aagcn_normalize_fn(data)  # N, C, T, V, M

        if sgn_preprocess:
            if aagcn_normalize:
                data = np.transpose(data, [0, 2, 4, 3, 1])  # N, T, M, V, C
            else:
                data = np.transpose(data, [0, 2, 1, 3, 4])  # N, T, M, V, C
            data = data.reshape((*data.shape[0:2], -1))  # N, T, MVC
            output = self.sgn_preprocess_fn(data)  # N,'T, MVC
            data = output[0]
            data = np.array(data, dtype=data[0].dtype)  # N, 'T, MVC

        return data
