import argparse
import json
import os
import numpy as np
import torch
import yaml
import time

from data_gen.ntu_gendata import (
    read_skeleton_filter,
    get_nonzero_std,
    training_cameras,
    training_subjects,
    max_frame,
    max_body_true,
    max_body_kinect,
    num_joint
)
from data_gen.preprocess import pre_normalization
from main import get_parser, import_class, init_seed
from visualize_ntu_skel import draw_skeleton


MAPPING = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
}


def get_datasplit_and_labels(
        data_path: str,
        ignored_sample_path: str = None,
        benchmark: str = 'xview',
        part: str = 'eval'):

    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    return sample_name, sample_label


def read_xyz(file, max_body=4, num_joint=25):
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data  # M, T, V, C


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
            self.data[:, 1:, :, :] = self.data[:, 0:-2, :, :]
            self.data[:, -2:, :, :] = data

    def select_skeletons(self, num_skels: int = 2) -> np.ndarray:
        """Select the `num_skels` most active skeletons. """
        # select two max energy body
        energy = np.array([get_nonzero_std(x) for x in self.data])
        index = energy.argsort()[::-1][0:num_skels]
        return self.data[index]  # m', T, V, C

    def normalize_data(self, data: np.ndarray) -> None:
        if data.ndim < 4 or data.ndim > 5:
            raise ValueError("Dimension not supported...")
        if data.ndim == 4:
            data = np.expand_dims(data, axis=0)
        data = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M
        data = pre_normalization(data)
        data = np.transpose(data, [0, 4, 2, 3, 1])  # N, M, T, V, C
        return data

    def select_skeletons_and_normalize_data(self,
                                            num_skels: int = 2) -> np.ndarray:
        data = self.select_skeletons(num_skels=num_skels)
        return self.normalize_data(data)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--data_path',
        default='./data/data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument(
        '--ignored_sample_path',
        default='./data/data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument(
        '--model_path',
        default='./data/model/211116110001/')
    parser.add_argument(
        '--model_config',
        default='./data/model/211116110001/config.yaml')
    parser.add_argument(
        '--out_folder',
        default='./data/data/ntu/')

    # load arg form config file ------------------------------------------------
    p = parser.parse_args()
    with open(p.model_config, 'r') as f:
        default_arg = yaml.safe_load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print(f'WRONG ARG: {k}')
            assert (k in key)
    parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    init_seed(0)

    # Data processor -----------------------------------------------------------
    DataProc = DataPreprocessor(num_joint, max_frame)

    # Prepare model ------------------------------------------------------------
    Model = import_class(arg.model)
    AAGCN = Model(**arg.model_args)
    AAGCN.eval()

    weight_file = [i for i in os.listdir(arg.model_path) if '.pt' in i]
    weight_file = os.path.join(arg.model_path, weight_file[0])
    weights = torch.load(weight_file)
    AAGCN.load_state_dict(weights)

    # Loop data ----------------------------------------------------------------
    for b in ['xsub', 'xview']:
        # for p in ['train', 'val']:
        for p in ['val']:
            print(f"Benchmark : {b}")
            print(f"Datasplit : {p}")

            # 0. Get all the relevant data and labels. -------------------------
            # Get the list of filenames and the labels from the dataset.
            sample_names, sample_labels = get_datasplit_and_labels(
                arg.data_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)

            # Loop through the recorded sequences.
            # Each sequence is in a file.
            for sample_name, sample_label in zip(sample_names, sample_labels):
                print(f"Processing : {sample_name}")

                # M, T, V, C
                data = read_xyz(os.path.join(arg.data_path, sample_name),
                                max_body=max_body_kinect,
                                num_joint=num_joint)

                DataProc.clear_data_array()

                start = time.time()

                # 1. Read raw frames. ------------------------------------------
                # Loop through the sequence.
                # Each sequence will be gradually added into the data processor.
                # This mimics the real setting where the frame is
                # continously fed into the system.
                for i in range(data.shape[1]):

                    # 2. Batch frames to fixed length.
                    DataProc.append_data(data[:, i:i+1, :, :])

                    # draw_skeleton(
                    #     data=np.transpose(data[:, i:i+1, :, :], [3, 1, 2, 0]),
                    #     action=MAPPING[sample_label+1]
                    # )

                    # 3. Normalization.
                    input_data = DataProc.select_skeletons_and_normalize_data(
                        max_body_true)

                    # N, C, T, V, M
                    input_data = np.transpose(input_data, [0, 4, 2, 3, 1])

                    # 4. Inference.
                    with torch.no_grad():
                        output = AAGCN(torch.Tensor(input_data))
                        _, predict_label = torch.max(output, 1)

                    if i % 10 == 0:
                        t = f"{(time.time() - start)/10:04.2f}s"
                        pred = f"{predict_label.item()}"
                        targ = f"{sample_label}"
                        print(f"{i:03d} :: {targ} :: {pred} :: {t}")
                        start = time.time()

                    # print(output.numpy().tolist())
                    # print(f"{sample_label} :: {predict_label.item()}")

                    # 5. View sequence + predicted action + GT action.
