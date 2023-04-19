import os
import numpy as np
from tqdm import tqdm

from data_gen.ntu_gendata import read_xyz


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 15
num_joint_ntu = 25
max_frame = 300

# openpose : ntu
joint_mapping = {
    0: 4,
    1: 21,
    2: 9,
    3: 10,
    4: 11,
    5: 5,
    6: 6,
    7: 7,
    8: 1,
    9: 17,
    10: 18,
    11: 19,
    12: 13,
    13: 14,
    14: 15,
}

# original : new labels
label_mapping = {
    1: 0,
    2: 0,
    8: 1,
    9: 2,
    27: 3,
    31: 4,
    43: 5,
    56: 6,
    59: 7,
    60: 8
}

output_folder = 'data/data_tmp/S003C001P002R001A031_15j'
os.makedirs(output_folder, exist_ok=True)

filename = 'data/data_tmp/S003C001P002R001A031.skeleton'
action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

# C, T, V, M
data = read_xyz(filename, max_body=max_body_kinect, num_joint=num_joint_ntu)

# C, T, V, M
data_new = np.zeros_like(data)[:, :, :num_joint, :]
for new_id, old_id in joint_mapping.items():
    data_new[:, :, new_id, :] = data[:, :, old_id-1, :]
data = data_new

# T, M, V, C
data = data.transpose(1, 3, 2, 0)
# T, M, VC
t, m, v, c = data.shape
data = data.reshape(t, m, v*c)

for i, data_i in enumerate(tqdm(data)):
    np.savetxt(os.path.join(output_folder, f"{i:012}.txt"),
               data_i,
               delimiter=',')
