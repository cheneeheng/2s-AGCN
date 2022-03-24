from .rotation import *
from tqdm import tqdm


def __verbose(x, out=None, verbose=False):
    if verbose:
        print(out)
        return tqdm(x)
    else:
        return x


def pre_normalization(data,
                      zaxis=[0, 1],
                      zaxis2=None,
                      xaxis=[8, 4],
                      pad=True,
                      center=True,
                      center_firstframe=False,
                      verbose=False,
                      tqdm=True):

    assert center_firstframe != center

    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    if verbose:
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0:
                print(f'Seq {i_s} has no skeleton')

    if pad:
        s_list = __verbose(s, out='pad the null frames with the previous frames', verbose=verbose)  # noqa
        for i_s, skeleton in enumerate(s_list):  # pad
            if skeleton.sum() == 0:
                continue
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                if person[0].sum() == 0:
                    index = (person.sum(-1).sum(-1) != 0)
                    tmp = person[index].copy()
                    person *= 0
                    person[:len(tmp)] = tmp
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        if person[i_f:].sum() == 0:
                            rest = len(person) - i_f
                            num = int(np.ceil(rest / i_f))
                            pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]  # noqa
                            s[i_s, i_p, i_f:] = pad
                            break

    if center:
        s_list = __verbose(s, out='sub the center joint #1 (spine joint in ntu and neck joint in kinetics)', verbose=verbose)  # noqa
        for i_s, skeleton in enumerate(s_list):
            if skeleton.sum() == 0:
                continue
            main_body_center = skeleton[0, :, 1:2, :].copy()
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                mask = (person.sum(-1) != 0).reshape(T, V, 1)
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    if center_firstframe:
        # based on SGN:
        # https://github.com/microsoft/SGN/blob/master/data/ntu/seq_transformation.py
        s_list = __verbose(s, out='sub the first valid frame center joint #1 (spine joint in ntu and neck joint in kinetics)', verbose=verbose)  # noqa
        for i_s, skeleton in enumerate(s_list):
            if skeleton.sum() == 0:
                continue
            i = 0  # get the "real" first frame of actor1
            while i < skeleton.shape[1]:
                if np.any(skeleton[0, i, :, :] != 0):
                    break
                i += 1
            main_body_center = skeleton[0, i:i+1, 1:2, :].copy()
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                mask = (person.sum(-1) != 0).reshape(T, V, 1)
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    if zaxis is not None:
        s_list = __verbose(s, out='parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis', verbose=verbose)  # noqa
        for i_s, skeleton in enumerate(s_list):
            if skeleton.sum() == 0:
                continue
            joint_bottom = skeleton[0, 0, zaxis[0]]
            joint_top = skeleton[0, 0, zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = rotation_matrix(axis, angle)
            # print(f"z angle {angle}")
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    if xaxis is not None:
        s_list = __verbose(s, out='parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis', verbose=verbose)  # noqa
        for i_s, skeleton in enumerate(s_list):
            if skeleton.sum() == 0:
                continue
            joint_rshoulder = skeleton[0, 0, xaxis[0]]
            joint_lshoulder = skeleton[0, 0, xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = rotation_matrix(axis, angle)
            # print(f"x angle {angle}")
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    if zaxis2 is not None:
        s_list = __verbose(s, out='parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis', verbose=verbose)  # noqa
        for i_s, skeleton in enumerate(s_list):
            if skeleton.sum() == 0:
                continue
            joint_bottom = skeleton[0, 0, zaxis2[0]]
            joint_top = skeleton[0, 0, zaxis2[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])  # N, M, T, V, C to N, C, T, V, M
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
