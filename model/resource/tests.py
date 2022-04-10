import torch
from tqdm import tqdm


def test(Model):
    batch_size = 64

    inputs = torch.ones(batch_size, 20, 75)
    subjects = torch.ones(batch_size, 20, 1)
    for in_position in tqdm([0, 1, 2, 3], position=9, leave=False):
        for in_velocity in tqdm([0, 1, 2, 3], position=8, leave=False):
            for in_part in tqdm([0, 1, 2, 3], position=7, leave=False):
                for in_part_type in tqdm([0, 1, 2], position=6, leave=False):  # noqa
                    for in_motion in tqdm([0, 1, 2, 3], position=5, leave=False):  # noqa
                        for in_par_mot_fusion in tqdm([0, 1], position=4, leave=False):  # noqa
                            for in_pos_vel_fusion in tqdm([0, 1], position=3, leave=False):  # noqa
                                for sem_part in tqdm([0, 1, 2, 3], position=2, leave=False):  # noqa
                                    for sem_position in tqdm([0, 1, 2, 3], position=1, leave=False):  # noqa
                                        for sem_frame in tqdm([0, 1, 2, 3], position=0, leave=False):  # noqa
                                            if in_position == 0 and in_velocity == 0 and in_part == 0 and in_motion == 0:  # noqa
                                                continue
                                            if in_position == 0 and sem_position > 0:  # noqa
                                                continue
                                            if in_part == 0 and sem_part > 0:  # noqa
                                                continue
                                            if in_pos_vel_fusion != 0 and in_par_mot_fusion == 0:  # noqa
                                                continue
                                            if in_par_mot_fusion != 0 and in_pos_vel_fusion == 0:  # noqa
                                                continue
                                            if in_pos_vel_fusion == 0 and (in_part == 0 or in_motion == 0):  # noqa
                                                continue
                                            if in_par_mot_fusion == 0 and (in_position == 0 or in_velocity == 0):  # noqa
                                                continue
                                            model = Model(
                                                num_segment=20,
                                                in_position=in_position,
                                                in_velocity=in_velocity,
                                                in_part=in_part,
                                                in_part_type=in_part_type,
                                                in_motion=in_motion,
                                                in_par_mot_fusion=in_par_mot_fusion,  # noqa
                                                in_pos_vel_fusion=in_pos_vel_fusion,  # noqa
                                                sem_part=sem_part,
                                                sem_position=sem_position,
                                                sem_frame=sem_frame,
                                            )
                                            model(inputs, subjects)
    # for g_shared in tqdm([True, False], position=3, leave=False):
    #     for g_proj_shared in tqdm([True, False], position=2, leave=False):
    #         for g_proj_dim in tqdm([32, 512], position=1, leave=False):  # noqa
    #             for gcn_t_kernel in tqdm([1, 5], position=0, leave=False):  # noqa
    #                 model = SGN(num_segment=20,
    #                             g_shared=g_shared,
    #                             g_proj_shared=g_proj_shared,
    #                             g_proj_dim=g_proj_dim,
    #                             gcn_t_kernel=gcn_t_kernel)
    #                 model(inputs, subjects)
    # for subject in tqdm([0, 1, 2, 3], position=4, leave=False):
    #     for t_kernel in tqdm([1, 5], position=3, leave=False):
    #         for t_max_pool in tqdm([0, 2], position=2, leave=False):
    #             for aspp in tqdm([None, [0, ], [0, 1, 5, 9]], position=1, leave=False):  # noqa
    #                 for norm_type in tqdm(['bn', 'ln'], position=0, leave=False):  # noqa
    #                     model = SGN(num_segment=20,
    #                                 subject=subject,
    #                                 t_kernel=t_kernel,
    #                                 t_max_pool=t_max_pool,
    #                                 aspp=aspp,
    #                                 norm_type=norm_type)
    #                     model(inputs, subjects)
