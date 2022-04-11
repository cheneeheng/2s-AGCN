import torch
from tqdm import tqdm


def test_sgn6(Model):
    bs = 64
    seg = 20

    inputs = torch.ones(bs, seg, 75)
    subjects = torch.ones(bs, seg, 1)

    c = 0
    for in_position in [0, 1, 2, 3]:
        for in_velocity in [0, 1, 2, 3]:
            for in_part in [0, 1, 2, 3]:
                for in_part_type in [0, 1, 2]:
                    for in_motion in [0, 1, 2, 3]:
                        for in_par_mot_fusion in [0, 1]:
                            for in_pos_vel_fusion in [0, 1]:
                                for sem_part in [0, 1, 2, 3]:
                                    for sem_position in [0, 1, 2, 3]:
                                        for sem_frame in [0, 1, 2, 3]:
                                            # all inputs are zero
                                            if in_position == 0 and in_velocity == 0 and in_part == 0 and in_motion == 0:  # noqa
                                                continue
                                            # need xyz for semantic
                                            if in_position == 0 and sem_position > 0:  # noqa
                                                continue
                                            # need parts for semantic
                                            if in_part == 0 and sem_part > 0:  # noqa
                                                continue
                                            # currently the fusion methods must be the same  # noqa
                                            if in_pos_vel_fusion != 0 and in_par_mot_fusion == 0:  # noqa
                                                continue
                                            # currently the fusion methods must be the same  # noqa
                                            if in_par_mot_fusion != 0 and in_pos_vel_fusion == 0:  # noqa
                                                continue
                                            # if concat, the part needs to be concat also   # noqa
                                            if in_pos_vel_fusion == 0 and ((in_part == 0 and in_motion != 0) or (in_part != 0 and in_motion == 0)):  # noqa
                                                continue
                                            # if concat, the xyz needs to be concat also   # noqa
                                            if in_par_mot_fusion == 0 and ((in_position == 0 and in_velocity != 0) or (in_position != 0 and in_velocity == 0)):  # noqa
                                                continue
                                            model = Model(
                                                num_segment=seg,
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
                                            print(f"Finished {c}")
                                            c += 1

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
