import math
import numpy as np
from multiprocessing import Pool, RLock
from tqdm import tqdm


__all__ = ['parallel_processing']


def check_arr_arg_length(arg, length: int):
    if length >= 0:
        assert len(arg) == length
    return len(arg)


def parallel_processing(func, num_of_processes: int, *args):

    arrays, non_arrays = [], []
    c = -1

    for arg in args:
        if isinstance(arg, list):
            c = check_arr_arg_length(arg, c)
            arrays.append(arg)
        elif isinstance(arg, tuple):
            c = check_arr_arg_length(arg, c)
            arrays.append(arg)
        elif isinstance(arg, np.ndarray):
            c = check_arr_arg_length(arg, c)
            arrays.append(arg)
        else:
            non_arrays.append(arg)

    num_ele_per_process = math.ceil(c / num_of_processes)

    argument_list = []
    for pid in range(num_of_processes):
        arg_tuple = []
        for ele in non_arrays:
            arg_tuple.append(ele)
        for ele in arrays:
            arg_tuple.append(ele[pid*num_ele_per_process:
                                 (pid+1)*num_ele_per_process])
        arg_tuple.append(pid)
        arg_tuple = tuple(arg_tuple)
        argument_list.append(arg_tuple)

    pool = Pool(processes=num_of_processes,
                initargs=(RLock(),),
                initializer=tqdm.set_lock)
    jobs = [pool.apply_async(func, args=arg) for arg in argument_list]
    pool.close()
    result_list = [job.get() for job in jobs]
    return result_list
