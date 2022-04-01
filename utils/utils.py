import argparse
import numpy as np
import os
import random
import torch

__all__ = [
    'print_arg',
    'init_seed',
    'str2bool',
    'bool2int',
    'import_class',
]


def print_arg(arg):
    def _dict_check(_k, _v, _i):
        if isinstance(_v, dict):
            print(f"AAA{_k} :".replace("AAA", " "*_i*4))
            for __k, __v in _v.items():
                _dict_check(__k, __v, _i+1)
        else:
            print(f"AAA{_k} : {_v}".replace("AAA", " "*_i*4))

    for k, v in list(vars(arg).items()):
        i = 0
        _dict_check(k, v, i)
    print()


def init_seed(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def bool2int(v):
    if isinstance(v, int):
        return v
    elif isinstance(v, bool):
        if v:
            return 1
        else:
            return 0
    elif isinstance(v, str):
        return bool2int(str2bool(v))
    else:
        raise ValueError('expects int, bool, str types only')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
