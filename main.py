#!/usr/bin/env python
# from __future__ import print_function

import yaml

from torch.utils import tensorboard

from main_processor import *
from main_utils import *


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WRONG ARG: {k}')
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    print_arg(arg)
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
    print()
