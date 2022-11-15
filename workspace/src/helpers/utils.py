# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-10-04 19:36:46
# @Last Modified by:   Condados
# @Last Modified time: 2022-10-20 14:44:36

import tensorflow as tf
from uuid import uuid4
from box import Box
import numpy as np
import random
import yaml
import os


def get_uuid():
    return uuid4().hex

def save_args(filename, args):
    args_dict = dict(vars(args))
    with open(filename, 'w') as f:
        for arg in args_dict:
            # print(arg, args_dict[arg])
            v = args_dict[arg]
            f.write( f'{arg}:{v}\n' )

def mkdirdirs_tmp(dirname):
    dirpath = os.path.join('/tmp', str(dirname))
    os.makedirs( dirpath )
    return dirpath

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def read_configuration(filename):
    with open(filename) as f:
        config = Box(yaml.safe_load(f.read()))
        return config