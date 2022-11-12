# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-11-12 14:12:47
# @Last Modified by:   Condados
# @Last Modified time: 2022-11-12 14:19:35
import tensorflow as tf

# ssl_ds_one.shuffle(1024, seed=SEED)
# .map(custom_augment, num_parallel_calls=AUTO)
# .batch(BATCH_SIZE)
# .prefetch(AUTO)


def get_dataloader(input_paths, target_paths, batch_size=32, transforms=[], train=True):
    data = tf.data.Dataset.from_tensor_slices( zip(input_paths, target_paths) )
    if train:
        data = data.shuffle(1024)
    for transform_f in transforms:
        data = data.map(transform_f, num_parallel_calls=AUTO)
    data = data.batch(batch_size)
    return data