# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-11-12 14:12:47
# @Last Modified by:   Condados
# @Last Modified time: 2022-11-12 14:19:35
from functools import partial
import tensorflow as tf
import numpy as np

@tf.function
def read_image(image_path):
    content = tf.io.read_file(image_path)
    ext = tf.strings.split(image_path, '.')[-1]
    ext = tf.strings.lower(ext)

    bmp_decode  = lambda: tf.image.decode_bmp(content, channels=3)
    png_decode  = lambda: tf.image.decode_png(content, channels=3)
    jpeg_decode = lambda: tf.image.decode_jpeg(content, channels=3, try_recover_truncated=True)

    image = tf.case([ (ext == tf.constant('bmp'), bmp_decode),
                    (ext == tf.constant('png'), png_decode),
                    (ext == tf.constant('jpg'), jpeg_decode),
                    (ext == tf.constant('jpeg'), jpeg_decode),
                  ], default=bmp_decode)
    image = tf.image.convert_image_dtype(image, dtype='float32') #will be in [0, 1]
    return image


@tf.function
def tf_resize(image, depth, mask, dsize=[256, 256]):
    image = tf.image.resize( image, size=dsize)
    depth = tf.image.resize( depth, size=dsize)
    mask = tf.image.resize( mask, size=dsize)
    return image, depth, mask

def get_tf_resize(dsize=[256, 256]):
    return partial(tf_resize, dsize=dsize)

@tf.function
def load_images(filenameA, filenameB):
    return read_image(filenameA), read_image(filenameB)

def load_numpy(filename):
    filename = filename.numpy().decode('utf-8')
    np_loaded = np.load(filename)
    return np_loaded

@tf.function
def load_image_depth_mask(input_path, depth_path, mask_path):
    # read as RGB and convert to [0, 1], float32
    input_image = read_image(input_path)
    h, w = input_image.shape[:2]
    # load depth map
    [depth,] = tf.py_function(load_numpy, [depth_path], [tf.float32])
    depth.set_shape( [h,w,1] )
    # load valid mask
    [mask,] = tf.py_function(load_numpy, [mask_path], [tf.float32])
    # put mask in [H,W,1] format
    mask = tf.expand_dims(mask, axis=-1)
    mask.set_shape( [h,w,1] )
    return input_image, depth, mask

def parser_DIODE_dataset(input_image, depth, mask):
    depth = tf.clip_by_value(depth, 0.6, 350.0) * mask
    depth = depth / 350.0
    return input_image, depth, mask

def build_tf_dataloader(input_paths,
                        depth_paths,
                        mask_paths,
                        batch_size=32,
                        transforms=[],
                        train=True):
    data = tf.data.Dataset.from_tensor_slices( (input_paths, depth_paths, mask_paths) )
    if train:
        data = data.shuffle(1024)
    data = data.map(load_image_depth_mask, num_parallel_calls=tf.data.AUTOTUNE)

    for transform_f in transforms:
        data = data.map(transform_f, num_parallel_calls=tf.data.AUTOTUNE)

    data = data.batch(batch_size)

    if train:
        data = data.repeat()
    return data