# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-11-12 14:12:47
# @Last Modified by:   Condados
# @Last Modified time: 2022-11-12 14:19:35
from functools import partial
import tensorflow as tf
import numpy as np
import cv2

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


# @tf.function
# def tf_resize(image, depth, mask, dsize=[256, 256]):
#     image = tf.image.resize( image, size=dsize)
#     depth = tf.image.resize( depth, size=dsize)
#     mask = tf.image.resize( mask, size=dsize)
#     return image, depth, mask

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

# @tf.function
# def load_image_depth_mask(input_path, depth_path, mask_path):
#     # read as RGB and convert to [0, 1], float32
#     input_image = read_image(input_path)
#     h, w = input_image.shape[:2]
#     # load depth map
#     [depth,] = tf.py_function(load_numpy, [depth_path], [tf.float32])
#     depth.set_shape( [h,w,1] )
#     # load valid mask
#     [mask,] = tf.py_function(load_numpy, [mask_path], [tf.float32])
#     # put mask in [H,W,1] format
#     mask = tf.expand_dims(mask, axis=-1)
#     mask.set_shape( [h,w,1] )
#     return input_image, depth, mask

def load_image_depth_mask(input_path, depth_path, mask_path):
    # read as RGB and convert to [0, 1], float32
    input_image = read_image(input_path)
    h, w = input_image.shape[:2]
    # load depth map
    [depth_map,] = tf.py_function(load_numpy, [depth_path], [tf.float32])
    depth_map.set_shape( [h,w,1] )
    # load valid mask
    [mask,] = tf.py_function(load_numpy, [mask_path], [tf.float32])
    # put mask in [H,W,1] format
    mask = tf.expand_dims(mask, axis=-1)
    mask.set_shape( [h,w,1] )

    min_depth = 0.1
    max_depth = 300

    depth_map = tf.clip_by_value(depth_map, min_depth, max_depth)
    depth_map = tf.math.log(depth_map) * mask
    depth_map = tf.clip_by_value(depth_map, np.log(min_depth), np.log(max_depth))
    return input_image, depth_map, mask

# def parser_DIODE_dataset(input_image, depth, mask):
#     min_depth = 0.1
#     max_depth = 300

#     mask = mask > 0

#     depth_map = np.clip(depth_map, min_depth, max_depth)
#     depth_map = np.log(depth_map, where=mask)

#     depth_map = np.ma.masked_where(~mask, depth_map)
#     depth_map = np.clip(depth_map, np.log(min_depth), np.log(max_depth))

#     return input_image, depth, mask

def build_tf_dataloader(input_paths,
                        depth_paths,
                        mask_paths,
                        batch_size=32,
                        transforms=[],
                        shuffle=False,
                        repeat=False,
                        ):
    data = tf.data.Dataset.from_tensor_slices( (input_paths, depth_paths, mask_paths) )
    if shuffle:
        data = data.shuffle(1024)
    data = data.map(load_image_depth_mask, num_parallel_calls=tf.data.AUTOTUNE)

    for transform_f in transforms:
        data = data.map(transform_f, num_parallel_calls=tf.data.AUTOTUNE)

    data = data.batch(batch_size)

    if repeat:
        data = data.repeat()
    return data

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(768, 1024), n_channels=3, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, image_path, depth_map, mask):
        """Load input and target image."""

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0

        max_depth = 256
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, np.log(self.min_depth), np.log(max_depth))
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def data_generation(self, batch):

        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["depth"][batch_id],
                self.data["mask"][batch_id],
            )

        return x, y