# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-11-12 14:10:18
# @Last Modified by:   Condados
# @Last Modified time: 2022-11-12 17:18:53

import tensorflow as tf

def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x

@tf.function
def random_rotate(image, depth, mask):
    """Rotates Images by 90 degrees."""
    # Outputs random values from uniform distribution in between 0 to 4
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    images = tf.concat([image, depth, mask], axis=-1)
    # Here rn signifies number of times the image(s) are rotated by 90 degrees
    images = tf.image.rot90(images, rn)
    image, depth, mask = tf.split(images, [3, 1, 1], axis=-1)
    return image, depth, mask

# def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
#     x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
#     x = tf.image.random_contrast(
#         x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
#     )
#     x = tf.image.random_saturation(
#         x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
#     )
#     x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
#     # Affine transformations can disturb the natural range of
#     # RGB images, hence this is needed.
#     x = tf.clip_by_value(x, 0, 255)
#     return x

# def color_drop(x):
#     x = tf.image.rgb_to_grayscale(x)
#     x = tf.tile(x, [1, 1, 3])
#     return x

@tf.function
def random_flip_left_right(image, depth, mask):
    """Flips Images to left and right."""

    # Outputs random values from a uniform distribution in between 0 to 1
    rn = tf.random.uniform(shape=(), maxval=1)
    # If rn is less than 0.5 it returns original lowres_img and highres_img
    # If rn is greater than 0.5 it returns flipped image
    images = tf.concat([image, depth, mask], axis=-1)
    images = tf.cond( rn < 0.5, lambda: tf.image.flip_left_right(images), lambda: images )
    image, depth, mask = tf.split(images, [3, 1, 1], axis=-1)
    return image, depth, mask

@tf.function
def random_crop(image, depth, mask, crop_size=[256,256]):
    concat = tf.concat( [image, depth, mask], axis=-1 )
    concat = tf.image.random_crop( concat, size=[*crop_size,3+1+1])
    image, depth, mask = tf.split( concat, [3,1,1], axis=-1 )
    return image, depth, mask

# @tf.function
# def random_resize(image, depth, mask, min_max_size=[32,256]):
#     images = tf.concat([image, depth, mask], axis=-1)

#     random_size = tf.random.uniform(shape=(), minval=min_max_size[0], maxval=min_max_size[1], dtype=tf.int32)
#     images = tf.image.resize( images, size=(random_size,min_max_size) )
#     image, depth, mask = tf.split(images, [3,1,1], axis=-1)
#     return image, depth, mask

# @tf.function
# def random_crop(input_image, target_image, crop_size):
#     input_image_shape = tf.shape(input_image)[:2]
#     inp_w = tf.random.uniform(shape=(), maxval=input_image_shape[1] - crop_size + 1, dtype=tf.int32)
#     inp_h = tf.random.uniform(shape=(), maxval=input_image_shape[0] - crop_size + 1, dtype=tf.int32)
#     target_w = inp_w
#     target_h = inp_h
#     input_image_cropped = input_image[inp_h : inp_h + crop_size, inp_w : inp_w + crop_size]
#     target_image_cropped = target_image[target_h : target_h + crop_size, target_w : target_w + crop_size]
#     return input_image_cropped, target_image_cropped