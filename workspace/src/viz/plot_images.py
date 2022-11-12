# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-11-12 14:21:09
# @Last Modified by:   Condados
# @Last Modified time: 2022-11-12 14:21:16


# # Ensure that the different versions of the dataset actually contain
# # identical images.
# sample_images_two = next(iter(ssl_ds_two))
# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(sample_images_two[n].numpy().astype("int"))
#     plt.axis("off")
# plt.show()


# 3D viz
# https://keras.io/examples/vision/nerf/