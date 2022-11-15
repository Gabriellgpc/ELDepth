# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-11-12 14:21:09
# @Last Modified by:   Condados
# @Last Modified time: 2022-11-12 14:21:16
import matplotlib.pyplot as plt
import numpy as np
import cv2

# # Ensure that the different versions of the dataset actually contain
# # identical images.
# sample_images_two = next(iter(ssl_ds_two))
# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(sample_images_two[n].numpy().astype("int"))
#     plt.axis("off")
# plt.show()


def visualize_depth_map(samples, test=False, model=None, save_at=''):
    images, depths, masks = samples

    images = images.numpy()
    depths = depths.numpy()
    masks = masks.numpy()

    max_depth = min(300, depths.max())
    depths = np.clip(depths, 0.1, max_depth)
    depths = np.log(depths)
    depths = np.ma.masked_where(~(masks > 0), depths)
    depths = np.clip(depths, 0.1, np.log(max_depth))

    cmap = plt.cm.get_cmap("jet").copy()
    # cmap = plt.cm.jet
    cmap.set_bad(color="black")

    fig, ax = plt.subplots(6, 3, figsize=(50, 50))
    if test:
        pred = model.predict(images)
        for i in range(6):
            ax[i, 0].imshow((images[i].squeeze()))
            ax[i, 1].imshow((depths[i].squeeze()), cmap=cmap)
            ax[i, 2].imshow((pred[i].squeeze()), cmap=cmap)
    else:
        for i in range(6):
            ax[i, 0].imshow((images[i].squeeze()))
            ax[i, 1].imshow((depths[i].squeeze()), cmap=cmap)
    if save_at != '':
        plt.savefig(save_at)
    return fig, ax

def random_region_highlight(images,
                            patch_shape=(128, 256),
                            scale=2,
                            ):
    # select the central region of the region of intest
    H, W, _ = images[0].shape
    h, w = patch_shape
    xmin = w//2
    xmax = W - (w//2)*scale
    ymin = h//2
    ymax = H - (h//2)*scale
    xc = np.random.uniform(low=xmin, high=xmax)
    yc = np.random.uniform(low=ymin, high=ymax)

    xi = int(xc - w//2)
    xf = int(xc + w//2)
    yi = int(yc - h//2)
    yf = int(yc + h//2)

    images_out = []
    # loop throught each image
    for image in images:
        # draw rectangle
        image_copy = image.copy()
        cv2.rectangle(image_copy, (xi,yi), (xf,yf), (0,255,0), 5)

        crop = image_copy[yi:yf,xi:xf,:]

        new_h = h*scale
        new_w = w*scale

        crop = cv2.resize(crop, (new_w, new_h))

        # overlaping the images
        image_copy[H-crop.shape[0]:,W-crop.shape[1]:,:] = crop
        images_out.append(image_copy)
    return images_out

# 3D viz
# https://keras.io/examples/vision/nerf/