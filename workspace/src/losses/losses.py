# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-11-12 17:56:52
# @Last Modified by:   Condados
# @Last Modified time: 2022-11-16 08:50:02
import tensorflow as tf

def charbonnier_loss(y_true, y_pred, eps=1e-3):
    loss = tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(eps)))
    return loss

def loss_RMSE(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    return rmse

def loss_iRMSE(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.01, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.01, 1.0)
    mse = tf.keras.losses.mean_squared_error(1.0/y_true, 1.0/y_pred)
    rmse = tf.sqrt(mse)
    return rmse

# class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
# from: https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py
#     def __init__(self):
#         super(SILogLoss, self).__init__()
#         self.name = 'SILog'

#     def forward(self, input, target, mask=None, interpolate=True):
#         if interpolate:
#             input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

#         if mask is not None:
#             input = input[mask]
#             target = target[mask]
#         g = torch.log(input) - torch.log(target)
#         # n, c, h, w = g.shape
#         # norm = 1/(h*w)
#         # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

#         Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
#         return 10 * torch.sqrt(Dg)

# def build_losses(self):
# from : https://github.com/cleinc/bts/blob/master/tensorflow/bts.py
#     with tf.variable_scope('losses', reuse=self.reuse_variables):

#         if self.params.dataset == 'nyu':
#             self.mask = self.depth_gt > 0.1
#         else:
#             self.mask = self.depth_gt > 1.0

#         depth_gt_masked = tf.boolean_mask(self.depth_gt, self.mask)
#         depth_est_masked = tf.boolean_mask(self.depth_est, self.mask)

#         d = tf.log(depth_est_masked) - tf.log(depth_gt_masked)  # Best

#         self.silog_loss = tf.sqrt(tf.reduce_mean(d ** 2) - 0.85 * (tf.reduce_mean(d) ** 2)) * 10.0
#         self.total_loss = self.silog_loss


def loss_SILog(y_true, y_pred):
    # ref.: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
    y_true = tf.clip_by_value(y_true, 0.001, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.001, 1.0)

    d = tf.math.log(y_true) - tf.math.log(y_pred)

    p1 = tf.reduce_mean(tf.square( d ))
    p2 = tf.square(tf.reduce_mean( d ))
    silog = (p1 - 0.85*p2)*10.0
    return silog

def loss_log10(y_true, y_pred):
    pass

def loss_siRMSE(y_true, y_pred):
    pass

def depth_final_loss(target, pred, max_depth=1.0, loss_weights=[1.0, 1.0, 0.1]):
    """
    # self.ssim_loss_weight = 0.85
    # self.l1_loss_weight = 0.1
    # self.edge_loss_weight = 0.9
    """
    # Edges
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))
    depth_smoothness_loss = tf.clip_by_value(depth_smoothness_loss, 0, 1)

    # Structural similarity (SSIM) index
    ssim_loss = 1 - tf.image.ssim(target, pred, max_val=max_depth) * 0.5
    ssim_loss = tf.reduce_mean(ssim_loss)
    ssim_loss = tf.clip_by_value(ssim_loss, 0, 1)
    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(target - pred))

    loss = (
          (loss_weights[0] * ssim_loss)
        + (loss_weights[1] * l1_loss)
        + (loss_weights[2] * depth_smoothness_loss)
    )

    return loss