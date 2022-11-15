import tensorflow as tf

def charbonnier_loss(y_true, y_pred, eps=1e-3):
    loss = tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(eps)))
    return loss

def depth_final_loss(target, pred, max_depth=350, loss_weights=[1.0, 1.0, 0.1]):
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