import tensorflow as tf

def metric_RMSE(y_true, y_pred):
    rmse = tf.sqrt( tf.reduce_mean( tf.square(y_pred - y_true) ) )
    return rmse

def metric_RMSElog(y_true, y_pred):
    diff = tf.math.log(y_pred) - tf.math.log(y_true)
    rmse_log = tf.sqrt( tf.reduce_mean( tf.square(diff) ) )
    return rmse_log

def metric_absRel(y_true, y_pred):
    diff = y_pred - y_true
    rel = diff / (y_true + 1e-8)
    return rel

def metric_SqRel(y_true, y_pred):
    diff = tf.square(y_pred - y_true)
    rel = diff / (y_true + 1e-8)
    return rel

def metric_SILog(y_true, y_pred):
    # ref.: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
    y_true = tf.clip_by_value(y_true, 0.001, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.001, 1.0)

    d = tf.math.log(y_true) - tf.math.log(y_pred)

    p1 = tf.reduce_mean(tf.square( d ))
    p2 = tf.square(tf.reduce_mean( d ))
    silog = (p1 - p2)
    return silog

def metric_Accuracies(y_true, y_pred, thr=1.25):
    diff_1 = y_pred / (y_true + 1e-8)
    diff_2 = y_true / (y_pred + 1e-8)

    max_d = tf.maximum( diff_1, diff_2 )
    # TODO