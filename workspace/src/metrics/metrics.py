import tensorflow as tf

# def compute_errors(gt, pred):

#     pred = recover_metric_depth(pred, gt)

#     thresh = np.maximum((gt / pred), (pred / gt))   #OKAY
#     d1 = (thresh < 1.25).mean()
#     d2 = (thresh < 1.25 ** 2).mean()
#     d3 = (thresh < 1.25 ** 3).mean()

#     rmse = (gt - pred) ** 2              #OKAY
#     rmse = np.sqrt(rmse.mean())

#     rmse_log = (np.log(gt) - np.log(pred)) ** 2     #OKAY
#     rmse_log = np.sqrt(rmse_log.mean())

#     abs_rel = np.mean(np.abs(gt - pred) / gt)   #OKAY
#     sq_rel = np.mean(((gt - pred) ** 2) / gt)   #OKAY

#     err = np.log(pred) - np.log(gt)
#     silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100 #OKAY

#     err = np.abs(np.log10(pred) - np.log10(gt))  #OKAY
#     log10 = np.mean(err)

#     return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def metric_log10(y_true, y_pred):
    err = tf.abs( log10(y_pred) - log10(y_true) )
    return tf.reduce_mean(err)

def metric_RMSE(y_true, y_pred):
    rmse = tf.sqrt( tf.reduce_mean( tf.square(y_true - y_pred) ) )
    return rmse

def metric_RMSElog(y_true, y_pred):
    diff = tf.math.log(y_true) - tf.math.log(y_pred)
    rmse_log = tf.sqrt( tf.reduce_mean( tf.square(diff) ) )
    return rmse_log

def metric_absRel(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    rel = tf.reduce_mean(diff / y_true)
    return rel

def metric_SqRel(y_true, y_pred):
    diff = tf.square(y_pred - y_true)
    rel = tf.reduce_mean(diff / y_true)
    return rel

def metric_SILog(y_true, y_pred):
    # ref.: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

    d = tf.math.log(y_true) - tf.math.log(y_pred)
    p1 = tf.reduce_mean(tf.square( d ))
    p2 = tf.square(tf.reduce_mean( d ))
    silog = (p1 - p2)
    return silog

def metric_Accuracies(y_true, y_pred):
    thresh = tf.maximum((y_true / y_pred), (y_pred / y_true))
    d1 = tf.reduce_mean( tf.cast(thresh < 1.25, dtype=tf.int32) )
    d2 = tf.reduce_mean( tf.cast(thresh < 1.25**2, dtype=tf.int32) )
    d3 = tf.reduce_mean( tf.cast(thresh < 1.25**3, dtype=tf.int32) )
    return d1, d2, d3