# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.
import tensorflow as tf
import math

def cosineAnnealingScheduler(epoch, initial_lr, reset_step=5, min_lr=1e-8):
    T = reset_step
    lr =  ((1 + math.cos( math.pi*( epoch%T /T) )) / 2) * (initial_lr - min_lr) + min_lr
    return lr

# class CosineAnnealing(keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, initial_lr, reset_step, min_lr):
#         super(CosineAnnealing, self).__init__()

#         self.initial_lr = initial_lr
#         self.T = reset_step
#         self.min_lr = min_lr

#     def __call__(self, step):
#         lr =  ((1 + tf.math.cos( tf.math.pi*( step%self.T /self.T) )) / 2) * (self.initial_lr - self.min_lr) + self.min_lr
#         return lr