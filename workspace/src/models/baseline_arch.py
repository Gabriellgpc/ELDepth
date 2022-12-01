import tensorflow as tf
from tensorflow.keras import layers

class DownscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.pool = layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor):
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x += d
        p = self.pool(x)
        return x, p


class UpscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        return x


class BottleNeckBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x


class BaselineModel(tf.keras.Model):
    def __init__(self, max_depth=300, ssim_loss_weight = 0.85, l1_loss_weight=0.1, edge_loss_weight=0.9, f=[16, 32, 64, 128, 256], **kwargs):
        super(BaselineModel, self).__init__(**kwargs)

        self.max_depth = max_depth

        self.ssim_loss_weight = ssim_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.edge_loss_weight = edge_loss_weight
        # self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.downscale_blocks = [
            DownscaleBlock(f[0]),
            DownscaleBlock(f[1]),
            DownscaleBlock(f[2]),
            DownscaleBlock(f[3]),
        ]
        self.bottle_neck_block = BottleNeckBlock(f[4])
        self.upscale_blocks = [
            UpscaleBlock(f[3]),
            UpscaleBlock(f[2]),
            UpscaleBlock(f[1]),
            UpscaleBlock(f[0]),
        ]
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="tanh")

    # def calculate_loss(self, target, pred):
    #     # Edges
    #     dy_true, dx_true = tf.image.image_gradients(target)
    #     dy_pred, dx_pred = tf.image.image_gradients(pred)
    #     weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    #     weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    #     # Depth smoothness
    #     smoothness_x = dx_pred * weights_x
    #     smoothness_y = dy_pred * weights_y

    #     depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

    #     # Structural similarity (SSIM) index
    #     ssim_loss = tf.reduce_mean(1 - tf.image.ssim(target, pred, max_val=self.max_depth, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2))
    #     # Point-wise depth
    #     l1_loss = tf.reduce_mean(tf.abs(target - pred))

    #     loss = (
    #         (self.ssim_loss_weight * ssim_loss)
    #         + (self.l1_loss_weight * l1_loss)
    #         + (self.edge_loss_weight * depth_smoothness_loss)
    #     )

    #     return loss

    # @property
    # def metrics(self):
    #     return [self.loss_metric]

    # def train_step(self, batch_data):
    #     input, target = batch_data
    #     with tf.GradientTape() as tape:
    #         pred = self(input, training=True)
    #         loss = self.calculate_loss(target, pred)

    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #     self.loss_metric.update_state(loss)
    #     return {
    #         "loss": self.loss_metric.result(),
    #     }

    # def test_step(self, batch_data):
    #     input, target = batch_data

    #     pred = self(input, training=False)
    #     loss = self.calculate_loss(target, pred)

    #     self.loss_metric.update_state(loss)
    #     return {
    #         "loss": self.loss_metric.result(),
    #     }

    def call(self, x):
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        bn = self.bottle_neck_block(p4)

        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)

        return self.conv_layer(u4)