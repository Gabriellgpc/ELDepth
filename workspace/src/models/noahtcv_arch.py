"""
    Network proposed by team NOAHTCV in the MAI 2021 Real Image Denoising challenge.
    report: https://arxiv.org/pdf/2105.08629.pdf


"[...] The authors especially emphasize the role of skip connections on
fidelity scores and the effect of upsampling and downsampling operations on the runtime results. The models were
trained on patches of size 256x256 pixels using Adam optimizer with a batch size of 64 for 200 epochs. The learning
rate was set to 1e-4 and decreased to 1e-5 by applying a cosine decay. L2 loss was used as the main fidelity metric."
"""

import tensorflow as tf
from keras.layers import Conv2D, \
                         Conv2DTranspose, \
                         Concatenate, \
                         Layer

from keras.models import Sequential, Model

class ResidualBlock(Layer):
    def __init__(self, n_filters, initializer='he_normal', groups=1, use_bias=False,**kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.groups = groups
        self.n_filters = n_filters
        self.initializer = initializer
        self.use_bias = use_bias

        self.double_conv = Sequential([
            Conv2D(n_filters,
                               kernel_size=(3,3),
                               strides=(1,1),
                               padding = "same",
                               activation = "relu",
                               kernel_initializer = self.initializer,
                               use_bias=self.use_bias,
                               groups=self.groups),

            Conv2D(n_filters,
                      kernel_size=(3,3),
                      strides=(1,1),
                      padding = "same",
                      activation = "relu",
                      kernel_initializer = self.initializer,
                      use_bias=self.use_bias,
                      groups=self.groups),
        ])

    def call(self, x):
        return self.double_conv(x) + x

class NOAHTCV(Model):
    def __init__(self, n_filters, initializer='he_normal', use_bias=False,**kwargs ):
        super(NOAHTCV, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.initializer = initializer
        self.use_bias = use_bias

        # for the "top" layer
        self.conv11 = Conv2D(n_filters, kernel_size=(3,3), strides=(1,1),
                          padding = "same",
                          activation = "relu",
                          kernel_initializer = self.initializer,
                          use_bias=self.use_bias)
        self.res11 = ResidualBlock(n_filters, initializer, groups=1, use_bias=use_bias)
        self.down11= Conv2D( n_filters, (3,3), strides=(2,2), padding='same', use_bias=use_bias, activation = "relu")


        # second layer
        self.conv21= Conv2D( n_filters, (3,3), strides=(1,1), padding='same', use_bias=use_bias, activation = "relu")
        self.res21 = ResidualBlock(n_filters, initializer, groups=1, use_bias=use_bias)
        self.down21= Conv2D( n_filters, (3,3), strides=(2,2), padding='same', use_bias=use_bias, activation = "relu")

        # last layer (lowest)
        self.res31 = ResidualBlock(n_filters, initializer, groups=1, use_bias=use_bias)
        self.up31 = Conv2DTranspose( n_filters,
                                     kernel_size=(1,1),
                                     strides=(2,2),
                                     use_bias=use_bias,
                                     kernel_initializer=initializer,
                                     activation = "relu"
                                     )


        # back to 2
        self.cat21 = Concatenate()
        self.conv22= Conv2D( n_filters, (3,3), strides=(1,1), padding='same', use_bias=use_bias, activation = "relu")
        self.res22 = ResidualBlock(n_filters, initializer, groups=1, use_bias=use_bias)
        self.up21 = Conv2DTranspose( n_filters,
                                     kernel_size=(1,1),
                                     strides=(2,2),
                                     use_bias=use_bias,
                                     kernel_initializer=initializer,
                                     activation = "relu")
        # back to 1/output
        self.cat11 = Concatenate()
        self.conv12= Conv2D( n_filters, (3,3), strides=(1,1), padding='same', use_bias=use_bias, activation = "relu")
        self.res12 = ResidualBlock(n_filters, initializer, groups=1, use_bias=use_bias)
        self.conv13= Conv2D( n_filters, (3,3), strides=(1,1), padding='same', use_bias=use_bias, activation = "relu")
        self.conv14= Conv2D( 3, (3,3), strides=(1,1), padding='same', use_bias=use_bias)

    def call(self, x):
        # top
        f1 = self.conv11(x)
        f2 = self.res11(f1)
        f3 = self.down11(f2)

        # mid
        f4 = self.res21(f3)
        # print('f4.shape', f4.shape)
        f5 = self.down21(f4)
        # print('f6.shape', f5.shape)

        # bot
        f6 = self.res31(f5)
        # print('f6.shape', f6.shape)
        f7 = self.up31(f6)
        # print('f8.shape', f7.shape)

        cat = self.cat21( [f4, f7] )
        # print('cat', cat.shape)
        f = self.conv22(cat)
        # print('f.shape', f.shape)
        f = self.res22(f)
        # print('f.shape', f.shape)
        f = self.up21(f)
        # print('f.shape', f.shape)

        cat = self.cat11( [f2, f] )
        # print('f.shape', f.shape)
        f = self.conv12(cat)
        # print('f.shape', f.shape)
        f = self.res12(f)
        # print('f.shape', f.shape)
        f = self.conv13(f)
        # print('f.shape', f.shape)
        f = self.conv14(f)
        # print('f.shape', f.shape)

        # return f + x
        return f