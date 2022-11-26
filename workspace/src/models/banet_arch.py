
import tensorflow as tf
from keras.layers import Layer, Conv2D, Conv2DTranspose
from keras.layers import ReLU, LeakyReLU
from keras.models import Model, Sequential

class AdaptiveAvgPool2D(tf.keras.layers.Layer):
    def __init__(self, output_size=(None, None)):
        super(AdaptiveAvgPool2D, self).__init__()
        # W2=(W1−F)/S+1
        # H2=(H1−F)/S+1
        # D2=D
        self.output_size = output_size
    def call(self, x):
        h, w = x.shape[1:3]
        fh = 1
        fw = 1
        if self.output_size[0] != None:
            fh = h - self.output_size[0] + 1
        if self.output_size[1] != None:
            fw = w - self.output_size[1] + 1
        ksize = (fh, fw)
        out = tf.nn.avg_pool( x, ksize, strides=(1,1), padding='VALID')
        return out

import tensorflow as tf
from keras.layers import Layer, Conv2D
from keras.layers import ReLU, LeakyReLU
from keras.models import Model, Sequential

class AdaptiveAvgPool2D(tf.keras.layers.Layer):
    def __init__(self, output_size=(None, None)):
        super(AdaptiveAvgPool2D, self).__init__()
        # W2=(W1−F)/S+1
        # H2=(H1−F)/S+1
        # D2=D
        self.output_size = output_size
    def call(self, x):
        h, w = x.shape[1:3]
        fh = 1
        fw = 1
        if self.output_size[0] != None:
            fh = h - self.output_size[0] + 1
        if self.output_size[1] != None:
            fw = w - self.output_size[1] + 1
        ksize = (fh, fw)
        out = tf.nn.avg_pool( x, ksize, strides=(1,1), padding='VALID')
        return out

class BA_Block(Layer):
    def __init__(self, outplanes):
        super(BA_Block, self).__init__()
        midplanes = int(outplanes//2)

        self.pool_1_h = AdaptiveAvgPool2D( (None,1) )

        self.pool_1_w = AdaptiveAvgPool2D((1, None))

        self.conv_1_h = Conv2D(midplanes, kernel_size=(3, 1), padding='SAME', use_bias=False)
        self.conv_1_w = Conv2D(midplanes, kernel_size=(1, 3), padding='SAME', use_bias=False)

        self.pool_3_h = AdaptiveAvgPool2D((None, 3))
        self.pool_3_w = AdaptiveAvgPool2D((3, None))
        self.conv_3 = Conv2D(midplanes, kernel_size=(3,3), padding='SAME', use_bias=False)

        self.pool_5_h = AdaptiveAvgPool2D((None, 5))

        self.pool_5_w = AdaptiveAvgPool2D((5, None))
        self.conv_5 = Conv2D(midplanes, kernel_size=(3,3), padding='SAME', use_bias=False)

        self.pool_7_h = AdaptiveAvgPool2D((None, 7))
        self.pool_7_w = AdaptiveAvgPool2D((7, None))
        self.conv_7 = Conv2D(midplanes, kernel_size=(3,3), padding='SAME', use_bias=False)

        self.fuse_conv = Conv2D(midplanes, kernel_size=(1,1), padding='VALID', use_bias=False)
        self.relu = ReLU()
        self.conv_final = Conv2D(outplanes, kernel_size=(1,1), use_bias=True)

        self.mask_conv_1 = Conv2D(outplanes, kernel_size=(3,3), padding='SAME')
        self.mask_relu = ReLU()
        self.mask_conv_2 = Conv2D(outplanes, kernel_size=(3,3), padding='SAME')

    def call(self, x):
        _, h, w, _ = x.shape
        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)

        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)

        x_3_h = self.pool_3_h(x)
        x_3_h = self.conv_3(x_3_h)
        x_3_h = tf.image.resize( x_3_h, (h,w) )

        x_3_w = self.pool_3_w(x)
        x_3_w = self.conv_3(x_3_w)
        x_3_w = tf.image.resize(x_3_w, (h, w))

        x_5_h = self.pool_5_h(x)
        x_5_h = self.conv_5(x_5_h)
        x_5_h = tf.image.resize(x_5_h, (h, w))

        x_5_w = self.pool_5_w(x)
        x_5_w = self.conv_5(x_5_w)

        x_5_w = tf.image.resize(x_5_w, (h, w))

        x_7_h = self.pool_7_h(x)
        x_7_h = self.conv_7(x_7_h)

        x_7_h = tf.image.resize(x_7_h, (h, w))

        x_7_w = self.pool_7_w(x)
        x_7_w = self.conv_7(x_7_w)

        x_7_w = tf.image.resize(x_7_w, (h, w))


        cat = tf.concat( [x_1_h + x_1_w, x_3_h + x_3_w, x_5_h + x_5_w, x_7_h + x_7_w], axis=-1)
        hx = self.relu(self.fuse_conv(cat))
        multi_scale_out = hx

        mask_1 = tf.sigmoid( self.conv_final(hx) )
        out1 = x * mask_1

        hx = self.mask_relu(self.mask_conv_1(out1))

        mask_2 = tf.sigmoid( self.mask_conv_2(hx) )
        hx = out1 * mask_2

        return hx, multi_scale_out

class BAM(Layer):
    def __init__(self, input_channel, channel_number, shortcut=False):
        super(BAM, self).__init__()
        self.shortcut = shortcut
        dilated_channel = channel_number // 2

        # Shortcut for residual
        if input_channel != channel_number:
            self.shortcut = True
            self.shortcut_conv = Conv2D(channel_number, kernel_size=(1,1), padding='SAME', use_bias=False)
        # Input conv
        self.input_conv = Conv2D(channel_number, kernel_size=(3,3), padding='SAME')
        self.input_relu = LeakyReLU(0.2)

        # PDC_1
        self.content_conv_d1 = Conv2D(dilated_channel, kernel_size=(3,3), padding='SAME', dilation_rate=2)
        self.content_conv_d3 = Conv2D(dilated_channel, kernel_size=(3,3), padding='SAME', dilation_rate=3)
        self.content_conv_d5 = Conv2D(dilated_channel, kernel_size=(3,3), padding='SAME', dilation_rate=5)
        # PDC_2
        self.content_conv_d1_2 = Conv2D(dilated_channel, kernel_size=(3,3), padding='SAME', dilation_rate=2)
        self.content_conv_d3_2= Conv2D(dilated_channel, kernel_size=(3,3), padding='SAME', dilation_rate=3)
        self.content_conv_d5_2 = Conv2D(dilated_channel, kernel_size=(3,3), padding='SAME', dilation_rate=5)
        # Bridge of CPDC
        self.fuse_conv_1 = Conv2D(channel_number, kernel_size=(3,3), padding='SAME')
        self.fuse_relu = LeakyReLU(0.2)
        # Fuse BA and CPDC
        self.final_conv = Conv2D(channel_number, kernel_size=(3,3), padding='SAME')
        self.fina_relu = LeakyReLU(0.2)

        # Blur-aware Attention
        self.BA = BA_Block(channel_number)

    def call(self, x):

        if self.shortcut:
            x = self.shortcut_conv(x)

        in_feature = self.input_relu((self.input_conv(x)))
        # BA
        BA_out, attn = self.BA(in_feature)

        # CPDC
        content_d1 = (self.content_conv_d1(in_feature))
        content_d3 = (self.content_conv_d3(in_feature))
        content_d5 = (self.content_conv_d5(in_feature))

        dilation_fusion_1 = self.fuse_relu(self.fuse_conv_1( tf.concat( [content_d1, content_d3, content_d5], axis=-1 ) ))
        content_d1_2 = (self.content_conv_d1_2(dilation_fusion_1))
        content_d3_2 = (self.content_conv_d3_2(dilation_fusion_1))
        content_d5_2 = (self.content_conv_d5_2(dilation_fusion_1))
        CPDC_out = tf.concat( (content_d1_2, content_d3_2, content_d5_2) , axis=-1)

        # Concatenate and Fuse
        BAM_out = self.final_conv(tf.concat((CPDC_out, BA_out), axis=-1))
        BAM_out = BAM_out + x
        BAM_out = self.fina_relu(BAM_out)

        return BAM_out, attn


class BANet(Model):
    def __init__(self, dim_1=64, dim_2=128, num_BAM_blocks=10, **kwargs):
        super(BANet, self).__init__(**kwargs)
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.num_BAM_blocks = num_BAM_blocks

        self.en_layer1 = Sequential(
            [Conv2D(dim_1, kernel_size=(3,3), padding='SAME'),
             LeakyReLU(0.2),]
        )

        self.en_layer2 = Sequential(
            [Conv2D(dim_2, kernel_size=(3,3), strides=(2,2), padding='SAME'),
             LeakyReLU(0.2),]
        )

        self.BAM_seq = [BAM(dim_2, dim_2) for _ in range(num_BAM_blocks)]
        self.de_layer1 = Sequential(
            [Conv2DTranspose(dim_1, kernel_size=(4,4), strides=(2,2), padding='SAME'),
             LeakyReLU(0.2)
            ]
        )
        self.de_layer2 = Sequential(
            [Conv2D(dim_1, kernel_size=(3,3), padding='SAME'),
             LeakyReLU(0.2),]
        )
        self.de_layer3 = Conv2D(3, kernel_size=(3,3), padding='SAME')

    def call(self, x):

        hx = self.en_layer1(x)

        residual = hx
        hx = self.en_layer2(hx)

        for BAM_i in self.BAM_seq:
            hx, attn_1 = BAM_i(hx)

        hx = self.de_layer1(hx)
        hx = self.de_layer2(tf.concat((hx, residual), axis=-1))
        hx = self.de_layer3(hx)
        # return hx + x
        return hx