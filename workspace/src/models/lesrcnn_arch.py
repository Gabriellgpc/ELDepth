from keras import Model
from keras.layers import Conv2D, \
                         ReLU, \
                         Concatenate


class LESRCNN(Model):
    def __init__(self):
        super(LESRCNN, self).__init__()

    def ResidualBlock(self, X, nfilters):
        out = Conv2D(nfilters, (3,3), padding='SAME')(X)
        out = ReLU()(out)
        out = Conv2D(nfilters, (3,3), padding='SAME')(out)
        return out + X

    def EResidualBlock(self, X, nfilters):
        out = Conv2D(nfilters, (3,3), padding='SAME')(X)
        out = ReLU()(out)
        out = Conv2D(nfilters, (3,3), padding='SAME')(X)
        out = ReLU()(out)
        out = Conv2D(nfilters, (1,1))(out)
        return out + X

    def UpsampleBlock(self, X, scale):
        pass


    def model(self, inp, ngblocks=4):
        res_conv_00 = self.ConvReLU(inp, filters=8, kernel_size=3)
        res_conv_01 = self.ConvReLU(inp, filters=8, kernel_size=3)
        res_conv_02 = self.ConvReLU(inp, filters=8, kernel_size=3)
        res_conv_03 = self.ConvReLU(inp, filters=8, kernel_size=3)
        res_conv_04 = self.ConvReLU(inp, filters=16, kernel_size=3)

        res_cat_0 = Concatenate(axis=-1)([res_conv_00, res_conv_01, res_conv_02, res_conv_03])

        res = Conv2D(filters=32, kernel_size=(1,1))(res_cat_0)
        for _ in range(ngblocks):
            res = self.Gblock(res, filters=32, groups=4)

        res = Concatenate(axis=-1)([res, res_conv_04])
        res = Conv2D(filters=32, kernel_size=(1,1))(res)
        res = ReLU()(res)
        res = Conv2D(filters=3, kernel_size=(3,3), padding='SAME')(res)

        return inp + res