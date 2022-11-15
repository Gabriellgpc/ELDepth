# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-10-05 18:39:50
# @Last Modified by:   Condados
# @Last Modified time: 2022-11-14 22:15:25

from keras import Model
from keras.layers import Conv2D, \
                         ReLU, \
                         Concatenate

class XLRD(Model):
    def __init__(self, num_gblocks=4, nfeat=32, **kwargs):
        super(XLRD, self).__init__(**kwargs)
        self.num_gblocks = num_gblocks
        self.nfeat = nfeat

    def ClippedReLU(self, X):
        X = ReLU(max_value=1)(X)
        return X

    def ConvReLU(self, X, filters, kernel_size):
        X = Conv2D(filters, (kernel_size, kernel_size), padding='SAME')(X)
        X = ReLU()(X)
        return X

    def Gblock(self, X, filters, kernel_size=3, groups=4):
        X = Conv2D(filters, (kernel_size, kernel_size), padding='SAME', groups=groups)(X)
        X = ReLU()(X)
        X = self.ConvReLU(X, filters, kernel_size=1)
        return X

    def model(self, inp):
        res_conv_00 = self.ConvReLU(inp, filters=self.nfeat, kernel_size=3)
        res_conv_01 = self.ConvReLU(inp, filters=self.nfeat, kernel_size=3)
        res_conv_02 = self.ConvReLU(inp, filters=self.nfeat, kernel_size=3)
        res_conv_03 = self.ConvReLU(inp, filters=self.nfeat, kernel_size=3)
        res_conv_04 = self.ConvReLU(inp, filters=self.nfeat, kernel_size=3)

        res_cat_0 = Concatenate(axis=-1)([res_conv_00, res_conv_01, res_conv_02, res_conv_03])

        res = Conv2D(filters=self.nfeat, kernel_size=(1,1))(res_cat_0)
        for i in range(self.num_gblocks):
            res = self.Gblock(res, filters=self.nfeat, groups= min(2**i, 8) )

        res = Concatenate(axis=-1)([res, res_conv_04])
        res = Conv2D(filters=self.nfeat, kernel_size=(1,1))(res)
        res = ReLU()(res)
        res = Conv2D(filters=3, kernel_size=(3,3), padding='SAME')(res)

        return inp + res