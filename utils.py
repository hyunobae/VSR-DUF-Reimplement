# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import ReLU, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import Input

def Model():
    stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
    sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
    relu = tf.keras.layers.Activation('relu')
    inputs = Input(shape=(128,))
    x = Conv3D(tf.pad(inputs, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1,1,1,1,1], 'VALID', name='conv1')
    uf = 4
    F = 64
    G = 16
    for r in range(0, 21):
        t = BatchNorm(x, is_train)
        t = ReLU(t)
        t = Conv3D(t, [1,1,1,F,F], [1,1,1,1,1], 'VALID')

        t = BatchNorm(t, is_train)
        t = ReLU(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3,3,3,F,G], [1,1,1,1,1], 'VALID')

        x = tf.concat([x,t], 4)
        F += G

    for r in range(21, 24):
        t = BatchNorm(x, is_train)
        t = ReLU(t)
        t = Conv3D(t, [1,1,1,f,f], [1,1,1,1,1], 'VALID')

        t = BatchNorm(t, is_train)
        t = ReLU(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3,3,3,F,G,], [1,1,1,1,1], 'VALID')
        x = tf.concat([x[:, 1:-1], t], 4)
        F+= G

    x = BatchNorm(x, is_train)
    x = ReLU(t)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1,3,3,448, 256], [1,1,1,1,1], 'VALID')
    x = ReLU(t)

    r = Conv3D(x, [1,1,1,256,256], [1,1,1,1,1], 'VALID')
    r = ReLU(t)
    r = Conv3D(x, [1,1,1,256,3*uf*uf], [1,1,1,1,1], 'VALID')

    f = Conv3D(x, [1,1,1,256,512], [1,1,1,1,1], 'VALID')
    f = ReLU(t)
    f = Conv3D(f, [1,1,1,512,1*5*5*uf*uf], [1,1,1,1,1], 'VALID')

    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], de_f[3], 25, uf*uf])
    f = Softmax(f, axis=4)

    #model_residual = Model(inputs, r)
    #model_filter = Model(inputs, f)
    #model = Model(inputs, [r,f]

    xc = []
    t = DynFilter3D(x[:, 7 // 2:7 // 2 + 1, :, :, c], f[:, 0, :, :, :, :], [1, 5, 5])  # [B,H,W,R*R]
    t = tf.depth_to_space(t, 4)
    xc += [t]
    x = tf.concat(xc, axis=3)

    x = tf.expand_dims(x, axis=1)
    r = depth_to_space_3D(r, 4)
    x += r

    model = Model(inputs, x)

    return model

#def G(x):
#    Fx, Rx = model(x)
#    x_c = []
#    t = DynFilter3D(x[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], [1, 5, 5])  # [B,H,W,R*R]
#    t = tf.depth_to_space(t, R)
#    x_c += [t]
#    x = tf.concat(x_c, axis=3)
#
#    x = tf.expand_dims(x, axis=1)
#    Rx = depth_to_space_3D(Rx, R)
#    x += Rx
#    return x
#

def LoadImage(path, color_mode='Y', channel_mean=None, modcrop=[0,0,0,0]):
    '''Load an image us    t = DynFilter3D(x[:
ing PIL and convert it into specified color space,
    and return it as an numpy array.

    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    '''
    ## Load image
    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')

    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:,:,0:1]

    ## To 0-1
    x *= 1.0/255.0

    if channel_mean:
        x[:,:,0] -= channel_mean[0]
        x[:,:,1] -= channel_mean[1]
        x[:,:,2] -= channel_mean[2]

    if modcrop[0]*modcrop[1]*modcrop[2]*modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]

    return x


he_normal_init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')

def BatchNorm(input, is_train, decay=0.999, name='BatchNorm'):

    return tf.keras.layers.BatchNormalization(momentum=decay, epsilon=1e-3, training=is_train)(input) #, tf.stack([mean[0], variance[0], beta[0], gamma[0]])


def Conv3D(input, kernel_shape, strides, padding, name='Conv3d', W_initializer=he_normal_init, bias=True):
    with tf.variable_scope(name):
        W = tf.get_variable("W", kernel_shape, initializer=W_initializer)
        if bias is True:
            b = tf.get_variable("b", (kernel_shape[-1]),initializer=tf.constant_initializer(value=0.0))
        else:
            b = 0

    return tf.keras.layers.Conv3d(kernel_size=W, strides=strides, padding=padding)(input) + b



def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], ds_x[4]])

    y = tf.depth_to_space(x, block_size)

    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x

def DynFilter3D(x, F, filter_size):
    '''
    3D Dynamic filtering
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    '''
    # make tower
    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
    filter_localexpand = tf.Variable(filter_localexpand_np, trainable=False, dtype='float32',name='filter_localexpand') 
    x = tf.transpose(x, perm=[0,2,3,1])
    x_localexpand = tf.nn.conv2d(x, filter_localexpand, [1,1,1,1], 'SAME') # b, h, w, 1*5*5
    x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, h, w, 1, 1*5*5
    x = tf.matmul(x_localexpand, F) # b, h, w, 1, R*R
    x = tf.squeeze(x, axis=3) # b, h, w, R*R

    return x

def Huber(y_true, y_pred, delta=0.01, axis=None):
    abs_error = tf.abs(y_pred - y_true)
    quadratic = tf.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses, axis=axis)

class CustomDataloader(Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), np.array(batch_y)

