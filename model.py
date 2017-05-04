import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
import ops

def build_block_source(img, C=32):
    with tf.name_scope("Block"):
        lt = [64, 40, 24, 14, 4]
        lc = [1, 2, 4, 8, 32]
        if C not in lc:
            C = 32
        out_depth = int(img.get_shape()[3])
        bottleneck = lt[lc.index(C)]
        hid_layers = range(C)
        for i in range(C):
            hid_layers[i] = ly.conv2d(img, bottleneck, 1)
            hid_layers[i] = ly.conv2d(hid_layers[i], bottleneck, 3)
        out = tf.concat(hid_layers, axis=3)
        out = ly.conv2d(out, out_depth, 1, activation_fn=tf.nn.tanh)
        out = out + img
    return out

def build_block(img, C=32):
    with tf.name_scope("Block"):
        lt = [64, 40, 24, 14, 4]
        lc = [1, 2, 4, 8, 32]
        if C not in lc:
            C = 32
        out_depth = int(img.get_shape()[3])
        bottleneck = lt[lc.index(C)]
        hid_layers = range(C)
        for i in range(C):
            hid_layers[i] = ly.conv2d(img, bottleneck, 1)
            hid_layers[i] = ly.conv2d(hid_layers[i], bottleneck, 3)
        out = tf.concat(hid_layers, axis=3)
    return out

def build_block_fukc(img):
    out = ly.conv2d(img, 32, 3)
    out = ly.conv2d(out, 32, 3)
    out = ly.conv2d(out, 32, 3)
    out = ly.conv2d(out, 32, 3)
    out = ly.conv2d(out, 32, 3)

    out = ly.conv2d(out, 32, 3)
    out = ly.conv2d(out, 32, 3)
    return out+img

def down_up_depth_feature(input):
    out = ly.conv2d(input, 16, 3, 2)
    out = ly.conv2d(out, 32, 3, 2)
    out = ly.convolution2d_transpose(out, 32, 3, 2)
    out = ly.convolution2d_transpose(out, 32, 3, 2)
    return out
