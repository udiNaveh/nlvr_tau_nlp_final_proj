'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Modified by Alane Suhr
'''

from __future__ import print_function

import math
import tensorflow as tf

### conv2d
# A convolutional layer.
# 
# Inputs:
#    x: input to layer.
#    W: kernel.
#    b: biases.
#    strides: stride to use.
#
# Outputs:
#    outputs of layer.
def conv2d(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)

### conv_net
# Convolutional network.
#
# Inputs:
#    x: input to network.
#    filter_sizes: filter sizes for each layer.
#    filter_insizes: sizes of filter results. 
#    filter_strides: strides to use.
#    out_size: size of output of convnet.
#
# Outputs:
#    output of convnet and weight TF variables created.
def conv_net(x, filter_sizes, filter_insizes, filter_strides, out_size):
  weights = [ ]

  for layer_num in range(len(filter_sizes)):
    size = filter_sizes[layer_num]
    insize = filter_insizes[layer_num]

    if layer_num < len(filter_sizes) - 1:
      outsize = filter_insizes[layer_num + 1]
    else:
      outsize = out_size

    stride = filter_strides[layer_num]

    # Kernel and biases
    k1 = tf.Variable(tf.random_normal([size, size, insize, outsize],
                                      stddev = 0.01))
    b1 = tf.Variable(tf.zeros([outsize]))
    weights.append(k1)

    x = conv2d(x, k1, b1, stride)
  return x, weights
