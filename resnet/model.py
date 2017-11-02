import sys
import numpy as np
import tensorflow as tf


def create_normal_variable(shape, name=None):
  normal_initializer = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(normal_initializer, name=name)


def softmax_layer(input, shape):
  weight = create_normal_variable(shape)
  bias = tf.Variable(tf.zeros([shape[1]]))
  predict = tf.matmul(input, weight) + bias
  softmax = tf.nn.softmax(predict)

  return softmax


def conv_bn_relu_layer(input, filter_shape, stride):
  output_channel_shape = filter_shape[3]

  # Convolution
  filter_weight = create_normal_variable(filter_shape)
  conv = tf.nn.conv2d(
      input,
      filter=filter_weight,
      strides=[1, stride, stride, 1],
      padding="SAME")

  # BN
  mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
  beta = tf.Variable(tf.zeros([output_channel_shape]), name="beta")
  gamma = create_normal_variable([output_channel_shape], name="gamma")

  batch_norm = tf.nn.batch_norm_with_global_normalization(
      conv, mean, var, beta, gamma, 0.001, scale_after_normalization=True)

  # ReLU
  output_layer = tf.nn.relu(batch_norm)

  return output_layer


def residual_block(input, output_shape_depth, is_down_sample):
  input_shape_depth = input.get_shape().as_list()[3]

  if is_down_sample:
    filter_ = [1, 2, 2, 1]
    input = tf.nn.max_pool(
        input, ksize=filter_, strides=filter_, padding='SAME')

  layer = conv_bn_relu_layer(input,
                             [3, 3, input_shape_depth, output_shape_depth], 1)
  layer = conv_bn_relu_layer(layer,
                             [3, 3, output_shape_depth, output_shape_depth], 1)

  if input_shape_depth != output_shape_depth:
    input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0],
                                 [0, output_shape_depth - input_shape_depth]])
  else:
    input_layer = input

  final_layer = layer + input_layer
  return final_layer


def resnet(input, layer_number):
  """
  Implement the ResNet from the paper.
  """

  if layer_number < 20 or (layer_number - 20) % 12 != 0:
    print("Error, the number of layers is invalid: {}".format(layer_number))
    sys.exit(1)

  num_conv = (layer_number - 20) / 12 + 1
  layers = []

  with tf.variable_scope('conv1'):
    conv1 = conv_bn_relu_layer(input, [3, 3, 3, 16], 1)
    layers.append(conv1)

  for i in range(num_conv):
    with tf.variable_scope('conv2_%d' % (i + 1)):
      conv2_x = residual_block(layers[-1], 16, False)
      # get_shape().as_list()[1:] == [32, 32, 16]
      conv2 = residual_block(conv2_x, 16, False)
      layers.append(conv2_x)
      layers.append(conv2)

  for i in range(num_conv):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv3_%d' % (i + 1)):
      conv3_x = residual_block(layers[-1], 32, down_sample)
      # get_shape().as_list()[1:] == [16, 16, 32]
      conv3 = residual_block(conv3_x, 32, False)
      layers.append(conv3_x)
      layers.append(conv3)

  for i in range(num_conv):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i + 1)):
      conv4_x = residual_block(layers[-1], 64, down_sample)
      # get_shape().as_list()[1:] == [8, 8, 64]
      conv4 = residual_block(conv4_x, 64, False)
      layers.append(conv4_x)
      layers.append(conv4)

  with tf.variable_scope('fc'):
    # get_shape().as_list()[1:] == [64]
    global_pool = tf.reduce_mean(layers[-1], [1, 2])
    layer = softmax_layer(global_pool, [64, 10])
    layers.append(layer)

  final_layer = layers[-1]
  return final_layer
