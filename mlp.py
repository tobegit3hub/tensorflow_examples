#!/usr/bin/env python

import tensorflow as tf
import math
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define the model
input_units = 784
hidden1_units = 10
hidden2_units = 20
output_units = 10

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

# Hidden 1
with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([input_units, hidden1_units],
                            stddev=1.0 / math.sqrt(float(input_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

# Hidden 2
with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

# Linear
with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, output_units],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([output_units]), name='biases')

    logits = tf.matmul(hidden2, weights) + biases

# Define loss and train op
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
learning_rate = 0.01
batch_size = 10

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init_op)

    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict={x: batch_x,
                                            y: batch_y})
        print(loss_value)
