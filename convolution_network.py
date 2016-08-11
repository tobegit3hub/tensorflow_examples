#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

learning_rate = 0.1
training_epochs = 1000
batch_size = 256
display_step = 10

n_input = 784
n_output = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Convolute layer
w1 = tf.Variable(tf.random_normal([3, 3, 1, 64]))
b1 = tf.Variable(tf.random_normal([64]))
input_r = tf.reshape(x, shape=[-1, 28, 28, 1])
conv1 = tf.nn.conv2d(input_r, w1, strides=[1, 1, 1, 1], padding="SAME")
conv2 = tf.nn.bias_add(conv1, b1)
conv3 = tf.nn.relu(conv2)
pool = tf.nn.max_pool(conv3,
                      ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1],
                      padding="SAME")

# Full connected layer
w2 = tf.Variable(tf.random_normal([14 * 14 * 64, n_output]))
b2 = tf.Variable(tf.random_normal([n_output]))
dense = tf.reshape(pool, [-1, w2.get_shape().as_list()[0]])
pred = tf.add(tf.matmul(dense, w2), b2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs,
                                              y: batch_ys}) / total_batch

        if epoch % display_step == 0:
            print("Epoch: {}, cost: {}".format(epoch, avg_cost))
