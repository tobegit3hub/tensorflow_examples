#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Train data
mnist = input_data.read_data_sets("./mnist/", one_hot=True)

# Hyper parameters
learning_rate = 0.1
batch_size = 10
train_epoch_number = 100
display_interval = 10

# The variables to compute
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]), name="weight")
b = tf.Variable(tf.zeros([10]), name="bias")

pred = tf.nn.softmax(tf.matmul(x, W) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(train_epoch_number):
        total_loss = 0
        batch_number = mnist.train.num_examples / batch_size

        for i in range(batch_size):
            # Train
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
            total_loss += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})

            # Print the loss
        if epoch % display_interval == 0:
            print("Epoch: {}, loss: {}".format(epoch, total_loss))
