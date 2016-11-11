#!/usr/bin/env python

import tensorflow as tf

print("Run TensorFlow to multiple numbers")

import pudb;pudb.set_trace()

number = tf.constant(3)

with tf.Session() as sess:
  print(sess.run(number))

print("End of TensorFlow")
