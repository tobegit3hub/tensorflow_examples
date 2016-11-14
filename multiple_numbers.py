#!/usr/bin/env python

import tensorflow as tf


import pudb;pudb.set_trace()

number1 = tf.constant(1.5)
number2 = tf.constant(3.5)
add_op = number1 + number2

with tf.Session() as sess:
  print(sess.run(add_op))

