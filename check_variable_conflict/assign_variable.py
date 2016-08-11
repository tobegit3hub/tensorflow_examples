#!/usr/bin/env python

import tensorflow as tf
import time

var1 = tf.Variable(0.0, name="var1")
init_op = tf.initialize_all_variables()
assign_op = tf.assign(var1, 2.0)

with tf.Session("grpc://localhost:44105") as sess:
    sess.run(init_op)

    sess.run(assign_op)

    while True:
        print("Var1 is {}, sleep 1 second".format(sess.run(var1)))
        time.sleep(1)
