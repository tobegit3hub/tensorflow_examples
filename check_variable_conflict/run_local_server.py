#!/usr/bin/env python

import tensorflow as tf

server = tf.train.Server.create_local_server()

server.join()
