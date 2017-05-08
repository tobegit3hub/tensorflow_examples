#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# 1:1 3:0.7 7:2.5
# 0:1.3 3:0 8:2.8
examples = tf.SparseTensor(indices=[[0, 1], [0, 3], [0, 7], [1, 0], [1, 3], [1, 8]], 
                           values=["a", "0.7", "2.5", "1.3", "0", "2.8"], 
                           shape=[2, 10])

vocabulary_size = 10
embedding_size = 1
batch_size = 2
valid_feature_number = 3

#embeddings = tf.Variable(tf.ones([vocabulary_size, embedding_size]))
embeddings = tf.Variable([0,1,4,9,16,25,36,49,64,81])

train_inputs = tf.placeholder(tf.int32, shape=[batch_size, valid_feature_number])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    batch_data = np.array([[1, 3, 7], [0, 3, 8]])
    print(sess.run(embeddings))
    print(sess.run(embed, feed_dict={train_inputs: batch_data}))
