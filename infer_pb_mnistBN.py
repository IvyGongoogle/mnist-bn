# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import os
import sys
#from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import cv2
from skimage import io

if __name__ == '__main__':

    ## for mnist-bn
    from tensorflow.examples.tutorials.mnist import input_data
    data_dir="/search/odin/gongzhenting/work/image/models/BN/mnist-bn/MNIST_data"
    batch_size=50
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    print ("~~~~~~~~~batch_xs.shape:",batch_xs.shape)

    with open("./mnist-bn.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    tf_sess = tf.Session(config=tf_config, graph=graph)
    tf_input = tf_sess.graph.get_tensor_by_name("Placeholder:0")
    # tf_output = tf_sess.graph.get_tensor_by_name("conv2/Relu:0")
    tf_output = tf_sess.graph.get_tensor_by_name("logits/batch_normalization/batchnorm/add_1:0")


    logits = tf_sess.run([tf_output], feed_dict={
        tf_input: np.reshape(batch_xs, [-1,28,28,1])
    })
    np.savetxt('./logits_pb/0.txt', logits, fmt = '%8f')
