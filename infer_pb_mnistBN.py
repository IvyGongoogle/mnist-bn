# coding:utf-8
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import os
import sys
#from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from skimage import io
import argparse

tf.app.flags.DEFINE_string('gpu_list', '3', '')
FLAGS = tf.app.flags.FLAGS

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="output/mnist_bn.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    #加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    #op.values() gives you a list of tensors it produces
    #op.name gives you the name
    #输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name, op.values())

    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    is_training = graph.get_tensor_by_name('prefix/Placeholder_3:0')
    y = graph.get_tensor_by_name('prefix/fc1/Relu:0')

    ## for mnist-bn
    from tensorflow.examples.tutorials.mnist import input_data
    data_dir="/search/odin/gongzhenting/work/image/models/BN/mnist-bn/MNIST_data"
    batch_size=50
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    batch_xs, batch_ys = mnist.test.next_batch(batch_size, shuffle=False)
    io.imsave('test.jpg', np.reshape(batch_xs[0], [28,28]))
    print ("~~~~~~~~~batch_xs.shape:",batch_xs.shape)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    with tf.Session(config=tf_config, graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: np.reshape(batch_xs, [-1,28,28,1]),
            is_training:False
        })
        print(y_out)
        np.savetxt('./output/logits_pb/0_bn.txt', y_out, fmt = '%8f')
    print ("finish")
