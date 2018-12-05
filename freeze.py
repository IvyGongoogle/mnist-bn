from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from skimage import io
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import numpy as np
import cv2
FLAGS = None

def model():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32, [])
    y_ = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, [])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):

        conv1 = slim.conv2d(x_image, 16, [5, 5], scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2 = slim.conv2d(pool1, 32, [5, 5], scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
        flatten = slim.flatten(pool2)
        fc = slim.fully_connected(flatten, 1024, scope='fc1')
        drop = slim.dropout(fc, keep_prob=keep_prob)
        logits = slim.fully_connected(drop, 10, activation_fn=None, scope='logits')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

    return {'x': x,
            'y_': y_,
            'keep_prob': keep_prob,
            'is_training': is_training,
            'train_step': train_step,
            'global_step': step,
            'accuracy': accuracy,
            'cross_entropy': cross_entropy,
            'logits': logits,
            'fc1': fc,
            'conv2': conv2}

# freeze the ExponentialMovingAverage of final variables
def freeze():

    # get the computation graph
    net = model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(0.99, net['global_step'])
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    ckpt = tf.train.latest_checkpoint("./checkpoints_bn/")
    if ckpt:
        saver.restore(sess, ckpt)
        print("restore from the checkpoint {0}".format(ckpt))

    output_node_names =["fc1/Relu"]

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)
    tf.graph_util.remove_training_nodes(frozen_graph_def)

    # Save the frozen graph
    with open('output/mnist_bn.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())


# freeze the final variables
def freeze_graph():

    meta_path = "./checkpoints_bn/mnist-conv-slim.meta"
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config = config) as sess:

        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        saver.restore(sess,tf.train.latest_checkpoint("./checkpoints_bn/"))

        # Output nodes
        output_node_names =["fc1/Relu"]

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)
        tf.graph_util.remove_training_nodes(frozen_graph_def)

        # Save the frozen graph
        with open('output/mnist_bn.pb', 'wb') as f:
          f.write(frozen_graph_def.SerializeToString())

if __name__ == '__main__':
    freeze()
