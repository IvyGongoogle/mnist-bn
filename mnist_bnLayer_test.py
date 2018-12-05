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

def model():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32, [])
    y_ = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, [])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training, 'momentum': 0.95}):

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

def test():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # get the computation graph
    net = model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        print("restore from the checkpoint {0}".format(ckpt))

    acc = 0.0
    batch_size = FLAGS.batch_size
    # batch_size = 1
    num_iter = 10000 // batch_size
    for i in range(1):
        batch_xs, batch_ys = mnist.test.next_batch(batch_size, shuffle=False)

        print ("~~~~~~~~~batch_xs[0].shape:",batch_xs[0].shape)
        io.imsave('test.jpg', np.reshape(batch_xs[0], [28,28]))
        feed_dict = {net['x']: np.reshape(batch_xs, [-1,28,28,1]),
                     net['y_']: np.reshape(batch_ys,[-1,10]),
                     net['keep_prob']: 1.0,
                     net['is_training']: False}
                     
        fc1 = sess.run(net['fc1'], feed_dict=feed_dict)
        print ("fc1.shape:{}".format(fc1.shape))
        np.savetxt('./output/fc1/'+str(i)+'_bn_layer.txt', fc1, fmt = '%8f')

    sess.close()


def main(_):
    if FLAGS.phase == 'train':
        train()
    else:
        test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MNIST_data',
                        help='Directory for storing input data')
    parser.add_argument('--phase', type=str, default='test',
                        help='Training or test phase, should be one of {"train", "test"}')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Training or test phase, should be one of {"train", "test"}')
    parser.add_argument('--train_log_dir', type=str, default='logs_bn_layer',
                        help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_bn_layer',
                        help='Directory for checkpoint file')
    FLAGS, unparsed = parser.parse_known_args()
    if not os.path.isdir(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
