# from tensorflow.python.tools import freeze_graph
# input_graph="/search/odin/gongzhenting/work/ml-tools/transfer/models/resnet/imagenet_resnet_v2_152.ckpt.meta"
# checkpoint_path="/search/odin/gongzhenting/work/ml-tools/transfer/models/resnet/imagenet_resnet_v2_152.ckpt"
# out_names="softmax"
# models_dir="./"
# input_binary=True
# model_filename="freeze_graph.pb"
# freeze_graph.freeze_graph(input_graph, '',
#                         input_binary, checkpoint_path, out_names,
#                         "save/restore_all", "save/Const:0",
#                         models_dir+model_filename, False, "")


# import tensorflow as tf
# from tensorflow.python.framework import graph_io
# # meta_path = '/search/odin/gongzhenting/work/ml-tools/Tensorflow/tensorflow-master/tensorflow/examples/tutorials/mnist/ckpt/lenet5_model.meta' # Your .meta file
# # meta_path = "/search/odin/gongzhenting/work/language/ocr/detection/Online_Detecttion/bestmodel/model.ckpt-487831"
# checkpoint_path="/search/odin/gongzhenting/work/language/ocr/detection/Online_Detecttion/bestmodel/model.ckpt-487831"
# config = tf.ConfigProto(allow_soft_placement = True)
# with tf.Session(config = config) as sess:
#     saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
#     saver.restore(sess, checkpoint_path)
#     input_ = tf.get_collection("input:0", scope="")[0]
#     output_ = tf.get_collection("output:0", scope="")[0]
#     input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
#     output = sess.run(output_, feed_dict={input_: input_images})
#     print "output", output

        # # -- this works too
        # output = sess.run("output:0", feed_dict={"input:0": np.arange(10, dtype=np.float32)})

# import tensorflow as tf
# gf = tf.GraphDef()
# print gf.ParseFromString(open('output_graph.pb','rb').read())
# print [n.name + '=>' +  n.op for n in gf.node if n.op in ( 'Softmax','Mul')]

import tensorflow as tf

# meta_path = '/search/odin/gongzhenting/work/ml-tools/Tensorflow/tensorflow-master-test/tensorflow/examples/tutorials/mnist/ckpt/lenet5_model.meta' # Your .meta file
# meta_path = "/search/odin/gongzhenting/work/language/ocr/detection/forgzt_v2_ztgong/models/model.ckpt-10101.meta"
meta_path = "./checkpoints/mnist-conv-slim.meta"
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config = config) as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint("./checkpoints/"))

    # Output nodes
    # output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
    # print (output_node_names)
    output_node_names =["logits/batch_normalization/batchnorm/add_1"]
    # output_node_names =["/Relu"]
    # output_node_names =["conv2/Relu"]

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)
    tf.graph_util.remove_training_nodes(frozen_graph_def)

    # Save the frozen graph
    with open('mnist-bn.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
