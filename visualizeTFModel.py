import tensorflow as tf
g = tf.Graph()

with g.as_default() as g:
    # tf.train.import_meta_graph('/Users/gongzhenting/work/image/models/mnist/ckpt/lenet5_model.meta')
    # tf.train.import_meta_graph('/Users/gongzhenting/work/language/ocr/detection/Online_Detecttion/bestmodel/model.ckpt-487831.meta')
    tf.train.import_meta_graph("./checkpoints/mnist-conv-slim.meta")    

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='./logs/', graph=g)
    file_writer.flush()
    file_writer.close()
