import tensorflow as tf
import tensorflow.contrib.slim.nets
import os
from tensorflow.python.tools import freeze_graph
from tensorflow.python import pywrap_tensorflow
resnet_v2 = tf.contrib.slim.nets.resnet_v2
slim = tf.contrib.slim
resnet_utils = tf.contrib.slim.nets.resnet_utils
from tensorflow.python.framework import graph_util


model_path  = "./pretraindata/resnet_v2_50.ckpt"

def showtensorname(ckptpath):
    checkpoint_path = ckptpath
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key)) # Remove this is you want to print only variable names

with tf.Graph().as_default() as g:
    input_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        net, endpoints=resnet_v2.resnet_v2_50(input_node)
    """
    trainable = tf.trainable_variables()
    variables_to_restore = [i for i in trainable if 'block' in i.name]
    restorer = tf.train.Saver(variables_to_restore)
    """
    glob_var = tf.global_variables()
    restorer = tf.train.Saver(glob_var)
    saver = tf.train.Saver()

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, model_path)
    #var_list = tf.trainable_variables()
    #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[var_list[i].name for i in range(len(var_list))])
    #tf.train.write_graph(constant_graph, './output', 'resnet_v2_50.pb', as_text=False)
    tf.train.write_graph(sess.graph_def, 'output', 'model.pb')
    freeze_graph.freeze_graph('output/model.pb', '', False, model_path, 'resnet_v2_50/pool5', 'save/restore_all',
                              'save/Const:0', 'output//frozen_model.pb', False, "")


def main():

    """


    tf.reset_default_graph()
    input_node = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    net, end_points = resnet_v2.resnet_v2_50(input_node)
    trainable = tf.trainable_variables()
    variables_to_restore = [i for i in trainable if 'block' in i.name]
    for i in range(variables_to_restore.__len__()):
        if variables_to_restore[i] == 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/biases':
            print("mmmmmmmmmmm",i)
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, model_path)



    print("done")
    :return:
    """

if __name__ == '__main__':
    main()