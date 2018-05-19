# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim.nets
import os
from tensorflow.python.tools import freeze_graph
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import gfile
resnet_v2 = tf.contrib.slim.nets.resnet_v2
slim = tf.contrib.slim
resnet_utils = tf.contrib.slim.nets.resnet_utils
from tensorflow.python.framework import graph_util


model_path  = "./pretraindata/resnet_v2_50.ckpt"

def test_graph_name(model_path='output//model.pb'):
    with tf.Graph().as_default() as graph:
        #model_path = 'output//model.pb'
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")
    gname = [n.name for n in graph.as_graph_def().node]
    print('name length =', gname.__len__())
    return gname


def showtensorname(ckptpath):
    checkpoint_path = ckptpath
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    """
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key)) # Remove this is you want to print only variable names    
    """
    return var_to_shape_map, reader

with tf.Graph().as_default() as g:
    input_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_node')

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        net, endpoints=resnet_v2.resnet_v2_50(input_node)


    #trainable = tf.trainable_variables()
    #variables_to_restore = [i for i in trainable if 'block' in i.name]
    #restorer = tf.train.Saver(variables_to_restore)
    #restorer = tf.train.Saver([i for i in tf.trainable_variables() if 'block' in i.name])
    restorer = tf.train.Saver([i for i in tf.trainable_variables()])
    """
    #glob_var = tf.global_variables()
    #restorer = tf.train.Saver(glob_var)
    """
    saver = tf.train.Saver()

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, model_path)
    a=sess.run(g.get_tensor_by_name('resnet_v2_50/conv1/weights:0'))
    #print(g.get_tensor_by_name('resnet_v2_50/postnorm/beta:0'))
    #var_list = tf.trainable_variables()
    #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[var_list[i].name for i in range(len(var_list))])
    #tf.train.write_graph(constant_graph, './output', 'resnet_v2_50.pb', as_text=False)
    tf.train.write_graph(sess.graph_def, '', 'resnet_model_binary.pb', as_text=False)

    freeze_graph.freeze_graph('resnet_model_binary.pb', '', True, model_path, 'resnet_v2_50/pool5', 'save/restore_all', 'save/Const:0', 'frozen_model.pb', False, "")

gname = test_graph_name(model_path='frozen_model.pb')

ckpt, reader = showtensorname(model_path)

b=reader.get_tensor('resnet_v2_50/conv1/weights')
print(a==b)

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
