from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from retrain import should_distort_images
from retrain import ensure_dir_exists
from retrain import create_model_info
tf.app.flags.DEFINE_string('architecture', 'resnet_v2_50', "define architecture")
tf.app.flags.DEFINE_string('summaries_dir', '../retrain_logs', "log")
tf.app.flags.DEFINE_integer('intermediate_store_frequency', 0, "descript3")
tf.app.flags.DEFINE_integer('testing_percentage', 20, "descript3")
tf.app.flags.DEFINE_integer('validation_percentage', 20, "descript3")
tf.app.flags.DEFINE_string('image_dir', '/home/zhongyi/PycharmProjects/GraduationThesis/generated_images/', "image directionary")
tf.app.flags.DEFINE_bool('flip_left_right', False, "image directionary")
tf.app.flags.DEFINE_integer('random_crop', 0, "image directionary")
tf.app.flags.DEFINE_integer('random_scale', 0, "image directionary")
tf.app.flags.DEFINE_integer('random_brightness', 0, "image directionary")
tf.app.flags.DEFINE_string('model_dir', False, "image directionary")


FLAGS = tf.app.flags.FLAGS
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_model_graph(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Args:
      model_info: Dictionary containing information about the model architecture.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        print('Model path: ', model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


def maybe_download_and_extract(data_url):
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.

    Args:
      data_url: Web location of the tar file containing the pretrained model.
    """
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded %s %d bytes.', filename,
                        statinfo.st_size)
        print('Extracting file from ', filepath)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print('Not extracting or downloading files, model already present in disk')


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return

def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it.

    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.

    Args:
      architecture: Name of a model architecture.

    Returns:
      Dictionary of information about the model, or None if the name isn't
      recognized

    Raises:
      ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()
    is_quantized = False
    if architecture == 'inception_v3':
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
    elif architecture.startswith('resnet_'):
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gzz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'net' ###
        bottleneck_tensor_size = 2048
        input_width = 224
        input_height = 224
        input_depth = 3
        resized_input_tensor_name = 'input'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128

    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'",
                             architecture)
            return None
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
                version_string != '0.5' and version_string != '0.25'):
            tf.logging.error(
                """"The Mobilenet version should be '1.0', '0.75', '0.5', or '0.25',
  but found '%s' for architecture '%s'""", version_string, architecture)
            return None
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
                size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
                size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quant':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                    architecture)
                return None
            is_quantized = True

        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/'
        model_name = 'mobilenet_v1_' + version_string + '_' + size_string
        if is_quantized:
            model_name += '_quant'
        data_url += model_name + '.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        resized_input_tensor_name = 'input:0'
        model_file_name = model_name + '_frozen.pb'

        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'quantize_layer': is_quantized,
    }


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['png', 'jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def main(_):
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)


    # Prepare necessary directories that can be used during training
    prepare_file_system()

    # Gather information about the model architecture we'll be using.
    model_info = create_model_info(FLAGS.architecture)
    if not model_info:
        tf.logging.error('Did not recognize architecture flag')
        return -1

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         FLAGS.image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    # See if the command-line flags mean we're applying any distortions.
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)

    # Set up the pre-trained graph.
    #maybe_download_and_extract(model_info['data_url'])
    graph, bottleneck_tensor, resized_image_tensor = (
        create_model_graph(model_info))

    # Add the new layer that we'll be training.
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
             class_count, FLAGS.final_tensor_name, bottleneck_tensor,
             model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
             True)

    with tf.Session(graph=graph) as sess:
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])

        if do_distort_images:
            # We will be applying distortions, so setup the operations we'll need.
            (distorted_jpeg_data_tensor,
             distorted_image_tensor) = add_input_distortions(
                 FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
                 FLAGS.random_brightness, model_info['input_width'],
                 model_info['input_height'], model_info['input_depth'],
                 model_info['input_mean'], model_info['input_std'])
        else:
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                              FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, FLAGS.architecture)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')

        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        train_saver = tf.train.Saver()

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS.how_many_training_steps):
            # Get a batch of input bottleneck values, either calculated fresh every
            # time with distortions applied, or from the cache stored on disk.
            if do_distort_images:
                (train_bottlenecks,
                 train_ground_truth) = get_random_distorted_bottlenecks(
                     sess, image_lists, FLAGS.train_batch_size, 'training',
                     FLAGS.image_dir, distorted_jpeg_data_tensor,
                     distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks,
                 train_ground_truth, _) = get_random_cached_bottlenecks(
                     sess, image_lists, FLAGS.train_batch_size, 'training',
                     FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                     decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                     FLAGS.architecture)
            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                (datetime.now(), i, train_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f' %
                                (datetime.now(), i, cross_entropy_value))
                # TODO(suharshs): Make this use an eval graph, to avoid quantization
                # moving averages being updated by the validation set, though in
                # practice this makes a negligable difference.
                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks(
                        sess, image_lists, FLAGS.validation_batch_size, 'validation',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                        FLAGS.architecture))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                (datetime.now(), i, validation_accuracy * 100,
                                 len(validation_bottlenecks)))

            # Store intermediate results
            intermediate_frequency = FLAGS.intermediate_store_frequency

            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                    and i > 0):
                # If we want to do an intermediate save, save a checkpoint of the train
                # graph, to restore into the eval graph.
                train_saver.save(sess, CHECKPOINT_NAME)
                intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                          'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' +
                                intermediate_file_name)
                save_graph_to_file(graph, intermediate_file_name, model_info,
                                   class_count)

        # After training is complete, force one last save of the train checkpoint.
        train_saver.save(sess, CHECKPOINT_NAME)

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        run_final_eval(sess, model_info, class_count, image_lists, jpeg_data_tensor,
                       decoded_image_tensor, resized_image_tensor,
                       bottleneck_tensor)

        # Write out the trained graph and labels with the weights stored as
        # constants.
        save_graph_to_file(graph, FLAGS.output_graph, model_info, class_count)
        with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        export_model(model_info, class_count, FLAGS.saved_model_dir)

if __name__ == '__main__':
    tf.app.run()