"""Train a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import deeplab_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug

import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=1000,
                    help='Number of training epochs.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=30000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, default='./dataset/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--pre_trained_model', type=str, default='./ini_checkpoints/resnet_v2_50/resnet_v2_50.ckpt',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 50
_HEIGHT = 200
_WIDTH = 200
_DEPTH = 1
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997

_NUM_IMAGES = {
    'train': 4596,
    'validation': 1148,
}


def get_filenames(is_training, data_dir):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return [os.path.join(data_dir, 'txt_train.record')]
  else:
    return [os.path.join(data_dir, 'txt_val.record')]


def parse_record(raw_record):
  """Parse image and label from a tf record."""
  keys_to_features = {
      'image/height':
      tf.FixedLenFeature((), tf.int64),
      'image/width':
      tf.FixedLenFeature((), tf.int64),
      'image/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
      'label/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
  }

  parsed = tf.parse_single_example(raw_record, keys_to_features)

  # height = tf.cast(parsed['image/height'], tf.int32)
  # width = tf.cast(parsed['image/width'], tf.int32)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
  image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
  image.set_shape([None, None, _DEPTH])

  label = tf.image.decode_image(
      tf.reshape(parsed['label/encoded'], shape=[]), 1)
  label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
  label.set_shape([None, None, 1])

  return image, label


def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    # image, label = preprocessing.random_rescale_image_and_label(
    #     image, label, _MIN_SCALE, _MAX_SCALE)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)

    # # Randomly flip the image and label horizontally.
    # image, label = preprocessing.random_flip_left_right_image_and_label(
    #     image, label)

    image.set_shape([_HEIGHT, _WIDTH, _DEPTH])
    label.set_shape([_HEIGHT, _WIDTH, 1])

  image = preprocessing.mean_image_subtraction(image)

  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: preprocess_image(image, label, is_training))
  dataset = dataset.prefetch(batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.clean_model_dir:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  # Set up a RunConfig to only save checkpoints once per training cycle.
  # run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=5000)
  # run_config = run_config.replace(save_summary_steps=1)

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': FLAGS.batch_size,
          'base_architecture': FLAGS.base_architecture,
          # 'pre_trained_model': FLAGS.pre_trained_model,
          'pre_trained_model': None,
          'batch_norm_decay': _BATCH_NORM_DECAY,
          'num_classes': _NUM_CLASSES,
          'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
          'weight_decay': FLAGS.weight_decay,
          'learning_rate_policy': FLAGS.learning_rate_policy,
          'num_train': _NUM_IMAGES['train'],
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'max_iter': FLAGS.max_iter,
          'end_learning_rate': FLAGS.end_learning_rate,
          'power': _POWER,
          'momentum': _MOMENTUM,
          'freeze_batch_norm': FLAGS.freeze_batch_norm,
          'initial_global_step': FLAGS.initial_global_step
      })

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
      'learning_rate': 'learning_rate',
      'cross_entropy': 'cross_entropy',
      'train_px_accuracy': 'train_px_accuracy',
      'train_mean_iou': 'train_mean_iou',
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    train_hooks = [logging_hook]

    if FLAGS.debug:
      debug_hook = tf_debug.LocalCLIDebugHook()
      train_hooks.append(debug_hook)

    tf.logging.info("Start training.")
    model.train(
        # input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, 1000),
        hooks=train_hooks,
        # steps=1  # For debug
    )


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
