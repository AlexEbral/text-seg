"""Utility functions for preprocessing data sets."""

from PIL import Image
import numpy as np
import tensorflow as tf

_R_MEAN = 127
_G_MEAN = 127
_B_MEAN = 127

# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


# colour map TEXT_DS 50 classes
label_colours = {0: (0, 0, 0), 1: (5, 4, 40), 2: (109, 33, 110),
                 3: (195, 241, 243), 4: (144, 122, 252), 5: (191, 226, 251),
                 6: (71, 196, 145), 7: (49, 134, 4), 8: (148, 9, 68),
                 9: (8, 210, 84), 10: (67, 238, 59), 11: (22, 56, 205),
                 12: (89, 27, 113), 13: (71, 114, 43), 14: (214, 212, 48),
                 15: (82, 129, 35), 16: (232, 240, 197), 17: (15, 59, 39),
                 18: (187, 180, 1), 19: (214, 85, 119), 20: (64, 149, 74),
                 21: (235, 12, 62), 22: (145, 206, 78), 23: (68, 174, 202),
                 24: (146, 176, 44), 25: (3, 244, 129), 26: (124, 114, 105),
                 27: (252, 154, 168), 28: (222, 162, 6), 29: (161, 8, 1),
                 30: (71, 114, 168), 31: (175, 121, 156), 32: (224, 128, 39),
                 33: (167, 179, 77), 34: (244, 178, 88), 35: (8, 69, 255),
                 36: (58, 20, 228), 37: (37, 14, 157), 38: (190, 54, 102),
                 39: (199, 136, 142), 40: (136, 85, 105), 41: (156, 213, 46),
                 42: (27, 130, 35), 43: (17, 48, 7), 44: (185, 117, 165),
                 45: (216, 229, 217), 46: (110, 45, 80), 47: (107, 110, 241),
                 48: (136, 159, 33), 49: (197, 161, 193)}

ALPHABET_DICT = {0: '/', 1: 'а', 2: 'б', 3: 'в', 4: 'г', 5: 'д', 6: 'е', 7: 'ё',
                 8: 'ж', 9: 'з', 10: 'и', 11: 'й', 12: 'к', 13: 'л', 14: 'м',
                 15: 'н', 16: 'о', 17: 'п', 18: 'р', 19: 'с', 20: 'т', 21: 'у',
                 22: 'ф', 23: 'х', 24: 'ц', 25: 'ч', 26: 'ш', 27: 'щ', 28: 'ъ',
                 29: 'ы', 30: 'ь', 31: 'э', 32: 'ю', 33: 'я', 34: '0', 35: '1',
                 36: '2', 37: '3', 38: '4', 39: '5', 40: '6', 41: '7', 42: '8',
                 43: '9', 44: ',', 45: ';', 46: '!', 47: ':', 48: '.', 49: '-'}

# colour map TEXT_DS 2 classes
# 0 - back 1 - text
# label_colours = {0: (0, 0, 0), 1: (255, 255, 255)}


def decode_labels(mask, num_images=1, num_classes=21):
  """Decode batch of segmentation masks.

  Args:
    mask: result of inference after taking argmax.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).

  Returns:
    A batch with num_images RGB images of the same size as the input.
  """
  n, h, w, c = mask.shape
  assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                            % (n, num_images)
  outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
  for i in range(num_images):
    img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
    pixels = img.load()
    for j_, j in enumerate(mask[i, :, :, 0]):
      for k_, k in enumerate(j):
        if k < num_classes:
            pixels[k_, j_] = label_colours[k]
    outputs[i] = np.array(img)
  return outputs


def mean_image_addition(image, mean=127):
  """Adds the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  img_shape = image.get_shape().as_list()
  num_channels = 3
  img_shape[2] = num_channels
  new_image = tf.fill(img_shape, 0.0)
  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=new_image)
  channels[0] = image[:, :, :]
  channels[0] += mean
  channels[1] = channels[0]
  channels[2] = channels[0]
  return tf.concat(axis=2, values=channels)


def mean_image_subtraction(image, mean=127):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  # if num_channels != 0:
  #   raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  # for i in range(num_channels):
  channels[0] -= mean
  return tf.concat(axis=2, values=channels)


def random_rescale_image_and_label(image, label, min_scale, max_scale):
  """Rescale an image and label with in target scale.

  Rescales an image and label within the range of target scale.

  Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[height, width, 1]`.
    min_scale: Min target scale.
    max_scale: Max target scale.

  Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
    If `labels` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, 1]`.
  """
  if min_scale <= 0:
    raise ValueError('\'min_scale\' must be greater than 0.')
  elif max_scale <= 0:
    raise ValueError('\'max_scale\' must be greater than 0.')
  elif min_scale >= max_scale:
    raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

  shape = tf.shape(image)
  height = tf.to_float(shape[0])
  width = tf.to_float(shape[1])
  scale = tf.random_uniform(
      [], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  image = tf.image.resize_images(image, [new_height, new_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
  # Since label classes are integers, nearest neighbor need to be used.
  label = tf.image.resize_images(label, [new_height, new_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return image, label


def random_crop_or_pad_image_and_label(image, label, crop_height, crop_width, ignore_label):
  """Crops and/or pads an image to a target width and height.

  Resizes an image to a target width and height by rondomly
  cropping the image or padding it evenly with zeros.

  Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[height, width, 1]`.
    crop_height: The new height.
    crop_width: The new width.
    ignore_label: Label class to be ignored.

  Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  label = label - ignore_label  # Subtract due to 0 padding.
  label = tf.to_float(label)
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  image_and_label = tf.concat([image, label], axis=2)
  image_and_label_pad = tf.image.pad_to_bounding_box(
      image_and_label, 0, 0,
      tf.maximum(crop_height, image_height),
      tf.maximum(crop_width, image_width))
  image_and_label_crop = tf.random_crop(
      image_and_label_pad, [crop_height, crop_width, 2])

  image_crop = image_and_label_crop[:, :, :1]
  label_crop = image_and_label_crop[:, :, 1:]
  label_crop += ignore_label
  label_crop = tf.to_int32(label_crop)

  return image_crop, label_crop


def random_flip_left_right_image_and_label(image, label):
  """Randomly flip an image and label horizontally (left to right).

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
  """
  uniform_random = tf.random_uniform([], 0, 1.0)
  mirror_cond = tf.less(uniform_random, .5)
  image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
  label = tf.cond(mirror_cond, lambda: tf.reverse(label, [1]), lambda: label)

  return image, label


def eval_input_fn(image_filenames, label_filenames=None, batch_size=1):
  """An input function for evaluation and inference.

  Args:
    image_filenames: The file names for the inferred images.
    label_filenames: The file names for the grand truth labels.
    batch_size: The number of samples per batch. Need to be 1
        for the images of different sizes.

  Returns:
    A tuple of images and labels.
  """
  # Reads an image from a file, decodes it into a dense tensor
  def _parse_function(filename, is_label):
    if not is_label:
      image_filename, label_filename = filename, None
    else:
      image_filename, label_filename = filename

    image_string = tf.read_file(image_filename)
    image = tf.image.decode_image(image_string)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 1])

    image = mean_image_subtraction(image)

    if not is_label:
      return image
    else:
      label_string = tf.read_file(label_filename)
      label = tf.image.decode_image(label_string)
      label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
      label.set_shape([None, None, 1])

      return image, label

  if label_filenames is None:
    input_filenames = image_filenames
  else:
    input_filenames = (image_filenames, label_filenames)

  dataset = tf.data.Dataset.from_tensor_slices(input_filenames)
  if label_filenames is None:
    dataset = dataset.map(lambda x: _parse_function(x, False))
  else:
    dataset = dataset.map(lambda x, y: _parse_function((x, y), True))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()

  if label_filenames is None:
    images = iterator.get_next()
    labels = None
  else:
    images, labels = iterator.get_next()

  return images, labels


def infer_input_fn(image_filenames, crop, batch_size=1):
  """An input function for evaluation and inference.

  Args:
    image_filenames: The file names for the inferred images.
    batch_size: The number of samples per batch. Need to be 1
        for the images of different sizes.

  Returns:
    A tuple of images and labels.
  """
  # Reads an image from a file, decodes it into a dense tensor
  def _parse_function(filename):
    image_filename, label_filename = filename, None

    image_string = tf.read_file(image_filename)
    image = tf.image.decode_image(image_string)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 1])
    image = tf.image.crop_to_bounding_box(image,
                                  crop[0],
                                  crop[1],
                                  crop[2],
                                  crop[3])
    image = mean_image_subtraction(image)
    return image

  dataset = tf.data.Dataset.from_tensor_slices(image_filenames)
  dataset = dataset.map(lambda x: _parse_function(x))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()

  images = iterator.get_next()
  return images, None