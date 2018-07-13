# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

layers_dict = {
    "conv1d": tf.layers.conv1d,
    "conv2d": tf.layers.conv2d
}

nn_dict = {
    "conv1d": tf.nn.conv1d,
    "conv2d": tf.nn.conv2d
}


def conv_actv(type, name, inputs, filters, kernel_size, activation_fn, strides,
              padding, regularizer, training, data_format):
  """Helper function that applies convolution and activation.
    Args:
      type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[type]

  conv = layer(
      name="{}".format(name),
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  output = conv
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def conv_bn_actv(type, name, inputs, filters, kernel_size, activation_fn, strides,
                 padding, regularizer, training, data_format, bn_momentum,
                 bn_epsilon):
  """Helper function that applies convolution, batch norm and activation.
    Accepts inputs in 'channels_last' format only.
    Args:
      type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[type]

  conv = layer(
      name="{}".format(name),
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  # trick to make batchnorm work for mixed precision training.
  # To-Do check if batchnorm works smoothly for >4 dimensional tensors
  squeeze = False
  if type == "conv1d":
    conv = tf.expand_dims(conv, axis=1)  # NWC --> NHWC
    squeeze = True

  bn = tf.layers.batch_normalization(
      name="{}/bn".format(name),
      inputs=conv,
      gamma_regularizer=regularizer,
      training=training,
      axis=-1 if data_format == 'channels_last' else 1,
      momentum=bn_momentum,
      epsilon=bn_epsilon,
  )

  if squeeze:
    bn = tf.squeeze(bn, axis=1)

  output = bn
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def conv_gn_actv(type, name, inputs, filters, kernel_size, activation_fn, strides,
                 padding, regularizer, training, data_format, bn_momentum,
                 bn_epsilon):
  """Helper function that applies convolution, group norm and activation.
    Accepts inputs in 'channels_last' format only.
    Args:
      type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[type]

  conv = layer(
      name="{}".format(name),
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  squeeze = False
  if type == "conv1d":
    conv = tf.expand_dims(conv, axis=-1)  # NWC --> NHWC
    squeeze = True

  bn = tf.layers.batch_normalization(
      name="{}/bn".format(name),
      inputs=conv,
      gamma_regularizer=regularizer,
      training=training,
      axis=-1 if data_format == 'channels_last' else 1,
      momentum=bn_momentum,
      epsilon=bn_epsilon,
  )

  if squeeze:
    bn = tf.squeeze(bn, axis=-1)

  output = bn
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def conv_wn_actv(type, name, inputs, filters, kernel_size, activation_fn, strides,
                 padding, regularizer, training, data_format):
  """Helper function that applies convolution, weight norm and activation.
    Accepts inputs in 'channels_last' format only.

    Args:
      type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = nn_dict[type]

  with tf.variable_scope(name):
    in_size_index = -1 if data_format == 'channels_last' else 1
    in_dim = int(inputs.get_shape()[in_size_index])
    out_dim = int(filters)

    # Initialize the weights by decoupling norm and direction
    V = tf.get_variable(
        '_V',
        shape=kernel_size + [in_dim, out_dim],
        initializer=tf.random_normal_initializer(
            mean=0, stddev=0.01),
        trainable=True
    )
    Vlen = len(V.get_shape().as_list())
    V_norm = tf.sqrt(
        tf.reduce_sum(
            tf.square(V.initialized_value()),
            axis=[i for i in range(Vlen - 1)]
        )
    )
    g = tf.get_variable(
        '_g',
        initializer=V_norm,
        trainable=True
    )
    b = tf.get_variable(
        '_b',
        shape=[out_dim],
        initializer=tf.zeros_initializer(),
        trainable=True
    )

    if type == "conv1d":
      strides = strides[0]

    W = tf.reshape(g, [1 for i in range(Vlen - 1)] + [out_dim]) * \
        tf.nn.l2_normalize(V, [i for i in range(Vlen - 1)])
    conv = tf.nn.bias_add(
        layer(
            name="{}".format(name),
            value=inputs,
            filters=W,
            stride=strides,
            padding=padding
        ),
        b
    )

  output = conv
  if activation_fn is not None:
    output = activation_fn(output)
  return output
