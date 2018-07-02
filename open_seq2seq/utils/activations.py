# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range
from six import string_types

import tensorflow as tf


def glu(features, axis=-1, name=None):
  """Computes the gated linear unit activation.
  Splits the input tensor in to two components A and B and apply gated activation as mentioned in https://arxiv.org/pdf/1612.08083.pdf.

  Args:
    features: input tensor.
    axis: axis along which the input tensor is split.
      default is -1.
  """

  with tf.name_scope(name):
    pre_activations, gate_inputs = tf.split(
        features, num_or_size_splits=2, axis=axis)
    gate_outputs = tf.sigmoid(gate_inputs)
    print(gate_outputs)
    features_nonlinear = tf.multiply(pre_activations, gate_outputs)
  return features_nonlinear
