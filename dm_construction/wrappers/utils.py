#!/usr/bin/python
#
# Copyright 2020 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Task utils."""

import warnings

import numpy as np


def pad_to_size(array, size, axis, repeat=False, error_on_trim=True,
                warning_on_trim=True):
  """Pads an array to the desired size in the specified axis.

  Args:
    array: Input array to be padded.
    size: desired dimension for the indictated axis.
    axis: axis to along which to pad the array.
    repeat: it False, it will be padded with zeros, otherwise, it will be padded
        with the last element of the array along that axis.
    error_on_trim: If True it will show an error if the size of the axis is
        larger than the desired size.
    warning_on_trim: If True and error_on_trim==False, it will show a warning
        instead of an error.

  Returns:
    The padded array.

  Raises:
    ValueError: If the array is larger than size along the specified axis.

  """
  if array.shape[axis] < size:
    if repeat:
      padding_slice = np.take(array, -1, axis=axis)
      missing_length = size - array.shape[axis]
      missing_block = np.stack([padding_slice]*missing_length, axis=axis)
    else:
      padding_shape = list(array.shape)
      padding_shape[axis] = size - padding_shape[axis]
      missing_block = np.zeros(padding_shape, array.dtype)
    array = np.concatenate(
        [array, missing_block],
        axis=axis)
  elif array.shape[axis] > size:
    if error_on_trim:
      raise ValueError("Trying to pad into a smaller size %d->%d"%
                       (array.shape[axis], size))

    if warning_on_trim:
      warnings.warn("Padding into a smaller size results on trimming %d->%d."%
                    (array.shape[axis], size))
    array = np.take(array, list(range(size)), axis=axis)

  return array
