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
"""A collection of observation wrappers for construction tasks."""

from dm_construction.wrappers import base
from dm_env import specs
import numpy as np


def _rescale_axis(array, target_size, axis=0, dtype=None):
  """Rescales the axis of an array to a target size using the mean.

  Args:
    array: input array.
    target_size: the desired size for axis. The target size must divide exactly
      the input size of the axis.
    axis: Axis to rescale to target_size.
    dtype: dtype of the output array. Tf None, the dtype input array will be
      preserved, although the mean will be calculated as np.float32 .

  Returns:
    array with axis rescaled to the size.

  """

  if dtype is None:
    dtype = array.dtype

  if array.shape[axis] == target_size:
    return array.astype(dtype)

  if array.shape[axis] % target_size != 0:
    raise ValueError("The target size {} should divide exactly "
                     "the input size {}."
                     .format(target_size, array.shape[axis]))
  downsize = array.shape[axis] // target_size

  leading_dims = array.shape[:axis]
  trailing_dims = array.shape[axis+1:]

  array_reshaped = np.reshape(
      array, leading_dims + (target_size, downsize) + trailing_dims)

  averaged_array = np.mean(
      array_reshaped, axis=axis+1, dtype=np.float32).astype(dtype)

  return averaged_array


def _rescale_frame(frame, target_size, axis_height=0, axis_width=1, dtype=None):
  """Rescales a frame to a target size using the mean..

  Args:
    frame: Frame with at least rank 2, corresponding to the axis spatial
      axis of the frame indicated by axis_height and axis_width.
    target_size: 2-tuple with the desired size for axis_height and axis_width
      respectively. The target size must divide exactly the input sizes of the
      corresponding axes.
    axis_height: Axis to rescale to target_size[0].
    axis_width: Axis to rescale to target_size[1].
    dtype: dtype of the output frame. Tf None, the dtype input frame will be
      preserved, although the mean will be calculated as np.float32 .

  Returns:
    Frame rescaled to the target size.

  """
  return _rescale_axis(
      _rescale_axis(frame, target_size[0], axis=axis_height, dtype=dtype),
      target_size[1], axis=axis_width, dtype=dtype)


class ContinuousAbsoluteImageWrapper(base.ConstructionBaseWrapper):
  """Rescales and exposes RGB observations with continuous absolute actions."""

  def __init__(self, env, output_resolution=(64, 64)):
    super(ContinuousAbsoluteImageWrapper, self).__init__(env=env)
    self._output_resolution = output_resolution

  def observation_spec(self):
    rgb_spec = self._env.observation_spec()["RGB"]
    shape = list(self._output_resolution) + [rgb_spec.shape[-1]]
    dtype = rgb_spec.dtype
    obs_spec = specs.Array(shape, dtype=dtype, name=rgb_spec.name)
    return obs_spec

  def action_spec(self):
    spec = self._env.action_spec().copy()
    spec["Sticky"] = specs.BoundedArray(
        [], dtype=np.float32, minimum=-1, maximum=1)
    return spec

  def _process_time_step(self, time_step):
    rgb_observation = time_step.observation["RGB"]
    # Remove extra time dimension returned by some environments (ie marble run)
    if rgb_observation.ndim == 4:
      rgb_observation = rgb_observation[0]
    observation = _rescale_frame(rgb_observation, self._output_resolution)
    return time_step._replace(observation=observation)

  def step(self, action):
    updated_action = action.copy()
    # Convert continuous sticky action to discrete.
    updated_action["Sticky"] = int(action["Sticky"] > 0)
    return super(ContinuousAbsoluteImageWrapper, self).step(updated_action)
