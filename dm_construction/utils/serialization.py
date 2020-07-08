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
"""Utils for serializing objects."""

import copy


def get_object_state(obj, ignore_attributes=()):
  """Returns a dictionary with the state of the attributes of an object.

  Note that this is not general. For example, it will throw an error for classes
  that define a __slots__ field (like namedtuples).

  Args:
    obj: a Python object
    ignore_attributes: list of attributes to ignore when getting the object
      state

  Returns:
    state: a dictionary representation of the object state.
  """
  state = {}
  for k, v in obj.__dict__.items():
    if k not in ignore_attributes:
      state[k] = copy.deepcopy(v)
  for k in ignore_attributes:
    if not hasattr(obj, k):
      raise ValueError("Ignored attribute `%s` does not exist in object." % k)
  return state


def set_object_state(obj, state, ignore_attributes=()):
  """Sets the state of an object obtained through `get_object_state`.

  Note that this is not general. For example, it will not work for classes
  that define a __slots__ field (like namedtuples).

  Args:
    obj: a Python object
    state: the state to set on the object (obtained with `get_object_state`).
    ignore_attributes: list of attributes to ignore when getting the object
      state.
  """
  for k, v in state.items():
    if k not in ignore_attributes:
      setattr(obj, k, copy.deepcopy(v))
  for k in ignore_attributes:
    if not hasattr(obj, k):
      raise ValueError("Ignored attribute `%s` does not exist in object." % k)
