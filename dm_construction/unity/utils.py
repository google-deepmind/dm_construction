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
"""Utitilies for dealing with the underlying Unity environment."""


def get_action_names_and_bounds(action_spec):
  """Extracts action names and their sizes from a Unity action spec."""
  scalar_action_names = action_spec[0].name.split("|")
  names_and_bounds = []
  bound_0 = 0
  while scalar_action_names:
    current_name_split = scalar_action_names.pop(0).split(".")
    current_name = current_name_split[0]
    bound_1 = bound_0 + 1
    if len(current_name_split) == 2:
      current_name = current_name_split[0]
      while scalar_action_names:
        next_name = scalar_action_names.pop(0)
        if next_name.split(".")[0] == current_name:
          bound_1 += 1
        else:
          scalar_action_names.insert(0, next_name)
          break
    names_and_bounds.append((current_name, (bound_0, bound_1)))
    bound_0 = bound_1
  return names_and_bounds
