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

from dm_construction.utils import serialization
import dm_env


class ConstructionBaseWrapper(dm_env.Environment):
  """A base class for wrappers around construction tasks."""

  def __init__(self, env):
    self._env = env
    super(ConstructionBaseWrapper, self).__init__()
    self._state_ignore_fields = ["_env"]
    self._last_time_step = None

  def close(self):
    self._env.close()

  def _process_time_step(self, time_step):
    return time_step

  def reset(self, *args, **kwargs):
    self._termination_reason = None
    self._last_time_step = self._process_time_step(
        self._env.reset(*args, **kwargs))
    return self._last_time_step

  def step(self, action):
    self._last_time_step = self._process_time_step(self._env.step(action))
    return self._last_time_step

  def action_spec(self):
    return self._env.action_spec()

  def observation_spec(self):
    return self._env.observation_spec()

  @property
  def core_env(self):
    return self._env.core_env

  def get_state(self):
    state = serialization.get_object_state(self, self._state_ignore_fields)
    state["_env"] = self._env.get_state()
    return state

  def set_state(self, state):
    serialization.set_object_state(self, state, self._state_ignore_fields)
    self._env.set_state(state["_env"])

  @property
  def last_time_step(self):
    return self._last_time_step

  @property
  def termination_reason(self):
    """A string indicating the reason why an episode was terminated."""
    return self.core_env.termination_reason

  @property
  def all_termination_reasons(self):
    """All possible termination reasons for this environment."""
    return self.core_env.all_termination_reasons

  @property
  def episode_logs(self):
    if hasattr(self.core_env, "episode_logs"):
      return self.core_env.episode_logs
    return {}
