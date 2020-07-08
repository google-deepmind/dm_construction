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
"""Tests the open source construction environments."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import dm_construction
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("backend", "docker", "")


def _make_random_action(action_spec, observation):
  """Makes a random action given an action spec and observation."""
  # Sample the random action.
  action = {}
  for name, spec in action_spec.items():
    if name == "Index":
      value = np.random.randint(observation["n_edge"])
    elif spec.dtype in (np.int32, np.int64, int):
      value = np.random.randint(spec.minimum, spec.maximum + 1)
    else:
      value = np.random.uniform(spec.minimum, spec.maximum)
    action[name] = value
  return action


def _random_unroll(env, seed=1234, num_steps=10, difficulty=5,
                   random_choice_before_reset=False):
  """Take random actions in the given environment."""
  np.random.seed(seed)
  action_spec = env.action_spec()
  if random_choice_before_reset:
    np.random.choice([8], p=[1.])
  timestep = env.reset(difficulty=difficulty)
  trajectory = [timestep]
  actions = [None]
  for _ in range(num_steps):
    if timestep.last():
      if random_choice_before_reset:
        np.random.choice([8], p=[1.])
      timestep = env.reset(difficulty=difficulty)
    action = _make_random_action(action_spec, timestep.observation)
    timestep = env.step(action)
    trajectory.append(timestep)
    actions.append(action)
  return trajectory, actions


class TestEnvironments(parameterized.TestCase):

  def _make_environment(
      self, problem_type, curriculum_sample, wrapper_type, backend_type=None):
    """Make the new version of the construction task."""
    if backend_type is None:
      backend_type = FLAGS.backend
    return dm_construction.get_environment(
        problem_type,
        unity_environment=self._unity_envs[backend_type],
        wrapper_type=wrapper_type,
        curriculum_sample=curriculum_sample)

  @classmethod
  def setUpClass(cls):
    super(TestEnvironments, cls).setUpClass()
    # Construct the unity environment.
    cls._unity_envs = {
        "docker": dm_construction.get_unity_environment("docker"),
    }

  @classmethod
  def tearDownClass(cls):
    super(TestEnvironments, cls).tearDownClass()
    for env in cls._unity_envs.values():
      env.close()

  @parameterized.named_parameters(
      ("covering", "covering"),
      ("covering_hard", "covering_hard"),
      ("connecting", "connecting"),
      ("silhouette", "silhouette"),
      ("marble_run", "marble_run"))
  def test_discrete_relative_environments_curriculum_sample(self, name):
    """Smoke test for discrete relative wrapper with curriculum_sample=True."""
    env = self._make_environment(name, True, "discrete_relative")
    _random_unroll(env, difficulty=env.core_env.max_difficulty)

  @parameterized.named_parameters(
      ("covering", "covering"),
      ("covering_hard", "covering_hard"),
      ("connecting", "connecting"),
      ("silhouette", "silhouette"),
      ("marble_run", "marble_run"))
  def test_continuous_absolute_environments_curriculum_sample(self, name):
    """Smoke test for continuous absolute wrapper w/ curriculum_sample=True."""
    env = self._make_environment(name, True, "continuous_absolute")
    _random_unroll(env, difficulty=env.core_env.max_difficulty)

  @parameterized.named_parameters(
      ("connecting_additional_layer", "connecting", "additional_layer"),
      ("connecting_mixed_height_targets", "connecting", "mixed_height_targets"),
      ("silhouette_double_the_targets", "silhouette", "double_the_targets"),)
  def test_generalization_modes(self, name, generalization_mode):
    """Smoke test for discrete relative wrapper with curriculum_sample=True."""
    env = self._make_environment(name, False, "discrete_relative")
    _random_unroll(env, difficulty=generalization_mode)


if __name__ == "__main__":
  absltest.main()
