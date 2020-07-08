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
"""Helpers to construct an inner enviromment from a config."""

from dm_construction.environments import connecting
from dm_construction.environments import covering
from dm_construction.environments import marble_run
from dm_construction.environments import silhouette
from dm_construction.unity import docker
from dm_construction.unity import environment
from dm_construction.wrappers import continuous_absolute
from dm_construction.wrappers import discrete_relative

_TASK_CLASSES = {
    "connecting": connecting.ConstructionConnecting,
    "covering_hard": covering.ConstructionCoveringHard,
    "covering": covering.ConstructionCovering,
    "marble_run": marble_run.ConstructionMarbleRun,
    "silhouette": silhouette.ConstructionSilhouette
}
ALL_TASKS = sorted(_TASK_CLASSES.keys())

_WRAPPER_CLASSES = {
    "continuous_absolute": continuous_absolute.ContinuousAbsoluteImageWrapper,
    "discrete_relative": discrete_relative.DiscreteRelativeGraphWrapper
}
ALL_WRAPPERS = sorted(_WRAPPER_CLASSES.keys())

_LOADERS = {
    "docker": docker.loader,
}
_DEFAULT_LOADER = "docker"


def get_unity_environment(backend=_DEFAULT_LOADER, **config):
  return environment.UnityConstructionEnv(loader=_LOADERS[backend], **config)


def get_task_environment(unity_environment, problem_type, **env_kwargs):
  """Returns a configured instance of a task environment."""
  env_cls = _TASK_CLASSES[problem_type]
  task_env = env_cls(unity_environment=unity_environment, **env_kwargs)
  return task_env


def get_wrapped_environment(task_environment, wrapper_type):
  """Wraps the environment with appropriate observations and actions."""
  wrapper_cls = _WRAPPER_CLASSES[wrapper_type]
  wrapped_environment = wrapper_cls(task_environment)
  return wrapped_environment


def get_environment(
    problem_type, unity_environment=None, wrapper_type="discrete_relative",
    unity_backend=_DEFAULT_LOADER, **env_kwargs):
  """Returns fully configured and wrapped environments."""
  if not unity_environment:
    unity_environment = get_unity_environment(backend=unity_backend)
  task_environment = get_task_environment(
      unity_environment, problem_type, **env_kwargs)
  agent_environment = get_wrapped_environment(task_environment, wrapper_type)
  return agent_environment
