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
"""Helper functions for loading the construction tasks through Docker."""

import codecs
import json
import re
import time

from absl import logging
from dm_construction.unity import utils
from dm_env import specs as dm_env_specs
import docker
import grpc
import numpy as np
import portpicker

from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import tensor_utils

# Maximum number of times to attempt gRPC connection.
_MAX_CONNECTION_ATTEMPTS = 10

# Port to expect the docker environment to internally listen on.
_DOCKER_INTERNAL_GRPC_PORT = 10000

# The name of the Docker image to pull.
_DEFAULT_DOCKER_IMAGE_NAME = (
    "gcr.io/deepmind-environments/dm_construction:v1.0.0")


class _ConstructionEnv(dm_env_adaptor.DmEnvAdaptor):
  """An implementation of dm_env_rpc.DmEnvAdaptor for construction tasks."""

  def __init__(self, connection, specs, channel, observations, container):
    """Initialize the docker Unity environment."""
    super().__init__(connection, specs, observations, nested_tensors=False)
    self._channel = channel
    self._observation_names = observations
    self._container = container
    self._action_names_and_bounds = utils.get_action_names_and_bounds(
        self.action_spec())

  def close(self):
    """Close the Unity environment."""
    if self._container:
      super().close()
      self._channel.close()
      try:
        self._container.kill()
      except (docker.errors.NotFound, docker.errors.APIError):
        pass  # Ignore, container has already been closed.
      self._container = None

  def read_property(self, name):
    """Read a property of the Unity environment."""
    properties = self._connection.send(
        dm_env_rpc_pb2.ReadPropertyRequest(keys=[name])).properties
    return tensor_utils.unpack_tensor(properties[name])

  def write_property(self, name, value):
    """Write a property of the Unity environment."""
    properties = {name: tensor_utils.pack_tensor(value)}
    self._connection.send(
        dm_env_rpc_pb2.WritePropertyRequest(properties=properties))

  def action_spec(self):
    """Build the action spec based on the underlying environment."""
    # Get the list of action names, in order.
    raw_actions = self._dm_env_rpc_specs.actions
    names = [raw_actions[i].name for i in range(len(raw_actions))]

    # Get the inner action spec, which is a dictionary.
    inner_action_spec = super().action_spec()

    # Convert the dictionary of specs to a single BoundedArray.

    minimums = []
    maximums = []
    dtypes = []
    for name in names:
      spec = inner_action_spec[name]
      minimums.append(spec.minimum)
      maximums.append(spec.maximum)
      dtypes.append(spec.dtype)
    shape = [len(names)]
    names = "|".join(names)
    dtypes = list(set(dtypes))
    assert len(dtypes) == 1
    minimums = np.array(minimums, dtype=dtypes[0])
    maximums = np.array(maximums, dtype=dtypes[0])
    return [dm_env_specs.BoundedArray(
        shape=shape, dtype=dtypes[0], name=names, minimum=minimums,
        maximum=maximums)]

  def observation_spec(self):
    """Build the observation spec based on the underlying environment."""
    # Get the inner observation spec, which is a dictionary.
    inner_obs_spec = super().observation_spec()
    # Convert it to a tuple of specs, in order of the observation names.

    flat_spec = []
    for name in self._observation_names:
      spec = inner_obs_spec[name]
      # For numerical specs, make sure they are an array and not a scalar.
      if spec.dtype != np.dtype("<U") and not spec.shape:
        spec = dm_env_specs.Array(shape=(1,), dtype=spec.dtype, name=spec.name)
      flat_spec.append(spec)
    return tuple(flat_spec)

  def step(self, flat_actions):
    """Step the Unity environment."""
    # Convert the action to a dictionary.
    action_dict = {}
    for name, (lower, upper) in self._action_names_and_bounds:
      action = flat_actions[0][lower:upper]
      if len(action) == 1:
        action = action[0]
        action_dict[name] = action
      else:
        for i in range(len(action)):
          action_dict["{}.{}".format(name, i)] = action[i]

    # Step the environment.
    time_step = super().step(action_dict)

    # Pack the observation into a tuple.
    flat_observation = []
    spec = self.observation_spec()
    for i, name in enumerate(self._observation_names):
      obs = time_step.observation[name]
      if spec[i].dtype == np.float64:
        obs = np.array([obs], dtype=spec[i].dtype)
      flat_observation.append(obs)
    flat_observation = tuple(flat_observation)

    return time_step._replace(observation=flat_observation)

  def reset(self):
    """Implements dm_env.Environment.reset."""
    response = self._connection.send(dm_env_rpc_pb2.ResetRequest())
    if self._dm_env_rpc_specs != response.specs:
      raise RuntimeError("Environment changed spec after reset")
    self._last_state = dm_env_rpc_pb2.EnvironmentStateType.INTERRUPTED


def _check_grpc_channel_ready(channel):
  """Helper function to check the gRPC channel is ready N times."""
  for _ in range(_MAX_CONNECTION_ATTEMPTS - 1):
    try:
      return grpc.channel_ready_future(channel).result(timeout=1)
    except grpc.FutureTimeoutError:
      pass
  return grpc.channel_ready_future(channel).result(timeout=1)


def _can_send_message(connection):
  """Returns if `connection` is healthy and able to process requests."""
  try:
    # This should return a response with an error unless the server isn't yet
    # receiving requests.
    connection.send(dm_env_rpc_pb2.StepRequest())
  except error.DmEnvRpcError:
    return True
  except grpc.RpcError:
    return False


def _create_channel_and_connection(port):
  """Returns a tuple of `(channel, connection)`."""
  for _ in range(_MAX_CONNECTION_ATTEMPTS):
    channel = grpc.secure_channel("localhost:{}".format(port),
                                  grpc.local_channel_credentials())
    _check_grpc_channel_ready(channel)
    connection = dm_env_rpc_connection.Connection(channel)
    if _can_send_message(connection):
      break
    else:
      # A gRPC server running within Docker sometimes reports that the channel
      # is ready but transitively returns an error (status code 14) on first
      # use.  Giving the server some time to breath and retrying often fixes the
      # problem.
      connection.close()
      channel.close()
      time.sleep(1.0)

  return channel, connection


def _parse_exception_message(message):
  """Returns a human-readable version of a dm_env_rpc json error message."""
  try:
    match = re.match(r"^message\:\ \"(.*)\"$", message)
    group = match.group(1)  # pytype: disable=attribute-error
    json_data = codecs.decode(group, "unicode-escape")  # pytype: disable=wrong-arg-types
    parsed_json_data = json.loads(json_data)
    return ValueError(json.dumps(parsed_json_data, indent=4))
  except:  # pylint: disable=bare-except
    return message


def _wrap_send(send):
  """Wraps `send` in order to reformat exceptions."""
  try:
    return send()
  except ValueError as e:
    e.args = [_parse_exception_message(e.args[0])]
    raise


def _connect_to_environment(port, settings):
  """Helper function for connecting to a running dm_construction environment."""
  channel, connection = _create_channel_and_connection(port)
  original_send = connection.send
  connection.send = lambda request: _wrap_send(lambda: original_send(request))

  all_settings = {
      key: tensor_utils.pack_tensor(val) for key, val in settings.items()}

  create_settings = {
      "levelName": all_settings["levelName"],
      "seed": tensor_utils.pack_tensor(0),
      "episodeId": tensor_utils.pack_tensor(0)
  }
  world_name = connection.send(
      dm_env_rpc_pb2.CreateWorldRequest(settings=create_settings)).world_name

  join_settings = all_settings.copy()
  del join_settings["levelName"]
  specs = connection.send(
      dm_env_rpc_pb2.JoinWorldRequest(
          world_name=world_name, settings=join_settings)).specs

  return channel, connection, specs


def loader(settings, observations, version, local_path=None):
  """Creates a construction unity environment connecting to docker."""
  del version  # unused
  client = docker.from_env()
  port = portpicker.pick_unused_port()

  if local_path:
    image_name = local_path
  else:
    image_name = _DEFAULT_DOCKER_IMAGE_NAME

  try:
    client.images.get(image_name)
  except docker.errors.ImageNotFound:
    logging.info("Downloading docker image '%s'...", image_name)
    client.images.pull(image_name)
    logging.info("Download finished.")

  container = client.containers.run(
      image_name,
      auto_remove=True,
      detach=True,
      ports={_DOCKER_INTERNAL_GRPC_PORT: port})

  channel, connection, specs = _connect_to_environment(port, settings)
  return _ConstructionEnv(
      connection=connection,
      specs=specs,
      channel=channel,
      observations=observations,
      container=container)
