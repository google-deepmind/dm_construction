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
r"""Wrapper for the true underlying unity meta environment.

The meta environment allows to place blocks in arbitrary positions.

Observation consists of variable size (0 in action spec means variable size)
"Blocks", "Balls" and "Contacts". It also contains a "ElapsedTime"
double containing simulation steps, "SpawnCollisionCount" with the number
of spawned collisions since the last Reset, and "CollisionStop" indicating
whether the last physics simulation ended due to a collision (only if
StopOnCollision is set).

It also includes an "RGB" image from a front ortographic camera with
configurable size and position and a "Segmentation" image containing
a 2d-array with the id of the object at each location. An additional
"ObserverRGB" image can be used for higher resolution 3d rendering.

Block Features:
-[0]: Object id.
-[1/2/3]: Horizontal/Vertical/Depth position.
-[4/5]: Cos(angle)/Sin(angle).
-[6/7/8]: Width/Height/Depth. Width/Height are scaling parameters and can be
  negative.
-[9/10/11/12]: R/G/B/A.
-[13/14]: Horizontal/Vertical velocity.
-[15]: Angular velocity.
-[16]: One hot indicator if it is a physical object.
-[17/18/19/20]: Collision mask.
-[21]: Density.
-[22]: Bounciness.
-[23]: Friction.
-[24]: Linear drag.
-[25]: Angular drag.
-[26]: One hot indicator if it is a free object.
-[27]: One hot indicator if it is glueable.
-[28]: One hot indicator if it is sticky.
-[29]: One hot indicator if it is glued (sticky only on the first contact).
-[30]: One hot indicator indicating if the object is still sticky.
-[31/32/33]: One hot indicator of the type of the shape.
-[34]: Simulation time when the block was spawned.
-[35]: Number of collisions that the block has experienced.

Contact Features:
-[0/1]: Object ids of the two blocks involved in the contact.
-[2]: One hot indicator if the contact is glued.
-[3/4/5]: Relative position of the contact point with respect to the parent
    object, evaluated when the objects enter the collision.

Actions (all of them float32 scalars, not enforced range in brackets) represent:

-SimulationSteps [0., 10000.]: Number of simulation steps (0.02s) to run.
  Rounded to int.
-Delete [0., 1.]: if > 0.5 Removes object with the "SelectId" id or at
  "Select" coordinates, if SelectId == 0.
-Reset [0., 1.]: if > 0.5 removes all objects from the scene.
-SpawnBlock [0., 1.]: if > 0.5 spawns one block at the "Set" coordinates.
-SetId [int]: Id of the spawned object. If == 0, a sequential negative Id is
    given. If > 0 the value (or next available value) will be used
    as Id. If < 0 the value (or previous available value) will be used
    as Id.
-SetPosX [-7.5, 7.5]: Horizontal position of the spawned object.
-SetPosY [0., 15.]: Vertical position of the spawned object.
-SetPosZ [-2., 2.]: Depth position of the spawned object.
-SetAngle [-pi, pi]: Angle spawned object.
-SetVelX [-10., 10.]: Horizontal velocity of the spawned moving block.
-SetVelY [-10., 10.]: Vertical velocity of the spawned moving block.
-SetVelAngle [-10., 10.]: Angular velocity of the spawned moving block.
-Shape [0., 1., 2.]: 0: box, 1: ball, 2: ramp.
-Width [-10., 10.]: Width of the spawned object. As it is a scaling parameter,
    negative values can be used to mirror the shapes horizontally.
-Height [-10., 10.]: Height of the spawned object, similar to width.
    As it is a scaling parameter, negative values can be used to mirror the
    shapes vertically. Unused for balls.
-Depth [0.1, 10.]: Depth of the spawned object, similar to width.
    Unused for balls. This is just for visualization purposes
    and does not affect the physical properties of the bodies.
-PhysicalBody [0., 1.]: if > 0.5, the spawned object is subject to physics,
    otherwise it does not interact with other bodies or gravity.
-CollisionMask [0b0000, 0b1111]: Bitmap with collision map, two bodies will
    collide if the bitwise "and" is positive ((mask_1 & mask_2) > 0).
-Density [0.001, 10000.]: Density of the created body.
-Bounciness [0., 1.]: Restitution coefficient of created body.
-Friction [0., 1.]: Friction of the material of the created body.
-LinearDrag [0., 100.]: Translation drag of the created body.
-AngularDrag [0., 100.]: Rotation drag of the created body.
-SelectId [int]: Id of selected objects.
-SelectPosX [-7.5, 7.5]: Horizontal position of selected objects.
-SelectPosY [0., 15.]: Vertical position of selected objects.
-RGBA [0., 1.]: Color components of the spawned element. Size 4.
-Glueable [0., 1.]: If 1. the object will be affected by glue.
-Sticky [0., 1.]: if > 0.5, the spawned block is sticky. (Only if Glueable).
-FreeBody [0., 1.]: if > 0.5, the spawned block moves freely, otherwise fixed.
-UseGlue [0., 1.]: if > 0.5, the spawned free block gets glues on the
  first collision. (Only if Glueable).
-GravityX/GravityY [-100., 100.]: Vector corresponding to the gravity.
-Timestep [0., 10.]: Duration of each simulation timestep.
-StopOnCollision [0., 1.]: If > 0.5 the simulation will be stop on the first
  collision, even if the number of simulation steps have not been reached.
-CameraPosX/CameraPosY [-100., 100.]: Position of the center of the camera view.
-CameraHeight [0., 1000.]: Height of the camera view.

Constants for block features and default actions are provided in constants.py.
"""

import time

from absl import logging
from dm_construction.unity import constants
from dm_construction.unity import utils
import dm_env
from dm_env import specs
import numpy as np

ACTION_DEFAULTS = constants.ACTION_DEFAULTS
OTHER_ACTION_DEFAULT = constants.OTHER_ACTION_DEFAULT

# Update together with new versions that are released.
LAST_PRODUCTION_LABEL = "production_v26"


class UnityConstructionEnv(dm_env.Environment):
  """Wrapper for the raw unity env that deserializes observations/actions."""

  def __init__(self, loader, version=None, include_agent_camera=True,
               width=128, height=128, include_segmentation_camera=False,
               num_segmentation_layers=3, include_observer_camera=False,
               observer_width=None, observer_height=None,
               observer_3d=False, show_set_cursor=True, show_select_cursor=True,
               max_simulation_substeps=0, local_path=None,
               max_collisions_per_simulation_step=10):
    """Inits the environment.

    Args:
      loader: a function that loads the environment, e.g. as a Docker image.
        This function should take the following arguments:
          - settings: a dictionary of settings for the Unity env
          - observations: a list of requested observations
          - version: the version to load
          - local_path: a path to a local version of the environment
        And it should return the loaded Unity environment.
      version: Label of the version of the construction mpm to be used. If None,
        the latest version of the mpm known to this module and stored in
        LAST_PRODUCTION_LABEL will be used.
      include_agent_camera: If True, an "RGB" field will contain a camera render
        as part fo the observation.
      width: Horizontal resolution of the agent camera observation.
      height: Vertical resolution of the agent camera observation.
      include_segmentation_camera: If True, a "Segmentation" camera will be
        provided as part of the observation.
      num_segmentation_layers: Maximum number of objects per pixel allowed
        in the segmented observation.
      include_observer_camera: If True, a separate "ObserverRGB" field will
        contain a second camera with potentically different resolution.
      observer_width: Horizontal resolution of the observer camera observation.
        If None it will default to `width`.
      observer_height: Vertical resolution of the observer camera observation.
        If None it will default to `height`.
      observer_3d: If True, the observer camera will render in 3d.
      show_set_cursor: If True, the set cursor will be visible.
      show_select_cursor: If True, the select cursor will be visible.
      max_simulation_substeps: If 0, the number of "SimulationSteps" will be
        executed at once together with the actions for maximum efficiency when
        training agents.
        If max_simulation_substeps > 0, It will proceed as follows:
        1. Store the "SimulationSteps" as pending simulation steps.
        2. Apply a first step to the environment with all passed actions except
           overriding SimulationSteps to 0.
        3. Keep applying environment steps with
           SimulationSteps = max_simulation_substeps, until there are no pending
           simulation steps. (Depending on rounding the last environment step
           may contain less than max_simulation_substeps).
        This allows to visualize trajectories in between agent steps by
        using an observer via the `add_observer` methods, together with this
        option.
      local_path: If provided, it will use a local build of the unity
        environment.
      max_collisions_per_simulation_step: The maximum number of new collisions
        that can happen within a single simulation step. A large number of new
        collisions occurring in a very short period of time usually indicates
        instability in the simulation. A MetaEnvironmentError is raised, and
        the environment is reset to an empty safe state.
    """

    if version is None:
      version = LAST_PRODUCTION_LABEL
    if observer_width is None:
      observer_width = width
    if observer_height is None:
      observer_height = height

    self._version = version
    self._include_agent_camera = include_agent_camera
    self._width = width
    self._height = height
    self._include_segmentation_camera = include_segmentation_camera
    self._include_observer_camera = include_observer_camera
    self._num_segmentation_layers = num_segmentation_layers
    self._observer_width = observer_width
    self._observer_height = observer_height
    self._observer_3d = observer_3d
    self._local_path = local_path
    self._show_set_cursor = show_set_cursor
    self._show_select_cursor = show_select_cursor
    self._max_collisions_per_simulation_step = (
        max_collisions_per_simulation_step)
    self._load_env(loader)
    self._raw_observations_observers = []
    self._max_simulation_substeps = max_simulation_substeps
    self._action_names_bounds = utils.get_action_names_and_bounds(
        self._env.action_spec())

    # The "IsAction" action is for internal use only.
    self._valid_action_names = [name for name, _ in self._action_names_bounds]
    self._valid_action_names.remove("IsAction")
    self._valid_action_names += constants.ADDITIONAL_HELPER_ACTIONS

    self._observation_spec = self._build_observation_spec()
    self._action_spec = self._build_action_spec()

    # Empirically we have observed that observations sometimes come empty
    # but only the very first time and right after instantiating the
    # environment. Sleeping and forcing reset seems to fix it.
    time.sleep(1)
    self.reset()
    time.sleep(1)

  def add_observer(self, observer):
    """Adds a raw observation observer.

    The observer will be notified for each new observation obtained from the
    mpm process. If the `max_simulation_substeps` argument is provided when
    instantiating the class, the observer will also be notified of additional
    intermediate observations corresponding to dynamics substeps within a
    single `step` call.

    Args:
      observer: A callable that takes as argument an observation.
    """
    self._raw_observations_observers.append(observer)

  def remove_observer(self, observer):
    """Removes a raw observation observer that was previously added.

    Args:
      observer: A callable that takes as argument an observation.
    """
    self._raw_observations_observers.remove(observer)

  def _get_simulation_steps_action(self, previous_action, num_steps=1):
    # We want to simply run simulation steps with cursors still pointing to
    # the same location used by a previous action, and same gravity/timestep.
    action_dict = {"SimulationSteps": float(num_steps)}
    actions_to_repeat = ["SelectPosX", "SelectPosY", "SetPosX",
                         "SetPosY", "GravityX", "GravityY", "Timestep",
                         "StopOnCollision", "CameraPosX", "CameraPosY",
                         "CameraHeight"]

    for action in actions_to_repeat:
      if action in previous_action:
        action_dict[action] = previous_action[action]
    return self._flatten_action_dict(action_dict)

  def _flatten_action_dict(self, action_dict):
    for action_name in action_dict:
      if action_name not in self._valid_action_names:
        raise ValueError("Unrecognized action {}, valid actions are {}."
                         .format(action_name, self._valid_action_names))
    action_dict = _replace_helper_actions(action_dict)
    action_values = {
        name: action_dict.get(
            name, ACTION_DEFAULTS.get(name, OTHER_ACTION_DEFAULT))
        for name, _ in self._action_names_bounds}
    action_list = [
        _prepare_action(name, action_values[name], bound_1-bound_0)
        for name, (bound_0, bound_1) in self._action_names_bounds]
    flat_actions = (np.concatenate(action_list, axis=0).astype(np.float32),)
    return flat_actions

  def _get_config_json(self):
    config_json = {"levelName": "ConstructionMetaEnvironment",
                   "ShowSetCursor": self._show_set_cursor,
                   "ShowSelectCursor": self._show_select_cursor,
                   "MaxCollisionsPerSimulationStep": (
                       self._max_collisions_per_simulation_step)}

    observations = [
        "Blocks", "Contacts", "ElapsedTime",
        "SpawnCollisionCount", "CollisionStop", "Message"]

    if self._include_agent_camera or self._include_segmentation_camera:
      height = self._height
      width = self._width
      config_json.update({
          "AgentCameraWidth": width,
          "AgentCameraHeight": height})

      if self._include_agent_camera:
        observations.append("AgentCamera")

      if self._include_segmentation_camera:
        config_json.update({
            "NumSegmentationLayers": self._num_segmentation_layers})
        observations.append("SegmentationCamera")

    if self._include_observer_camera:
      height = self._observer_height
      width = self._observer_width
      config_json.update({
          "ObserverCameraWidth": width,
          "ObserverCameraHeight": height,
          "ObserverCamera3D": self._observer_3d})
      observations.append("ObserverCamera")

    self._obs_to_ind_map = {
        obs: index for index, obs in enumerate(observations)}

    return config_json, observations

  def _load_env(self, loader):
    config_json, observations = self._get_config_json()
    self._env = loader(
        config_json, observations, self._version, local_path=self._local_path)

    # Verify that the version is consistent.
    version = self._env.read_property("Version")

    if version != self._version:
      raise ValueError("Wrong version loaded: required `{}`, got `{}`."
                       .format(self._version, version))
    else:
      msg = ("Construction meta-environment running at version `%s`." %
             version)
      if self._local_path:
        msg += " (Local build)"
      logging.info(msg)

  def __del__(self):
    if hasattr(self, "_env"):
      self._env.close()

  def close(self):
    return self._env.close()

  def hard_reset(self):
    # Perform a hard reset of the environment, which does not return anything.
    # Then, perform a soft reset so we can actually get a timestep.
    self._env.reset()
    return self.reset()

  def reset(self):
    return self.step({"Reset": 1.})

  @property
  def last_observation(self):
    """Returns the last observation."""
    return self._last_observation

  def restore_state(self,
                    observation,
                    verify_restored_state=True,
                    verification_threshold_abs=1e-3,
                    verification_threshold_rel=1e-5,
                    verify_velocities=True):
    """Restores the environment to the state given by an observation.

    Args:
      observation: Environment observation.
      verify_restored_state: If True, it will verify that the observation
          after restoring is consistent with the observation set.
      verification_threshold_abs: Maximum absolute difference disagreement
          between features in the input observation and the restore observation
          allowed when `verify_restored_state==True`.
      verification_threshold_rel: Maximum relative difference disagreement
          between features in the input observation and the restore observation
          allowed when `verify_restored_state==True`.
      verify_velocities: If False, the velocities will not be verified. This is
          sometimes required in environments that make extensive use of glue,
          as the velocities cannot always be set correctly for constraints.

    Returns:
      A timestep with the first observation after restoring the state. All
      fields should be equal to those in the observation passed as argument
      (except for numerical precision errors). Camera renders are just copied
      from the input observation. As the observation is not informative enough
      to tell the placement of the cameras, and also cursors may be in different
      positions.

    Raises:
      ValueError if the shapes or values of the observation after restoring
      are different than those being restored.

    """
    serialized_blocks = _serialize_array(observation["Blocks"])
    serialized_contacts = _serialize_array(observation["Contacts"])
    string_spawn_collision_count = "%d" % observation["SpawnCollisionCount"]
    string_elapsed_time = "%g" % observation["ElapsedTime"]
    string_collision_stop = "%d" % int(observation["CollisionStop"])

    empty_markers = ""  # For deprecated marker behavior.
    serialized_observation = "/".join([
        empty_markers, serialized_blocks, serialized_contacts,
        string_spawn_collision_count, string_elapsed_time,
        string_collision_stop])
    self._env.write_property(
        "RestoreObservation", serialized_observation)
    # We need to send a null action after setting the property with
    # the sequence, to run the effects, and get the observation back.

    restored_timestep = self._one_step({})
    if verify_restored_state:
      _verify_restored_observation(
          observation,
          restored_timestep.observation,
          difference_threshold_abs=verification_threshold_abs,
          difference_threshold_rel=verification_threshold_rel,
          verify_velocities=verify_velocities)
    if self._include_agent_camera:
      restored_timestep.observation["RGB"] = observation["RGB"].copy()
    if self._include_observer_camera:
      restored_timestep.observation["ObserverRGB"] = (
          observation["ObserverRGB"].copy())

    self._last_observation = restored_timestep.observation.copy()
    return restored_timestep

  def step(self, actions):
    """Applies the actions to the environment.

    Args:
      actions: Dictionary of actions containing an action set as indicated by
        the action spec. Keys that are not specified will take the default
        value as contained in ACTION_DEFAULTS. Limits of the actions are not
        enforced for additional flexibility. Optionally, a list of dictionaries
        may be passed instead, in which case the entire sequence of action sets
        will be sent and processed by the unity backend in a single interaction,
        speeding up the execution of the sequence.

    Returns:
      TimeStep with the final observation resulting from applying the action
        set, or the entire action set list.

    Raises:
      ValueError: if the actions are not a dictionary or a list.
      MetaEnvironmentError: if an error occurs in the underlying Unity
        environment.

    """
    if isinstance(actions, list):
      return self._multiple_steps(actions)
    elif isinstance(actions, dict):
      return self._one_step(actions)
    else:
      raise ValueError("Unrecognized action type {}, should be a list or a dict"
                       .format(type(actions)))

  def _multiple_steps(self, action_dict_list):
    if action_dict_list:
      # If we are storing timesteps, we actually actions one by one.
      if self._max_simulation_substeps > 0:
        for action_dict in action_dict_list:
          time_step = self._one_step(action_dict)
        return time_step

      # Otherwise, we pack all of the actions, and send them as one.
      flat_actions_list = [self._flatten_action_dict(action_dict)[0]
                           for action_dict in action_dict_list]
      serialized_action_sequence = _serialize_array(flat_actions_list)
      self._env.write_property(
          "ChainedActionSequence", serialized_action_sequence)

    # We need to send a null action after setting the property with
    # the sequence, to run the effects, and get the observation back.
    # If the list is empty this will just return the observation with the
    # current state.
    return self._one_step({})

  def _one_step(self, action_dict):
    # If we want to explicitly run simulation steps, we set them to 0, and
    # run them later in a loop.
    if self._max_simulation_substeps:
      action_dict = action_dict.copy()
      num_pending_simulation_steps = action_dict.get("SimulationSteps", 0)
      num_pending_simulation_steps = int(round(num_pending_simulation_steps))
      action_dict["SimulationSteps"] = 0.
    else:
      num_pending_simulation_steps = 0

    time_step = self._process_and_store_timestep(
        self._env.step(self._flatten_action_dict(action_dict)))

    # We simulate exactly `num_extra_simulation_steps` by sending multiple
    # environment steps, each with not more than self._explicit_simulation_steps
    if num_pending_simulation_steps:
      while (num_pending_simulation_steps > 0 and
             not time_step.observation["CollisionStop"]):
        if num_pending_simulation_steps >= self._max_simulation_substeps:
          num_substeps = self._max_simulation_substeps
          num_pending_simulation_steps -= self._max_simulation_substeps
        else:
          num_substeps = num_pending_simulation_steps
          num_pending_simulation_steps = 0
        time_step = self._process_and_store_timestep(
            self._env.step(self._get_simulation_steps_action(
                action_dict, num_substeps)))
    return time_step

  def _build_action_spec(self):
    # Separate each of the actions into a dictionary.
    flat_action_spec = self._env.action_spec()[0]
    action_spec = {}
    for name, (bound_0, bound_1) in self._action_names_bounds:
      size = bound_1 - bound_0
      if size <= 1:
        shape = []
        index = bound_0
      else:
        shape = [size]
        index = slice(bound_0, bound_1)
      action_spec[name] = specs.BoundedArray(
          shape, dtype=np.float32,
          minimum=flat_action_spec.minimum[index],
          maximum=flat_action_spec.maximum[index])
    del action_spec["IsAction"]
    return action_spec

  def action_spec(self, *args, **kwargs):
    return self._action_spec

  def _build_observation_spec(self):
    parent_observation_spec = self._env.observation_spec()
    observation_spec = {
        "Blocks": specs.Array(
            [0, constants.BLOCK_SIZE], dtype=np.float32, name="Blocks"),
        "Contacts": specs.Array(
            [0, constants.CONTACT_SIZE], dtype=np.float32, name="Contacts"),
        "ElapsedTime": specs.Array(
            (), dtype=np.float32,
            name="ElapsedTime"),
        "SpawnCollisionCount": specs.Array(
            (), dtype=np.int32,
            name="SpawnCollisionCount"),
        "CollisionStop": specs.Array(
            (), dtype=np.bool,
            name="SpawnCollisionCount")
    }

    if self._include_agent_camera:
      observation_spec["RGB"] = parent_observation_spec[
          self._obs_to_ind_map["AgentCamera"]]

    if self._include_observer_camera:
      observation_spec["ObserverRGB"] = parent_observation_spec[
          self._obs_to_ind_map["ObserverCamera"]]

    if self._include_segmentation_camera:
      raw_spec = parent_observation_spec[
          self._obs_to_ind_map["SegmentationCamera"]]
      observation_spec["Segmentation"] = specs.Array(
          raw_spec.shape, dtype=np.int32, name="Segmentation")

    return  observation_spec

  def observation_spec(self, *args, **kwargs):
    return self._observation_spec

  def _process_message(self, message):
    messages = message.split(";")
    for message in messages:
      if not message:
        continue
      if message.startswith("E:"):
        raise constants.MetaEnvironmentError(message)
      else:
        logging.info(message)

  def _process_and_store_timestep(self, time_step):
    """Deserialize string observations into arrays, removing ignored ones."""
    blocks = _deserialize_array(
        time_step.observation[self._obs_to_ind_map["Blocks"]],
        expected_size=constants.BLOCK_SIZE)
    contacts = _deserialize_array(
        time_step.observation[self._obs_to_ind_map["Contacts"]],
        expected_size=constants.CONTACT_SIZE)
    new_observation = {
        "Blocks": blocks,
        "Contacts": contacts,
        "ElapsedTime": time_step.observation[
            self._obs_to_ind_map["ElapsedTime"]][0].astype(np.float32),
        "SpawnCollisionCount": np.array(
            round(time_step.observation[
                self._obs_to_ind_map["SpawnCollisionCount"]][0]),
            dtype=np.int32),
        "CollisionStop": np.array(
            round(time_step.observation[
                self._obs_to_ind_map["CollisionStop"]][0]), dtype=np.bool)}

    if self._include_agent_camera:
      new_observation["RGB"] = time_step.observation[
          self._obs_to_ind_map["AgentCamera"]]

    if self._include_observer_camera:
      new_observation["ObserverRGB"] = time_step.observation[
          self._obs_to_ind_map["ObserverCamera"]]

    if self._include_segmentation_camera:
      new_observation["Segmentation"] = time_step.observation[
          self._obs_to_ind_map["SegmentationCamera"]].astype(np.int32)

    message = str(time_step.observation[self._obs_to_ind_map["Message"]])
    self._process_message(message)

    for observer in self._raw_observations_observers:
      observer(new_observation.copy())

    self._last_observation = new_observation.copy()
    return time_step._replace(observation=new_observation,
                              discount=np.array(time_step.discount,
                                                dtype=np.float32))


def block_to_actions(block, delete_existing=False):
  """Converts a block vector representation into actions that create the block.

  The idea here is that a block with the properties of `block` will be created
  when the returned actions are executed in the unity environment.

  Note that if delete_existing=False, and an object already exists with that id,
  the actions will still create an object, but it will have a different id than
  the one specified in `block`.

  Args:
    block: a vector of block properties
    delete_existing: whether to delete an existing block with the given id

  Returns:
    action: a dictionary of actions to create the block
  """
  action = {
      "SpawnBlock": 1.,
      "SetId": block[constants.ID_FEATURE_INDEX],
      "SetPosX": block[constants.POSITION_X_FEATURE_INDEX],
      "SetPosY": block[constants.POSITION_Y_FEATURE_INDEX],
      "SetPosZ": block[constants.POSITION_Z_FEATURE_INDEX],
      "SetAngle": np.arctan2(block[constants.SINE_ANGLE_FEATURE_INDEX],
                             block[constants.COSINE_ANGLE_FEATURE_INDEX]),
      "Width": block[constants.WIDTH_FEATURE_INDEX],
      "Height": block[constants.HEIGHT_FEATURE_INDEX],
      "Depth": block[constants.DEPTH_FEATURE_INDEX],
      "RGBA": np.asarray([block[constants.RED_CHANNEL_FEATURE_INDEX],
                          block[constants.GREEN_CHANNEL_FEATURE_INDEX],
                          block[constants.BLUE_CHANNEL_FEATURE_INDEX],
                          block[constants.ALPHA_CHANNEL_FEATURE_INDEX]]),
      "SetVelX": block[constants.VELOCITY_X_FEATURE_INDEX],
      "SetVelY": block[constants.VELOCITY_Y_FEATURE_INDEX],
      "SetVelAngle": block[constants.ANGULAR_VELOCITY_FEATURE_INDEX],
      "PhysicalBody": block[constants.PHYSICAL_OBJECT_FEATURE_INDEX],
      "CollisionMask": (block[constants.COLLISION_MASK_FEATURE_SLICE]*
                        np.power(2, np.arange(constants.NUM_LAYERS))).sum(),
      "Density": block[constants.DENSITY_FEATURE_INDEX],
      "Bounciness": block[constants.BOUNCINESS_FEATURE_INDEX],
      "Friction": block[constants.FRICTION_FEATURE_INDEX],
      "LinearDrag": block[constants.LINEAR_DRAG_FEATURE_INDEX],
      "AngularDrag": block[constants.ANGULAR_DRAG_FEATURE_INDEX],
      "FreeBody": block[constants.FREE_OBJECT_FEATURE_INDEX],
      "Glueable": block[constants.GLUEABLE_FEATURE_INDEX],
      "Sticky": block[constants.STICKY_FEATURE_INDEX],
      "UseGlue": block[constants.GLUED_FEATURE_INDEX],
      "Shape": np.argmax(block[constants.SHAPE_FEATURE_SLICE]),
  }

  if delete_existing:
    action["Delete"] = 1.
    action["SelectId"] = action["SetId"]

  return action


def _deserialize_array(string, expected_size=2, dtype=np.float32):
  if not string:
    return np.zeros([0, expected_size], dtype=dtype)

  return np.array(
      [[float(item_element)
        for item_element in item.split(",")]
       for item in str(string).split(";")],
      dtype=dtype)


def _serialize_array(array):
  return ";".join([",".join(["%g" % e for e in row]) for row in array])


def _verify_restored_observation(
    input_observation, restored_observation,
    difference_threshold_abs=1e-3, difference_threshold_rel=1e-5,
    verify_velocities=True):
  """Verifies if a restored observation is equal to an input observation."""
  observation_names = list(input_observation.keys())
  error_messages = []
  for observation_name in observation_names:
    # We ignore cameras, as they are just copied from the inputs.
    if observation_name in ["RGB", "ObserverRGB"]:
      continue

    input_ = input_observation[observation_name]
    restored = restored_observation[observation_name]

    # This can happen if there are a different number of contacts.
    if input_.shape != restored.shape:
      error_messages.append(
          "Shape for the restored observation {} is different than the shape "
          "for the input observation {} for observation `{}`."
          .format(restored.shape, input_.shape, observation_name))
      continue

    if not input_.size:
      continue

    target = input_.copy().astype(np.float32)
    comparison = restored.copy().astype(np.float32)

    if not verify_velocities and observation_name == "Blocks":
      idx = [
          constants.VELOCITY_X_FEATURE_INDEX,
          constants.VELOCITY_Y_FEATURE_INDEX,
          constants.ANGULAR_VELOCITY_FEATURE_INDEX
      ]
      target[:, idx] = 0
      comparison[:, idx] = 0

    threshold = (
        difference_threshold_abs + difference_threshold_rel * np.abs(target))
    too_far = np.abs(target - comparison) > threshold
    if too_far.any():
      difference = np.abs(target - comparison) * too_far

      if difference.shape:
        max_diff_index = np.unravel_index(
            np.argmax(difference), difference.shape)
        max_difference = difference[max_diff_index]
        difference_threshold = threshold[max_diff_index]
        input_value = input_[max_diff_index]
      else:
        max_diff_index = None
        max_difference = difference
        difference_threshold = threshold
        input_value = input_

      error_messages.append(
          "Feature at index {} of `{}` observation with shape {} differs by "
          "{} (more than {}) from the input observation with value {}."
          .format(max_diff_index, observation_name, input_.shape,
                  max_difference, difference_threshold, input_value))

  if error_messages:
    raise constants.RestoreVerificationError("\n".join(error_messages))


def _prepare_action(name, value, size):
  """Adds a leading axis to scalars and verifies the size."""
  value = np.asarray(value)
  if not value.shape:
    value = value[np.newaxis]

  if value.shape[0] != size:
    raise ValueError("Invalid size value for %s, expected %d, got %d"%
                     (name, size, value.shape[0]))
  return value


def _verify_mutually_exclusive_actions(
    action_dict, action_name, invalid_action_names):
  for other_name in invalid_action_names:
    if other_name in action_dict:
      raise ValueError("Got %s action, but %d was already provided" %
                       (action_name, other_name))


def _replace_helper_actions(action_dict):
  """Replaces helper actions by the corresponding actions."""
  _replace_color_helper_actions(action_dict)
  return action_dict


def _replace_color_helper_actions(action_dict):
  """Replaces all color-related helper actions ensuring on RGBA is left."""
  if "RGBA" in action_dict:
    _verify_mutually_exclusive_actions(
        action_dict, "RGBA", ["RGB", "R", "G", "B", "A"])
  else:
    if "A" in action_dict:
      alpha = action_dict["A"]
      del action_dict["A"]
    else:
      alpha = ACTION_DEFAULTS.get("A", OTHER_ACTION_DEFAULT)

    if "RGB" in action_dict:
      _verify_mutually_exclusive_actions(action_dict, "RGB", ["R", "G", "B"])
      action_dict["RGBA"] = np.concatenate(
          [action_dict["RGB"], np.asarray(alpha)[None]], axis=0)
      del action_dict["RGB"]
    else:
      channel_values = []
      for channel in ["R", "G", "B"]:
        if channel in action_dict:
          value = action_dict[channel]
          del action_dict[channel]
        else:
          value = ACTION_DEFAULTS.get(channel, OTHER_ACTION_DEFAULT)
        channel_values.append(value)
      channel_values.append(alpha)
      action_dict["RGBA"] = np.stack(channel_values, axis=0)
  return action_dict
