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

from dm_construction.unity import constants as unity_constants
from dm_construction.utils import constants
from dm_construction.wrappers import base
import dm_env
from dm_env import specs
import numpy as np


# Used to avoid putting actions at exactly the limits of the scene.
_SMALL_EPSILON = 1e-6

# Added to y-coordinates of actions, to avoid triggering a collision between the
# object being placed and the object below it. We cannot use a value too low,
# because of Unity's collision behavior
_Y_MARGIN = 4e-2


def _discrete_array_spec(shape, base_name):
  return specs.Array(shape, dtype=np.int32, name=base_name + "_spec")


def _continuous_array_spec(shape, base_name):
  return specs.Array(shape, dtype=np.float32, name=base_name + "_spec")


def _slices_and_indices_to_indices(slices_or_indices):
  indices = []
  for slice_or_index in slices_or_indices:
    if isinstance(slice_or_index, slice):
      if slice_or_index.step not in [None, 1]:
        raise ValueError("slices should only use a step size of 1")
      indices.extend(list(range(slice_or_index.start, slice_or_index.stop)))
    else:
      indices.append(slice_or_index)
  return sorted(indices)


def _get_relative_discretization_grid(point_counts_inside):
  if point_counts_inside > 0:
    extra_side_width = max(1. / point_counts_inside, _Y_MARGIN)
  else:
    extra_side_width = _Y_MARGIN
  bound = 1. + extra_side_width
  # Create a linspace that allows to stack blocks on the sides of each other
  # as well.
  return np.linspace(-bound, bound, point_counts_inside + 3)


class DiscreteRelativeGraphWrapper(base.ConstructionBaseWrapper):
  """Creates graph-based observations with discrete relative actions."""

  def __init__(self,
               env,
               allow_reverse_action=False,
               max_x_width=2.0,
               max_y_width=2.0,
               discretization_steps=12,
               invalid_edge_penalty=0.,
               enable_y_action=False,
               enable_glue_action=True,
               enable_selection_action=True):
    """Wraps an environment with graph-structured observations and actions.

    The environment support the following modes of discrete actions:
      - Agent only providing "x_action" (defining the x-coordinate of the placed
      block, starting from the leftmost position).
      - Agent additionally providing "y_action" (defining the y-coordinate of
      the placed block, starting from the bottom most position). To use this
      mode, set enable_y_position to True.

    Args:
      env: An instance of `ConstructionBaseWrapper`.
      allow_reverse_action: Whether to allow the action to be attached to an
        edge in the reverse direction. If this option is set to True and that
        the edge to which the action is attached is going from a valid moved
        block to a valid base block (instead of the reverse direction), then the
        action corresponding to the reverse edge will be taken. Otherwise, the
        episode will end with an `invalid_edge` termination.
      max_x_width: The accessible width along the x axis, centered on the chosen
        block center.
      max_y_width: The accessible width along the y axis, centered on the chosen
        block center.
      discretization_steps: The number of discrete steps along the x and y axis.
      invalid_edge_penalty: The penalty received when selecting an invalid edge
        (a positive number; the reward will be minus that).
      enable_y_action: Whether the agent also select the y-coordinate. If False,
        the y coordinate is set to be a small margin on top of the block, at the
        given y coordinate.
      enable_glue_action: Whether the agent select whether to glue or not. If
        False, glue is always applied.
      enable_selection_action: Whether the agent selects the order of the
        blocks.
    """
    super(DiscreteRelativeGraphWrapper, self).__init__(env=env)
    self._allow_reverse_action = allow_reverse_action
    self._discretization_steps = discretization_steps
    assert invalid_edge_penalty > -1e-6
    self._invalid_edge_penalty = invalid_edge_penalty
    self._enable_y_action = enable_y_action
    self._enable_glue_action = enable_glue_action
    self._enable_selection_action = enable_selection_action

    self._init_observation_wrapping()
    self._init_action_wrapping()

  def _init_observation_wrapping(self):
    """Sets up attributes needed for wrapping observations."""
    # Which keys from the underlying observation to include as nodes in the
    # graph observation.
    self._node_types = [
        constants.BLOCK, constants.AVAILABLE_BLOCK, constants.OBSTACLE,
        constants.TARGET]
    if constants.BALL in self._env.observation_spec():
      self._node_types.append(constants.BALL)

    # We will first concatenate on one hots, then cherry pick the node features
    # that we want. Before doing the cherry picking, these will be the indices
    # of the one hot node types.
    self._one_hot_feature_slice = slice(
        unity_constants.BLOCK_SIZE,
        unity_constants.BLOCK_SIZE + len(self._node_types))

    # Which features from the underlying observation to include in the node
    # attributes.
    self._node_features = _slices_and_indices_to_indices([
        unity_constants.POSITION_FEATURE_SLICE,
        unity_constants.ORIENTATION_FEATURE_SLICE,
        unity_constants.WIDTH_FEATURE_INDEX,
        unity_constants.HEIGHT_FEATURE_INDEX,
        unity_constants.LINEAR_VELOCITY_FEATURE_SLICE,
        unity_constants.ANGULAR_VELOCITY_FEATURE_INDEX,
        unity_constants.STICKY_FEATURE_INDEX,
        unity_constants.FREE_OBJECT_FEATURE_INDEX,
        unity_constants.SHAPE_FEATURE_SLICE,
        self._one_hot_feature_slice
    ])

  def _init_action_wrapping(self):
    """Sets up attributes needed for wrapping actions."""
    valid_base_block_types = [
        constants.BLOCK, constants.OBSTACLE, constants.TARGET]

    if "Balls" in self._env.observation_spec():
      valid_base_block_types.append(constants.BALL)
    self._valid_base_block_one_hots = [
        self._get_node_one_hot_index(x)
        for x in valid_base_block_types
    ]
    self._valid_moved_block_one_hots = [
        self._get_node_one_hot_index(x)
        for x in [constants.AVAILABLE_BLOCK]
    ]
    self._non_physical_one_hots = [
        self._get_node_one_hot_index(x)
        for x in [constants.TARGET]
    ]

    self._x_feature_index = self._get_feature_index(
        unity_constants.POSITION_X_FEATURE_INDEX)
    self._y_feature_index = self._get_feature_index(
        unity_constants.POSITION_Y_FEATURE_INDEX)
    self._height_feature_index = (
        self._get_feature_index(
            unity_constants.HEIGHT_FEATURE_INDEX))

    standard_action_spec = self._env.action_spec()

    if "Selector" not in standard_action_spec:
      self._enable_selection_action = False

    self._min_x = (
        float(standard_action_spec["Horizontal"].minimum) + _SMALL_EPSILON)
    self._min_y = (
        float(standard_action_spec["Vertical"].minimum) + _SMALL_EPSILON)
    self._max_x = (
        float(standard_action_spec["Horizontal"].maximum) - _SMALL_EPSILON)
    self._max_y = (
        float(standard_action_spec["Vertical"].maximum) - _SMALL_EPSILON)
    self._num_x_actions = self._discretization_steps + 3
    self._num_y_actions = self._discretization_steps + 3
    self._relative_x_positions = _get_relative_discretization_grid(
        self._discretization_steps)
    self._relative_y_positions = _get_relative_discretization_grid(
        self._discretization_steps)

    # Ignoring attributes with nested structure that are constant to avoid
    # unnecessary deepcopies of those when restoring states. This is not
    # technically necessary (e.g. we do not bother with scalar attributes).
    self._state_ignore_fields.extend([
        "_valid_base_block_one_hots", "_valid_moved_block_one_hots",
        "_non_physical_one_hots", "_relative_x_positions",
        "_relative_y_positions"
    ])

  def _get_feature_index(self, core_index):
    return self._node_features.index(core_index)

  def _get_node_one_hot_index(self, object_type):
    # Get the index just in the one-hots
    base_index = self._node_types.index(object_type)
    # Get the feature index into node_features
    features = _slices_and_indices_to_indices([self._one_hot_feature_slice])
    feature_index = features[base_index]
    # Look up the actual index
    one_hot_index = self._node_features.index(feature_index)
    return one_hot_index

  def action_spec(self):
    edge_spec = {
        "Index": specs.Array([], dtype=np.int32),
        "x_action": specs.BoundedArray(
            [], np.int32, 0, self._num_x_actions - 1)
    }
    if self._enable_y_action:
      edge_spec.update({
          "y_action": specs.BoundedArray(
              [], np.int32, 0, self._num_y_actions - 1)
      })
    if self._enable_glue_action:
      edge_spec.update({"sticky": specs.BoundedArray([], np.int32, 0, 1)})

    return edge_spec

  def observation_spec(self):
    """The observation spec as a graph.

    Note that while this method returns a dictionary, it is compatible with the
    GraphsTuple data structure from the graph_nets library. To convert the spec
    from this method to a GraphsTuple:

      from graph_nets import graphs
      spec = graphs.GraphsTuple(**env.observation_spec())

    Returns:
      spec: the observation spec as a dictionary
    """
    node_size = len(self._node_features)
    nodes_spec = _continuous_array_spec([0, node_size], "nodes")
    edges_spec = _continuous_array_spec([0, 1], "edges")
    senders_spec = _discrete_array_spec([0], "senders")
    receivers_spec = _discrete_array_spec([0], "receivers")
    globals_spec = _continuous_array_spec([1, 1], "globals")
    n_node_spec = _discrete_array_spec([1], "n_node")
    n_edge_spec = _discrete_array_spec([1], "n_edge")
    observation_spec = dict(
        nodes=nodes_spec,
        edges=edges_spec,
        globals=globals_spec,
        n_node=n_node_spec,
        n_edge=n_edge_spec,
        receivers=receivers_spec,
        senders=senders_spec
    )
    return observation_spec

  def _get_nodes(self, observation):
    """Returns node attributes."""
    objects = []
    for i, key in enumerate(self._node_types):
      # Remove extra time dimension returned by some environments
      # (like marble run)
      features = observation[key]
      if features.ndim == 3:
        features = features[:, 0]

      # Add a one-hot indicator of the node type.
      one_hot = np.zeros(
          (features.shape[0], len(self._node_types)), dtype=np.float32)
      one_hot[:, i] = 1

      features = np.concatenate([features, one_hot], axis=1)
      objects.append(features)

    return np.concatenate(objects, axis=0)

  def _get_edges(self, nodes):
    sender_node_inds = np.arange(len(nodes))
    receiver_node_inds = np.arange(len(nodes))
    senders, receivers = np.meshgrid(sender_node_inds, receiver_node_inds)
    senders, receivers = senders.flatten(
        ).astype(np.int32), receivers.flatten().astype(np.int32)

    # This removes self-edges.
    same_index = senders == receivers
    senders = senders[~same_index]
    receivers = receivers[~same_index]

    edge_content = np.zeros([senders.shape[0], 1], dtype=np.float32)
    return edge_content, senders, receivers

  def _get_globals(self):
    return np.zeros([1], dtype=np.float32)

  def _order_nodes(self, observation):
    """Order nodes based on object id."""
    indices = observation["nodes"][
        :, unity_constants.ID_FEATURE_INDEX].astype(int)
    ordering = np.argsort(indices)

    # update nodes
    nodes = observation["nodes"][ordering]

    # update senders/receivers
    ordering = list(ordering)
    inverse_ordering = np.array(
        [ordering.index(i) for i in range(len(ordering))], dtype=np.int32)

    if observation["senders"] is not None:
      senders = inverse_ordering[observation["senders"]]
    else:
      senders = None
    if observation["receivers"] is not None:
      receivers = inverse_ordering[observation["receivers"]]
    else:
      receivers = None

    new_observation = observation.copy()
    new_observation.update(dict(
        nodes=nodes,
        senders=senders,
        receivers=receivers))
    return new_observation

  def _select_node_features(self, observation):
    """Cherry-pick desired node features."""
    nodes = observation["nodes"][:, self._node_features]
    new_observation = observation.copy()
    new_observation["nodes"] = nodes
    return new_observation

  def _process_time_step(self, time_step):
    nodes = self._get_nodes(time_step.observation)
    edges, senders, receivers = self._get_edges(nodes)
    globals_ = self._get_globals()
    observation = dict(
        nodes=nodes,
        edges=edges,
        globals=globals_[np.newaxis],
        n_node=np.array([nodes.shape[0]], dtype=int),
        n_edge=np.array([edges.shape[0]], dtype=int),
        receivers=receivers,
        senders=senders)
    observation = self._order_nodes(observation)
    observation = self._select_node_features(observation)
    time_step = time_step._replace(observation=observation)
    return time_step

  def _compute_continuous_action(self, base_pos, base_length, moved_length,
                                 offset, min_pos, max_pos):
    ratio = (base_length + moved_length) / 2.
    return np.clip(base_pos + offset * ratio, min_pos, max_pos)

  def reset(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    """Reset the environment.

    Note that while this method returns observations as a dictionary, they are
    compatible with the GraphsTuple data structure from the graph_nets library.
    To convert the observations returned by this method to a GraphsTuple:

      from graph_nets import graphs
      timestep = env.reset()
      timestep = timestep._replace(
          observation=graphs.GraphsTuple(**timestep.observation))

    Args:
      *args: args to pass to super
      **kwargs: args to pass to super

    Returns:
      timestep: a dm_env.TimeStep
    """
    return super(DiscreteRelativeGraphWrapper, self).reset(*args, **kwargs)

  def step(self, action):
    """Step the environment.

    Note that while this method returns observations as a dictionary, they are
    compatible with the GraphsTuple data structure from the graph_nets library.
    To convert the observations returned by this method to a GraphsTuple:

      from graph_nets import graphs
      timestep = env.step(action)
      timestep = timestep._replace(
          observation=graphs.GraphsTuple(**timestep.observation))

    Args:
      action: the action to take in the environment.

    Returns:
      timestep: a dm_env.TimeStep
    """
    valid_action, base_block, moved_block = self._validate_edge_index(
        int(action["Index"]))

    if not valid_action:
      self._termination_reason = constants.TERMINATION_INVALID_EDGE
      self._last_time_step = dm_env.TimeStep(
          step_type=dm_env.StepType.LAST,
          observation=self._last_time_step.observation,
          reward=-self._invalid_edge_penalty,
          discount=0)
      return self._last_time_step

    block_x = base_block[self._x_feature_index]
    block_y = base_block[self._y_feature_index]
    selector = moved_block[self._x_feature_index]

    width_index = self._get_feature_index(
        unity_constants.WIDTH_FEATURE_INDEX)
    base_width = np.abs(base_block[width_index])
    moved_width = np.abs(moved_block[width_index])

    base_height = np.abs(base_block[self._height_feature_index])
    moved_height = np.abs(moved_block[self._height_feature_index])

    x_continuous_action = self._compute_continuous_action(
        base_pos=block_x,
        base_length=base_width,
        moved_length=moved_width,
        offset=self._relative_x_positions[action["x_action"]],
        min_pos=self._min_x,
        max_pos=self._max_x)

    if self._enable_y_action:
      y_continuous_action = self._compute_continuous_action(
          base_pos=block_y,
          base_length=base_height,
          moved_length=moved_height,
          offset=self._relative_y_positions[action["y_action"]],
          min_pos=self._min_y,
          max_pos=self._max_y)
    else:
      y_continuous_action = block_y + _Y_MARGIN
      if all(base_block[self._non_physical_one_hots] < 0.5):
        y_continuous_action += (base_height + moved_height) / 2.

    updated_action = {
        "Horizontal": np.array(x_continuous_action, dtype=np.float32),
        "Vertical": np.array(y_continuous_action, dtype=np.float32),
        "Sticky": np.array(1., dtype=np.int32),
    }

    if self._enable_glue_action:
      updated_action["Sticky"] = action["sticky"]

    if self._enable_selection_action:
      updated_action["Selector"] = selector

    self._last_time_step = self._process_time_step(
        self._env.step(updated_action))
    return self._last_time_step

  def _validate_edge_index(self, edge_index):
    """Checks that an action connecting first_node to second_node is valid.

    An action is valid if it connects a marker or block to an avaible block.

    Args:
      edge_index: Index of the edge to apply the action relatively with.

    Returns:
      is_valid: A boolean indicating whether the action was valid.
      base_block: The features of the base block, or None.
      moved_block: The features of the moved block, or None.
    """
    previous_observation = self._last_time_step.observation
    edges = list(
        zip(previous_observation["senders"], previous_observation["receivers"]))

    edge = edges[edge_index]
    nodes = previous_observation["nodes"]
    first_node_features = nodes[edge[0]]
    second_node_features = nodes[edge[1]]

    if not self._enable_selection_action:
      first_movable_block = next((i for i, x in enumerate(nodes)
                                  if x[self._valid_moved_block_one_hots] > 0.5),
                                 None)
      if edge[0] != first_movable_block and edge[1] != first_movable_block:
        return False, None, None

    if self._allow_reverse_action and any(
        first_node_features[self._valid_base_block_one_hots] > 0.5):
      base_block = first_node_features
      moved_block = second_node_features
    elif any(second_node_features[self._valid_base_block_one_hots] > 0.5):
      base_block = second_node_features
      moved_block = first_node_features
    else:
      return False, None, None  # Not a valid base block.
    if not any(moved_block[self._valid_moved_block_one_hots] > 0.5):
      return False, None, None  # Not a valid moved block.

    return True, base_block, moved_block

  @property
  def termination_reason(self):
    if self._termination_reason:
      return self._termination_reason
    return super(DiscreteRelativeGraphWrapper, self).termination_reason

  @property
  def all_termination_reasons(self):
    return self.core_env.all_termination_reasons + [
        constants.TERMINATION_INVALID_EDGE]
