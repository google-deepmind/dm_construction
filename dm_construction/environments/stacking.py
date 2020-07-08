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
"""The base environment for Construction tasks.

See: Bapst, V., Sanchez-Gonzalez, A., Doersch, C., Stachenfeld, K., Kohli, P.,
Battaglia, P., & Hamrick, J. (2019, May). Structured agents for physical
construction. In International Conference on Machine Learning (pp. 464-474).

See: Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Pfaff, T., Weber, T.,
Buesing, L., & Battaglia, P. W. (2020). Combining Q-Learning and Search with
Amortized Value Estimates. ICLR 2020.
"""

import abc

from absl import logging
from dm_construction.unity import constants as unity_constants
from dm_construction.utils import block as block_utils
from dm_construction.utils import constants
from dm_construction.utils import serialization
import dm_env
from dm_env import specs
import numpy as np
from scipy import stats


_OBJECT_TYPE_NAMES = [
    constants.BLOCK,
    constants.OBSTACLE,
    constants.TARGET,
    constants.AVAILABLE_BLOCK
]


def _find_value_in_array(value, array):
  index = np.where(value == array)[0]
  if index.shape[0] == 0:
    return None
  if index.shape[0] > 1:
    raise ValueError("Found more than one {} in {}".format(value, array))
  return index[0]


def _build_segmentation_mask_for_id(segmentation_array, target_id):
  """Builds a binary mask for target_id."""
  return np.any(segmentation_array == target_id, axis=2)


def build_segmentation_masks_for_ids(
    segmentation_array, target_ids):
  if target_ids:
    return np.stack([_build_segmentation_mask_for_id(segmentation_array, id_)
                     for id_ in target_ids])
  else:
    return np.zeros((0,) + segmentation_array.shape[:2], dtype=np.bool)
  return


def _obstacle_has_been_hit(obstacles):
  return np.sum(
      obstacles[:, unity_constants.COLLISION_COUNT_FEATURE_INDEX]) > 0.


def _calculate_contact_pairs_and_features(placed_blocks_ids, contacts):
  """Returns pairs of blocks in contact and the corresponding features."""
  placed_blocks_ids = np.round(placed_blocks_ids).astype(np.int32)
  senders = np.round(contacts[:, 0]).astype(np.int32)
  receivers = np.round(contacts[:, 1]).astype(np.int32)
  # We are only going to provide the feature that tells if the there is glue,
  # but not the features indicating the position of the glue/contact.
  features = contacts[:, 2:3]

  contact_pairs = []
  contact_features = []
  for sender, receiver, feature in zip(senders, receivers, features):
    sender_ind = _find_value_in_array(sender, placed_blocks_ids)
    receiver_ind = _find_value_in_array(receiver, placed_blocks_ids)
    if sender_ind is None or receiver_ind is None:
      continue
    contact_pairs.append(np.array([sender_ind, receiver_ind], np.int32))
    contact_features.append(feature)

  if not contact_pairs:
    contact_pairs = np.zeros([0, 2], dtype=np.int32)
    contact_features = np.zeros_like(features[:0])
  else:
    contact_pairs = np.stack(contact_pairs, axis=0)
    contact_features = np.stack(contact_features, axis=0)

  return contact_pairs, contact_features


class ConstructionStacking(dm_env.Environment):
  """A base class for the construction tasks."""

  def __init__(self,
               unity_environment,
               block_replacement,
               sticky_penalty,
               max_difficulty,
               progress_threshold,
               bad_choice_penalty=0.0,
               spawn_collision_penalty=0.0,
               hit_obstacle_penalty=0.0,
               difficulty=None,
               curriculum_sample=True,
               curriculum_sample_geom_p=0.,
               bad_choice_termination=True,
               num_simulation_steps=1000,
               target_color=(0., 0., 1., 0.5),
               block_color=(0., 0., 1., 1.),
               sticky_block_color=(0., 0.8, 1., 1.),
               obstacle_color=(1., 0., 0., 1.),
               ball_color=(0., 1., 0., 1.),
               generator_width=200,
               random_state=None):
    """Inits the environment.

    Args:
      unity_environment: To be used to run the environment. Should be created
        with unity/environment.py.
      block_replacement: if True, the same block can be used multiple times.
      sticky_penalty: value to be subtracted from the score for each sticky
        block used.
      max_difficulty: the maximum curriculum difficulty level.
      progress_threshold: the fraction of maximum reward that needs to be
        obtained for the task to be considered "solved".
      bad_choice_penalty: value to be subtracted from the score each time the
        agent does not select an object correctly.
      spawn_collision_penalty: Reward to be passed to the agent when it
        terminates the episode early by placing an object overlapping with
        another object.
      hit_obstacle_penalty: Reward to be passed to the agent when it
        terminates the episode early by hitting an obstacle.
      difficulty: Difficulty of the environment. If None, it will be required
        to be passed in the reset method instead. It will usually be an integer
        between 0 and `max_difficulty`. Some base classes may accept a string
        as documented in their docstring to indicate a generalization level.
      curriculum_sample: If `True`, then when doing the curriculum, difficulties
        up to the current difficulty are sampled. If None, it will
        be required to be passed in the reset method instead. It cannot be set
        to true when the difficulty is passed as a string.
      curriculum_sample_geom_p: Parameter of the geometric distribution used
        to sample difficulty levels when curriculum_sample = True. A value of
        0.6, indicates that approximately 0.6 of the episodes run at the
        `current` difficulty, 0.6 of the remaining episodes run at `current-1`,
        0.6 of the remaining at `current-2`, etc. Since the probabilities
        are normalized, a small value can be used here for uniform distribution.
        A value of 1, is equivalent to curriculum_sample=False, and a value of 0
        is equivalent to uniform sampling.
      bad_choice_termination: If True, episodes terminate when an agent tries to
        select an available object that is no longer available.
      num_simulation_steps: number of simulation steps to run every time an
        object is placed.
      target_color: Color of the targets.
      block_color: Color of the blocks.
      sticky_block_color: Color of the sticky blocks.
      obstacle_color: Color of the obstacles.
      ball_color: Color of the balls.
      generator_width: Width discretization unit for generator.
      random_state: a np.random.RandomState object.
    """
    self._unity_environment = unity_environment
    self._random_state = random_state or np.random
    # This number is the width discretization unit.
    self._generator_width = generator_width

    # Maximum displacement from the center of the image to display available
    # objects. Units are the same as object horizontal positions. Camera view
    # covers roughtly between -7 and +7.
    self._display_limit = 7.
    generator_scale = self._display_limit * 2. / self._generator_width
    self._generator_scale = np.array((generator_scale, generator_scale))
    self._generator_offset = np.array((
        -self._generator_width*self._generator_scale[0]/2, 0.))

    # Force boolean parameters to not be passed as None.
    assert block_replacement is not None

    self._block_replacement = block_replacement
    self._sticky_penalty = sticky_penalty
    self._bad_choice_penalty = bad_choice_penalty
    self._spawn_collision_penalty = spawn_collision_penalty
    self._hit_obstacle_penalty = hit_obstacle_penalty
    self._bad_choice_termination = bad_choice_termination
    self._progress_threshold = progress_threshold

    self._target_color = target_color
    self._block_color = block_color
    self._sticky_block_color = sticky_block_color
    self._obstacle_color = obstacle_color
    self._ball_color = ball_color

    assert sticky_penalty > -1e-6
    assert bad_choice_penalty > -1e-6
    assert sticky_penalty > -1e-6
    assert hit_obstacle_penalty > -1e-6

    self._num_simulation_steps = num_simulation_steps

    self._init_curriculum_sample = curriculum_sample
    self._init_difficulty = difficulty
    if curriculum_sample_geom_p < 0.:
      raise ValueError("`curriculum_sample_geom_p (%g) should be >= 0.`"
                       % curriculum_sample_geom_p)
    self._curriculum_sample_geom_p = curriculum_sample_geom_p
    self._max_difficulty = max_difficulty
    self._termination_reason = None
    self._state_ignore_fields = [
        "_unity_environment", "_random_state", "_generator"]

    # Contains the overall level of difficulty.
    self._overall_difficulty = None
    # Contains the overall level of difficulty of the current episode instance.
    # Equal to `self._overall_difficulty` when curriculum sample is False.
    self._episode_difficulty = None

    # For the frame observer.
    self._frames_list = None
    self._frame_observer = None

    self._initialize()

  def close(self):
    self._unity_environment.close()

  @property
  def max_difficulty(self):
    return self._max_difficulty

  def get_state(self, ignore_unity_state=False):
    state = serialization.get_object_state(self, self._state_ignore_fields)
    if not ignore_unity_state:
      state["_unity_environment"] = self._unity_environment.last_observation
    state["_generator"] = self._generator.get_state()
    return state

  def get_reset_state(self):
    """Reset state to pass to reset method to restart an identical episode."""
    return self._reset_state

  def set_state(self, state, ignore_unity_state=False):
    serialization.set_object_state(self, state, self._state_ignore_fields)
    # In scenes with many constraints (glue) it is not always possible to
    # the state fully accurately, leading to different velocities. This should
    # not be much of a problem, since the state that is restored should
    # only have objects with velocities close to 0 (settled blocks, without
    # the ball).
    if not ignore_unity_state:
      self._unity_environment.restore_state(
          state["_unity_environment"], verify_velocities=False)
    self._generator.set_state(state["_generator"])

  def _split_available_obstacles_placed(self, blocks):
    """Splits observations for available blocks, obstacles and placed blocks."""
    num_remaining_display_blocks = len(self._remaining_indices)
    num_obstacles = self._num_obstacles
    num_targets = self._num_targets

    # Because of the order in which the objects were added, we know the
    # obstacles come first, available objects next, and all remaining
    # objects are blocks placed by the agent.
    object_offset = 0
    obstacles = blocks[:num_obstacles]
    object_offset += num_obstacles
    targets = blocks[object_offset:object_offset+num_targets]
    object_offset += num_targets
    available = blocks[object_offset:object_offset+num_remaining_display_blocks]
    object_offset += num_remaining_display_blocks
    placed = blocks[object_offset:]
    return available, targets, obstacles, placed

  def _maybe_add_segmentation_masks(self, observation):
    if "Segmentation" not in list(observation.keys()):
      return
    segmentation = observation["Segmentation"]
    for name in _OBJECT_TYPE_NAMES:
      obs_name = "SegmentationMasks" + name
      ids = list(np.round(observation[name][:, 0]))
      observation[obs_name] = build_segmentation_masks_for_ids(
          segmentation, ids)
    del observation["Segmentation"]

  def _set_observation_and_termination(
      self, time_step, default_step_type=dm_env.StepType.MID):

    new_observation = time_step.observation.copy()
    # We split the different types of blocks.
    (available, targets, obstacles,
     placed) = self._split_available_obstacles_placed(
         time_step.observation["Blocks"])

    new_observation[constants.AVAILABLE_BLOCK] = available
    new_observation[constants.BLOCK] = placed
    new_observation[constants.OBSTACLE] = obstacles
    new_observation[constants.TARGET] = targets

    contact_pairs, contact_features = _calculate_contact_pairs_and_features(
        placed[:, unity_constants.ID_FEATURE_INDEX],
        new_observation["Contacts"])

    del new_observation["Contacts"]
    new_observation["ContactPairs"] = contact_pairs
    new_observation["ContactFeatures"] = contact_features

    self._maybe_add_segmentation_masks(new_observation)

    # Evaluate termination conditions.
    # If we have placed as many objects as there are in display, or have reached
    # the maximum number of steps
    if not available.shape[0] or self._num_steps >= self._max_steps:
      self._end_episode(constants.TERMINATION_MAX_STEPS)

    # If there was a Spawn collision. A Spawn collision means the agent placed
    # an object overlapping with another object. We also override the reward.
    penalty_reward = 0.
    block_reward = 0.
    if (time_step.observation["SpawnCollisionCount"] >
        self._initial_spawn_collision_count):
      self._end_episode(constants.TERMINATION_SPAWN_COLLISION)
      penalty_reward = -self._spawn_collision_penalty
    # If we hit an obstacle, we also end the episode and override the reward.
    elif _obstacle_has_been_hit(obstacles):
      self._end_episode(constants.TERMINATION_OBSTACLE_HIT)
      penalty_reward = -self._hit_obstacle_penalty
    else:
      # We remove the floor before evaluating the score.
      blocks = new_observation[constants.BLOCK][1:]
      self._num_sticky_blocks = np.sum(
          blocks[:, unity_constants.STICKY_FEATURE_INDEX])
      self._progress = self._get_task_reward(
          new_observation[constants.OBSTACLE],
          new_observation[constants.TARGET],
          blocks)
      total_cost = self._get_cost(blocks)
      total_score = self._progress
      cost = total_cost - self._previous_cost
      self._previous_cost = total_cost
      block_reward = total_score - self._previous_score
      self._previous_score = total_score
      block_reward -= cost

      if self._enough_progress(self._progress):
        self._end_episode(constants.TERMINATION_COMPLETE)

    if self._is_end_of_episode:
      step_type = dm_env.StepType.LAST
      discount = time_step.discount * 0.
    else:
      step_type = default_step_type
      discount = time_step.discount

    reward = penalty_reward + block_reward

    self._last_time_step = time_step._replace(
        observation=new_observation,
        step_type=step_type,
        discount=discount,
        reward=reward)

    return self._last_time_step

  def _get_cost(self, blocks):
    # The number of bad choices can be inferred from the total number of blocks.
    num_bad_choices = self._num_steps - len(blocks)
    total_cost = self._bad_choice_penalty * num_bad_choices
    total_cost += self._sticky_penalty * self._num_sticky_blocks
    return total_cost

  def observation_spec(self, *args, **kwargs):
    new_spec = self._unity_environment.observation_spec().copy()
    # The block observation is exactly as we get it
    block_obs_shape = [0, new_spec[constants.BLOCK].shape[1]]
    block_obs_dtype = new_spec[constants.BLOCK].dtype
    # We know the observation is the same for all block types.
    for name in _OBJECT_TYPE_NAMES:
      new_spec[name] = specs.Array(
          block_obs_shape, dtype=block_obs_dtype, name=name)

    if "Segmentation" in list(new_spec.keys()):
      segmentation_resolution = new_spec["Segmentation"].shape[:2]
      segmentation_obs_shape = (0,) + segmentation_resolution
      for name in _OBJECT_TYPE_NAMES:
        obs_name = "SegmentationMasks" + name
        new_spec[obs_name] = specs.Array(
            segmentation_obs_shape, dtype=np.bool, name=obs_name)
      del new_spec["Segmentation"]

    new_spec.update({"ContactPairs": specs.Array(
        [0, 2], dtype=np.int32, name="ContactPairs")})
    new_spec.update({"ContactFeatures": specs.Array(
        [0, 1], dtype=new_spec["Contacts"].dtype, name="ContactFeatures")})

    del new_spec["Contacts"]
    return new_spec

  def action_spec(self, *args, **kwargs):
    action_spec = {}
    # The action spec of the unity_environment is documented in
    # unity/environment.py.
    unity_action_spec = self._unity_environment.action_spec()
    action_spec["Horizontal"] = unity_action_spec["SetPosX"]
    action_spec["Vertical"] = unity_action_spec["SetPosY"]
    action_spec["Sticky"] = specs.DiscreteArray(num_values=2)
    action_spec["Selector"] = specs.BoundedArray(
        [], dtype=np.float32,
        minimum=-self._display_limit,
        maximum=self._display_limit)

    return action_spec

  def step(self, actions):
    if self._is_end_of_episode:
      raise ValueError("Calling step on a closed episode")

    self._num_steps += 1

    slot_index = self._selector_value_to_slot_index(actions["Selector"])
    horizontal = actions["Horizontal"]
    vertical = actions["Vertical"]

    # Dictionary for the actions that are going to be applied to the core env.
    actions_apply = {}

    # To move the cursor to the object picked by the agent and the location
    # picked by the agent.
    display_coordinates = self._display_coordinates[slot_index]
    actions_apply.update({"SelectPosX": display_coordinates[0],
                          "SelectPosY": display_coordinates[1],
                          "SetPosX": horizontal,
                          "SetPosY": vertical})

    # If the selected block is not available, nothing else happens.
    if slot_index not in self._remaining_indices:
      time_step = self._unity_environment.step(actions_apply)
      if self._bad_choice_termination:
        self._end_episode(constants.TERMINATION_BAD_CHOICE)
      return self._set_observation_and_termination(time_step)

    # If  there is no replacement, remove the objects from remaining objects
    # and append the delete action.
    if not self._block_replacement:
      self._remaining_indices.remove(slot_index)
      display_object_id = self._display_ids[slot_index]
      actions_apply.update({
          "Delete": 1.,
          "SelectId": display_object_id,
          "SetId": display_object_id
      })
    else:
      actions_apply["SetId"] = self._next_object_id
      self._next_object_id += 1

    # Setting the actions necessary to add the new block.
    new_block = self._initial_available_objects[slot_index]
    size_x = new_block.width
    size_y = new_block.height

    if actions["Sticky"]:
      actions_apply["Sticky"] = 1.
      actions_apply.update({"RGBA": self._sticky_block_color})
    else:
      actions_apply.update({"RGBA": self._block_color})

    actions_apply.update(
        {"Width": size_x,
         "Height": size_y,
         "Shape": new_block.shape,
         "SimulationSteps": float(self._num_simulation_steps),
         "FreeBody": 1.,
         "SpawnBlock": 1.})

    try:
      time_step = self._unity_environment.step(actions_apply)
    except unity_constants.MetaEnvironmentError as e:
      logging.info(e)
      self._end_episode(constants.TERMINATION_BAD_SIMULATION)
      self._last_time_step = self._last_time_step._replace(
          discount=self._last_time_step.discount * 0.,
          reward=self._last_time_step.reward * 0.,
          step_type=dm_env.StepType.LAST)
      return self._last_time_step
    else:
      out = self._set_observation_and_termination(time_step)
      return out

  def _initialize(self):
    # Initializes the env by forcing a reset. This is important to be
    # able to get and set states, so all attributes are instantiated
    # and a generator is put in place.
    self.reset(
        difficulty=None if self._init_difficulty is not None else 0,
        curriculum_sample=(
            None if self._init_curriculum_sample is not None else False)
        )

  def reset(self, reset_state=None, difficulty=None, curriculum_sample=None):
    """Resets the generator.

    Args:
      reset_state: A full state that guarantees that an environment will be
        reset to the same initial conditions as a past episode.
      difficulty: Difficulty of the environment.
      curriculum_sample: If `True`, then when doing the curriculum, difficulties
        up to the current difficulty are sampled.

    Returns:
      time_step: The initial time_step.
    """
    while True:
      found, time_step = self._try_to_reset(
          reset_state=reset_state,
          difficulty=difficulty,
          curriculum_sample=curriculum_sample)
      if reset_state is not None:
        # We should always be able to reset from a reset state
        # in a single attempt.
        assert found
      if found:
        return time_step

  def _clip_slot_index(self, slot_index):
    if slot_index < 0:
      slot_index = 0
    elif slot_index >= len(self._initial_available_objects):
      slot_index = len(self._initial_available_objects) - 1
    return slot_index

  def _selector_value_to_slot_index(self, selector_value):
    slot_index = int(np.digitize(selector_value, self._display_edges)-1)
    return self._clip_slot_index(slot_index)

  def _end_episode(self, reason):
    if reason not in self.all_termination_reasons:
      raise ValueError("invalid termination reason: {}".format(reason))
    self._termination_reason = reason
    self._is_end_of_episode = True

  @property
  def termination_reason(self):
    return self._termination_reason

  @property
  def all_termination_reasons(self):
    return [
        constants.TERMINATION_MAX_STEPS,
        constants.TERMINATION_SPAWN_COLLISION,
        constants.TERMINATION_OBSTACLE_HIT,
        constants.TERMINATION_COMPLETE,
        constants.TERMINATION_BAD_SIMULATION,
        constants.TERMINATION_BAD_CHOICE,
    ]

  @property
  def core_env(self):
    return self

  @property
  def difficulty(self):
    """Returns the overall current difficulty passed to init or reset method.

     If `curriculum_sample` is True, the difficulty of the current episode will
     be sampled from 0 up to this value, and can be obtained via
     `episode_difficulty`.
    """
    return self._overall_difficulty

  @property
  def episode_difficulty(self):
    """Returns the actual difficulty of the present episode.

    If `curriculum_sample` is False, this will always be equal to `difficulty`.
    Otherwise, it will be `0 <= episode_difficulty <= difficulty`.
    """
    return self._episode_difficulty

  @property
  def episode_logs(self):
    """A dictionnary of logs for a completed episode."""
    normalized_glue_points = 0.
    if self._num_steps > 0:
      normalized_glue_points = self._num_sticky_blocks/float(self._num_steps)
    return dict(
        score=self._previous_score,
        num_steps=self._num_steps,
        glue_points=self._num_sticky_blocks,
        normalized_score=self._previous_score/self._max_episode_reward,
        normalized_glue_points=normalized_glue_points)

  @property
  def last_time_step(self):
    return self._last_time_step

  # Abstract methods below.

  def _enough_progress(self, progress):
    """Whether enough reward has been obtained."""
    return progress > self._max_episode_reward * self._progress_threshold

  @abc.abstractmethod
  def _get_generator(self, difficulty):
    """Will return a generator for the required difficulty."""

  @abc.abstractmethod
  def _get_task_reward(self, obstacles, targets, blocks):
    """Returns the score for this set of obstacles, targets and blocks."""

  @abc.abstractmethod
  def _maybe_update_max_steps(self):
    """Update max_num_steps based on the current instance properties."""

  def _get_sampled_episode_difficulty(
      self, difficulty, curriculum_sample):
    """Returns a value of the difficulty to be used for the next episode."""

    if not curriculum_sample:
      # If we don't do curriculum sample, we just return the passed difficulty.
      return difficulty

    # Will be sampling from a difficulty value from 0 up to difficulty.
    candidate_difficulties = list(range(difficulty + 1))
    num_candidate_difficulties = len(candidate_difficulties)

    # And define the probabilities that we will sampling from each level.
    if self._curriculum_sample_geom_p > 0.:
      distribution = stats.distributions.geom(
          p=self._curriculum_sample_geom_p)
      # Geometrical distribution pmf starts at 1.
      probs = distribution.pmf(np.arange(1, num_candidate_difficulties+1))
      # Geometrical distributions goes from high to low, but we want the
      # opposite (higher probability for the highest level).
      probs = probs[::-1]
    else:
      # A value of 0. corresponds to uniform distribution among all
      # candidate difficulties.
      probs = np.ones([num_candidate_difficulties], dtype=np.float32)

    # Normalize probabilities.
    candidate_difficulties_probs = probs / probs.sum()

    # Sample a difficulty according to their probabilities.
    sampled_difficulty = int(np.random.choice(
        candidate_difficulties, p=candidate_difficulties_probs))
    return sampled_difficulty

  def _get_new_starting_configuration(
      self, difficulty, curriculum_sample):
    sampled_difficulty = self._get_sampled_episode_difficulty(
        difficulty, curriculum_sample)

    self._generator = self._get_generator(sampled_difficulty)
    self._episode_difficulty = sampled_difficulty
    blocks_observation = self._generator.generate_one()

    # Rescale the blocks observation.
    blocks_observation = block_utils.transform_blocks_observation(
        blocks_observation, self._generator_scale, self._generator_offset)
    return blocks_observation

  def _get_difficulty_and_curriculum_sample(
      self, reset_difficulty, reset_curriculum_sample):
    if not ((reset_difficulty is None) ^
            (self._init_difficulty is None)):
      raise ValueError(
          "A difficulty value must be passed to the constructor (%s) or "
          "to the reset method (%s) and never to both." % (
              self._init_difficulty, reset_difficulty))
    if not ((reset_curriculum_sample is None) ^
            (self._init_curriculum_sample is None)):
      raise ValueError(
          "A curriculum_sample value must be passed to the constructor (%s) or "
          "to the reset method (%s) and never to both." % (
              self._init_curriculum_sample, reset_curriculum_sample))

    if reset_difficulty is not None:
      difficulty = reset_difficulty
    else:
      difficulty = self._init_difficulty

    if reset_curriculum_sample is not None:
      curriculum_sample = reset_curriculum_sample
    else:
      curriculum_sample = self._init_curriculum_sample

    if isinstance(difficulty, int):
      if difficulty > self._max_difficulty or  difficulty < 0:
        raise ValueError("Trying to set a value of the difficulty (%d) larger "
                         "than the maximum difficulty (%d) or smaller than 0" %(
                             difficulty, self._max_difficulty))
    elif isinstance(difficulty, str):
      if curriculum_sample:
        raise ValueError(
            "`difficulty` can only be a passed as a string when using "
            "`curriculum_sample==False`, got `difficulty==%s`" % difficulty)
    else:
      raise ValueError(
          "Difficulty must be `int` or `str`, got (%s) with type (%s)" %
          (str(difficulty), type(difficulty)))

    return difficulty, curriculum_sample

  def _try_to_reset(self, reset_state, difficulty, curriculum_sample):
    """Tries to generate a new episode.

    Args:
      reset_state: A full state that guarantees that an environment will be
        reset to the same initial conditions as a past episode.
      difficulty: Difficulty of the environment.
      curriculum_sample: If `True`, then when doing the curriculum, difficulties
        up to the current difficulty are sampled.

    Returns:
      1. A boolean indicating whether the scene generation was successful.
      2. A time_step corresponding to the beginning of an episode, if the
         generation was successful, or None.
    """

    if reset_state is None:
      (difficulty,
       curriculum_sample) = self._get_difficulty_and_curriculum_sample(
           difficulty, curriculum_sample)

      self._overall_difficulty = difficulty

      self._initial_scene = self._get_new_starting_configuration(
          difficulty, curriculum_sample)
      self._initial_available_objects = self._initial_scene.blocks[1:]

      self._maybe_update_max_steps()

      # It is assumed that from here on, everything is deterministic, so it is
      # a safe point to obtain the reset_state.
      self._reset_state = None  # So we don't get this as part of the state.
      self._reset_state = self.get_state(ignore_unity_state=True)
    else:
      if difficulty is not None:
        raise ValueError(
            "`difficulty` should be None when `reset_state` is passed.")
      if curriculum_sample is not None:
        raise ValueError(
            "`curriculum_sample` should be None when `reset_state` is passed.")

      self.set_state(reset_state, ignore_unity_state=True)
      # This is the only thing that would not have been restored.
      self._reset_state = reset_state

    return self._deterministic_reset()

  def _deterministic_reset(self):
    """Set-up work for the episode that is fully deterministic on the state."""

    # Start setting up the scene in Unity.
    setup_actions = []
    self._unity_environment.reset()

    # Indices corresponding to the _initial_available_objects still available.
    # (All of them are available at the beginning of the episode).
    self._remaining_indices = {
        i for i in range(len(self._initial_available_objects))}

    # Place the obstacles.
    self._num_obstacles = len(self._initial_scene.obstacles)
    self._num_targets = len(self._initial_scene.targets)
    self._max_episode_reward = self._compute_max_episode_reward(
        self._initial_scene)
    self._progress = None
    object_index = len(self._initial_available_objects) + 1

    obstacle_color = self._obstacle_color
    for obstacle in self._initial_scene.obstacles:
      setup_actions.append(
          {"SetId": object_index,
           "SetPosX": obstacle.x, "SetPosY": obstacle.y,
           "Width": obstacle.width,
           "Height": obstacle.height,
           "SetAngle": obstacle.angle,
           "Shape": obstacle.shape,
           "SpawnBlock": 1.,
           "RGBA": obstacle_color})
      object_index += 1

    target_color = self._target_color
    for target in self._initial_scene.targets:
      setup_actions.append(
          {"SetId": object_index,
           "SetPosX": target.x, "SetPosY": target.y,
           "Width": target.width,
           "Height": target.height,
           # By default, collision masks are 0b0001, so by using 0b0010 target
           # will not collide with any block, unless their mask matches 0b??1?.
           "CollisionMask": 0b10,
           "SetAngle": target.angle,
           "Shape": target.shape,
           "SpawnBlock": 1.,
           "RGBA": target_color})
      object_index += 1

    # Add the balls only for display purposes.
    self._ball_ids = []
    for ball in self._initial_scene.balls:
      self._ball_ids.append(object_index)
      setup_actions.append({
          "SpawnBlock": 1.,
          "PhysicalBody": 0.,
          "Shape": ball.shape,
          "SetId": object_index,
          "SetPosX": ball.x,
          "SetPosY": ball.y,
          "Width": ball.width,
          "Height": ball.height,
          "RGBA": np.array(list(self._ball_color[:3]) + [0.5]),
      })
      object_index += 1

    self._display_ids = []
    self._display_coordinates = []
    blocks_starts = []
    blocks_ends = []
    for display_index, block in enumerate(self._initial_available_objects):
      # We give explicit positive ids to the display objects,
      # so we can remove them later using their ids.
      display_id = (display_index+1)
      y_display = -1.
      x_display = block.x
      setup_actions.append(
          {"SetId": display_id,
           "SetPosX": block.x, "SetPosY": block.y,
           "Width": block.width,
           "Height": block.height,
           "SetAngle": block.angle,
           "Shape": block.shape,
           "SpawnBlock": 1., "RGBA": self._block_color})
      self._display_ids.append(display_id)
      self._display_coordinates.append((x_display, y_display))
      blocks_starts.append(block.x-np.abs(block.width)/2.)
      blocks_ends.append(block.x+np.abs(block.width)/2.)

    # Compute the edge between two blocks as the center between the end of the
    # previous block and the start of the next block
    edges = [(x + y) / 2. for x, y in zip(blocks_ends[:-1], blocks_starts[1:])]
    self._display_edges = [-self._display_limit] + edges + [self._display_limit]

    # Place the floor.
    floor = self._initial_scene.blocks[0]
    setup_actions.append(
        {"SetId": object_index,
         "SetPosX": floor.x, "SetPosY": floor.y,
         "SetAngle": floor.angle,
         "Shape": floor.shape,
         "Width": floor.width, "Height": floor.height,
         "SpawnBlock": 1., "R": 0., "G": 0., "B": 0., "A": 1.})

    self._next_object_id = object_index + 1
    self._previous_cost = 0
    self._previous_score = 0.
    self._num_steps = 0
    self._is_end_of_episode = False

    time_step = self._unity_environment.step(setup_actions)
    self._initial_spawn_collision_count = time_step.observation[
        "SpawnCollisionCount"]

    first_time_step = self._set_observation_and_termination(
        time_step, default_step_type=dm_env.StepType.FIRST)
    time_step = first_time_step

    self._termination_reason = None

    return True, time_step._replace(step_type=first_time_step.step_type)

  def enable_frame_observer(self):
    """Enables a frame observer on the Unity environment.

    This observer will gather frames from the Unity observer camera, which
    typically produces higher-res images than agent observations.
    """
    if self._frame_observer is not None:
      raise ValueError("the frame observer is already enabled")
    obs_spec = self._unity_environment.observation_spec()
    if "ObserverRGB" not in obs_spec:
      raise ValueError(
          "the observer camera in the Unity environment is not enabled")
    self._frames_list = []
    self._frame_observer = (
        lambda obs: self._frames_list.append(obs["ObserverRGB"]))
    self._unity_environment.add_observer(self._frame_observer)

  def disable_frame_observer(self):
    """Disables the frame observer on the Unity environment.

    This observer will gather frames from the Unity observer camera, which
    typically produces higher-res images than agent observations.
    """
    if self._frame_observer is None:
      return
    self._unity_environment.remove_observer(self._frame_observer)
    self._frames_list = None
    self._frame_observer = None

  def pop_observer_frames(self):
    """Queries frames from the frame observer, and empties the frame list.

    Returns:
      observations: list of RGB frames
    """
    if self._frame_observer is None:
      raise ValueError("the frame observer is not enabled")
    observations = self._frames_list.copy()
    self._frames_list[:] = []
    return observations


class GenerationError(Exception):
  pass


class StackingGenerator(metaclass=abc.ABCMeta):
  """Abstract base class for construction generators."""

  def __init__(self,
               num_blocks_range,
               scene_width,
               random_state,
               height=10,
               margin=5,
               num_small=3,
               num_medium=3,
               num_large=1):
    """Initialize the generator.

    Args:
      num_blocks_range: a tuple indicating the range of obstacles
        that will be in the generated towers, from low (inclusive) to high
        (exclusive).
      scene_width: the width of the scene.
      random_state: a np.random.RandomState object
      height: the height of a block
      margin: the space between blocks
      num_small: the number of small available blocks
      num_medium: the number of medium available blocks
      num_large: the number of large available blocks
    """
    self.num_blocks_range = num_blocks_range
    self.scene_width = scene_width
    self.random_state = random_state
    self._state_ignore_fields = ["random_state"]

    self.scene_height = self.scene_width

    # Width of small, medium, and large blocks.
    self.small_width = constants.SMALL_WIDTH
    self.medium_width = constants.MEDIUM_WIDTH
    self.large_width = constants.LARGE_WIDTH

    self.height = height
    self.margin = margin

    self._num_small = num_small
    self._num_medium = num_medium
    self._num_large = num_large

  def get_state(self):
    return serialization.get_object_state(self, self._state_ignore_fields)

  def set_state(self, state):
    serialization.set_object_state(self, state, self._state_ignore_fields)

  def _place_available_objects(self):
    """Create the set of objects that can be picked up."""
    # compute the margins between available blocks
    available_width = self.scene_width
    available_width -= self._num_small * self.small_width
    available_width -= self._num_medium * self.medium_width
    available_width -= self._num_large * self.large_width
    num_available = self._num_small + self._num_medium + self._num_large
    if num_available > 1:
      margin = available_width / (num_available - 1)
    else:
      margin = available_width
    assert margin >= 1
    margin = np.floor(margin)

    current_position = dict(x=0, y=-2 * (self.margin + self.height))
    def add_block(width):
      block = block_utils.Block(
          x=current_position["x"] + width / 2,
          y=current_position["y"],
          width=width,
          height=self.height)
      current_position["x"] += width + margin
      return block

    observation_blocks = [
        add_block(self.small_width) for _ in range(self._num_small)]
    observation_blocks += [
        add_block(self.medium_width) for _ in range(self._num_medium)]
    observation_blocks += [
        add_block(self.large_width) for _ in range(self._num_large)]

    assert current_position["x"] - margin <= self.scene_width
    return observation_blocks

  def _place_floor(self):
    floor_height = self.height / 2
    floor = block_utils.Block(
        x=self.scene_width / 2., y=-floor_height / 2.,
        height=floor_height, width=self.scene_width * 2)
    return floor

  @abc.abstractmethod
  def generate_one(self):
    """Generate a single scene.

    Returns:
      A BlocksObservation object
    """
    pass

