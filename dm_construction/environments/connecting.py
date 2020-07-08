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
"""An environment for stacking blocks from the floor to targets.

See: Bapst, V., Sanchez-Gonzalez, A., Doersch, C., Stachenfeld, K., Kohli, P.,
Battaglia, P., & Hamrick, J. (2019, May). Structured agents for physical
construction. In International Conference on Machine Learning (pp. 464-474).
"""

from dm_construction.environments import stacking
from dm_construction.unity import constants as unity_constants
from dm_construction.utils import block as block_utils
from dm_construction.utils import constants
from dm_construction.utils import geometry
import numpy as np


def _is_target_reached(block, target):
  """Returns `True` if block overlaps with target, `False` otherwise."""
  target_center_x = target[unity_constants.POSITION_X_FEATURE_INDEX]
  target_center_y = target[unity_constants.POSITION_Y_FEATURE_INDEX]
  center_x = block[unity_constants.POSITION_X_FEATURE_INDEX]
  center_y = block[unity_constants.POSITION_Y_FEATURE_INDEX]
  cos_theta = block[unity_constants.COSINE_ANGLE_FEATURE_INDEX]
  sin_theta = block[unity_constants.SINE_ANGLE_FEATURE_INDEX]
  width = block[unity_constants.WIDTH_FEATURE_INDEX]
  height = block[unity_constants.HEIGHT_FEATURE_INDEX]
  x1, y1 = geometry.rotate_rectangle_corner(
      center_x + width / 2, center_y - height / 2,
      center_x, center_y, cos_theta, sin_theta)
  x2, y2 = geometry.rotate_rectangle_corner(
      center_x + width / 2, center_y + height / 2,
      center_x, center_y, cos_theta, sin_theta)
  x3, y3 = geometry.rotate_rectangle_corner(
      center_x - width / 2, center_y + height / 2,
      center_x, center_y, cos_theta, sin_theta)
  x4, y4 = geometry.rotate_rectangle_corner(
      center_x - width / 2, center_y - height / 2,
      center_x, center_y, cos_theta, sin_theta)
  return geometry.is_point_in_rectangle(
      x1, y1, x2, y2, x3, y3, x4, y4, target_center_x, target_center_y,)


def _count_targets_reached(targets, blocks):
  """Returns the reward for the connecting task."""
  targets_reached = 0
  for target in targets:
    for block in blocks:
      is_reached = _is_target_reached(block, target)
      if is_reached:
        targets_reached += 1
        break
  return targets_reached


class ConstructionConnecting(stacking.ConstructionStacking):
  """Environment for the Connecting task.

  In the Connecting task, the agent must stack blocks to connect the floor to
  three different target locations, avoiding randomly positioned obstacles
  arranged in layers. The reward function is: +1 for each target whose center is
  touched by at least one block, and 0 (no penalty) for each block set to
  sticky. The task-specific termination criterion is achieved when all targets
  are connected to the floor.

  Generalization levels:
    * `"mixed_height_targets"`: Scenes where different targets may be at
      different vertical positions interleaved with the obstacle layers (in
      other levels all three targets are always on top of the highest layer of
      obstacles).
    * `"additional_layer"`: Scenes with 4 layers of obstacles, with targets
      above the new highest obstacle layer (maximum number of obstacle
      layers for other levels is 3).
  """

  def __init__(self,
               unity_environment,
               sticky_penalty=0.0,
               **stacking_kwargs):
    """Inits the environment.

    Args:
      unity_environment: See base class.
      sticky_penalty: See base class.
      **stacking_kwargs: keyword arguments passed to
        covering.ConstructionStacking.
    """

    self._max_steps = None
    super(ConstructionConnecting, self).__init__(
        unity_environment=unity_environment,
        block_replacement=True,
        sticky_penalty=sticky_penalty,
        max_difficulty=9,
        progress_threshold=0.98,
        **stacking_kwargs)

  def _compute_max_episode_reward(self, obs):
    return len(obs.targets)

  def _maybe_update_max_steps(self):
    pass

  def _get_task_reward(self, obstacles, targets, blocks):
    """Computes the current score based on the targets and placed blocks."""
    del obstacles
    targets_reached = _count_targets_reached(targets, blocks)
    return targets_reached

  def _get_generator(self, difficulty):
    offset = 0

    if isinstance(difficulty, int):
      min_num_targets = 3
      max_num_targets = 3
      # Targets on floor level, no obstacles
      if difficulty >= 0:
        min_num_obstacles = 0
        max_num_obstacles = 0
        obstacles_ys_range = [(0,)]
        targets_ys_range = [(0,)]
        self._max_steps = 7
      # Targets one level higher
      if difficulty >= 1:
        targets_ys_range = [(1,)]
        self._max_steps = 7
      # Obstacles on the floor, targets one level higher
      if difficulty >= 2:
        min_num_obstacles = 1
        max_num_obstacles = 1
        obstacles_ys_range = [(0,)]
        targets_ys_range = [(1 + offset,)]
        self._max_steps = 7
      # More obstacles
      if difficulty >= 3:
        max_num_obstacles = 2
      # Even more obstacles (3 per layer), and more targets
      if difficulty >= 4:
        min_num_obstacles = 2
        max_num_obstacles = 3
      # Make the targets higher
      if difficulty >= 5:
        targets_ys_range = [(2 + offset,)]
        self._max_steps = 14
      # Even higher, and more obstacles
      if difficulty >= 6:
        min_num_obstacles = 3
        targets_ys_range = [(3 + offset,)]
        self._max_steps = 21
      # Second layer of obstacles, and more targets
      if difficulty >= 7:
        obstacles_ys_range = [(0, 2)]
      # Move targets higher
      if difficulty >= 8:
        targets_ys_range = [(4 + offset,)]
      # More obstacles, higher targets
      if difficulty >= 9:
        obstacles_ys_range = [(0, 2, 4)]
        targets_ys_range = [(5 + offset,)]
    elif difficulty == "mixed_height_targets":
      # Targets at different heights, instead of a single height.
      min_num_targets = 1
      max_num_targets = 1
      min_num_obstacles = 3
      max_num_obstacles = 3
      obstacles_ys_range = [(0, 2, 4)]
      targets_ys_range = [(1 + offset, 3 + offset, 5 + offset,)]
      self._max_steps = 26
    elif difficulty == "additional_layer":
      # Targets at a new height
      min_num_targets = 3
      max_num_targets = 3
      min_num_obstacles = 3
      max_num_obstacles = 3
      obstacles_ys_range = [(0, 2, 4, 6)]
      targets_ys_range = [(7 + offset,)]
      self._max_steps = 26
    else:
      raise ValueError("Unrecognized difficulty: %s" % difficulty)

    return ConnectingGenerator(
        num_obstacles_range=(min_num_obstacles, max_num_obstacles + 1),
        num_targets_range=(min_num_targets, max_num_targets+1),
        scene_width=self._generator_width,
        random_state=self._random_state,
        obstacles_ys_range=obstacles_ys_range,
        targets_ys_range=targets_ys_range,
        min_obstacles_interdistance=constants.SMALL_WIDTH * 2,
        min_targets_interdistance=0.)


class ConnectingGenerator(stacking.StackingGenerator):
  """Generates a set of horizontal obstacles and targets."""

  def __init__(self,
               num_obstacles_range,
               num_targets_range,
               obstacles_ys_range,
               targets_ys_range,
               scene_width,
               random_state,
               obstacles_width_range=(10, 40),
               use_legacy_obstacles_heights=False,
               targets_side=5,
               obstacles_height=5,
               min_obstacles_interdistance=0.,
               min_targets_interdistance=0.,
               **kwargs):
    """Initialize the generator.

    Args:
      num_obstacles_range: a tuple indicating the range of obstacles
        that will be in the generated scene, from low (inclusive) to high
        (exclusive). This counts the number of obstacles per height.
      num_targets_range: a tuple indicating the range of targets
        that will be in the generated scene, from low (inclusive) to high
        (exclusive). This counts the total number of targets.
      obstacles_ys_range: y-position to draw the obstacle from. A tuple of
        y-positions will be sampled from this range. This is scalled
        appropriately, so that -1 corresponds to an object below the floor, 0
        to an object on the floor, etc.
      targets_ys_range: y-position to draw the targets from. A tuple of
        y-positions will be sampled from this range. This is scalled
        appropriately, so that -1 corresponds to an object below the floor, 0
        to an object on the floor, etc.
      scene_width: the width of the scene.
      random_state: a np.random.RandomState object
      obstacles_width_range: the range of widths for obstacles, from low
        (inclusive) to high (exclusive).
      use_legacy_obstacles_heights: In the first versions, obstacles would be
        placed with less margin compared to a corresponding stack of blocks.
        With the new versions, obstacles are thiner and there is therefore more
        margin around them. This makes the task easier, and using glue more
        helpful.
      targets_side: The width and height of targets. Only used when
        `use_legacy_obstacles_heights` is set to False.
      obstacles_height: The height of the obstacles. Only used when
        `use_legacy_obstacles_heights` is set to False.
      min_obstacles_interdistance: The minimal horizontal distance between
        obstacles at the same height. Default=0.
      min_targets_interdistance: The minimal horizontal distance between
        targets at the same height. Default=0.
      **kwargs: additional keyword arguments passed to super
    """
    super(ConnectingGenerator, self).__init__(
        num_blocks_range=None,
        scene_width=scene_width,
        random_state=random_state,
        **kwargs)

    self._num_obstacles_range = num_obstacles_range
    self._num_targets_range = num_targets_range
    self._targets_side = targets_side

    self._min_obstacles_interdistance = min_obstacles_interdistance
    self._min_targets_interdistance = min_targets_interdistance

    self._obstacles_width_range = obstacles_width_range
    self._use_legacy_obstacles_heights = use_legacy_obstacles_heights

    if use_legacy_obstacles_heights:
      self.obstacles_height = self.height
      self._targets_height = self.height
      self._obstacles_ys_range = [
          self._scale_y_range(obstacles_ys, margin=self.margin)
          for obstacles_ys in obstacles_ys_range]
      self._targets_ys_range = [
          self._scale_y_range(targets_ys, offset=self.height/2.)
          for targets_ys in targets_ys_range]
    else:
      self.obstacles_height = obstacles_height
      self._targets_height = targets_side
      scale_y_fn = lambda y: (y + 0.5) * self.height
      scale_ys_fn = lambda ys: tuple([scale_y_fn(y) for y in ys])
      self._obstacles_ys_range = [
          scale_ys_fn(obstacles_ys) for obstacles_ys in obstacles_ys_range]
      self._targets_ys_range = [
          scale_ys_fn(targets_ys) for targets_ys in targets_ys_range]

  def _generate_line_of_blocks(self,
                               available_widths,
                               num_blocks,
                               min_available_width=0,
                               min_interdistance=0.):
    if num_blocks == 0:
      return [], []
    # Pick a set of block widths, and check that the sum of the widths is
    # not greater than the scene width plus some buffer room. keep regenerting
    # blocks until this is the case.
    available_width = -1
    while available_width < min_available_width:
      blocks_lengths = self.random_state.choice(
          available_widths, size=[num_blocks])
      available_width = self.scene_width - np.sum(blocks_lengths)

    # Compute the left and right edges of each block, assuming the blocks are
    # all placed right next to each other beginning from the left side of the
    # scene.
    blocks_begins = np.concatenate(
        [np.array([0], dtype=np.int32), np.cumsum(blocks_lengths)[:-1]])
    blocks_ends = np.cumsum(blocks_lengths)

    # available_width now is the amount of space left on the floor, not taken
    # up by obstacles. we split this into a few chunks of random size to space
    # the obstacles out along the floor
    while True:
      relative_shifts = self.random_state.uniform(0., 1., size=[num_blocks + 1])
      relative_shifts /= np.sum(relative_shifts)
      if len(relative_shifts) < 3 or (
          np.min(relative_shifts[1:-1]) > min_interdistance / available_width):
        break
    relative_shifts = np.floor(relative_shifts * available_width)
    shifts = np.cumsum(relative_shifts.astype(np.int32))[:-1]
    blocks_begins += shifts
    blocks_ends += shifts
    return blocks_begins, blocks_ends

  def _scale_y_range(self, y_range, offset=0., margin=0.):
    return tuple([offset + y * (margin + self.height) for y in y_range])

  def generate_one(self):
    """Generate a single scene.

    Returns:
      observation: a block_utils.BlocksObservation object
      solution: a list of Block objects in their final locations
    """
    # Pick the set of y-positions we want for our obstacles
    idx = np.arange(len(self._obstacles_ys_range))
    obstacles_ys_range = self._obstacles_ys_range[self.random_state.choice(idx)]
    # Place the obstacles at each level, going from bottom to top
    obstacles = []
    for y in obstacles_ys_range:
      available_widths = np.arange(*self._obstacles_width_range)
      obstacles_begins, obstacles_ends = self._generate_line_of_blocks(
          available_widths=available_widths,
          num_blocks=self.random_state.randint(*self._num_obstacles_range),
          min_available_width=self.small_width + 1,
          min_interdistance=self._min_obstacles_interdistance)
      # Now actually create the obstacles
      for obstacle_begin, obstacle_end in zip(obstacles_begins, obstacles_ends):
        center = (obstacle_begin + obstacle_end) // 2
        width = obstacle_end - obstacle_begin
        obstacle = block_utils.Block(
            x=center, y=y, width=width, height=self.obstacles_height)
        obstacles.append(obstacle)

    # Pick y positions for the targets
    idx = np.arange(len(self._targets_ys_range))
    targets_ys_range = self._targets_ys_range[self.random_state.choice(idx)]
    targets = []
    for y in targets_ys_range:
      available_widths = [self.small_width, self.medium_width]
      if not self._use_legacy_obstacles_heights:
        available_widths = [self._targets_side]
      num_targets = self.random_state.randint(*self._num_targets_range)
      targets_begins, targets_ends = self._generate_line_of_blocks(
          available_widths=available_widths,
          num_blocks=num_targets,
          min_interdistance=self._min_targets_interdistance)
      for target_begin, target_end in zip(targets_begins, targets_ends):
        center = (target_begin + target_end) // 2
        width = target_end - target_begin
        target = block_utils.Block(
            x=center, y=y, width=width, height=self._targets_height)
        targets.append(target)

    observation_blocks = self._place_available_objects()
    floor = self._place_floor()

    observation = block_utils.BlocksObservation(
        blocks=[floor] + observation_blocks,
        obstacles=obstacles,
        targets=targets,
        balls=[])

    return observation
