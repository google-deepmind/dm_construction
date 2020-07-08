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
"""A construction environment where the task is to cover obstacles.

See: Bapst, V., Sanchez-Gonzalez, A., Doersch, C., Stachenfeld, K., Kohli, P.,
Battaglia, P., & Hamrick, J. (2019, May). Structured agents for physical
construction. In International Conference on Machine Learning (pp. 464-474).
"""

from dm_construction.environments import stacking
from dm_construction.unity import constants as unity_constants
from dm_construction.utils import block as block_utils
from dm_construction.utils import geometry
import numpy as np


class _CoveringTracker(object):
  """Keeps track of how much of an obstacle has been covered."""

  def __init__(self, obstacle):
    self._obstacle = obstacle
    self._segment_list = []

  def _add(self, xmin, xmax):
    """Track coverage in the segment (xmin, xmax)."""
    for segment_xmin, segment_xmax in self._segment_list:
      # We already have a segment spanning the entire range we are trying to
      # add, so it is redundant.
      if segment_xmin <= xmin and segment_xmax >= xmax:
        return
      # The segment is disjoint from the one we are trying to add
      if segment_xmin > xmax or segment_xmax < xmin:
        continue
      # The segments overlap, so we merge them together.
      else:
        self._segment_list.remove((segment_xmin, segment_xmax))
        self._add(min((segment_xmin, xmin)), max((segment_xmax, xmax)))
        return

    # At this point, it means the segment we're adding is totally disjoint from
    # all the others, so we add it to the list.
    self._segment_list.append((xmin, xmax))

  def add_block(self, block):
    """Computes the xmin and xmax of the block boundaries and tracks it."""
    cos_index = unity_constants.COSINE_ANGLE_FEATURE_INDEX
    sin_index = unity_constants.SINE_ANGLE_FEATURE_INDEX
    x_index = unity_constants.POSITION_X_FEATURE_INDEX
    y_index = unity_constants.POSITION_Y_FEATURE_INDEX
    width_index = unity_constants.WIDTH_FEATURE_INDEX
    height_index = unity_constants.HEIGHT_FEATURE_INDEX

    # The block is below the obstacle.
    if block[y_index] < self._obstacle[y_index]:
      return

    x = block[x_index]
    angle = np.arctan2(block[sin_index], block[cos_index])
    width = block[width_index]
    height = block[height_index]
    projected_width = geometry.rect_projected_width(width, height, angle)
    self._add(x - projected_width / 2., x + projected_width / 2.)

  def compute_amount_covered(self):
    """Computes the amount of the obstacle that is covered."""
    width_index = unity_constants.WIDTH_FEATURE_INDEX
    x_index = unity_constants.POSITION_X_FEATURE_INDEX

    obstacle_width = self._obstacle[width_index]
    xmin = self._obstacle[x_index] - obstacle_width / 2.
    xmax = self._obstacle[x_index] + obstacle_width / 2.
    value = 0
    for segment_xmin, segment_xmax in self._segment_list:
      if segment_xmin <= xmax and segment_xmax >= xmin:
        xmax_ = min(xmax, segment_xmax)
        xmin_ = max(xmin, segment_xmin)
        value += xmax_ - xmin_
    return value


def _compute_covered_length(obstacles, blocks):
  """Compute the length of `o` in `obstacles` covered by any `b` in `blocks`."""
  total_covered_length = 0
  for obstacle in obstacles:
    tracker = _CoveringTracker(obstacle)
    for block in blocks:
      tracker.add_block(block)
    total_covered_length += tracker.compute_amount_covered()
  return total_covered_length


class ConstructionCovering(stacking.ConstructionStacking):
  """Construction task consisting of covering obstacles laying on the ground.

  In the Covering task, the agent must build a shelter that covers all obstacles
  from above, without touching them. The reward function is: +L, where L is the
  sum of the lengths of the top surfaces of the obstacles which are sheltered by
  blocks placed by the agent; and -2 for each block set as sticky. The task-
  specific termination criterion is achieved when at least 99% of the summed
  obstacle surfaces are covered. The layers of obstacles are well-separated
  vertically so that the agent can build structures between them.
  """

  def __init__(self,
               unity_environment,
               sticky_penalty=2.0,
               **stacking_kwargs):
    """Inits the environment.

    Args:
      unity_environment: See base class.
      sticky_penalty: See base class.
      **stacking_kwargs: keyword arguments passed to
        covering.ConstructionStacking.
    """
    default_stacking_kwargs = dict(
        block_replacement=True,
        max_difficulty=2,
        progress_threshold=0.99)
    default_stacking_kwargs.update(stacking_kwargs)
    super(ConstructionCovering, self).__init__(
        unity_environment=unity_environment,
        sticky_penalty=sticky_penalty,
        **default_stacking_kwargs)

  def _compute_max_episode_reward(self, obs):
    # Assuming you don't need glue.
    max_episode_reward = 0
    for obstacle in obs.obstacles:
      max_episode_reward += obstacle.width
    return max_episode_reward

  def _maybe_update_max_steps(self):
    self._max_steps = len(self._initial_available_objects) * 4

  def _get_task_reward(self, obstacles, targets, blocks):
    del targets
    covered_length = _compute_covered_length(obstacles, blocks)
    return covered_length

  def _get_generator(self, difficulty):
    if isinstance(difficulty, str):
      raise ValueError("Unrecognized difficulty: %s" % difficulty)

    # Up to `difficulty+1` layers of obstacles, interleaved with layers
    # with no obstacles.
    obstacles_ys_range = [tuple(np.arange(difficulty+1) * 2)]
    return CoveringGenerator(
        num_blocks_range=(1, 3),
        scene_width=self._generator_width,
        random_state=self._random_state,
        obstacles_ys_range=obstacles_ys_range,
        obstacles_width_range=(10, 40))


class ConstructionCoveringHard(ConstructionCovering):
  """Hard version of the covering task.

  In the Covering Hard task, the agent must build a shelter, but the task is
  modified to encourage longer term planning: there is a finite supply of
  movable blocks, the distribution of obstacles is denser, and the cost of
  stickiness is lower (-0.5 per sticky block). The reward function and
  termination criterion are the same as in Covering.
  """

  def __init__(self,
               unity_environment,
               sticky_penalty=0.5,
               **covering_kwargs):
    """Inits the environment.

    Args:
      unity_environment: See base class.
      sticky_penalty: See base class.
      **covering_kwargs: keyword arguments passed to
        covering.ConstructionCovering.
    """
    super(ConstructionCoveringHard, self).__init__(
        unity_environment=unity_environment,
        sticky_penalty=sticky_penalty,
        block_replacement=False,
        max_difficulty=1,
        **covering_kwargs)

  def _get_generator(self, difficulty):
    if isinstance(difficulty, str):
      raise ValueError("Unrecognized difficulty: %s" % difficulty)

    # Up to `difficulty+1` layers of obstacles.
    obstacles_ys_range = [tuple(range(difficulty+1))]

    return CoveringGenerator(
        num_blocks_range=(1, 3),
        scene_width=self._generator_width,
        random_state=self._random_state,
        obstacles_ys_range=obstacles_ys_range,
        obstacles_width_range=(10, 50))

  def _maybe_update_max_steps(self):
    self._max_steps = len(self._initial_available_objects) * 2


class CoveringGenerator(stacking.StackingGenerator):
  """Generates a set of obstacles for the covering task."""

  def __init__(self,
               num_blocks_range,
               scene_width,
               random_state,
               obstacles_width_range=(10, 50),
               obstacles_ys_range=None,
               obstacles_height=5,
               **kwargs):
    """Initialize the generator.

    Args:
      num_blocks_range: a tuple indicating the range of obstacles
        that will be in the generated towers, from low (inclusive) to high
        (exclusive).
      scene_width: the width of the scene.
      random_state: a np.random.RandomState object
      obstacles_width_range: the range of widths for obstacles, from low
        (inclusive) to high (exclusive).
      obstacles_ys_range: y-position to draw the obstacle from. A tuple of
        y-positions will be sampled from this range. This is scalled
        appropriately, so that -1 corresponds to an object below the floor, 0
        to an object on the floor.
      obstacles_height: The height of the obstacles.
      **kwargs: additional keyword arguments passed to super
    """
    super(CoveringGenerator, self).__init__(
        num_blocks_range=num_blocks_range,
        scene_width=scene_width,
        random_state=random_state,
        **kwargs)

    obstacles_ys_range = obstacles_ys_range or [(0,)]

    self.obstacle_height = obstacles_height
    scale_y_fn = lambda y: (y + 0.5) * self.height
    scale_ys_fn = lambda ys: tuple([scale_y_fn(y) for y in ys])
    self._obstacles_ys_range = [
        scale_ys_fn(obstacles_ys) for obstacles_ys in obstacles_ys_range]

    self._obstacles_width_range = obstacles_width_range

  def generate_one(self):
    """Generate a single scene.

    Returns:
      observation: a BlocksObservation object
      solution: a list of Block objects in their final locations
    """
    # pick the set of y-positions we want for our obstacles
    idx = np.arange(len(self._obstacles_ys_range))
    obstacles_ys = self._obstacles_ys_range[self.random_state.choice(idx)]

    # place the obstacles at each level, going from bottom to top
    obstacles = []
    for y in obstacles_ys:

      # get the number of obstacles at this layer
      num_obstacles = self.random_state.randint(*self.num_blocks_range)

      # pick a set of obstacle widths, and check that the sum of the widths is
      # not greater than the scene width plus some buffer room. keep regenerting
      # obstacles until this is the case.
      available_width = 0
      while available_width < self.small_width + 1:
        available_widths = np.arange(*self._obstacles_width_range)
        obstacles_lengths = self.random_state.choice(
            available_widths, size=[num_obstacles])
        available_width = self.scene_width - np.sum(obstacles_lengths)

      # compute the left and right edges of each obstacle, assuming the
      # obstacles are all placed right next to each other beginning from the
      # left side of the scene.
      obstacles_begins = np.concatenate(
          [np.array([0], dtype=np.int32), np.cumsum(obstacles_lengths)[:-1]])
      obstacles_ends = np.cumsum(obstacles_lengths)

      # available_width now is the amount of space left on the floor, not taken
      # up by obstacles. we split this into a few chunks of random size to space
      # the obstacles out along the floor
      relative_shifts = self.random_state.uniform(
          0., 1., size=[num_obstacles + 1])
      relative_shifts /= np.sum(relative_shifts)
      relative_shifts = np.floor(relative_shifts * available_width)
      shifts = np.cumsum(relative_shifts.astype(np.int32))[:-1]
      obstacles_begins += shifts
      obstacles_ends += shifts

      # now actually create the obstacles
      for obstacle_begin, obstacle_end in zip(obstacles_begins, obstacles_ends):
        center = (obstacle_begin + obstacle_end) // 2
        width = obstacle_end - obstacle_begin
        obstacle = block_utils.Block(
            x=center, y=y, width=width, height=self.obstacle_height)
        obstacles.append(obstacle)

    observation_blocks = self._place_available_objects()
    floor = self._place_floor()

    observation = block_utils.BlocksObservation(
        blocks=[floor] + observation_blocks,
        obstacles=obstacles,
        targets=[],
        balls=[])

    return observation
