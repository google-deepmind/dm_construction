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
"""A construction environment where the task is to fill in silhouettes.

See: Bapst, V., Sanchez-Gonzalez, A., Doersch, C., Stachenfeld, K., Kohli, P.,
Battaglia, P., & Hamrick, J. (2019, May). Structured agents for physical
construction. In International Conference on Machine Learning (pp. 464-474).
"""

from dm_construction.environments import stacking
from dm_construction.unity import constants as unity_constants
from dm_construction.utils import block as block_utils
import numpy as np


# Approximate vertical correction to leave verically between blocks, due
# to contact precision.
_VERTICAL_CORRECTION = 0.2


def _get_horizontal_blocks(
    blocks, index_getter=lambda x: x, threshold_sine=0.05):
  sine_index = index_getter(unity_constants.SINE_ANGLE_FEATURE_INDEX)
  return blocks[np.abs(blocks[:, sine_index]) < threshold_sine]


def _map_slice(slice_, mapper):
  """Applies `mapper` to a contiguous slice `slice_`."""
  assert slice_.step is None or slice_.step == 1
  new_slice = slice(mapper(slice_.start), 1 + mapper(slice_.stop -1))
  assert slice_.stop - slice_.start == new_slice.stop - new_slice.start
  return new_slice


def _target_fraction(
    target, horizontal_blocks, index_getter=lambda x: x, threshold_size=1e-3):
  """Returns the fraction of a target covered by a block of the same size."""

  if horizontal_blocks.shape[0] == 0:
    return 0.

  target_width = target[index_getter(unity_constants.WIDTH_FEATURE_INDEX)]
  target_height = target[index_getter(unity_constants.HEIGHT_FEATURE_INDEX)]

  position_slice = _map_slice(
      unity_constants.POSITION_FEATURE_SLICE, index_getter)

  distances = np.linalg.norm(
      horizontal_blocks[:, position_slice] - target[position_slice], axis=1)
  closest_index = np.argmin(distances)
  closest_block = horizontal_blocks[closest_index]
  closest_block_width = closest_block[
      index_getter(unity_constants.WIDTH_FEATURE_INDEX)]
  closest_block_height = closest_block[
      index_getter(unity_constants.HEIGHT_FEATURE_INDEX)]

  if (np.abs(closest_block_width-target_width) > threshold_size or
      np.abs(closest_block_height-target_height) > threshold_size):
    return 0.

  # Calculate the fraction of the target_area that is covered by the closest
  # block, assuming they are both horizontal, and of the same size.
  vector_distance = (closest_block[position_slice] - target[position_slice])

  covered_area = np.prod(
      np.maximum([target_width, target_height] - np.abs(vector_distance), 0.))
  target_area = target_width * target_height
  return covered_area / target_area


class ConstructionSilhouette(stacking.ConstructionStacking):
  """Task consisting of filling in silhouettes, avoiding obstacles.

  In the Silhouette task, the agent must place blocks to overlap with target
  blocks in the scene, while avoiding randomly positioned obstacles. The reward
  function is: +1 for each placed block which overlaps at least 90% with a
  target block of the same size; and -0.5 for each block set as sticky. The
  task-specific termination criterion is achieved when there is at least 90%
  overlap with all targets.

  Generalization levels:
    * `"double_the_targets"`: Scenes with 16 target blocks (8 is maximum number
      of target blocks in other levels).

  """

  def __init__(self,
               unity_environment,
               sticky_penalty=0.5,
               num_allowed_extra_steps=0,
               **stacking_kwargs):
    """Inits the environment.

    Args:
      unity_environment: See base class.
      sticky_penalty: See base class.
      num_allowed_extra_steps: Number of extra-steps to allow the agent to take
        in an episode, additionaly to a number of steps equal to the number of
        targets. Defaults to zero, meaning that (assuming cheap enough glue) an
        optimal agent should place blocks exactly on the targets location only.
      **stacking_kwargs: keyword arguments passed to
        covering.ConstructionStacking.

    Raises:
      ValueError: If curriculum_type is not in ["ys"].
    """

    self._num_allowed_extra_steps = num_allowed_extra_steps
    self._max_steps = None

    super(ConstructionSilhouette, self).__init__(
        unity_environment=unity_environment,
        sticky_penalty=sticky_penalty,
        block_replacement=True,
        max_difficulty=7,
        target_color=(0., 1., 0., 0.3),
        progress_threshold=0.98,
        **stacking_kwargs)

  def _compute_max_episode_reward(self, obs):
    return len(self._initial_scene.targets)

  def _maybe_update_max_steps(self):
    self._max_steps = (
        len(self._initial_scene.targets) + self._num_allowed_extra_steps)

  def _get_task_reward(self, obstacles, targets, blocks):
    """Computes the current score based on the targets and placed blocks."""
    del obstacles

    targets_score = 0.
    targets_fraction = 0.
    # Filter to keep only horizontal blocks.
    horizontal_blocks = _get_horizontal_blocks(blocks)

    targets_to_reward = targets
    for _, target in enumerate(targets_to_reward):
      target_fraction = _target_fraction(
          target, horizontal_blocks)
      targets_fraction += target_fraction
      targets_score += 1. if target_fraction > 0.9 else 0.

    return targets_score

  def _get_generator(self, difficulty):
    min_num_obstacles = 1

    if isinstance(difficulty, int):
      # One additional target per difficulty up to 8 (difficulty=7)
      max_num_targets = difficulty + 1

      # Max number of levels added one by one, capped at 6.
      max_num_levels = min(difficulty + 1, 6)

      # Max number of obstacles added one by one starting at difficulty=2,
      # and capped at 4.
      max_num_obstacles = max(0, min(difficulty - 1, 4))

      min_num_targets = max_num_targets
      min_num_obstacles = min(min_num_obstacles, max_num_obstacles)
    elif difficulty == "double_the_targets":
      # Twice as many targets
      max_num_levels = 6
      min_num_targets = 16
      max_num_targets = 16
      max_num_obstacles = 4
    else:
      raise ValueError("Unrecognized difficulty: %s" % difficulty)

    num_levels_range = (max_num_levels, max_num_levels+1)
    num_targets_range = (min_num_targets, max_num_targets+1)
    num_obstacles_range = (min_num_obstacles, max_num_obstacles+1)

    return SilhouetteGenerator(
        num_obstacles_range=num_obstacles_range,
        num_targets_range=num_targets_range,
        scene_width=self._generator_width,
        random_state=self._random_state,
        num_levels_range=num_levels_range)


def _almost_equal(a, b, threshold=1e-3):
  return np.abs(a - b) < threshold


def _bounds_to_center_dims_1d(bds):
  """Get midpoints of bounds.

  Args:
    bds: list of numbers denoting intervals (bds[i] = start of interval i,
      end of interval i-1)

  Returns:
    midpoints: midpoints of intervals. Note len(midpoints) = len(bds)-1
      because bds contains both start + end bounds, meaning there will be one
      more bds entry than there are intervals.
  """
  return .5 * (bds[:-1] + bds[1:]), np.diff(bds)


def _center_dims_to_bounding_boxes(cxy_dxy):
  """Convert from center-dimensions coordinates to bounding box coordinates.

  Args:
    cxy_dxy: n x 4 matrix describing rectangles in terms of center coordinates
      and dimensions in x, y directions
      (cxy_dxy[i] = [center_x_i, center_y_i, dimension_x_i, dimension_y_i])

  Returns:
    lxy_uxy: n x 4 matrix describing rectangles  in terms of lower/upper bounds
      in x, y directions
      (lxy_uxy[i] = [lower_bd_x_i, lower_bd_y_i, upper_bd_x_i, upper_bd_y_i])
  """
  if not list(cxy_dxy): return []
  dim = len(cxy_dxy.shape)
  if dim == 1: cxy_dxy = cxy_dxy.reshape(1, -1)

  c_xy = cxy_dxy[:, :2]
  d_xy = cxy_dxy[:, 2:]

  if dim == 1:
    return np.concatenate([c_xy - d_xy * .5, c_xy + d_xy * .5],
                          axis=1).reshape(-1)
  else:
    return np.concatenate([c_xy - d_xy * .5, c_xy + d_xy * .5], axis=1)


def _get_bounds_1d_discrete(stack_max_length, discrete_values,
                            random_state, offset=0.):
  """Get bounds of blocks in 1 dimension.

  Args:
    stack_max_length: max length of blocks
    discrete_values: discrete values used to partition stack_max_length.
    random_state: np.random.RandomState(seed)
    offset: how much to offset first point by (default=0)

  Returns:
    pts: array of starts, ends of blocks
      [s_block_0, s_block_1/e_block_1, ..., s_block_n/e_block_n-1, e_block_n]

  Raises:
    ValueError: if random_state is None or block_range is incorrectly specified
  """
  # check inputs
  if random_state is None:
    raise ValueError("random_state must be supplied and not None.")

  n_max = int(np.ceil(stack_max_length * 1. / np.min(discrete_values)))
  dim = random_state.choice(discrete_values, size=n_max)
  y_bds = np.insert(np.cumsum(dim), 0, 0) + offset
  # sanity checks
  assert y_bds[-1] >= stack_max_length
  assert np.all(np.diff(y_bds) > 0.)

  y_bds_lt_max_len = y_bds[y_bds < stack_max_length]

  return y_bds_lt_max_len


def _tessellate_discrete_rectangles_by_row_from_options(
    scene_width, scene_height=None,
    random_state=None,
    discrete_widths=(5, 10, 20, 40),
    discrete_heights=(5, 10),
    do_x_offset=True, return_as="cxy_dxy"):
  """Method 1 for tessellating plane with rectangles.

  Tessellate with rule:
  1) sample row heights for current row.
  4) for each row, sample block widths from the discrete widhts until scene
    is filled horizontally.
  5) for each row: select random width for each block uniformly from
    discrete_widths; offset blocks from left wall by random amount if
    do_x_offset
  6) return as cxy_dxy (center/dimensions) or lxy_uxy (lower/upper bounds)

  Args:
    scene_width: width of scene
    scene_height: height of scene (default=scene_width)
    random_state: np.random.RandomState(seed)
    discrete_widths: Iterable for values of the block widths.
    discrete_heights: Iterable for values of the block heights.
    do_x_offset: jitter x offset of blocks in each row (default=True)
    return_as: "cxy_dxy" for center (x,y), dimensions (x,y) format
      (cxy_dxy[i] = [cx, cy, dx, dy])
      "lxy_uxy" for lower (x,y) bounds, upper (x,y) bounds format
      (lxy_uxy[i] = [lx, ly, ux, uy])
      (default="cxy_dxy")

  Returns:
    discrete_widths: as a np.array
    discrete_heights: as a np.array
    rectangle_coords: in format "lxy_uxy" or "cxy_dxy" as specified

  Raises:
    ValueError: if random_state is not supplied or is None; if
      discrete_heights or discrete_widths is incorrectly specified
  """

  y_bds = _get_bounds_1d_discrete(scene_height, discrete_heights, random_state)
  y_c, y_dim = _bounds_to_center_dims_1d(y_bds)
  ny = len(y_c)

  coords = []
  for iy in range(ny):
    # get widths of blocks in row
    if do_x_offset:
      x_offset = random_state.rand() * np.max(discrete_widths)
    else: x_offset = 0.
    x_bds = _get_bounds_1d_discrete(
        scene_width, discrete_widths, random_state, offset=x_offset)
    x_c, x_dim = _bounds_to_center_dims_1d(x_bds)

    # add x + y features of rectangles to box
    coords_i = np.concatenate([ii.reshape(-1, 1) for ii in
                               [x_c, np.repeat(y_c[iy], len(x_c)),
                                x_dim, np.repeat(y_dim[iy], len(x_c))]],
                              axis=1)
    coords.append(coords_i)
  if return_as == "cxy_dxy":
    return np.concatenate(coords, axis=0)
  elif return_as == "lxy_uxy":
    return _center_dims_to_bounding_boxes(np.concatenate(coords, axis=0))
  else:
    raise ValueError("return_as type '{}' not recognized.".format(return_as) +
                     " Should be 'cxy_dxy' or 'lxy_uxy'")


class SilhouetteGenerator(stacking.StackingGenerator):
  """Generates a set of horizontal obstacles and targets."""

  def __init__(self,
               num_obstacles_range,
               num_targets_range,
               num_levels_range,
               scene_width,
               random_state,
               height_distribution_exponent=1.75,
               obstacles_height=5,
               **kwargs):
    """Initialize the generator.

    Args:
      num_obstacles_range: a tuple indicating the range of obstacles
        that will be in the generated scene, from low (inclusive) to high
        (exclusive). This counts the number of obstacles per height.
      num_targets_range: a tuple indicating the range of targets
        that will be in the generated scene, from low (inclusive) to high
        (exclusive). This counts the total number of targets.
      num_levels_range: a tuple indicating the range of levels that will be in
        the generated scene, from low (inclusive) to high (exclusive).
      scene_width: the width of the scene.
      random_state: a np.random.RandomState object
      height_distribution_exponent: probability of chosing objects at different
        levels will be proportional to level**height_distribution_parameter for
        each object.
      obstacles_height: The height of the obstacles. Only used when
        `use_legacy_obstacles_heights` is set to False.
      **kwargs: additional keyword arguments passed to super
    """
    super(SilhouetteGenerator, self).__init__(
        num_blocks_range=None,
        scene_width=scene_width,
        random_state=random_state,
        **kwargs)

    self._num_obstacles_range = num_obstacles_range
    self._num_targets_range = num_targets_range
    self._height_distribution_exponent = height_distribution_exponent

    self.obstacles_height = obstacles_height
    self._num_levels_range = num_levels_range
    self._corrected_height = self.height + _VERTICAL_CORRECTION

  def _get_supported_blocks(
      self, tessellation_blocks, existing_blocks, min_overlap):
    """Returns tessellation blocks on top of previous blocks or the floor.

    Args:
      tessellation_blocks: Array of blocks left in the tessellation
        with shape [num_blocks, 4], where the last axis indicates: center_x,
        center_y, width, height.
      existing_blocks: List of existing blocks, formatted as the rows in
        tessellation_blocks
      min_overlap: minimum overlap between an existing block and a tessellation
        block, for the second one to the considered supported.

    Returns:
      List of tuples (index, block), corresponding to rows of
      `tessellation_blocks` satifying the support condition.

    """

    # Blocks on the floor.
    possible_blocks = [
        (i, candidate_block)
        for i, candidate_block in enumerate(tessellation_blocks)
        if _almost_equal(candidate_block[1], self._corrected_height/2)]

    # Blocks overlapping with other blocks underneath.
    for existing_block in existing_blocks:
      for i, candidate_block in enumerate(tessellation_blocks):
        # If it is one level above an existing block.
        if _almost_equal(candidate_block[1],
                         existing_block[1] + self._corrected_height):
          distance_between_centers = np.abs(
              candidate_block[0] - existing_block[0])

          combined_half_length = (candidate_block[2]/2 + existing_block[2]/2 -
                                  self.margin)

          # If it overlaps enough with the block at the level below.
          if distance_between_centers < combined_half_length - min_overlap:
            possible_blocks.append((i, candidate_block))

    return possible_blocks

  def _sample_candidates(self, candidates, num_samples=1):
    candidates_heights = np.array([candidate[1][1] for candidate in candidates])
    # Levels as 1, 2, 3
    candidates_levels_num = (candidates_heights+self.height/2)/self.height
    if not candidates:
      return []
    # We want the probability to increase with height, to build more
    # tower like structures, and less flat structures.
    probs = candidates_levels_num**self._height_distribution_exponent
    probs = probs/probs.sum()
    chosen_indices = self.random_state.choice(
        np.arange(len(candidates)), p=probs, size=num_samples, replace=False)
    return [candidates[i] for i in chosen_indices]

  def generate_one(self):
    """Generate a single scene.

    Returns:
      observation: a block_utils.BlocksObservation object
      solution: a list of Block objects in their final locations
    """

    # Tessellate space, fractions of the blocks lengths, with a small margin.
    # And with the standard block height.

    discrete_heights = (self._corrected_height,)
    discrete_widths = (self.small_width+self.margin,
                       self.medium_width+self.margin,
                       self.large_width+self.margin)

    # Sample a maximum height for the scene.
    num_levels = self.random_state.randint(*self._num_levels_range)
    this_scene_height = self.height * (num_levels+0.5)

    # Set the targets.
    num_targets = self.random_state.randint(*self._num_targets_range)
    max_attempts = 10
    min_overlap = 0.9 * self.small_width
    for _ in range(max_attempts):
      # Tessellate space. This returns a list of blocks in the tesselation
      # with shape [num_blocks, 4], where the last axis, indicats, center_x,
      # center_y, width, height.
      tessellation_blocks = (
          _tessellate_discrete_rectangles_by_row_from_options(
              scene_width=self.scene_width, scene_height=this_scene_height,
              discrete_widths=discrete_widths,
              discrete_heights=discrete_heights,
              random_state=self.random_state,
              do_x_offset=True, return_as="cxy_dxy"))

      # Pick num_targets blocks from possible options.
      existing_blocks = []
      for _ in range(num_targets):
        candidates = self._get_supported_blocks(
            tessellation_blocks, existing_blocks, min_overlap=min_overlap)
        if not candidates:
          break
        block_i, block = self._sample_candidates(candidates, num_samples=1)[0]
        tessellation_blocks = np.delete(tessellation_blocks, block_i, axis=0)
        existing_blocks.append(block)
      else:
        # If we successfully added as many targets as we needed, we do not need
        # to keep attempting by breaking the loop.
        break
    else:
      # If we got here, is because we did not break out of the loop, and
      # we have exhausted all attempts.
      raise ValueError(
          "Maximum number of attempts reached to generate silhouette.")

    targets = [block_utils.Block(
        x=b[0], y=b[1]+_VERTICAL_CORRECTION/2, width=b[2]-self.margin,
        height=b[3]-_VERTICAL_CORRECTION) for b in existing_blocks]

    # Set the obstacles.
    num_obstacles = self.random_state.randint(*self._num_obstacles_range)
    # We only require negative overlap for obstacles.
    candidates = self._get_supported_blocks(
        tessellation_blocks, existing_blocks, min_overlap=-2.)
    sampled_candidates = self._sample_candidates(
        candidates, num_samples=min(num_obstacles, len(candidates)))
    obstacles = [block_utils.Block(
        x=block[0], y=block[1], width=block[2]-self.margin,
        height=self.obstacles_height) for _, block in sampled_candidates]
    tessellation_blocks = np.delete(
        tessellation_blocks,
        [block_i for block_i, _ in sampled_candidates], axis=0)
    observation_blocks = self._place_available_objects()
    floor = self._place_floor()

    observation = block_utils.BlocksObservation(
        blocks=[floor] + observation_blocks,
        obstacles=obstacles,
        targets=targets,
        balls=[])

    return observation
