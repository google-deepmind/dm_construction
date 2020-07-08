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
"""A construction environment where the task is to get a ball from A to B.

See: Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Pfaff, T., Weber, T.,
Buesing, L., & Battaglia, P. W. (2020). Combining Q-Learning and Search with
Amortized Value Estimates. ICLR 2020.
"""

import collections
import itertools

from dm_construction.environments import stacking
from dm_construction.unity import constants as unity_constants
from dm_construction.utils import block as block_utils
from dm_construction.utils import constants
from dm_construction.utils import geometry
import dm_env
from dm_env import specs
import numpy as np


_OBJECT_TYPE_NAMES = [
    constants.BLOCK,
    constants.OBSTACLE,
    constants.TARGET,
    constants.AVAILABLE_BLOCK,
    constants.BALL
]


def _rect_or_ramp_blocks_collision(block0, block1, epsilon=1e-4):
  """Returns true if there is a collision between ramps/boxes."""

  x0 = block0[unity_constants.POSITION_X_FEATURE_INDEX]
  y0 = block0[unity_constants.POSITION_Y_FEATURE_INDEX]
  w0 = block0[unity_constants.WIDTH_FEATURE_INDEX]
  h0 = block0[unity_constants.HEIGHT_FEATURE_INDEX]
  angle0 = _get_angle(block0)

  x1 = block1[unity_constants.POSITION_X_FEATURE_INDEX]
  y1 = block1[unity_constants.POSITION_Y_FEATURE_INDEX]
  w1 = block1[unity_constants.WIDTH_FEATURE_INDEX]
  h1 = block1[unity_constants.HEIGHT_FEATURE_INDEX]
  angle1 = _get_angle(block1)

  if (block0[unity_constants.IS_BOX_FEATURE_INDEX] and
      block1[unity_constants.IS_BOX_FEATURE_INDEX]):
    return geometry.rect_overlap(x0, y0, w0, h0, angle0,
                                 x1, y1, w1, h1, angle1) > epsilon
  elif (block0[unity_constants.IS_RAMP_FEATURE_INDEX] and
        block1[unity_constants.IS_RAMP_FEATURE_INDEX]):
    return geometry.ramp_overlap(x0, y0, w0, h0, angle0,
                                 x1, y1, w1, h1, angle1) > epsilon
  elif (block0[unity_constants.IS_BOX_FEATURE_INDEX] and
        block1[unity_constants.IS_RAMP_FEATURE_INDEX]):
    return geometry.rect_ramp_overlap(x0, y0, w0, h0, angle0,
                                      x1, y1, w1, h1, angle1) > epsilon
  elif (block0[unity_constants.IS_RAMP_FEATURE_INDEX] and
        block1[unity_constants.IS_BOX_FEATURE_INDEX]):
    return geometry.rect_ramp_overlap(x1, y1, w1, h1, angle0,
                                      x0, y0, w0, h0, angle1) > epsilon
  else:
    raise ValueError()


def _rect_or_ramp_ball_blocks_collision(block_box, block_ball, epsilon=1e-4):
  """Returns true if there is a collision between a box/ramp and a ball."""
  x0 = block_box[unity_constants.POSITION_X_FEATURE_INDEX]
  y0 = block_box[unity_constants.POSITION_Y_FEATURE_INDEX]
  w0 = block_box[unity_constants.WIDTH_FEATURE_INDEX]
  h0 = block_box[unity_constants.HEIGHT_FEATURE_INDEX]
  angle0 = _get_angle(block_box)

  x1 = block_ball[unity_constants.POSITION_X_FEATURE_INDEX]
  y1 = block_ball[unity_constants.POSITION_Y_FEATURE_INDEX]
  w1 = block_ball[unity_constants.WIDTH_FEATURE_INDEX]

  if block_box[unity_constants.IS_BOX_FEATURE_INDEX]:
    return geometry.rect_ball_overlap(
        x0, y0, w0, h0, angle0, x1, y1, w1) > epsilon
  elif block_box[unity_constants.IS_RAMP_FEATURE_INDEX]:
    return geometry.ramp_ball_overlap(
        x0, y0, w0, h0, angle0, x1, y1, w1) > epsilon
  else:
    raise ValueError()


def _blocks_collision(block1, block2):
  if (not block1[unity_constants.IS_BALL_FEATURE_INDEX] and
      not block2[unity_constants.IS_BALL_FEATURE_INDEX]):
    return _rect_or_ramp_blocks_collision(block1, block2)
  elif block2[unity_constants.IS_BALL_FEATURE_INDEX]:
    return _rect_or_ramp_ball_blocks_collision(block1, block2)
  elif block1[unity_constants.IS_BALL_FEATURE_INDEX]:
    return _rect_or_ramp_ball_blocks_collision(block2, block1)
  else:
    raise ValueError(
        "Only collisions between boxes, and box and ball are supported")


def _get_angle(block):
  return np.arctan2(block[unity_constants.SINE_ANGLE_FEATURE_INDEX],
                    block[unity_constants.COSINE_ANGLE_FEATURE_INDEX])


MarbleRunEpisodeParams = collections.namedtuple(
    "MarbleRunEpisodeParams",
    ["num_obstacles",  # Number of obstacles in the episode.
     "num_targets",  # Number of targets in the episode.
     "num_balls",  # Number of balls in the episode.
     "min_height_obstacle",  # Vertical position of the lowest obstacle.
     "max_height_obstacle",  # Vertical position of the highest obstacle.
     "max_height_target",  # Vertical position of the highest target.
     # Maximum horizontal distance between a ball and its closest target in the
     # episode. This distance is discretized as follows:
     # 0: less than 1/3 the width of the scene.
     # 1: between 1/3 and 2/3 the width of the scene.
     # 2: more than 2/3 the width of the scene.
     "discretized_ball_target_distance",
    ]
    )


class ConstructionMarbleRun(stacking.ConstructionStacking):
  """Environment for Marble Run task.

  The goal in Marble Run is to stack blocks to enable a marble to get from its
  original starting position to a goal location, while avoiding obstacles. At
  each step, the agent may choose from a number of differently shaped
  rectangular blocks as well as ramp shapes, and may choose to make these blocks
  "sticky" (for a price) so that they stick to other objects in the scene. The
  episode ends once the agent has created a structure that would get the marble
  to the goal. The agent receives a reward of one if it solves the scene, and
  zero otherwise.
  """

  def __init__(self,
               unity_environment,
               length_dynamics=1,
               reward_per_target=1.,
               max_num_bounces=8,
               curriculum_sample_geom_p=0.5,
               sticky_penalty=0.0,
               **stacking_kwargs):
    """Inits the environment.

    Args:
      unity_environment: See base class.
      length_dynamics: Length of the augmented observation of the ball. The
        observation will be a sequence of the state of the system between
        bounces of the ball.
      reward_per_target: Reward associated with colliding with each target.
      max_num_bounces: Maximum number of objects the ball can bounce off/
        interact with to evaluate the reward. Must be >= length_dynamics - 1 .
      curriculum_sample_geom_p: See base class.
      sticky_penalty: See base class.
      **stacking_kwargs: keyword arguments passed to
        stacking.ConstructionStacking.
    """
    # Bounciness of the balls.
    self._bounciness = 0.
    # Linear drag of the balls.
    self._linear_drag = 0.
    # Angular drag of the balls.
    self._angular_drag = 0.
    # Density of the ball.
    self._ball_density = 1.
    # If True the ball also gets glued to sticky objects.
    self._glueable_ball = False
    # Maximum number of timesteps a ball can can spend in a long bounce, before
    # the bounce is split into two.
    self._max_simulation_steps_per_bounce = 1000
    # If True, the dynamics are only evaluated until one of the termination
    # conditions is reached, otherwise, the dynamics are always evaluated up to
    # max_num_bounces. If True, the observation sequence is padded with the last
    # observation up to length_dynamics, whenever the simulation is too short.
    self._end_dynamics_on_termination = True

    self._length_dynamics = length_dynamics
    self._reward_per_target = reward_per_target
    self._max_num_bounces = max_num_bounces

    if self._max_num_bounces < self._length_dynamics - 1:
      raise ValueError(
          "`max_num_bounces` cannot be smaller than `length_dynamics - 1`.")

    super(ConstructionMarbleRun, self).__init__(
        unity_environment=unity_environment,
        sticky_penalty=sticky_penalty,
        block_replacement=True,
        max_difficulty=8,
        generator_width=250,
        curriculum_sample_geom_p=curriculum_sample_geom_p,
        progress_threshold=0.99,
        **stacking_kwargs)

  def _split_available_obstacles_placed_balls(self, blocks):
    """Splits observations for available blocks, obstacles and placed blocks."""

    (available, targets, obstacles, placed
    ) = self._split_available_obstacles_placed(blocks)

    # We know that the balls are always last, since they are instantitated in
    # the scene last.
    num_balls = len(self._initial_scene.balls)
    balls = placed[-num_balls:]
    placed = placed[:-num_balls]
    return available, targets, obstacles, placed, balls

  def _check_ball_collision(self, blocks):
    """Verifies if any of the balls is colliding with the placed blocks."""
    # We check spawn collisions manually because currently the Unity Env only
    # detects spawn collisions after a simulation has been run. However
    # it there is a spawn collisition with the bumper, we don't even want to run
    # a simulation.
    (unused_available, unused_targets, unused_obstacles, placed, balls
    ) = self._split_available_obstacles_placed_balls(blocks)
    for ball in balls:
      for other_block in placed:
        if _blocks_collision(ball, other_block):
          return True
    return False

  def _add_balls_to_scene(self):
    reset_ball_actions = []
    for ball_i, ball in enumerate(self._initial_scene.balls):
      reset_ball_actions.append({
          # Remove the display ball.
          "Delete": 1.,
          "SelectId": self._ball_ids[ball_i],
          # Add the actual ball.
          "SpawnBlock": 1.,
          "Shape": ball.shape,
          "FreeBody": 1,
          "SetId": self._ball_ids[ball_i],
          "SetPosX": ball.x,
          "SetPosY": ball.y,
          "Width": ball.width,
          "Height": ball.height,
          "Density": self._ball_density,
          "SetAngle": ball.angle,
          "RGBA": self._ball_color,
          "Bounciness": self._bounciness,
          "LinearDrag": self._linear_drag,
          "AngularDrag": self._angular_drag,
          "Glueable": 1. if self._glueable_ball else 0.,
          "CollisionMask": 0b11,  # Balls can collide with targets too.
          "Friction": 0
      })
    return self._unity_environment.step(reset_ball_actions)

  def _check_collisions_and_run_ball_dynamics(self, initial_observation):
    """Runs the jumpy simulation between collisions from the current state.

    Simulation terminates when `max_num_bounces` is reached, or may terminate
    early if the ball was spawned on top of another object, reaches an obstacle,
    or reaches the target.

    Args:
      initial_observation: previous observation from the core environment. Used
          to initialize the observation list returned by the function.

    Returns:
      core_observation_list: List of the observations observed during the jumpy.
          In case of early termination the last observation is repeated to
          indicate stationary dynamics after termination.
      simulation_on_list: Boolean bask of the same length as
          `core_observation_list`. A True in the i-th position indicates that
          physics were enabled when transitioning from core_observation_list[i]
          and core_observation_list[i+1]. Otherwise, the dynamics between
          core_observation_list[i] and core_observation_list[i+1] are purely
          stationary due to early simulation termination.
      spawn_collision: Indicates an object was placed on top of the ball, or
          and existing object.
      hit_obstacle: Indicates an obstacle was hit even before the ball
          simulation.
      hit_obstacle_dynamics: Indicates an obstacle was hit during the ball
          simulation.
      hit_goals: Indicates all goals were hit.
    """

    # We will construct a trajectory of the dynamics for up to
    # self._max_num_bounces and trim it later if required.
    core_observation_list = [initial_observation]
    spawn_collision = (
        (initial_observation["SpawnCollisionCount"] >
         self._initial_spawn_collision_count) or
        self._check_ball_collision(initial_observation["Blocks"]))

    # Check if any of the obstacles have been hit already
    blocks = initial_observation["Blocks"]
    (unused_available, targets, obstacles, unused_placed, balls
    ) = self._split_available_obstacles_placed_balls(blocks)
    initial_obstacle_hits = obstacles[
        :, unity_constants.COLLISION_COUNT_FEATURE_INDEX]
    hit_obstacle = True if np.any(initial_obstacle_hits) else False

    # We will flip these flags to true when the conditions are satified.
    hit_goals = False
    hit_obstacle_dynamics = False

    if spawn_collision or hit_obstacle:
      # If there is a spawn collision, we do not even simulate.
      simulation_on_list = [False]
      return (core_observation_list, simulation_on_list,
              spawn_collision, hit_obstacle, hit_obstacle_dynamics, hit_goals)

    simulation_on_list = [True]

    previous_ball_hits = balls[:, unity_constants.COLLISION_COUNT_FEATURE_INDEX]

    simulation_timestep = 0.01
    max_time_one_bounce = (
        self._max_simulation_steps_per_bounce * simulation_timestep)

    current_time = initial_observation["ElapsedTime"]
    for _ in range(self._max_num_bounces):

      # Run steps until we either get a collision of the ball reach the
      # timeout, or get an obstacle collision, or no balls are moving.
      finishing_time = current_time + max_time_one_bounce
      num_remaining_steps = np.round((finishing_time - current_time) /
                                     simulation_timestep)

      # Even though we run always with "StopOnCollision", it may be that this
      # collision is between two blocks and does not involve the balls.
      # So we keep simulating until we get a ball collision or we timeout.
      while num_remaining_steps:
        core_time_step = self._unity_environment.step(
            {"StopOnCollision": 1.0,
             "SimulationSteps": num_remaining_steps,
             "Timestep": simulation_timestep})
        blocks = core_time_step.observation["Blocks"]
        (unused_available, targets, obstacles, unused_placed, balls
        ) = self._split_available_obstacles_placed_balls(blocks)

        current_time = core_time_step.observation["ElapsedTime"]

        ball_hits = balls[:, unity_constants.COLLISION_COUNT_FEATURE_INDEX]
        obstacle_hits = obstacles[
            :, unity_constants.COLLISION_COUNT_FEATURE_INDEX]

        # No need to continue if an obstacle was hit, or if the collision
        # was with one of the balls.
        if obstacle_hits.any():
          hit_obstacle_dynamics = True
          break
        if (ball_hits - previous_ball_hits).any():
          break

        num_remaining_steps = np.round((finishing_time - current_time) /
                                       simulation_timestep)

      goal_hits = targets[:, unity_constants.COLLISION_COUNT_FEATURE_INDEX]
      if goal_hits.all():
        hit_goals = True

      core_observation_list.append(core_time_step.observation)
      if (self._end_dynamics_on_termination and
          (hit_obstacle_dynamics or hit_goals)):
        # Terminate the dynamics when ball collides with obstacle or targets.
        # (Later will be padded with zeros, repeating the last observation).
        # Models will be able to learn the transition from simulation_on=True
        # to simulation_on=False when one of these conditions happen.
        simulation_on_list.append(False)

        # By repeating the last observation, they will also be able to learn
        # that if simulation_on=False in the input, the output should just be
        # the same as the input and simulation_on will also be False after that.
        # Allowing them to chain constant predictions after termination
        # conditions.
        core_observation_list.append(core_observation_list[-1])
        simulation_on_list.append(False)
        break
      else:
        simulation_on_list.append(True)

      previous_ball_hits = ball_hits

      epsilon_velocity = 1e-4
      ball_velocities = balls[
          :, (unity_constants.VELOCITY_X_FEATURE_INDEX,
              unity_constants.VELOCITY_Y_FEATURE_INDEX,
              unity_constants.ANGULAR_VELOCITY_FEATURE_INDEX)]
      if np.all(np.abs(ball_velocities.flatten()) < epsilon_velocity):
        break

    return (core_observation_list, simulation_on_list,
            spawn_collision, hit_obstacle, hit_obstacle_dynamics, hit_goals)

  def _set_observation_and_termination(
      self, time_step, default_step_type=dm_env.StepType.MID):

    # Save the current state to restore it later, this corresponds to the scene
    # with the last placed block already relaxed and at its final location,
    # which will be the starting point for the next step (undoing all damage
    # that the ball simulation may cause to the existing structures).
    time_step_before_balls = time_step

    # Adding the balls and simulating the dynamics.
    time_step = self._add_balls_to_scene()

    (core_observation_list, simulation_on_list, spawn_collision,
     hit_obstacle, unused_hit_obstacle_dynamics, hit_goals
    ) = self._check_collisions_and_run_ball_dynamics(time_step.observation)

    new_observation = self._build_observation(
        core_observation_list, simulation_on_list)

    time_step = time_step._replace(observation=core_observation_list[-1])

    # We split the different types of blocks.
    blocks = time_step.observation["Blocks"]
    (available, targets, obstacles, placed, unused_balls
    ) = self._split_available_obstacles_placed_balls(blocks)

    # Evaluate termination conditions.
    # If we have placed as many objects as there are in display, or have reached
    # the maximum number of steps
    if not available.shape[0] or self._num_steps >= self._max_steps:
      self._end_episode(constants.TERMINATION_MAX_STEPS)

    # If there was a Spawn collision. A Spawn collision means the agent placed
    # an object overlapping with another object. We also override the reward.
    penalty_reward = 0.
    block_reward = 0.
    if spawn_collision > 0:
      self._end_episode(constants.TERMINATION_SPAWN_COLLISION)
      penalty_reward = -self._spawn_collision_penalty
    # If we hit an obstacle, we also end the episode and override the reward.
    elif hit_obstacle:
      self._end_episode(constants.TERMINATION_OBSTACLE_HIT)
      penalty_reward = -self._hit_obstacle_penalty
    else:
      # We remove the floor before evaluating the score.
      placed_blocks = placed[1:]
      self._num_sticky_blocks = np.sum(
          placed_blocks[:, unity_constants.STICKY_FEATURE_INDEX])
      self._progress = self._get_task_reward(
          obstacles, targets, placed_blocks)
      total_cost = self._get_cost(blocks)
      total_score = self._progress
      cost = total_cost - self._previous_cost
      self._previous_cost = total_cost
      block_reward = total_score - self._previous_score
      self._previous_score = total_score
      block_reward -= cost

      if hit_goals:
        self._end_episode(constants.TERMINATION_COMPLETE)

    if self._is_end_of_episode:
      step_type = dm_env.StepType.LAST
      discount = time_step.discount * 0.
    else:
      step_type = default_step_type
      discount = time_step.discount

    reward = penalty_reward + block_reward

    # Restore the state to what if was before the ball, to be able to repeat the
    # simulation, without any potential changes that the ball made to the
    # blocks.
    if not self._is_end_of_episode:
      self._unity_environment.restore_state(
          time_step_before_balls.observation,
          verify_restored_state=False)

    self._last_time_step = time_step._replace(
        observation=new_observation,
        step_type=step_type,
        discount=discount,
        reward=reward)

    return self._last_time_step

  def _build_observation(self, core_observation_list, simulation_on_list):

    new_observation = core_observation_list[-1].copy()

    # Trim or repeat to match the desired dynamics length.
    num_missing_steps = self._length_dynamics - len(core_observation_list)
    if num_missing_steps > 0:
      time_mask = [True]*len(core_observation_list) + [False]*num_missing_steps
      core_observation_list = core_observation_list[:]
      core_observation_list += [core_observation_list[-1]] * num_missing_steps
      simulation_on_list = simulation_on_list[:]
      simulation_on_list += [False] * num_missing_steps
    else:
      core_observation_list = core_observation_list[:self._length_dynamics]
      simulation_on_list = simulation_on_list[:self._length_dynamics]
      time_mask = [True] * self._length_dynamics

    # Get observations for each step and stack them.
    step_observation_dict_list = [
        self._build_observation_each_simulation_step(obs)
        for obs in core_observation_list]
    for key in step_observation_dict_list[0].keys():
      if key in ["RGB", "ObserverRGB"]:
        # For observations without an entity axis, the time axis will be
        # the first axis.
        axis = 0
      else:
        # The rest of the observations from each simulation timestep have
        # a leading entity axis, so the time axis should be the second one.
        axis = 1
      new_observation[key] = np.stack(
          [obs[key] for obs in step_observation_dict_list], axis=axis)

    new_observation["SimulationOn"] = np.array(simulation_on_list, np.bool)
    new_observation["TimeMask"] = np.array(time_mask, np.bool)

    del new_observation["Contacts"]
    del new_observation["SpawnCollisionCount"]
    del new_observation["CollisionStop"]
    del new_observation["ElapsedTime"]
    if "Segmentation" in new_observation:
      del new_observation["Segmentation"]

    return new_observation

  def _build_observation_each_simulation_step(self, core_observation):
    (available, targets, obstacles, placed, balls
    ) = self._split_available_obstacles_placed_balls(core_observation["Blocks"])
    new_observation = {}
    new_observation[constants.AVAILABLE_BLOCK] = available
    new_observation[constants.BLOCK] = placed
    new_observation[constants.OBSTACLE] = obstacles
    new_observation[constants.TARGET] = targets
    new_observation[constants.BALL] = balls

    for key in ["RGB", "ObserverRGB"]:
      if key in core_observation:
        new_observation[key] = core_observation[key]

    if "Segmentation" in core_observation:
      self._add_segmentation_masks(new_observation,
                                   core_observation["Segmentation"])
    return new_observation

  def _add_segmentation_masks(self, observation, segmentation):
    for name in _OBJECT_TYPE_NAMES:
      obs_name = "SegmentationMasks" + name
      ids = list(np.round(observation[name][:, 0]))
      observation[obs_name] = (
          stacking.build_segmentation_masks_for_ids(
              segmentation, ids))

  def observation_spec(self, *args, **kwargs):
    new_spec = self._unity_environment.observation_spec().copy()
    # The block observation is exactly as we get it
    block_obs_shape = [0, self._length_dynamics, new_spec["Blocks"].shape[1]]
    block_obs_dtype = new_spec["Blocks"].dtype
    # We know the observation is the same for all block types.
    for name in _OBJECT_TYPE_NAMES:
      new_spec[name] = specs.Array(
          block_obs_shape, dtype=block_obs_dtype, name=name)

    for key in ["RGB", "ObserverRGB"]:
      if key in new_spec:
        prev_spec = new_spec[key]
        new_spec[key] = specs.Array(
            (self._length_dynamics,) + prev_spec.shape,
            dtype=prev_spec.dtype, name=prev_spec.name)

    if "Segmentation" in list(new_spec.keys()):
      segmentation_resolution = new_spec["Segmentation"].shape[:2]
      segmentation_obs_shape = (
          0, self._length_dynamics) + segmentation_resolution
      for name in _OBJECT_TYPE_NAMES:
        obs_name = "SegmentationMasks" + name
        new_spec[obs_name] = specs.Array(
            segmentation_obs_shape, dtype=np.bool, name=obs_name)
      del new_spec["Segmentation"]

    new_spec.update({
        "SimulationOn": specs.Array(
            [self._length_dynamics], dtype=np.bool, name="SimulationOn"),
        "TimeMask": specs.Array(
            [self._length_dynamics], dtype=np.bool, name="TimeMask"),
    })

    del new_spec["Contacts"]
    del new_spec["SpawnCollisionCount"]
    del new_spec["CollisionStop"]
    del new_spec["ElapsedTime"]
    return new_spec

  def _compute_max_episode_reward(self, obs):
    return len(obs.targets) * self._reward_per_target

  def _maybe_update_max_steps(self):
    block_height = self._initial_available_objects[0].height
    target_level_height = int(
        self._initial_scene.targets[0].y // block_height)
    self._max_steps = max(15, 5 * (target_level_height + 1))

  def _get_task_reward(self, obstacles, targets, blocks):
    del obstacles, blocks
    num_targets_hit = np.sum(
        (targets[:, unity_constants.COLLISION_COUNT_FEATURE_INDEX] > 0.))
    target_score = num_targets_hit * self._reward_per_target
    return target_score

  def _get_generator(self, difficulty):
    if isinstance(difficulty, str):
      raise ValueError("Unrecognized difficulty: %s" % difficulty)
    # By having a single ball level at 0, the ball will always be placed
    # a fixed number of levels above the target.
    ball_levels = [0]

    difficulty_distance = min(difficulty, 4)
    max_height = max(0, difficulty - 4)
    min_num_obstacles = 1
    max_num_obstacles = difficulty // 2 + min_num_obstacles

    distance_ranges = [(0.03, 0.3),
                       (0.36, 0.49),
                       (0.50, 0.63),
                       (0.69, 0.82),
                       (0.83, 1.)]

    max_rel_distance = distance_ranges[difficulty_distance][1]
    min_rel_distance = distance_ranges[difficulty_distance][0]

    horizontal_distance_range = [min_rel_distance, max_rel_distance]
    target_levels = [max_height]
    num_obstacles_range = [max_num_obstacles, max_num_obstacles + 1]

    return MarbleRunGenerator(
        target_levels=target_levels,
        ball_levels=ball_levels,
        scene_width=self._generator_width,
        random_state=self._random_state,
        num_obstacles_range=num_obstacles_range,
        rel_horizontal_distance_range=horizontal_distance_range)

  @property
  def episode_params(self):
    """Returns discrete parameters of the current episode.

    This can be used to implement a dynamic curriculum by clustering episode
    outcomes according to these, and by asking the agent to do well on all
    clusters/parameter setting combinations, before it can progress to the next
    difficulty level.
    """
    # First two obstacles are the fixed walls.
    obstacles = self._initial_scene.obstacles[2:]
    balls = self._initial_scene.balls
    targets = self._initial_scene.targets
    num_obstacles = len(obstacles)
    num_targets = len(targets)
    num_balls = len(balls)
    max_height_target = max([target.y for target in targets])

    obstacle_heights = [obstacle.y for obstacle in obstacles]
    min_height_obstacle = min(obstacle_heights)
    max_height_obstacle = max(obstacle_heights)

    # Calculate the maximum horizontal distance between all targets and
    # their closest ball and digitize it.
    scene_width = self._display_limit * 2.
    max_min_target_distance_per_target = max([
        min([np.abs(ball.x-target.x) for ball in balls])  # pylint: disable=g-complex-comprehension
        for target in targets])
    horizontal_distance_bin = int(np.digitize(
        max_min_target_distance_per_target, [scene_width/3, scene_width*2/3]))
    return MarbleRunEpisodeParams(
        num_obstacles=num_obstacles,
        num_targets=num_targets,
        num_balls=num_balls,
        min_height_obstacle=min_height_obstacle,
        max_height_obstacle=max_height_obstacle,
        max_height_target=max_height_target,
        discretized_ball_target_distance=horizontal_distance_bin)


class MarbleRunGenerator(stacking.StackingGenerator):
  """Generates a set of horizontal obstacles and targets."""

  def __init__(self,
               target_levels,
               ball_levels,
               num_obstacles_range,
               rel_horizontal_distance_range=(0., 1.),
               min_ball_target_level_diff=4,
               targets_side=5,
               obstacles_height=5,
               **kwargs):
    """Initialize the generator.

    Args:
      target_levels: List of discrete height levels (starting from 0) at which
        the targets can be located.
      ball_levels: List of discrete height levels (starting from 0) at which
        the balls can be located. See min_ball_target_level_diff.
      num_obstacles_range: a tuple indicating the range of targets
        that will be in the generated scene, from low (inclusive) to high
        (exclusive). This counts the total number of obstacles.
      rel_horizontal_distance_range: Range of horizontal distances between
        the target and the ball, relative to the scene width.
      min_ball_target_level_diff: Minimum number of levels that the ball will
        be above the target. This will define the lower limit for the position
        the ball level to be `(target_level + min_ball_target_level_diff)`,
        which may cause the ball to be higher than any of the levels specified
        in ball_levels.
      targets_side: The width and height of targets. Only used when
        `use_legacy_obstacles_heights` is set to False.
      obstacles_height: The height of the obstacles. Only used when
        `use_legacy_obstacles_heights` is set to False.
      **kwargs: additional keyword arguments passed to super
    """
    super(MarbleRunGenerator, self).__init__(
        num_blocks_range=None,
        **kwargs)

    self._targets_side = targets_side
    self.obstacles_height = obstacles_height
    self._targets_height = targets_side
    self._target_levels = target_levels
    self._num_obstacles_range = num_obstacles_range
    self._min_ball_target_level_diff = min_ball_target_level_diff
    self._rel_horizontal_distance_range = rel_horizontal_distance_range
    self._ball_levels = ball_levels

  def _place_available_objects(self):
    """Returns the available blocks to the agent."""

    def create_block(width, height, shape):
      """Returns a block with the specified properties."""
      block = block_utils.Block(
          width=width, height=height, angle=0., shape=shape,
          x=0, y=0)  # x and y will be set later.
      return block

    observation_blocks = [
        create_block(
            self.small_width, self.height, unity_constants.BOX_SHAPE),
        create_block(
            2*self.small_width, self.height, unity_constants.BOX_SHAPE),
        create_block(
            self.small_width, 2*self.height, unity_constants.BOX_SHAPE),
        create_block(
            self.medium_width, self.height*2/3, unity_constants.BOX_SHAPE),
        create_block(
            self.large_width, self.height/10*3, unity_constants.BOX_SHAPE),
        create_block(
            -self.medium_width, self.height, unity_constants.RAMP_SHAPE),
        create_block(
            self.medium_width, self.height, unity_constants.RAMP_SHAPE),
    ]

    # Calculate margin of blocks.
    block_abs_widths = [np.abs(block.width) for block in observation_blocks]
    empty_width = self.scene_width - sum(block_abs_widths)

    if empty_width <= 0:
      raise ValueError("Not enough space between available objects.")

    horizontal_margin = empty_width / (len(observation_blocks) - 1)

    # Update the position of the blocks using the margin.
    observation_block_with_positions = []
    current_x = 0
    display_y_pos = -2 * (self.margin + self.height)
    for block in observation_blocks:
      abs_width = np.abs(block.width)
      display_x_pos = current_x + abs_width / 2
      observation_block_with_positions.append(
          block._replace(x=display_x_pos, y=display_y_pos))
      current_x += abs_width + horizontal_margin

    assert current_x - horizontal_margin <= self.scene_width
    return observation_block_with_positions

  def _build_tessellation(self, num_levels):
    """"Tessellates space blocks.

    The width of the blocks in the tessellation will be either
    `self.small_width` or `2*self.small_width`.

    Args:
      num_levels: Number of layers of blocks in the tesselation.

    Returns:
      2-D Array containing the blocks in the tesselation with shape
      [num_blocks, 3]. Where the trailing dimension contains horizontal
      position (floating point bounded between 0 and self.scene_width),
      vertical discrete position (integer in range(num_levels)) and width (
      one of `self.small_width` or `2*self.small_width`).

    """

    valid_widths = [self.small_width, 2*self.small_width]
    blocks = []
    for level in range(num_levels):
      accumulated_width = self.random_state.uniform(0, self.small_width)
      while True:
        block_width = self.random_state.choice(valid_widths)
        block_x = accumulated_width + block_width / 2
        block_y = level
        accumulated_width += block_width
        if accumulated_width > self.scene_width:
          break
        blocks.append([block_x, block_y, block_width])

    blocks_xyw = np.stack(blocks, axis=0)
    return blocks_xyw

  def _place_walls(self, relative_wall_margin=0.05):
    """Returns blocks with walls at the edges of the scene, with some margin."""
    wall_width = self.height / 2
    margin = self.scene_width * relative_wall_margin
    scene_height = self.scene_width * 2/3
    right_wall = block_utils.Block(
        x=self.scene_width + margin + wall_width/2., y=scene_height/2,
        height=scene_height, width=wall_width)
    left_wall = block_utils.Block(
        x=-margin-wall_width/2., y=scene_height/2,
        height=scene_height, width=wall_width)
    return [right_wall, left_wall]

  def _scale_vertical_positions(self, blocks):
    """Transformcs integer level vertical positions into continuous."""
    return [block._replace(y=(block.y+0.5)*self.height)
            for block in blocks]

  def _remove_nearby_blocks(self, reference_block, blocks_xyw):
    """Removes blocks that are nearby to a reference block from a tessellation.

    A block is considered nearby if it is within the same layer or up to two
    layers abover or below the reference block, and the horitonzal distance
    to the reference block is less than self.small_width * 1.5.

    Args:
      reference_block: reference block_utils.Block object.
      blocks_xyw: Block in the tessellation.

    Returns:
      Updated tessellation where the nearby blocks have been removed.

    """

    # We will for sure keep all blocks that are more than two levels away
    # from the reference block.
    mask_keep_rows = np.abs(blocks_xyw[:, 1] - reference_block.y) > 2.5

    margin = self.small_width * 2

    # We will also keep blocks whose left side is far enough to the right of the
    # right side of the reference block, and blocks whose right side is far
    # enough to the left from the left side of the reference block.
    right_side_reference = reference_block.x + reference_block.width/2
    right_side_blocks = blocks_xyw[:, 0] + blocks_xyw[:, 2]/2

    left_side_reference = reference_block.x - reference_block.width/2
    left_side_blocks = blocks_xyw[:, 0] - blocks_xyw[:, 2]/2

    mask_keep_margin = np.logical_or(
        left_side_blocks > right_side_reference + margin,
        right_side_blocks < left_side_reference - margin)

    mask_keep = np.logical_or(mask_keep_margin, mask_keep_rows)
    return blocks_xyw[mask_keep]

  def _sample_block_from_tessellation(
      self, tessellation_blocks_xyw, balls, targets):
    """Samples a block from the tessellation according to different criteria."""

    probabilities = []
    # Start by assigning uniform probability to each block.
    weight_uniform = 1.
    probabilities_uniform = np.ones([tessellation_blocks_xyw.shape[0]])
    probabilities.append((weight_uniform, probabilities_uniform))

    # Probabilities near ball and target
    weight_near = 1.
    temperature_length = self.scene_width / 10
    for ball in balls:
      distances = np.abs(tessellation_blocks_xyw[:, 0] - ball.x)
      probabilities_near_ball = 1 / ((distances/temperature_length)**2 + 0.1)
      probabilities.append((weight_near, probabilities_near_ball))
    for target in targets:
      distances = np.abs(tessellation_blocks_xyw[:, 0] - target.x)
      probabilities_near_target = 1 / ((distances/temperature_length)**2 + 0.1)
      probabilities.append((weight_near, probabilities_near_target))

    # Higher probabilities for objects laying exactly on top of the floor,
    # in the region between the ball and targets.
    weight_floor = 4.
    blocks_on_floor = tessellation_blocks_xyw[:, 1] == 0
    balls_and_targets_x = [ball_or_target.x for ball_or_target in balls+targets]
    min_x, max_x = min(balls_and_targets_x), max(balls_and_targets_x)
    blocks_between = ((tessellation_blocks_xyw[:, 0] > min_x) *
                      (tessellation_blocks_xyw[:, 0] < max_x))
    blocks_floor_between = blocks_on_floor * blocks_between

    if np.any(blocks_floor_between):
      probabilities_on_floor = np.where(blocks_floor_between, 1., 0.)
      probabilities.append((weight_floor, probabilities_on_floor))

    # Probabilities near the middle point between ball and target:
    weight_middle = 1.
    temperature_length = self.scene_width / 5
    for ball, target in itertools.product(balls, targets):
      middle_x = (ball.x + target.x) / 2
      distances = np.abs(tessellation_blocks_xyw[:, 0] - middle_x)
      probabilities_near_middle = 1 / ((distances/temperature_length)**2 + 0.1)
      probabilities.append((weight_middle, probabilities_near_middle))

    weights, probabilities = list(zip(*probabilities))
    # Stack multipliers and probabilities.
    weights = np.stack(weights, axis=0)

    probabilities = np.stack(probabilities, axis=0)

    # Normalize each row of probabilities to 1 and weight to each category.
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    probabilities = weights[:, np.newaxis] * probabilities

    # Sum the probabilities and normalize again.
    probabilities_merged = probabilities.sum(axis=0)
    probabilities_merged /= np.sum(probabilities_merged)

    block_indices = np.arange(tessellation_blocks_xyw.shape[0])
    block_ind = self.random_state.choice(block_indices, p=probabilities_merged)
    return tessellation_blocks_xyw[block_ind]

  def _try_to_generate_one(self):
    """Generates a single marble_run scene.

    Returns:
      observation: a block_utils.BlocksObservation object
      solution: a list of Block objects in their final locations. This generator
        in particular always returns an empty list here. Unused
    """

    # Generate ball and target.
    # Make sure that there is enough horizontal separation.
    min_horiz_dist = self._rel_horizontal_distance_range[0]*self.scene_width
    max_horiz_dist = self._rel_horizontal_distance_range[1]*self.scene_width

    # Start by assuming that the ball will go left to right.
    # Sample the position of the ball to make sure that is it at least
    # min_horiz_dist away from the right.
    ball_x = self.random_state.uniform(0, self.scene_width-min_horiz_dist)

    min_target_x = ball_x + min_horiz_dist
    # The minimum target separation is the size of the target and the ball,
    # so the ball cannot drop directly on the target.
    min_target_x = max(self._targets_height + self.height, min_target_x)

    max_target_x = ball_x + max_horiz_dist
    max_target_x = min(self.scene_width, max_target_x)

    target_x = self.random_state.uniform(min_target_x, max_target_x)

    assert target_x - ball_x < max_horiz_dist
    assert target_x - ball_x > min_horiz_dist

    # We flip a coin to exchage the positions of target and ball
    if self.random_state.randint(2):
      target_x, ball_x = ball_x, target_x

    # Make sure that the ball is at least two levels above the target.
    target_level = np.random.choice(self._target_levels)
    ball_level = np.random.choice(self._ball_levels)
    ball_level = max(ball_level,
                     target_level + self._min_ball_target_level_diff)
    targets = [block_utils.Block(
        x=target_x, y=target_level, shape=unity_constants.BALL_SHAPE,
        width=self._targets_height, height=self._targets_height)]
    balls = [block_utils.Block(
        x=ball_x, y=ball_level, shape=unity_constants.BALL_SHAPE,
        width=self.height, height=self.height)]

    # Generating obstacles.
    obstacles = []
    # Tesselate space and remove blocks too close to balls and targets.
    tessellation_blocks_xyw = self._build_tessellation(
        num_levels=ball_level+1)
    for ball in balls:
      tessellation_blocks_xyw = self._remove_nearby_blocks(
          ball, tessellation_blocks_xyw)
    for target in targets:
      tessellation_blocks_xyw = self._remove_nearby_blocks(
          target, tessellation_blocks_xyw)

    # Add obstacles sequentially.
    num_obstacles = np.random.randint(*self._num_obstacles_range)
    for _ in range(num_obstacles):
      if tessellation_blocks_xyw.shape[0] == 0:
        raise stacking.GenerationError(
            "Not possible to generate all obstacles.")
      block = self._sample_block_from_tessellation(
          tessellation_blocks_xyw, balls, targets)
      obstacles.append(
          block_utils.Block(
              x=block[0], y=block[1],
              width=block[2], height=self.obstacles_height))
      tessellation_blocks_xyw = self._remove_nearby_blocks(
          obstacles[-1], tessellation_blocks_xyw)

    floor = self._place_floor()
    observation_blocks = self._place_available_objects()
    fixed_blocks = [floor] + observation_blocks

    # Convert discrete vertical positions into continuous values.
    obstacles = self._scale_vertical_positions(obstacles)
    targets = self._scale_vertical_positions(targets)
    balls = self._scale_vertical_positions(balls)

    walls = self._place_walls()

    observation = block_utils.BlocksObservation(
        balls=balls,
        blocks=fixed_blocks,
        obstacles=walls + obstacles,
        targets=targets)

    return observation

  def generate_one(self):
    """Generates a single marble_run scene.

    Returns:
      observation: a block_utils.BlocksObservation object
      solution: a list of Block objects in their final locations. This generator
        in particular always returns an empty list here. Unused
    """

    max_attempts_generate = 100
    for _ in range(max_attempts_generate):
      try:
        return self._try_to_generate_one()
      except stacking.GenerationError:
        continue
    raise stacking.GenerationError("Max number of generation attempts reached.")
