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
"""Tests for unity.environment."""

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from dm_construction.unity import constants
from dm_construction.unity import docker
from dm_construction.unity import environment as unity_environment
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_boolean("skip_local", False, "")
flags.DEFINE_boolean("skip_mpm", False, "")
flags.DEFINE_string("local_path", None, "")
flags.DEFINE_string("loader", "docker", "")

FRAME_WIDTH = 30
FRAME_HEIGHT = 40
OBSERVER_WIDTH = 60
OBSERVER_HEIGHT = 80

_LOADERS = {
    "docker": docker.loader,
}

# To make the run tests faster, we are going to have a single test that runs
# everything loading the meta environment only once.


def print_func(fn):
  """Prints a method name function before calling it."""
  fn_name = fn.__name__

  def decorated_fn(*args, **kwargs):
    logging.info(fn_name)
    output = fn(*args, **kwargs)
    logging.info("%s: done", fn_name)
    return output
  return decorated_fn


class CoreTest(parameterized.TestCase):

  def _get_local_path(self):
    if FLAGS.local_path is not None:
      return FLAGS.local_path
    else:
      raise ValueError("local path is not defined")

  def _get_loader(self):
    return _LOADERS[FLAGS.loader]

  @print_func
  def _create_environment_features(self, is_local):
    kwargs = {}
    if is_local:
      kwargs["local_path"] = self._get_local_path()
    return unity_environment.UnityConstructionEnv(
        loader=self._get_loader(), include_agent_camera=False, **kwargs)

  @print_func
  def _create_environment_pixels(self, is_local):
    kwargs = {}
    if is_local:
      kwargs["local_path"] = self._get_local_path()
    return unity_environment.UnityConstructionEnv(
        loader=self._get_loader(),
        include_agent_camera=True,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        **kwargs)

  @print_func
  def _create_environment_video(self, is_local):
    kwargs = {}
    if is_local:
      kwargs["local_path"] = self._get_local_path()
    return unity_environment.UnityConstructionEnv(
        loader=self._get_loader(),
        include_agent_camera=True,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        include_observer_camera=True,
        observer_3d=True,
        observer_width=OBSERVER_WIDTH,
        observer_height=OBSERVER_HEIGHT,
        max_simulation_substeps=20,
        **kwargs)

  @parameterized.named_parameters(
      ("LocalPath", True),
      ("MPM", False),
      )
  def test_meta_environment(self, use_local_path):

    if FLAGS.skip_local and use_local_path:
      logging.info("Skipping local test")
      return

    if FLAGS.skip_mpm and not use_local_path:
      logging.info("Skipping mpm test")
      return

    self._unity_env_features = self._create_environment_features(use_local_path)
    self._unity_env_pixels = self._create_environment_pixels(use_local_path)
    self._unity_env_video = self._create_environment_video(use_local_path)

    # Test for specific features.
    self._stop_on_collision_feature_test(self._unity_env_features)
    self._collision_masks_test(self._unity_env_features)
    self._spawn_collision_test(self._unity_env_features)

    # Test restoration of states.
    self._restore_test(self._unity_env_features)

    # Test that multiple modes give same results.
    actions_setup, actions_dynamics = self._stop_on_collision_actions()
    self._different_types_test(actions_setup + actions_dynamics)
    self._action_list_test(actions_setup + actions_dynamics)

    self._verify_restored_observation_test()

    self._unity_env_features.close()
    self._unity_env_pixels.close()
    self._unity_env_video.close()

  def _rollout_environment(self, environment, actions_list,
                           send_actions_as_list=False):
    reset_observation = environment.reset().observation
    if send_actions_as_list:
      return [reset_observation,
              environment.step(actions_list).observation]
    else:
      return ([reset_observation] +
              [environment.step(action).observation
               for action in actions_list])

  @print_func
  def _different_types_test(self, actions_list):
    # Verify that observation sequence is consistent across environment modes.
    observations_1 = self._rollout_environment(
        self._unity_env_features, actions_list)
    observations_2 = self._rollout_environment(
        self._unity_env_pixels, actions_list)
    observations_3 = self._rollout_environment(
        self._unity_env_video, actions_list)

    for obs_1, obs_2, obs_3 in zip(observations_1, observations_2,
                                   observations_3):
      unity_environment._verify_restored_observation(obs_1, obs_2)
      unity_environment._verify_restored_observation(obs_1, obs_3)

  @print_func
  def _action_list_test(self, actions_list):

    # Verify that final observation is the same regardless whether actions
    # were sent one by one, or as a single list.
    observations_1 = self._rollout_environment(
        self._unity_env_features, actions_list, send_actions_as_list=False)
    observations_2 = self._rollout_environment(
        self._unity_env_pixels, actions_list, send_actions_as_list=False)
    observations_3 = self._rollout_environment(
        self._unity_env_video, actions_list, send_actions_as_list=False)

    final_observation_1 = self._rollout_environment(
        self._unity_env_features, actions_list, send_actions_as_list=True)[-1]
    final_observation_2 = self._rollout_environment(
        self._unity_env_pixels, actions_list, send_actions_as_list=True)[-1]
    final_observation_3 = self._rollout_environment(
        self._unity_env_video, actions_list, send_actions_as_list=True)[-1]

    unity_environment._verify_restored_observation(
        observations_1[-1], final_observation_1)
    unity_environment._verify_restored_observation(
        observations_2[-1], final_observation_2)
    unity_environment._verify_restored_observation(
        observations_3[-1], final_observation_3)

  @print_func
  def _collision_masks_test(self, unity_env):
    """Compares final position of the blocks after falling."""
    actions = self._collision_masks_actions()
    unity_env.reset()
    for action in actions:
      observation = unity_env.step(action).observation

    expected_y_coordinates = (0., 2.5, 5.0, 7.5, 8.114995,
                              5.61499691, 3.11499906, 0.61499971)

    self.assertEqual(8, observation["Blocks"].shape[0])
    delta = 1e-4
    for block_i in range(8):
      self.assertAlmostEqual(
          expected_y_coordinates[block_i],
          observation["Blocks"][block_i, constants.POSITION_Y_FEATURE_INDEX],
          delta=delta)

  @print_func
  def _restore_test(self, unity_env):
    """Compares final position of the blocks after falling."""
    actions = self._multiple_balls_actions()
    unity_env.reset()
    observation_sequence = []
    for action in actions:
      observation = unity_env.step(action).observation
      observation_sequence.append(observation)

    # Restore the state from each of the observations, and compare the
    # observation for all subsequent steps.
    for restore_index, initial_observation in enumerate(observation_sequence):
      unity_env.reset()
      obs_restored = unity_env.restore_state(initial_observation)
      restored_observations = [obs_restored.observation]
      extra_actions = actions[restore_index+1:]

      restored_observations += [unity_env.step(action).observation
                                for action in extra_actions]

      for restored_observation, original_observation in zip(
          restored_observations, observation_sequence[restore_index:]):
        unity_environment._verify_restored_observation(
            original_observation, restored_observation)

  @print_func
  def _stop_on_collision_feature_test(self, unity_env):
    """Compares the set of collisions."""
    actions_setup, actions_dynamics = self._stop_on_collision_actions()

    unity_env.reset()
    for action in actions_setup:
      observation = unity_env.step(action).observation

    observation_sequence = (
        [observation] +
        [unity_env.step(action).observation for action in actions_dynamics])

    expected_x_coordinates = [
        (7.5, -7.5, -4.0, 0.0),
        (7.5, -7.5, -0.489688, 0.0896859),
        (7.5, -7.5, -0.489688, 7.11094),
        (7.5, -7.5, -0.57874, -9.19681e-09),
        (7.5, -7.5, -7.12189, -9.19681e-09),
        (7.5, -7.5, -0.489688, 0.0677929),
        (7.5, -7.5, -0.489688, 7.13284),
    ]

    expected_y_coordinates = [
        (5.0, 5.0, 5.5, 5.5),
        (5.0, 5.0, 5.5, 5.5),
        (5.0, 5.0, 5.5, 5.5),
        (5.0, 5.0, 5.5, 5.5),
        (5.0, 5.0, 5.5, 5.5),
        (5.0, 5.0, 5.5, 5.5),
        (5.0, 5.0, 5.5, 5.5),
    ]

    expected_num_collisions = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 2.0),
        (1.0, 0.0, 2.0, 3.0),
        (1.0, 1.0, 3.0, 3.0),
        (1.0, 1.0, 4.0, 4.0),
        (2.0, 1.0, 4.0, 5.0),
    ]

    delta = 1e-4
    for time, observation in enumerate(observation_sequence):
      self.assertEqual(4, observation["Blocks"].shape[0])
      for block_i in range(4):
        self.assertAlmostEqual(
            expected_x_coordinates[time][block_i],
            observation["Blocks"][block_i, constants.POSITION_X_FEATURE_INDEX],
            delta=delta)
        self.assertAlmostEqual(
            expected_y_coordinates[time][block_i],
            observation["Blocks"][block_i, constants.POSITION_Y_FEATURE_INDEX],
            delta=delta)
        self.assertAlmostEqual(
            expected_num_collisions[time][block_i],
            observation["Blocks"][
                block_i, constants.COLLISION_COUNT_FEATURE_INDEX],
            delta=delta)

  @print_func
  def _spawn_collision_test(self, unity_env):
    """Compares final position of the blocks after falling."""

    # Remove the final action, since we do not want to run the physics.
    actions = self._collision_masks_actions()[:-1]

    # We know there are objects at y = 10 and x = -3, -1, 1, 3, with collision
    # masks 0b0001, 0b0010, 0b0100 and 0b1000.
    x_positions = [-3, -1, 1, 3]
    shared_action = {
        "SpawnBlock": 1.0, "Width": 0.5, "Height": 0.5, "SetPosY": 10.0,
        "FreeBody": 1}

    # Try adding objects that do should not cause an spawn collision.
    masks_no_collisions = [0b1110, 0b1101, 0b1011, 0b0111]
    masks_collisions = [0b0001, 0b0010, 0b0100, 0b1000]
    for expected_spawn_collision_count, masks in [(0, masks_no_collisions),
                                                  (1, masks_collisions)]:
      for x_pos, mask in zip(x_positions, masks):
        self._rollout_environment(
            unity_env, actions, send_actions_as_list=True)
        action = shared_action.copy()
        action.update({"SetPosX": x_pos, "CollisionMask": mask})
        new_observation = unity_env.step(action).observation
        self.assertEqual(expected_spawn_collision_count,
                         new_observation["SpawnCollisionCount"])

  @print_func
  def _verify_restored_observation_test(self):
    unity_env = self._unity_env_features
    actions = self._collision_masks_actions()
    unity_env.reset()
    observation = unity_env.step(actions).observation

    unity_environment._verify_restored_observation(
        observation, observation, difference_threshold_abs=1e-9)

    for observation_name in observation.keys():
      if observation_name in ["RGB", "ObserverRGB"]:
        continue

      observation_item = observation[observation_name]
      observation_item_flat = observation_item.flatten().copy()

      if not observation_item_flat.size:
        continue

      if observation_item.dtype == np.bool:
        observation_item_flat[0] = not observation_item_flat[0]
      elif observation_item.dtype == np.int32:
        observation_item_flat[0] += 1
      elif observation_item.dtype == np.float32:
        observation_item_flat[0] += 5e-4
      else:
        raise ValueError("Unknown observation type.")
      bad_observation_item = np.reshape(
          observation_item_flat, observation_item.shape)

      bad_observation = observation.copy()
      bad_observation[observation_name] = bad_observation_item

      if observation_item.dtype == np.float32:
        # This should not fail, since it is under the threshold.
        unity_environment._verify_restored_observation(
            observation, bad_observation, difference_threshold_abs=1e-3)

      with self.assertRaisesRegex(
          constants.RestoreVerificationError,
          "`%s` observation with shape" % observation_name):
        unity_environment._verify_restored_observation(
            observation, bad_observation, difference_threshold_abs=1e-4)

    # Check that verify_velocities == False does ignore block velocities.
    bad_observation = observation.copy()
    bad_observation["Blocks"] = observation["Blocks"].copy()
    bad_observation["Blocks"][:, constants.VELOCITY_X_FEATURE_INDEX] = 1000
    bad_observation["Blocks"][:, constants.VELOCITY_Y_FEATURE_INDEX] = 1000
    bad_observation["Blocks"][
        :, constants.ANGULAR_VELOCITY_FEATURE_INDEX] = 1000
    unity_environment._verify_restored_observation(
        observation, bad_observation, difference_threshold_abs=1e-9,
        verify_velocities=False)
    with self.assertRaisesRegex(constants.RestoreVerificationError,
                                "`Blocks` observation with shape"):
      unity_environment._verify_restored_observation(
          observation, bad_observation, difference_threshold_abs=100.,
          verify_velocities=True)

  def _collision_masks_actions(self):
    """Generates 4 floors and 4 blocks falling with custom collision masks."""
    actions = [
        {"SpawnBlock": 1.0, "SetPosY": 0.0, "Width": 50.0,
         "Height": 0.2, "R": 0.2, "CollisionMask": 0b1111},
        {"SpawnBlock": 1.0, "SetPosY": 2.5, "Width": 50.0,
         "Height": 0.2, "R": 0.2, "CollisionMask": 0b0111},
        {"SpawnBlock": 1.0, "SetPosY": 5.0, "Width": 50.0,
         "Height": 0.2, "R": 0.2, "CollisionMask": 0b0011},
        {"SpawnBlock": 1.0, "SetPosY": 7.5, "Width": 50.0,
         "Height": 0.2, "R": 0.2, "CollisionMask": 0b0001},
        {"SpawnBlock": 1.0, "Width": 1., "Height": 1., "SetPosX": -3.0,
         "SetPosY": 10.0, "R": 0.4, "CollisionMask": 0b0001, "FreeBody": 1},
        {"SpawnBlock": 1.0, "Width": 1., "Height": 1., "SetPosX": -1.0,
         "SetPosY": 10.0, "R": 0.4, "CollisionMask": 0b0010, "FreeBody": 1},
        {"SpawnBlock": 1.0, "Width": 1., "Height": 1., "SetPosX": 1.0,
         "SetPosY": 10.0, "R": 0.4, "CollisionMask": 0b0100, "FreeBody": 1},
        {"SpawnBlock": 1.0, "Width": 1., "Height": 1., "SetPosX": 3.0,
         "SetPosY": 10.0, "R": 0.4, "CollisionMask": 0b1000, "FreeBody": 1},
        {"SimulationSteps": 400}]
    return actions

  def _multiple_balls_actions(self):
    """Spawns a floor, some bumpers, and som balls."""

    # Floor and bumpers.
    setup_actions = [
        {"SpawnBlock": 1.0, "SetPosX": 0.0, "SetPosY": 0.0, "Width": 50.0,
         "Height": 0.2, "R": 0.2},
        {"SpawnBlock": 1.0, "SetPosX": 3.0, "SetPosY": 4.0, "SetAngle": 3.14/4,
         "Width": 2.0, "Height": 0.1, "R": 0.4, "Sticky": 0},
        {"SpawnBlock": 1.0, "SetPosX": 1.0, "SetPosY": 2.0, "SetAngle": 3.14/7,
         "Width": 2.0, "Height": 1.0, "R": 0.4},
        {"SpawnBlock": 1.0, "Shape": 2.0, "SetPosX": -2.0, "SetPosY": 0.5,
         "SetAngle": 0., "Width": 5.0, "Height": 1.0, "R": 0.4},
    ]

    # Balls spawned at intervals.
    periodic_actions = [
        {"SpawnBlock": 1.0, "Shape": 1.0, "FreeBody": 1,
         "SetPosX": 3.0, "SetPosY": 5, "Width": 0.5, "Height": 0.5, "G": 1.0,
         "LinearDrag": 0.5},
        {"SimulationSteps": 100, "StopOnCollision": 1.},
        {"SimulationSteps": 200, "StopOnCollision": 0.}]
    return setup_actions + periodic_actions * 2

  def _stop_on_collision_actions(self):
    """Generates a set of actions of two balls bouncing."""

    simulation_steps = 600
    size_ball_1 = 0.5
    size_ball_2 = 0.5
    density_ball_1 = 1.0
    density_ball_2 = 1.0
    bounciness = 1.0
    linear_drag = 0.0
    angular_drag = 1000
    actions_setup = [{  # Right wall.
        "SpawnBlock": 1.0,
        "SetPosX": 7.5,
        "SetPosY": 5.0,
        "Width": 0.2,
        "Height": 10,
        "Bounciness": bounciness,
        "LinearDrag": linear_drag,
        "G": 0.3
    }, {  # Left wall.
        "SpawnBlock": 1.0,
        "SetPosX": -7.5,
        "SetPosY": 5.0,
        "Width": 0.2,
        "Height": 10,
        "Bounciness": bounciness,
        "LinearDrag": linear_drag,
        "G": 0.3
    }]
    actions_setup.append({  # Left ball.
        "SpawnBlock": 1.0,
        "Shape": constants.BALL_SHAPE,
        "FreeBody": 1,
        "SetPosX": -4.0,
        "SetPosY": 5.5,
        "Width": size_ball_1,
        "Height": size_ball_1,
        "Sticky": 0.0,
        "R": 1.0,
        "SetVelX": 5.0,
        "SetVelY": 0.0,
        "Bounciness": bounciness,
        "LinearDrag": linear_drag,
        "AngularDrag": angular_drag,
        "Density": density_ball_1
    })
    actions_setup.append({  # Right ball.
        "SpawnBlock": 1.0,
        "Shape": constants.BALL_SHAPE,
        "FreeBody": 1,
        "SetPosX": 0.0,
        "SetPosY": 5.5,
        "Width": size_ball_2,
        "Height": size_ball_2,
        "Sticky": 0.0,
        "G": 1.0,
        "SetVelX": 0.0,
        "SetVelY": 0.0,
        "Bounciness": bounciness,
        "LinearDrag": linear_drag,
        "AngularDrag": angular_drag,
        "Density": density_ball_2
    })

    actions_dynamics = []
    for _ in range(6):
      actions_dynamics.append({
          "SimulationSteps": simulation_steps,
          "StopOnCollision": 1.0,
          "GravityY": 0.0,
          "GravityX": 0.0
      })

    return actions_setup, actions_dynamics


if __name__ == "__main__":
  absltest.main()
