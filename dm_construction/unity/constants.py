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
r"""Constants for environment.py .

"""

import types

BLOCK_SIZE = 36
CONTACT_SIZE = 6
NUM_LAYERS = 4


ID_FEATURE_INDEX = 0
POSITION_FEATURE_SLICE = slice(1, 3)
POSITION_3D_FEATURE_SLICE = slice(1, 4)
POSITION_X_FEATURE_INDEX = POSITION_FEATURE_SLICE.start
POSITION_Y_FEATURE_INDEX = POSITION_FEATURE_SLICE.start + 1
POSITION_Z_FEATURE_INDEX = POSITION_FEATURE_SLICE.start + 2
ORIENTATION_FEATURE_SLICE = slice(4, 6)
COSINE_ANGLE_FEATURE_INDEX = ORIENTATION_FEATURE_SLICE.start
SINE_ANGLE_FEATURE_INDEX = ORIENTATION_FEATURE_SLICE.start + 1
SIZE_FEATURE_SLICE = slice(6, 9)
WIDTH_FEATURE_INDEX = SIZE_FEATURE_SLICE.start
HEIGHT_FEATURE_INDEX = SIZE_FEATURE_SLICE.start + 1
DEPTH_FEATURE_INDEX = SIZE_FEATURE_SLICE.start + 2
COLOR_FEATURE_SLICE = slice(9, 13)
RED_CHANNEL_FEATURE_INDEX = COLOR_FEATURE_SLICE.start
GREEN_CHANNEL_FEATURE_INDEX = COLOR_FEATURE_SLICE.start + 1
BLUE_CHANNEL_FEATURE_INDEX = COLOR_FEATURE_SLICE.start + 2
ALPHA_CHANNEL_FEATURE_INDEX = COLOR_FEATURE_SLICE.start + 3
LINEAR_VELOCITY_FEATURE_SLICE = slice(13, 15)
VELOCITY_X_FEATURE_INDEX = LINEAR_VELOCITY_FEATURE_SLICE.start
VELOCITY_Y_FEATURE_INDEX = LINEAR_VELOCITY_FEATURE_SLICE.start + 1
ANGULAR_VELOCITY_FEATURE_INDEX = 15
PHYSICAL_OBJECT_FEATURE_INDEX = 16
COLLISION_MASK_FEATURE_SLICE = slice(17, 17 + NUM_LAYERS)
DENSITY_FEATURE_INDEX = 21
BOUNCINESS_FEATURE_INDEX = 22
FRICTION_FEATURE_INDEX = 23
LINEAR_DRAG_FEATURE_INDEX = 24
ANGULAR_DRAG_FEATURE_INDEX = 25
FREE_OBJECT_FEATURE_INDEX = 26
GLUEABLE_FEATURE_INDEX = 27
STICKY_FEATURE_INDEX = 28
GLUED_FEATURE_INDEX = 29
REMAINING_GLUE_FEATURE_INDEX = 30
SHAPE_FEATURE_SLICE = slice(31, 34)
IS_BOX_FEATURE_INDEX = SHAPE_FEATURE_SLICE.start
IS_BALL_FEATURE_INDEX = SHAPE_FEATURE_SLICE.start + 1
IS_RAMP_FEATURE_INDEX = SHAPE_FEATURE_SLICE.start + 2
START_TIME_FEATURE_INDEX = 34
COLLISION_COUNT_FEATURE_INDEX = 35

BOX_SHAPE = 0
BALL_SHAPE = 1
RAMP_SHAPE = 2


ACTION_DEFAULTS = types.MappingProxyType(dict(
    GravityY=-9.8,
    Friction=0.4,
    AngularDrag=0.05,
    Glueable=1.,
    Timestep=0.02,
    Density=1.,
    Depth=1.,
    Shape=BOX_SHAPE,
    PhysicalBody=1.,
    CollisionMask=1.,
    CameraPosY=5.,
    CameraHeight=16.,
    A=1.,
    # Every time we send an action, we have to manually set this to 1.
    # The user however is not allowed to set this action.
    IsAction=1.,
))
OTHER_ACTION_DEFAULT = 0.0

ADDITIONAL_HELPER_ACTIONS = (
    "R", "G", "B", "A", "RGB",
)


class RestoreVerificationError(Exception):
  """Exception to raise if verification of restored observations fails."""
  pass


class MetaEnvironmentError(Exception):
  """Exception to raise when the metaenvironment is in a bad state."""
  pass

