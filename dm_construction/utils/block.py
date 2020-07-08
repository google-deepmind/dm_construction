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
"""Utilities for handling manipulating building blocks."""

import collections
from dm_construction.unity import constants as unity_constants

# Block is a namedtuple which contains the relevant attributes for a block:
# its x and y coordinates of the center of the block, and the width and height
# of the block (should all be floats).
Block = collections.namedtuple("Block", ["x", "y", "width", "height",
                                         "angle", "shape"])
# Default values for backwards compatibility: angle (0.) and shape (box).
Block.__new__.__defaults__ = (0., unity_constants.BOX_SHAPE)


BlocksObservation = collections.namedtuple(
    "BlocksObservation",
    ["blocks", "obstacles", "targets", "balls"])


def transform_block(block, scale, translation):
  """Transform a block by a specified scale and translation.

  This scales BOTH the width/height as well as the x and y positions, and THEN
  performs the translation.

  Args:
    block: Block object
    scale: a tuple/list of length two, corresponding to the scale of the x and
      y dimensions
    translation: a tuple/list of length two, corresponding to the translation
      in the x and y dimensions

  Returns:
    block: a scaled Block object
  """
  if block is None:
    return None
  if block.x is None:
    x = None
  else:
    x = (block.x * scale[0]) + translation[0]
  if block.y is None:
    y = None
  else:
    y = (block.y * scale[1]) + translation[1]
  width = block.width * scale[0]
  height = block.height * scale[1]

  return block._replace(x=x, y=y, width=width, height=height)


def transform_blocks_observation(observation, scale, translation):
  """Scale and translate a blocks observation by a specified amount.

  This scales BOTH the width/height as well as the x and y positions, and THEN
  performs the translation.

  Args:
    observation: a BlocksObservation object
    scale: a tuple/list of length two, corresponding to the scale of the x and
      y dimensions
    translation: a tuple/list of length two, corresponding to the translation
      in the x and y dimensions

  Returns:
    observation: a scaled BlocksObservation object
  """
  transform = lambda b: transform_block(b, scale, translation)
  return BlocksObservation(
      [transform(b) for b in observation.blocks],
      [transform(b) for b in observation.obstacles],
      [transform(b) for b in observation.targets],
      [transform(b) for b in observation.balls])
