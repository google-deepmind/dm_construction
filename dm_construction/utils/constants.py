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
"""Constant values."""

# Available block sizes.
SMALL_WIDTH = 10
MEDIUM_WIDTH = 30
LARGE_WIDTH = 50

# Types of objects in the ConstructionStacking base environment.
BLOCK = "Blocks"
AVAILABLE_BLOCK = "AvailableBlocks"
OBSTACLE = "Obstacles"
TARGET = "Targets"
BALL = "Balls"

# ConstructionStacking termination types.
TERMINATION_MAX_STEPS = "max_steps"
TERMINATION_SPAWN_COLLISION = "spawn_collision"
TERMINATION_OBSTACLE_HIT = "obstacle_hit"
TERMINATION_COMPLETE = "complete"
TERMINATION_BAD_SIMULATION = "bad_simulation"
TERMINATION_BAD_CHOICE = "bad_choice"
# Termination types of the DiscreteRelativeGraphWrapper.
TERMINATION_INVALID_EDGE = "invalid_edge"
