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

from absl import flags
from absl.testing import absltest
from dm_construction.utils import geometry

FLAGS = flags.FLAGS


class GeometryTest(absltest.TestCase):

  def test_area(self):
    # Rectangular triangle
    self.assertAlmostEqual(geometry._area(0, 0, 0, 1, 1, 0), 0.5)
    # Translate it alongside the x-axis
    self.assertAlmostEqual(geometry._area(10, 0, 10, 1, 11, 0), 0.5)
    # Translate it alongside the y-axis
    self.assertAlmostEqual(geometry._area(0, 10, 0, 11, 1, 10), 0.5)
    # Rotate it
    self.assertAlmostEqual(geometry._area(0, 0, 1, 0, 0.5, 1), 0.5)
    # Non-rectangular anymore
    self.assertAlmostEqual(geometry._area(0, 0, 2, 0, 0.5, 1), 1.)

  def test_rotation(self):
    # No rotation
    x, y = geometry.rotate_rectangle_corner(2, -1, 0, 0, 1., 0.)
    self.assertAlmostEqual(x, 2)
    self.assertAlmostEqual(y, -1)
    # 90 degrees
    x, y = geometry.rotate_rectangle_corner(2, -1, 0, 0, 0., 1.)
    self.assertAlmostEqual(x, 1)
    self.assertAlmostEqual(y, 2)

  def test_is_point_in_rectangle(self):
    self.assertTrue(
        geometry.is_point_in_rectangle(-1, -1, -1, 1, 1, 1, 1, -1, 0, 0))
    # Just on the boundary
    self.assertTrue(
        geometry.is_point_in_rectangle(-1, -1, -1, 1, 1, 1, 1, -1, 1, 1))
    # Outside
    self.assertFalse(
        geometry.is_point_in_rectangle(-1, -1, -1, 1, 1, 1, 1, -1, 1, 1.1))


if __name__ == "__main__":
  absltest.main()
