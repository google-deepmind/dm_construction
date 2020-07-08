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
"""Geometry utility functions."""

import numpy as np
import shapely.affinity
import shapely.geometry


def _area(x1, y1, x2, y2, x3, y3):
  """Heron's formula."""
  a = np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
  b = np.sqrt(pow(x3 - x2, 2) + pow(y3 - y2, 2))
  c = np.sqrt(pow(x1 - x3, 2) + pow(y3 - y1, 2))
  s = (a + b + c) / 2
  return np.sqrt(s * (s - a) * (s - b) * (s - c))


def is_point_in_rectangle(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
  """Whether (x, y) is within the rectange defined by (x1, y1), .. (x4, y4)."""
  # Calculate area of rectangle ABCD
  rect_area = (_area(x1, y1, x2, y2, x3, y3) + _area(x1, y1, x4, y4, x3, y3))
  # Areas of each 4 triangles.
  area_1 = _area(x, y, x1, y1, x2, y2)  # Calculate area of triangle PAB
  area_2 = _area(x, y, x2, y2, x3, y3)  # Calculate area of triangle PBC
  area_3 = _area(x, y, x3, y3, x4, y4)  # Calculate area of triangle PCD
  area_4 = _area(x, y, x1, y1, x4, y4)  # Calculate area of triangle PAD
  # Check if the sum of triangle areas is the same as the rectange area.
  epsilon = rect_area * 0.001
  return rect_area >= area_1 + area_2 + area_3 + area_4 - epsilon


def rotate_rectangle_corner(x, y, center_x, center_y, cos_theta, sin_theta):
  temp_x = x - center_x
  temp_y = y - center_y
  rotated_x = temp_x * cos_theta - temp_y * sin_theta
  rotated_y = temp_x * sin_theta + temp_y * cos_theta
  x = rotated_x + center_x
  y = rotated_y + center_y
  return (x, y)


def rect_projected_width(width, height, angle):
  """Get the x projection of an angle-rotated rect. with sides width, height."""
  # Angle between floor and first diag.
  first_angle = np.arctan2(height, width)
  # Angle between floor and second diagonal
  second_angle = np.pi - first_angle
  diagonal_length = np.sqrt(height * height + width * width)
  first_angle += angle  # Rotate the cube.
  second_angle += angle
  projected_width = np.max(
      [np.abs(np.cos(first_angle)), np.abs(np.cos(second_angle))])
  return diagonal_length * projected_width


def rect_projected_height(width, height, angle):
  """Get the y projection of an angle-rotated rect. with sides width, height."""
  return rect_projected_width(width, height, angle + np.pi/2.)


def rect_bounding_frame(x, y, w, h, angle):
  """Returns the bounding frame with x_end > x_begin and y_end > y_begin."""
  projected_width = rect_projected_width(w, h, angle)
  projected_height = rect_projected_height(w, h, angle)
  block_x_begin, block_x_end = x - projected_width/2., x + projected_width/2.
  block_y_begin, block_y_end = y - projected_height/2., y + projected_height/2.
  return block_x_begin, block_x_end, block_y_begin, block_y_end


def circle_bounding_frame(x, y, w):
  """Returns the bounding frame with x_end > x_begin and y_end > y_begin."""
  r = w/2
  return x-r, x+r, y-r, y+r


def bounding_circles_overlap(x0, y0, w0, x1, y1, w1):
  center_distance = np.linalg.norm([x1-x0, y1-y0])
  return center_distance < w0/2 + w1/2


def bounding_box_overlap(x0_begin, x0_end, y0_begin, y0_end,
                         x1_begin, x1_end, y1_begin, y1_end):
  if (x1_begin > x0_end or x0_begin > x1_end or
      y1_begin > y0_end or y0_begin > y1_end):
    return False
  return True


def rect_overlap(x0, y0, w0, h0, angle0, x1, y1, w1, h1, angle1):
  """Calculates the overlap area between two rectangles."""

  # Check if bounding spheres do not intersect.
  dw0 = _rect_diagonal(w0, h0)
  dw1 = _rect_diagonal(w1, h1)
  if not bounding_circles_overlap(x0, y0, dw0, x1, y1, dw1):
    return 0.

  # Check if bounding boxes do not intersect.
  x0_begin, x0_end, y0_begin, y0_end = rect_bounding_frame(
      x0, y0, w0, h0, angle0)
  x1_begin, x1_end, y1_begin, y1_end = rect_bounding_frame(
      x1, y1, w1, h1, angle1)
  if not bounding_box_overlap(x0_begin, x0_end, y0_begin, y0_end,
                              x1_begin, x1_end, y1_begin, y1_end):
    return 0.

  # Otherwise, calculate proper intersection.
  rect_1 = _build_shapely_rectangle(x0, y0, w0, h0, angle0)
  rect_2 = _build_shapely_rectangle(x1, y1, w1, h1, angle1)
  return rect_1.intersection(rect_2).area


def ramp_overlap(x0, y0, w0, h0, angle0, x1, y1, w1, h1, angle1):
  """Calculates the overlap area between two ramps."""

  # Check if bounding spheres do not intersect.
  dw0 = _rect_diagonal(w0, h0)
  dw1 = _rect_diagonal(w1, h1)
  if not bounding_circles_overlap(x0, y0, dw0, x1, y1, dw1):
    return 0.

  # Check if bounging boxes do not intersect.
  x0_begin, x0_end, y0_begin, y0_end = rect_bounding_frame(
      x0, y0, w0, h0, angle0)
  x1_begin, x1_end, y1_begin, y1_end = rect_bounding_frame(
      x1, y1, w1, h1, angle1)
  if not bounding_box_overlap(x0_begin, x0_end, y0_begin, y0_end,
                              x1_begin, x1_end, y1_begin, y1_end):
    return 0.

  # Otherwise, calculate proper intersection.
  rect_1 = _build_shapely_ramp(x0, y0, w0, h0, angle0)
  rect_2 = _build_shapely_ramp(x1, y1, w1, h1, angle1)
  return rect_1.intersection(rect_2).area


def rect_ramp_overlap(x0, y0, w0, h0, angle0, x1, y1, w1, h1, angle1):
  """Calculates the overlap area between two rectangles."""

  # Check if bounding spheres do not intersect.
  dw0 = _rect_diagonal(w0, h0)
  dw1 = _rect_diagonal(w1, h1)
  if not bounding_circles_overlap(x0, y0, dw0, x1, y1, dw1):
    return 0.

  # Check if bounging boxes do not intersect.
  x0_begin, x0_end, y0_begin, y0_end = rect_bounding_frame(
      x0, y0, w0, h0, angle0)
  x1_begin, x1_end, y1_begin, y1_end = rect_bounding_frame(
      x1, y1, w1, h1, angle1)
  if not bounding_box_overlap(x0_begin, x0_end, y0_begin, y0_end,
                              x1_begin, x1_end, y1_begin, y1_end):
    return 0.

  # Otherwise, calculate proper intersection.
  rect_1 = _build_shapely_rectangle(x0, y0, w0, h0, angle0)
  rect_2 = _build_shapely_ramp(x1, y1, w1, h1, angle1)
  return rect_1.intersection(rect_2).area


def rect_ball_overlap(x0, y0, w0, h0, angle0, x1, y1, w1):
  """Calculates the overlap area between a rectangles and a ball."""

  # Check if bounding spheres do not intersect.
  dw0 = _rect_diagonal(w0, h0)
  if not bounding_circles_overlap(x0, y0, dw0, x1, y1, w1):
    return 0.

  # Check if bounging boxes do not intersect.
  x0_begin, x0_end, y0_begin, y0_end = rect_bounding_frame(
      x0, y0, w0, h0, angle0)
  x1_begin, x1_end, y1_begin, y1_end = circle_bounding_frame(
      x1, y1, w1)
  if not bounding_box_overlap(x0_begin, x0_end, y0_begin, y0_end,
                              x1_begin, x1_end, y1_begin, y1_end):
    return 0.

  # Otherwise, calculate proper intersection.
  rect = _build_shapely_rectangle(x0, y0, w0, h0, angle0)
  circle = _build_shapely_circle(x1, y1, w1)
  return rect.intersection(circle).area


def ramp_ball_overlap(x0, y0, w0, h0, angle0, x1, y1, w1):
  """Calculates the overlap area between a ramp and a ball."""

  # Check if bounding spheres do not intersect.
  dw0 = _rect_diagonal(w0, h0)
  if not bounding_circles_overlap(x0, y0, dw0, x1, y1, w1):
    return 0.

  # Check if bounding boxes do not intersect.
  x0_begin, x0_end, y0_begin, y0_end = rect_bounding_frame(
      x0, y0, w0, h0, angle0)
  x1_begin, x1_end, y1_begin, y1_end = circle_bounding_frame(
      x1, y1, w1)
  if not bounding_box_overlap(x0_begin, x0_end, y0_begin, y0_end,
                              x1_begin, x1_end, y1_begin, y1_end):
    return 0.

  # Otherwise, calculate proper intersection.
  rect = _build_shapely_ramp(x0, y0, w0, h0, angle0)
  circle = _build_shapely_circle(x1, y1, w1)
  return rect.intersection(circle).area


def _rect_diagonal(w, h):
  """Calculates the radius of a rectangle."""
  return np.sqrt(w**2 + h**2)


def _build_shapely_rectangle(x, y, w, h, angle):
  """Creates a shapely object representing a rectangle."""
  centered = shapely.geometry.box(-w/2, -h/2, w/2, h/2)
  rotated = shapely.affinity.rotate(centered, angle/np.pi*180)
  return shapely.affinity.translate(rotated, x, y)


def _build_shapely_ramp(x, y, w, h, angle):
  """Creates a shapely object representing a ramp."""
  centered = shapely.geometry.Polygon([(-w/2, -h/2), (-w/2, h/2), (w/2, -h/2)])
  rotated = shapely.affinity.rotate(centered, angle/np.pi*180)
  return shapely.affinity.translate(rotated, x, y)


def _build_shapely_circle(x, y, w):
  """Creates a shapely object representing a rectangle."""
  return shapely.geometry.Point(x, y).buffer(w/2)
