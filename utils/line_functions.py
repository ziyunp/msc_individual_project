import math
import matplotlib.pyplot as plt

def get_midpoint(line):
  x = (line["p1"][0] + line["p2"][0]) / 2
  y = (line["p1"][1] + line["p2"][1]) / 2
  return (x, y)

def distance_btw_two_points(p1, p2):
  return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def construct_line(point1, point2, m, c):
  length = distance_btw_two_points(point1, point2)
  return { "p1": point1, "p2": point2, "m": m, "c": c, "len": length }

def get_perpendicular_distance(shorter_line, longer_line):
  """
    Rotate shorter_line around the midpoint so that it is parallel to the longer_line, and get the perpendicular distance between rotated lines
    1. get midpoint and perpendicular gradient
    2. construct a line with the midpoint and the intersection point on ln 
    3. get length of the line
  """
  midpoint = get_midpoint(shorter_line)
  intersecting_point = find_perpendicular_intersect(longer_line, midpoint)
  perp_distance = distance_btw_two_points(midpoint, intersecting_point)
  return perp_distance

def find_perpendicular_intersect(line, point):
  x, y = point
  m = line["m"]
  c = line["c"]
  if m == 0:
    # horizontal line
    x, y = point
    return (x, line["p1"][1])
  perp_m = -1 / m
  c_perp = y - perp_m * x
  # find intersection by equating the eq of longer_line and perp line
  x_intersect = (c_perp - c) / (m - perp_m)
  y_intersect = perp_m * x_intersect + c_perp
  return (x_intersect, y_intersect)

def get_parallel_distance(shorter_line, longer_line):
  d = shorter_line["len"]
  m = longer_line["m"]
  mid_x, mid_y = get_midpoint(shorter_line)
  c = mid_y - m * mid_x
  # find points at two ends = half the distance from midpoint
  x1 = mid_x - (d/2) / math.sqrt(1 + m**2)
  x2 = mid_x + (d/2) / math.sqrt(1 + m**2)
  y1 = m * x1 + c
  y2 = m * x2 + c
  left_shorter = (x1, y1)
  right_shorter = (x2, y2)

  # determine the left and right points on the longer line
  left_longer, right_longer = determine_left_and_right_ends(longer_line["p1"], longer_line["p2"])

  # find the intersection point on longer line from the ends of the shorter line
  left_intersect = find_perpendicular_intersect(longer_line, left_shorter)
  right_intersect = find_perpendicular_intersect(longer_line, right_shorter)
  # find parallel shift
  left_shift = distance_btw_two_points(left_intersect, left_longer)
  right_shift = distance_btw_two_points(right_intersect, right_longer)
  return min(left_shift, right_shift)

def determine_left_and_right_ends(point1, point2):
  x1 = point1[0]
  x2 = point2[0]

  if x1 < x2:
    left = point1
    right = point2
  elif x2 < x1:
    left = point2
    right = point1
  else:
    # vertical line
    y1 = point1[1]
    y2 = point2[1]
    if y1 < y2:
      left = point1
      right = point2
    else:
      right = point2
      left = point1
  return left, right


def plot_points(xs, ys, c=None):
  plt.scatter(xs, ys, color=c)

def plot_line(line, c=None):
  xs = [line["p1"][0]] + [line["p2"][0]]
  ys = [line["p1"][1]] + [line["p2"][1]]
  plt.plot(xs, ys, color=c)

def show_plot():
  plt.show()