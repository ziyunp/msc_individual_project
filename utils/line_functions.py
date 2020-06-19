from math import sin, cos, asin, sqrt, radians
import matplotlib.pyplot as plt

OFFSET = 1e-15

def get_midpoint(line):
  Bx = cos(x2) * cos(y2-y1)
  By = cos(x2) * sin(y2-y1)
  xm = atan2(sin(x1) + sin(x2), sqrt((cos(x1)+Bx)**2 + By**2))
  ym = y1 + atan2(By, cos(x1) + Bx)
  return (xm, ym)

def distance_btw_two_points(p1, p2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    x1, y1 = p1
    x2, y2 = p2
    # convert decimal degrees to radians 
    x1, y1, x2, y2 = map(radians, [x1, y1, x2, y2])
    # haversine formula 
    dy = y2 - y1 
    dx = x2 - x1 
    a = sin(dx/2)**2 + cos(x1) * cos(x2) * sin(dy/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def construct_line(point1, point2):
  length = distance_btw_two_points(point1, point2)
  x1, y1 = point1
  x2, y2 = point2
  diff_x = x2 - x1
  if diff_x == 0:
    diff_x = OFFSET
    x2 += OFFSET
  m = (y2 - y1) / diff_x
  c = y1 - m * x1
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
  x1 = mid_x - (d/2) / sqrt(1 + m**2)
  x2 = mid_x + (d/2) / sqrt(1 + m**2)
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