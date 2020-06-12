import math

def midpoint(line):
  x = (line["p1"][0] + line["p2"][0]) / 2
  y = (line["p1"][1] + line["p2"][1]) / 2
  return (x, y)

def distance_between_two_points(p1, p2):
  return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def construct_line(point1, point2, m, c):
  return { "p1": point1, "p2": point2, "m": m, "c": c }

def get_perpendicular_distance(shorter_line, longer_line):
  """
    Rotate shorter_line around the midpoint so that it is parallel to the longer_line, and get the perpendicular distance between rotated lines
    1. get midpoint and perpendicular gradient
    2. construct a line with the midpoint and the intersection point on ln 
    3. get length of the line
  """
  parallel_m = longer_line["m"]
  c_longer = longer_line["c"]
  mid_x, mid_y = midpoint(shorter_line)
  perp_m = -1 / parallel_m 
  c_perp = mid_y - perp_m * mid_x
  intersecting_x = (c_perp - c_longer) / (parallel_m - perp_m)
  intersecting_y = perp_m * intersecting_x + c_perp
  perp_distance = distance_between_two_points((mid_x, mid_y), (intersecting_x, intersecting_y))
