from math import sin, cos, asin, sqrt, radians, atan2, pi, degrees
import utils.config as config

OFFSET = 1e-15

def get_midpoint(p1, p2):
  x = (p1[0] + p2[0]) / 2
  y = (p1[1] + p2[1]) / 2
  return (x, y)

def distance_btw_two_points(p1, p2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    x1, y1 = convert_coords_to_radians(p1)
    x2, y2 = convert_coords_to_radians(p2)
    # haversine formula 
    dy = y2 - y1 
    dx = x2 - x1 
    a = sin(dx/2)**2 + cos(x1) * cos(x2) * sin(dy/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * config.CONSTANTS["earth_radius"] # in km

def construct_line(point1, point2, dttm, road1, road2):
  length = distance_btw_two_points(point1, point2)
  x1, y1 = point1
  x2, y2 = point2
  diff_x = x2 - x1
  if diff_x == 0:
    diff_x = OFFSET
    x2 += OFFSET
  m = (y2 - y1) / diff_x
  c = y1 - m * x1
  midpoint = get_midpoint(point1, point2)
  return { "p1": point1, "p2": point2, "m": m, "c": c, "len": length, "midpoint": midpoint, "dttm": dttm, "road1": road1, "road2": road2 }

def get_perpendicular_distance(shorter_line, longer_line):
  """
    Rotate shorter_line around the midpoint so that it is parallel to the longer_line, and get the perpendicular distance between rotated lines
    1. get midpoint and perpendicular gradient
    2. construct a line with the midpoint and the intersection point on ln 
    3. get length of the line
  """
  midpoint = shorter_line["midpoint"]
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
  d = shorter_line["len"] / config.CONSTANTS["earth_radius"]
  m = longer_line["m"]
  mid_x, mid_y = shorter_line["midpoint"]
  c = mid_y - m * mid_x
  mid_x, mid_y, c = map(radians, [mid_x, mid_y, c])
  # find points at two ends = half the distance from midpoint
  x1 = mid_x - (d/2) / sqrt(1 + m**2)
  x2 = mid_x + (d/2) / sqrt(1 + m**2)
  y1 = m * x1 + c
  y2 = m * x2 + c
  x1, y1, x2, y2 = map(degrees, [x1, y1, x2, y2])

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
      left = point2
      right = point1
  return left, right

def bearing(line):
  x1, y1 = convert_coords_to_radians(line["p1"])
  x2, y2 = convert_coords_to_radians(line["p2"])
  y = sin(y2-y1) * cos(x2)
  x = cos(x1)*sin(x2) - sin(x1)*cos(x2)*cos(y2-y1)
  angle = atan2(y, x)
  return degrees(angle)

def convert_coords_to_radians(coord):
  x, y = coord
  x_rad, y_rad = map(radians, [x, y])
  return (x_rad, y_rad)

def longest_common_subsequnce(M , N): 
	m = len(M) 
	n = len(N)
	# bottom-up dynamic programming method
	L = [[None] * (n+1) for i in range(m + 1)] 
	for i in range(m + 1): 
		for j in range(n + 1): 
			if i == 0 or j == 0: 
				L[i][j] = 0
			elif M[i-1] == N[j-1]: 
				L[i][j] = L[i-1][j-1]+1
			else: 
				L[i][j] = max(L[i-1][j] , L[i][j-1]) 
	return L[m][n] 