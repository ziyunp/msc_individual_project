import pandas as pd
import numpy as np
import logging
import os
import utils.column_names as cn
import utils.config as config
import utils.helpers as hp
import utils.tree as tree
import utils.make_map as mm
from tqdm import tqdm
from statistics import mean
from math import sin, atan, pi, radians
import evaluation as ev
import sys
import pickle

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def make_lines(df):
    """
    Takes in a dataframe with the lat/lon points in Event_Lat and Event_Long columns
    Returns a list of the lines for the modified line Hausdorff distance function
    :param df: dataframe with lat/lon points in separate columns
    :return: list of lines with { "p1": point1, "p2": point2, "m": gradient, "c": y-intercept, "len": length, "midpoint": midpoint, "dttm": dttm } representing a line
    """
    lines = []
    df_coords = df[[cn.EVENT_LAT, cn.EVENT_LONG]].drop_duplicates()
    df_filtered = df[df.index.isin(df_coords.index)]
    for i in range(len(df_coords) - 1):
      point1 = df_coords.iloc[i]
      point2 = df_coords.iloc[i + 1]
      dttm1 = df_filtered.iloc[i][cn.EVENT_DTTM]
      dttm2 = df_filtered.iloc[i + 1][cn.EVENT_DTTM]
      road1 = df_filtered.iloc[i][cn.ROAD_NAME]
      road2 = df_filtered.iloc[i + 1][cn.ROAD_NAME]
      avg_dttm = dttm1 + (dttm2 - dttm1) / 2
      line = hp.construct_line(point1, point2, avg_dttm, road1, road2)
      lines.append(line)
    return np.asarray(lines)

def angle_distance_btw_two_lines(lm, ln):
  bearing_m = hp.bearing(lm)
  bearing_n = hp.bearing(ln)
  angle = bearing_m - bearing_n if bearing_m >= bearing_n else bearing_n - bearing_m
  # smallest angle
  if angle > 180:
    angle = 360 - angle
  if angle > 90:
    angle = (360 - 2 * angle) / 2
  assert angle >= 0
  return min(lm["len"], ln["len"]) * sin(radians(angle))

def perpendicular_distance_btw_two_lines(lm, ln):
  lm_length = lm["len"]
  ln_length = ln["len"]
  if ln_length >= lm_length:
    return hp.get_perpendicular_distance(lm, ln)
  return hp.get_perpendicular_distance(ln, lm)

def parallel_distance_btw_two_lines(lm, ln):
  if ln["len"] >= lm["len"]:
    return hp.get_parallel_distance(lm, ln)
  return hp.get_parallel_distance(ln, lm)

def road_distance_btw_two_lines(lm, ln):
  if lm["road1"] == ln["road1"] and lm["road2"] == ln["road2"]:
    return 0
  if lm["road1"] != ln["road1"] and lm["road1"] != ln["road2"] and lm["road2"] != ln["road1"] and lm["road2"] != ln["road2"]:
    return lm["len"]
  return 0.5 * lm["len"]

def collective_angle_distance(lm, N_lines):
  """
    Calculates the angle distance between a line and the set of neighboring lines
    :param lm: line, N_lines: the set of neighboring lines of lm
    :return: angle distance between a line and the set of neighboring lines
  """
  total_angle_distance = 0
  for ln in N_lines:
    total_angle_distance += angle_distance_btw_two_lines(lm, ln)
  return total_angle_distance

def collective_perpendicular_distance(lm, N_lines):
  """
    Calculates the perpendicular distance between a line and the set of neighboring lines
    :param lm: line, N_lines: the set of neighboring lines of lm
    :return: perpendicular distance between a line and the set of neighboring lines
  """
  max_perp_distance = 0
  for ln in N_lines:
    max_perp_distance = max(max_perp_distance, perpendicular_distance_btw_two_lines(lm, ln))
  return max_perp_distance

def collective_parallel_distance(lm, N_lines):
  """
    Calculates the parallel distance between a line and the set of neighboring lines
    :param lm: line, N_lines: the set of neighboring lines of lm
    :return: parallel distance between a line and the set of neighboring lines
  """
  min_parallel = parallel_distance_btw_two_lines(lm, N_lines[0])
  for ln in N_lines:
    parallel_dist = parallel_distance_btw_two_lines(lm, ln)
    min_parallel = min(min_parallel, parallel_dist)
  return min_parallel

def collective_compensation_distance(lm, N_lines):
  """
    Calculates the compensation distance between a line and the set of neighboring lines
    :param lm: line, N_lines: the set of neighboring lines of lm
    :return: compensation distance between a line and the set of neighboring lines
  """
  total_N_lengths = 0
  for ln in N_lines:
    total_N_lengths += ln["len"]
  diff = lm["len"] - total_N_lengths
  if diff < 0:
    return 0
  return diff

def collective_road_distance(lm, N_lines):
  total_road_distance = 0
  for ln in N_lines:
    total_road_distance += road_distance_btw_two_lines(lm, ln)
  return 1/len(N_lines) * total_road_distance

def within_neighborhood(lm, lines_N, tree_N):
  d_penalty = 0
  radius = 0.5 * lm["len"]
  index = tree.query_balltree_radius(tree_N, [lm["midpoint"]], radius)
  if len(index[0]) == 0:
    dist, index = tree.query_balltree_knn(tree_N, [lm["midpoint"]], 1, True)
    d_penalty = dist[0][0] * config.CONSTANTS["earth_radius"] - radius
  nearest_lines = []
  for i in index[0]:
    nearest_lines.append(lines_N[i])
  return nearest_lines, d_penalty

# def within_neighborhood_original(lm, lines_N, Rm):
#   m_length = lm["len"]
#   perp_distances = {}
#   for i in range(len(lines_N)):
#     perp_distances[i] = abs(perpendicular_distance_btw_two_lines(lm, lines_N[i]))
#   # sort by absolute perpendicular distance
#   sorted_perp_distances = sorted(perp_distances.items(), key=lambda x: x[1])
#   total_N_length = 0
#   n_of_lines = 0
#   for d in sorted_perp_distances:
#     index = d[0]
#     perp_distance = d[1]
#     ln = lines_N[index]
#     length = ln["len"]
#     total_N_length += length
#     n_of_lines += 1
#     if n_of_lines > 1:
#       if perp_distance > 0.5 * Rm * m_length or total_N_length > m_length:
#         n_of_lines -= 1
#         break
#   filtered_indices = [x[0] for x in sorted_perp_distances[0:n_of_lines]]
#   return [lines_N[i] for i in filtered_indices]

def compute_MLHD(lines_M, lines_N, tree_N, make_map = False, u_idx="", v_idx=""): 
  # find Rm
  x_min = lines_M[0]["p1"][0]
  y_min = lines_M[0]["p1"][1]
  x_max = lines_M[-1]["p2"][0]
  y_max = lines_M[-1]["p2"][1]
  Rm = hp.distance_btw_two_points((x_min, y_min), (x_max, y_max)) / 2
  total_M_length = 0
  total_prod_of_length_distance = 0
  i = 0
  for lm in lines_M: 
    i += 1
    m_length = lm["len"]
    total_M_length += m_length
    N_neighbors, d_penalty = within_neighborhood(lm, lines_N, tree_N)
    # N_neighbors = within_neighborhood_original(lm, lines_N, Rm)
    assert d_penalty >= 0
    d_penalty = min(d_penalty, lm["len"])
    d_angle = min(collective_angle_distance(lm, N_neighbors), m_length)
    assert d_angle >= 0
    d_perp = min(collective_perpendicular_distance(lm, N_neighbors), m_length)
    assert d_perp >= 0
    d_comp = min(collective_compensation_distance(lm, N_neighbors), m_length)
    assert d_comp >= 0
    d_road = min(collective_road_distance(lm, N_neighbors), m_length)
    assert d_road >= 0
    # d_parallel = collective_parallel_distance(lm, N_neighbors)
    # assert d_parallel >= 0
    distance = d_angle + d_perp + d_comp + d_penalty + d_road
    if make_map:
      map_file_name = u_idx + "-" + v_idx + "_" + str(i) 
      mm.make_map_with_line_segments([lm], N_neighbors, True, True, map_file_name, str(distance))
    total_prod_of_length_distance += m_length * distance
  return 1/total_M_length * total_prod_of_length_distance

def make_hausdorff_matrix(df, saved=False):
  leg_ids = df[cn.LEG_ID].unique()
  n = len(leg_ids)
  distances = np.zeros((n, n))
  labels = np.zeros(n)
  # for every leg, make lines and create a ball tree
  filename = 'trees_and_lines_' + leg_ids[0] + '.dictionary'
  if not saved:
    trees_and_lines = {}
    for id in leg_ids:
      data = df[df[cn.LEG_ID] == id]
      lines = make_lines(data)
      midpoints = [line["midpoint"] for line in lines]
      ball_tree = tree.construct_balltree(midpoints)
      trees_and_lines[id] = { "lines": lines, "tree": ball_tree }
    with open(filename, 'wb') as trees_and_lines_file:
      pickle.dump(trees_and_lines, trees_and_lines_file)
  else:
    with open(filename, 'rb') as trees_and_lines_file:
      trees_and_lines = pickle.load(trees_and_lines_file)
 
  for r in range(n):
    u_id = leg_ids[r]
    data = df[df[cn.LEG_ID] == u_id]
    labels[r] = data[cn.CLUSTER].unique()
    for c in range(r + 1, n):
      make_map = False
      v_id = leg_ids[c]
      u = trees_and_lines[u_id]["lines"]
      v = trees_and_lines[v_id]["lines"]
      u_tree = trees_and_lines[u_id]["tree"]
      v_tree = trees_and_lines[v_id]["tree"]
      distances[r, c] = max(compute_MLHD(u, v, v_tree), compute_MLHD(v, u, u_tree))
      distances[c, r] = distances[r, c]
      # Plot a full map
      if make_map:
        map_file_name = "Full_" + str(r) + "-" + str(c)
        mm.make_map_with_line_segments(u, v, True, True, map_file_name, str(distances[r, c]))
  return distances, labels
