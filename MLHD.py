import pandas as pd
import numpy as np
import logging
import os
import utils.column_names as cn
import utils.config as config
import utils.line_functions as lf
import utils.tree as tree
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
    :return: list of lines with { "p1": (x1, y1), "p2": (x2, y2), "m": m, "c": c } representing a line
    """
    lines = []
    df = df[[cn.EVENT_LAT, cn.EVENT_LONG]].drop_duplicates()
    for i in range(len(df) - 1):
      point1 = df.iloc[i]
      point2 = df.iloc[i + 1]
      line = lf.construct_line(point1, point2)
      lines.append(line)
    return np.asarray(lines)

def angle_distance_btw_two_lines_bearing(lm, ln):
  """
    Experimental: alternative to angle_distance_btw_two_lines
  """
  bearing_m = lf.bearing(lm)
  bearing_n = lf.bearing(ln)
  if bearing_m > bearing_n:
    angle = bearing_m - bearing_n
  else:
    angle = bearing_n - bearing_m
  return min(lm["len"], ln["len"]) * sin(radians(angle))

def angle_distance_btw_two_lines(lm, ln):
  """
    Calculates the angle distance between two lines.
    The smallest positive intersecting angle between the two lines is calculated by using the gradient of the two lines.
    :param lm: line1, ln: line2
    :return: angle distance between two lines
  """
  m1 = lm["m"]
  m2 = ln["m"]
  if m1 * m2 == -1:
    # perpendicular line
    angle = pi / 2
  else: 
    angle = atan(abs((m2 - m1)/(1 + m1 * m2)))
  return min(lm["len"], ln["len"]) * sin(angle)

def perpendicular_distance_btw_two_lines(lm, ln):
  lm_length = lm["len"]
  ln_length = ln["len"]
  if ln_length >= lm_length:
    return lf.get_perpendicular_distance(lm, ln)
  perp_distance = lf.get_perpendicular_distance(ln, lm)
  return (ln_length / lm_length) * perp_distance

def parallel_distance_btw_two_lines(lm, ln):
  if ln["len"] >= lm["len"]:
    return lf.get_parallel_distance(lm, ln)
  return lf.get_parallel_distance(ln, lm)

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
  total_perp_distance = 0
  for ln in N_lines:
    total_perp_distance += perpendicular_distance_btw_two_lines(lm, ln)
  return total_perp_distance

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

def within_neighborhood(lm, lines_N, tree_N):
  index = tree_N.query_radius([lm["midpoint"]], r=lm["len"])
  if len(index[0]) == 0:
    index = tree_N.query([lm["midpoint"]], return_distance=False, k=1)
  nearest_lines = []
  for i in index[0]:
    nearest_lines.append(lines_N[i])
  return nearest_lines

def compute_MLHD(lines_M, lines_N, tree_N): 
  xm = [m["p1"][0] for m in lines_M] + [m["p2"][0] for m in lines_M]
  ym = [m["p1"][1] for m in lines_M] + [m["p2"][1] for m in lines_M]
  total_M_length = 0
  total_prod_of_length_distance = 0
  for lm in lines_M: 
    m_length = lm["len"]
    total_M_length += m_length
    N_neighbors = within_neighborhood(lm, lines_N, tree_N)
    d_angle = collective_angle_distance(lm, N_neighbors)
    d_perp = collective_perpendicular_distance(lm, N_neighbors)
    d_parallel = collective_parallel_distance(lm, N_neighbors)
    distance = d_angle + d_perp + d_parallel
    total_prod_of_length_distance += m_length * distance
  return 1/total_M_length * total_prod_of_length_distance
  
def make_MLHD_matrix(df, saved=False):
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
      ball_tree = tree.construct_balltree(lines)
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
    for c in range(r+1, n):
      v_id = leg_ids[c]
      u = trees_and_lines[u_id]["lines"]
      v = trees_and_lines[v_id]["lines"]
      v_tree = trees_and_lines[v_id]["tree"]
      distances[r, c] = compute_MLHD(u, v, v_tree)
      distances[c, r] = distances[r, c]
  return distances, labels

def main():
  data_file = config.FILENAMES["all_data_cleaned"]
  log.info("Reading in data with leg_ids from  from {}".format(data_file))

  df = pd.read_csv(data_file, parse_dates=[cn.EVENT_DTTM])

  log.info("Getting unique from/to RM site combinations that don't arrive and depart at the same place")
  df_from_to = df[[cn.FROM_DEPOT, cn.TO_DEPOT]].drop_duplicates()
  df_from_to_no_loop = df_from_to[df_from_to[cn.FROM_DEPOT] != df_from_to[cn.TO_DEPOT]]
  df_from_to_no_loop.dropna(inplace=True)
  log.info("There are {} unique from/to combinations".format(len(df_from_to_no_loop)))

  log.info("Clustering routes between each pair of sites")

  i = 0
  for row in tqdm(df_from_to_no_loop.itertuples(), total=df_from_to_no_loop.shape[0]):
    i += 1
    df_sub = df[(df.from_depot == row.from_depot) & (df.to_depot == row.to_depot)]
    distance_matrix, labels = make_MLHD_matrix(df_sub, True)
    save_file = "distance_matrix_" + str(i) + ".csv"
    np.savetxt(save_file, distance_matrix, delimiter=",")
    silhouette = ev.silhouette_score(labels, distance_matrix)
    print(row.from_depot, "-", row.to_depot, ": ", silhouette)

if __name__ == "__main__":
  main()
