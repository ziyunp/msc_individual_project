import pandas as pd
import numpy as np
import logging
import os
import utils.column_names as cn
import utils.config as config
import utils.line_functions as lf
from tqdm import tqdm
from statistics import mean
import math
import evaluation as ev
import sys

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
      point1 = (point1[cn.EVENT_LAT], point1[cn.EVENT_LONG])
      point2 = (point2[cn.EVENT_LAT], point2[cn.EVENT_LONG])
      line = lf.construct_line(point1, point2)
      lines.append(line)
    return lines

def angle_distance_btw_two_lines(lm, ln):
  """
    Calculates the angle distance between two lines.
    The smallest positive intersecting angle between the two lines is calculated by using the gradient of the two lines.
    :param lm: line1, ln: line2
    :return: angle distance between two lines
  """
  m1 = lm["m"]
  m2 = ln["m"]
  angle = math.atan(abs((m2 - m1)/(1 + m1 * m2)))
  return min(lm["len"], ln["len"]) * math.sin(angle)

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

def within_neighborhood(lm, Rm, lines_N):
  m_length = lm["len"]
  perp_distances = {}
  for i in range(len(lines_N)):
    perp_distances[i] = abs(perpendicular_distance_btw_two_lines(lm, lines_N[i]))
  # sort by absolute perpendicular distance
  sorted_perp_distances = sorted(perp_distances.items(), key=lambda x: x[1])
  total_N_length = 0
  n_of_lines = 0
  for d in sorted_perp_distances:
    index = d[0]
    perp_distance = d[1]
    ln = lines_N[index]
    length = ln["len"]
    total_N_length += length
    n_of_lines += 1
    if n_of_lines > 1:
      if perp_distance > 0.5 * Rm * m_length:
        n_of_lines -= 1
        break
  filtered_indices = [x[0] for x in sorted_perp_distances[0:n_of_lines]]
  return [lines_N[i] for i in filtered_indices]

def compute_MLHD(lines_M, lines_N): 
  xm = [m["p1"][0] for m in lines_M] + [m["p2"][0] for m in lines_M]
  ym = [m["p1"][1] for m in lines_M] + [m["p2"][1] for m in lines_M]
  # find Rm
  x_min = min(xm)
  x_max = max(xm)
  y_min = min(ym)
  y_max = max(ym)
  Rm = lf.distance_btw_two_points((x_min, y_min), (x_max, y_max)) / 2
  total_M_length = 0
  total_prod_of_length_distance = 0
  for lm in lines_M: 
    m_length = lm["len"]
    total_M_length += m_length
    N_neighbors = within_neighborhood(lm, Rm, lines_N)
    d_angle = collective_angle_distance(lm, N_neighbors)
    d_perp = collective_perpendicular_distance(lm, N_neighbors)
    d_parallel = collective_parallel_distance(lm, N_neighbors)
    d_comp = collective_compensation_distance(lm, N_neighbors)
    distance = d_angle + d_perp + d_parallel + d_comp
    total_prod_of_length_distance += m_length * distance
  return 1/Rm * 1/total_M_length * total_prod_of_length_distance
  
def make_MLHD_matrix(df, symm=False):
  leg_ids = df[cn.LEG_ID].unique()
  n = len(leg_ids)
  distances = np.zeros((n, n))
  labels = np.zeros(n)
  for r in range(n):
    data = df[df[cn.LEG_ID] == leg_ids[r]]
    labels[r] = data[cn.CLUSTER].unique()
    for c in range(r+1, n):
      u_id = leg_ids[r]
      v_id = leg_ids[c]
      u = df[df[cn.LEG_ID] == u_id]
      v = df[df[cn.LEG_ID] == v_id]
      u = make_lines(u)
      v = make_lines(v)
      distances[r, c] = compute_MLHD(u, v)
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
