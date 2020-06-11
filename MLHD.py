import pandas as pd
import numpy as np
import logging
import os
import utils.column_names as cn
import utils.config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mean

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

PING_INTERVAL = 5

def find_best_fit(xs, ys):
  # find slope
  m = (mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs) * mean(xs) - mean(xs * xs))
  # find intercept
  c = mean(ys) - m * mean(xs)
  
  x = []
  y = []
  # Take first and last points t c
  x1 = xs[0]
  x2 = xs[-1]
  y1 = m * xs[0] + c
  y2 = m * xs[-1] + c
  line = { "p1": (x1, y1), "p2": (x2, y2), "m": m, "c": c }
  return line

def make_lines(df):
    """
    Takes in a dataframe with the lat/lon points in Event_Lat and Event_Long columns
    Returns a list of the lines for the modified line Hausdorff distance function
    :param df: dataframe with lat/lon points in separate columns
    :return: list of lines with { "p1": (x1, y1), "p2": (x2, y2), "m": m, "c": c } representing a line
    """
    lines = []
    df = df[[cn.LEG_ID, cn.EVENT_LAT, cn.EVENT_LONG, cn.EVENT_DTTM]]
    
    # for every 5 min interval, get the set of points and construct a line
    start_time = df.iloc[0][cn.EVENT_DTTM]
    end_time = df.iloc[-1][cn.EVENT_DTTM]
    prev_point = df.iloc[0][cn.EVENT_DTTM]

    while start_time < end_time:
      last_ping = pd.to_datetime(start_time + pd.Timedelta(PING_INTERVAL, "minute"))
      
      lower_bound = prev_point if prev_point < start_time else start_time
      points = df[(df[cn.EVENT_DTTM] >= lower_bound) & (df[cn.EVENT_DTTM] <= last_ping)]

      if len(points) < 2:
        prev_point = start_time
        start_time = last_ping
      else:
        start_time = points.iloc[-1][cn.EVENT_DTTM]
        prev_point = start_time
        xs = points[cn.EVENT_LAT].to_numpy()
        ys = points[cn.EVENT_LONG].to_numpy()

        line = find_best_fit(xs, ys)
        lines.append(line)
        # x1, y1 = line["p1"]
        # x2, y2 = line["p2"]
        # x = [ x1, x2 ]
        # y = [ y1, y2 ]
        # plt.scatter(xs, ys)
        # plt.plot(x, y)
        # plt.show() 
    return lines

def make_MLHD_matrix(df, symm=False):
  leg_ids = df[cn.LEG_ID].unique()
  n = len(leg_ids)
  distances = np.zeros((n, n))
  for r in range(n):
    for c in range(n):
      u_id = leg_ids[r]
      v_id = leg_ids[c]
      u = df[df[cn.LEG_ID] == u_id]
      v = df[df[cn.LEG_ID] == v_id]
      u = make_lines(u)
      v = make_lines(v)

      # if symm:
      #     distances[r, c] = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
      # else:
      #     distances[r, c] = directed_hausdorff(u, v)[0]
      # distances[c, r] = distances[r, c]
  return distances    

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

  for row in tqdm(df_from_to_no_loop.itertuples(), total=df_from_to_no_loop.shape[0]):
    row = df_from_to_no_loop.iloc[0]
    df_sub = df[(df.from_depot == row.from_depot) & (df.to_depot == row.to_depot)]
    make_MLHD_matrix(df_sub, True)


if __name__ == "__main__":
  main()
