import utils.column_names as cn
import utils.config as config
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import logging
import os
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import sys

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

"""
  TODO:
  1. Group pings by legs
  2. Get regular hausdorff distance
  3. Run plot_kdist for pings in one leg
  4. Look for modification to use MLHD
"""

def make_points(df):
    """
    Takes in a dataframe with the lat/lon points in Event_Lat and Event_Long columns
    Returns a list of the zipped lat/lon for the directed Hausdorff distance function
    :param df: dataframe with lat/lon points in separate columns
    :return: list of np.arrays with (lat, lon) inside
    """
    return [np.array(item) for item in zip(df[cn.EVENT_LAT], df[cn.EVENT_LONG])]

def make_hausdorff_matrix(df, symm=False):
  cache = {}
  leg_ids = df[cn.LEG_ID].unique()
  n = len(leg_ids)
  distances = np.zeros((n, n))
  for r in range(n):
    for c in range(n):
      if (c in cache) and (r in cache[c]):
        distances[r, c] = cache[c][r]
      else:
        u_id = leg_ids[r]
        v_id = leg_ids[c]
        u = df[df[cn.LEG_ID] == u_id]
        v = df[df[cn.LEG_ID] == v_id]
        u = make_points(u)
        v = make_points(v)
        if symm:
            distances[r, c] = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        else:
            distances[r, c] = directed_hausdorff(u, v)[0]
        if r != c:
          cache[r] = { c: distances[r, c] }
  return distances    


def main():
  data_file = config.PATHS["labelled_data"]# file stored locally
  log.info("Reading in data with leg_ids from  from {}".format(data_file))
  df = pd.read_csv(data_file, parse_dates=[cn.EVENT_DTTM])

  log.info("Getting unique from/to RM site combinations that don't arrive and depart at the same place")
  df_from_to = df[[cn.FROM_DEPOT, cn.TO_DEPOT]].drop_duplicates()
  df_from_to_no_loop = df_from_to[df_from_to[cn.FROM_DEPOT] != df_from_to[cn.TO_DEPOT]]
  df_from_to_no_loop.dropna(inplace=True)
  log.info("There are {} unique from/to combinations".format(len(df_from_to_no_loop)))

  log.info("Clustering routes between each pair of sites")

  count = 1
  for row in tqdm(df_from_to_no_loop.itertuples(), total=df_from_to_no_loop.shape[0]):
      df_sub = df[(df.from_depot == row.from_depot) & (df.to_depot == row.to_depot)]
      fig = "Fig" + str(count) + ".png"
      count += 1
      make_hausdorff_matrix(df_sub, True)

if __name__ == "__main__":
  main()
