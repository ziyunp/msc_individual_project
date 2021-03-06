import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import logging
import os
import utils.column_names as cn

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def make_points(df):
    """
    Takes in a dataframe with the lat/lon points in Event_Lat and Event_Long columns
    Returns a list of the zipped lat/lon for the directed Hausdorff distance function
    :param df: dataframe with lat/lon points in separate columns
    :return: list of np.arrays with (lat, lon) inside
    """
    return [np.array(item) for item in zip(df[cn.EVENT_LAT], df[cn.EVENT_LONG])]

def make_hausdorff_matrix(df):
  leg_ids = df[cn.LEG_ID].unique()
  n = len(leg_ids)
  distances = np.zeros((n, n))
  labels = np.zeros(n)
  for r in range(n):
    data = df[df[cn.LEG_ID] == leg_ids[r]]
    labels[r] = data[cn.CLUSTER].unique()
    for c in range(r + 1, n):
      u_id = leg_ids[r]
      v_id = leg_ids[c]
      u = df[df[cn.LEG_ID] == u_id]
      v = df[df[cn.LEG_ID] == v_id]
      u = make_points(u)
      v = make_points(v)
      distances[r, c] = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
      distances[c, r] = distances[r, c]
  return distances, labels 
