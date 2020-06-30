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
import plot_kdist as kdist
import evaluation as ev
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn import metrics
import utils.make_map as mm
import clustering

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

def make_points(df):
    """
    Takes in a dataframe with the lat/lon points in Event_Lat and Event_Long columns
    Returns a list of the zipped lat/lon for the directed Hausdorff distance function
    :param df: dataframe with lat/lon points in separate columns
    :return: list of np.arrays with (lat, lon) inside
    """
    return [np.array(item) for item in zip(df[cn.EVENT_LAT], df[cn.EVENT_LONG])]

def make_hausdorff_matrix(df, symm=False):
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
      if symm:
          distances[r, c] = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
      else:
          distances[r, c] = directed_hausdorff(u, v)[0]
      distances[c, r] = distances[r, c]
  return distances, labels 
