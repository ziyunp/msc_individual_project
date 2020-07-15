"""
  This is a temporary file to get distance matrix from the rtf files given by Alex
"""
import utils.config as config
import numpy as np
import pandas as pd
from striprtf.striprtf import rtf_to_text
import utils.column_names as cn

def read_file(filename):
  raw_data = open(filename).read()
  data = rtf_to_text(raw_data).splitlines()
  data = np.array([line.split(',') for line in data])
  # First row is the leg_ids
  leg_ids = data[0, 1:]
  # First col is the cluster labels
  clusters = data[1:, 0].astype(np.float)
  # The rest is the distance matrix
  distances = data[1:, 1:].astype(np.float)
  return leg_ids, clusters, distances
 
def arrange_matrix_by_leg_ids(matrix, labels, leg_ids, df):
  ordered_legs = df[cn.LEG_ID].unique()

  # Filter extra legs
  extra_legs = []
  for i in range(len(leg_ids)):
    if leg_ids[i] not in ordered_legs:
      extra_legs.append(i)

  if len(extra_legs) > 0:
    matrix = np.delete(matrix, extra_legs, axis=0)
    matrix = np.delete(matrix, extra_legs, axis=1)
    labels = np.delete(labels, extra_legs)
    leg_ids = np.delete(leg_ids, extra_legs)

  assert len(matrix) == len(labels) == len(leg_ids)
  
  order = []
  missing_legs = [] # if there are missing legs, remove from df
  for id in ordered_legs:
    if id in leg_ids:
      ind = list(leg_ids).index(id)
      order.append(ind)
    else:
      missing_legs.append(id)
  
  if len(missing_legs) > 0:
    for leg in missing_legs:
      df = df.drop(df[df[cn.LEG_ID] == leg].index)
  
  length = len(order)
  order_rows = np.tile(order, (length, 1))
  ordered_matrix = np.array(list(map(lambda x, y: y[x], order_rows, matrix)))
  ordered_matrix = ordered_matrix[order]
  ordered_labels = labels[order]
  return ordered_matrix, ordered_labels, df


def get_distance_matrix(df, from_to):
  distance_file = config.FILENAMES["dist_mat_RM"] + from_to + "_train.rtf"
  leg_ids, clusters, distances = read_file(distance_file)

  # Rearrange matrix based on our data
  ordered_matrix, ordered_labels, missing_legs = arrange_matrix_by_leg_ids(distances, clusters, leg_ids, df)
  return ordered_matrix, ordered_labels, missing_legs
