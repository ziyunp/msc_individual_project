from sklearn.neighbors import BallTree
import numpy as np

def construct_balltree(lines):
  midpoints = np.asarray([np.array(line["midpoint"]) for line in lines])
  return BallTree(midpoints, metric="haversine")