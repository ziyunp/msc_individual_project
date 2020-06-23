from sklearn.neighbors import BallTree
import numpy as np

def construct_balltree(points):
  return BallTree(points, metric="haversine")
