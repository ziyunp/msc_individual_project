from sklearn.neighbors import BallTree, KDTree
import numpy as np
import utils.helpers as hp
import utils.config as config

def construct_balltree(points):
  points = map(hp.convert_coords_to_radians, points)
  return BallTree(points, metric="haversine")

def construct_kdtree(points):
  return KDTree(points)

def query_balltree_radius(tree, point, radius, ret_distance=False):
  point = hp.convert_coords_to_radians(point)
  radius = radius / config.CONSTANTS["earth_radius"] # convert to radians
  return tree.query_radius([point], r=radius, return_distance=ret_distance)

def query_balltree_knn(tree, point, _k, ret_distance=False):
  point = hp.convert_coords_to_radians(point)
  return tree.query([point], return_distance=ret_distance, k=_k)
