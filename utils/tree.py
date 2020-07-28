from sklearn.neighbors import BallTree, KDTree
import numpy as np
import utils.helpers as hp
import utils.config as config

def construct_balltree(points):
  points = np.array(list(map(hp.convert_coords_to_radians, points)))
  return BallTree(points, metric="haversine")

def construct_kdtree(points):
  return KDTree(points)

def query_balltree_radius(tree, points, radius, ret_distance=False, sort=False):
  points = list(map(hp.convert_coords_to_radians, points))
  radius = radius / config.CONSTANTS["earth_radius"] # convert to radians
  return tree.query_radius(points, r=radius, return_distance=ret_distance, sort_results=sort)

def query_balltree_knn(tree, points, _k, ret_distance=False):
  points = list(map(hp.convert_coords_to_radians, points))
  return tree.query(points, return_distance=ret_distance, k=_k)
