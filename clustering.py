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

def apply_dbscan(df, distances, eps, min_samples):
  leg_ids = df[cn.LEG_ID].unique()
  clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distances)
  
  db_clusters = defaultdict(list)
  # for every cluster, append the leg_id to the list of legs that belong to a cluster
  for i, cluster_number in enumerate(clustering.labels_):
      leg_id = leg_ids[i]
      db_clusters[cluster_number].append(leg_id)
  return db_clusters

def cluster(df, distances, eps, min_samples = 1):
  db_clusters = apply_dbscan(df, distances, eps, min_samples)
  leg_to_cluster = defaultdict(int)

  for cluster_number, leg_ids in db_clusters.items():
    for leg_id in leg_ids:
      leg_to_cluster[leg_id] = cluster_number

  df.loc[:, cn.ASSIGNED_CLUSTER] = df[cn.LEG_ID].map(leg_to_cluster)
  return df

def main():
  data_file = config.FILENAMES["all_data_cleaned"] # file stored locally
  log.info("Reading in data with leg_ids from {}".format(data_file))
  df = pd.read_csv(data_file, parse_dates=[cn.EVENT_DTTM])

  log.info("{} pings at start of clustering.py".format(len(df)))
  log.info("{} leg_ids at start of clustering.py".format(df[cn.LEG_ID].nunique()))
  log.info("Getting unique from/to RM site combinations that don't arrive and depart at the same place")
  df_from_to = df[[cn.FROM_DEPOT, cn.TO_DEPOT]].drop_duplicates()
  df_from_to_no_loop = df_from_to[df_from_to[cn.FROM_DEPOT] != df_from_to[cn.TO_DEPOT]]
  df_from_to_no_loop.dropna(inplace=True)
  log.info("There are {} unique from/to combinations".format(len(df_from_to_no_loop)))

  log.info("Clustering routes between each pair of sites")
  result_list = []
  i = 0
  for row in tqdm(df_from_to_no_loop.itertuples(), total=df_from_to_no_loop.shape[0]):
    i += 1
    df_sub = df[(df.from_depot == row.from_depot) & (df.to_depot == row.to_depot)]
    distances, labels = make_hausdorff_matrix(df_sub, True)
    fig = "Fig" + str(i) + ".png"
    elbow = kdist.locate_elbow(distances, fig, 4)
    
    # Evaluate silhouette score before clustering
    silhouette = ev.silhouette_score(labels, distances)
    log.info("Silhouette score of {} - {}: {}".format(row.from_depot, row.to_depot, silhouette))

    # Clustering
    df_clustered = cluster(df_sub, distances, elbow)
    result_list.append(df_clustered)
    
    # Visualise on map
    map_file = "Map_" + str(i)
    mm.make_map(df_clustered, cn.ASSIGNED_CLUSTER, save=True, map_file_name=map_file)

    # Evaluate with homogeneity, completeness and v_measure scores
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(df_clustered[cn.CLUSTER], df_clustered[cn.ASSIGNED_CLUSTER])
    print(homogeneity, completeness, v_measure)
  
  result = pd.concat(result_list)

  log.info("{} pings at end of clustering.py".format(len(result)))
  log.info("{} leg_ids at end of clustering.py".format(result[cn.LEG_ID].nunique()))

  export_path = config.FILENAMES["all_data_clustered"]
  log.info("Writing out the clustered Isotrak data to {}".format(export_path))

  result.to_csv(export_path, index=False)
  log.info("Isotrak data with clusters exported")


if __name__ == "__main__":
  main()
