import numpy as np
import pandas as pd
import logging
import os
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict
import utils.make_map as mm
import basic_HD as HD
import MLHD 
import utils.column_names as cn
import utils.config as config
import utils.evaluation as ev
import utils.plot_kdist as kdist

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def apply_dbscan(df, distances, eps, min_samples, used_labels):
  leg_ids = df[cn.LEG_ID].unique()
  clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distances)
  max_used_labels = max(used_labels)
  if max_used_labels != -1:
    for i, lbl in enumerate(clustering.labels_):
      if lbl != -1:
        clustering.labels_[i] += max_used_labels + 1
  
  # Create a dict of leg_id -> cluster label
  leg_to_clusters = defaultdict(int)
  for i, cluster_number in enumerate(clustering.labels_):
    leg_to_clusters[leg_ids[i]] = cluster_number
  return leg_to_clusters

def cluster(df, distances, eps, min_samples):
  leg_ids = df[cn.LEG_ID].unique()
  df_unclustered = df[df[cn.ASSIGNED_CLUSTER] == -1]
  # All data is clustered
  if len(df_unclustered) == 0:
    return df
  # Filter out clustered legs from the distance matrix
  unclustered_leg_ids = df_unclustered[cn.LEG_ID].unique()
  clustered_legs = []
  for i, leg_id in enumerate(leg_ids):
    if leg_id not in unclustered_leg_ids:
      clustered_legs.append(i)
  distances = np.delete(distances, clustered_legs, axis=0)
  distances = np.delete(distances, clustered_legs, axis=1)

  assert distances.shape[0] == df_unclustered[cn.LEG_ID].nunique()
  assert distances.shape[1] == df_unclustered[cn.LEG_ID].nunique()

  # Perform DBSCAN, returns a dict of leg_id -> cluster label
  existing_labels = df[cn.ASSIGNED_CLUSTER].unique()
  leg_to_clusters = apply_dbscan(df_unclustered, distances, eps, min_samples, existing_labels)

  # Assign cluster labels to all GPS pings by their leg Ids
  for k, v in leg_to_clusters.items():
    df.loc[df[cn.LEG_ID] == k, cn.ASSIGNED_CLUSTER] = v
  return df

def clustering_multi_eps(df, distance_matrix, elbows, reversed=False):
  start = len(elbows) - 1 if reversed else 0
  stop = 0 if reversed else len(elbows) - 1
  step = -1 if reversed else 1
  stop += step
  for e in range(start, stop, step):
    elbow = elbows[e]
    min_pts = 2
    if elbow == elbows[stop - step]:
      min_pts = 1
    df = cluster(df, distance_matrix, elbow, min_pts)
  return df

def clustering_multi_minpts(df, distance_matrix, elbows, minpts_arr):
  elbow = elbows[0] # not varying elbows
  for min_pts in minpts_arr:
    df = cluster(df, distance_matrix, elbow, min_pts)
  return df

def main(distance_metric, clustering_algorithm, k=3):
  data_file = config.FILENAMES["all_data_with_road_names_cleaned"] 
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
    df_sub = df[(df.from_depot == row.from_depot) & (df.to_depot == row.to_depot)].copy()

    if distance_metric == "HD":
      distances, labels = HD.make_hausdorff_matrix(df_sub)
    elif distance_metric == "MLHD":
      distances, labels = MLHD.make_hausdorff_matrix(df_sub, modified=False)
    else:
      log.error("No matching distance metric")
      return 

    # Save distance matrix for heatmap
    distance_file = "distance_matrix_" + str(i) + ".csv"
    labels_file = "labels_" + str(i) + ".csv"
    np.savetxt(distance_file, distances, delimiter=",")
    np.savetxt(labels_file, labels, delimiter=",")

    # Evaluate silhouette score before clustering
    silhouette = ev.silhouette_score(labels, distances)
    log.info("Silhouette score of {} - {}: {}".format(row.from_depot, row.to_depot, silhouette))

    # Clustering
    fig = "Elbow_" + str(i) + ".png"
    if clustering_algorithm == "DBSCAN":
      elbows = kdist.locate_elbow(distances, k, multiple=False, fig_name=fig)
    elif clustering_algorithm == "DMDBSCAN":
      elbows = kdist.locate_elbow(distances, k, multiple=True, fig_name=fig)

    assert len(elbows) > 0

    df_sub.loc[:, cn.ASSIGNED_CLUSTER] = -1

    df_sub = clustering_multi_eps(df_sub, distances, elbows)

    result_list.append(df_sub)
    
    # Visualise on map
    map_file = "Map_" + str(i)
    mm.make_map(df_sub, cn.ASSIGNED_CLUSTER, save=True, map_file_name=map_file)

    # Evaluate with homogeneity, completeness and v_measure scores
    homogeneity, completeness, v_measure = ev.homogeneity_completeness_v_measure(df_sub)
    log.info("Homogeneity, Completeness, V-measure: {}, {}, {}".format(homogeneity, completeness, v_measure))

  result = pd.concat(result_list)

  log.info("{} pings at end of clustering.py".format(len(result)))
  log.info("{} leg_ids at end of clustering.py".format(result[cn.LEG_ID].nunique()))

  export_path = config.FILENAMES["all_data_clustered"]
  log.info("Writing out the clustered Isotrak data to {}".format(export_path))

  result.to_csv(export_path, index=False)
  log.info("Isotrak data with clusters exported")

# DBSCAN + multi_minpts or DMDBSCAN + multi_eps / multi_eps_reversed
if __name__ == "__main__":
  main("HD", "DBSCAN", 5)
