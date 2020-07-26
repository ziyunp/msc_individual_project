import pandas as pd
import numpy as np
import logging
import os
import utils.column_names as cn
import utils.config as config
import utils.tree as tree
"""
  Filter legs with two consecutive pings of more than 10 min away
"""
log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

MISSING_PING_THRESHOLD = 10 # defines a missing ping if two consecutive pings are >= 10 min apart
ROAD_THRESHOLD = 0.25 # km

def has_missing_pings(data):
  total = len(data)
  for i in range(total - 1):
    this_ping = data.iloc[i]
    next_ping = data.iloc[i + 1]
    diff = next_ping[cn.EVENT_DTTM] - this_ping[cn.EVENT_DTTM]
    if diff >= pd.Timedelta(MISSING_PING_THRESHOLD, 'minute'):
      return True
  return False

def assign_road(data, roads_tree, roads_data):
  points = list(zip(data[cn.EVENT_LAT], data[cn.EVENT_LONG]))
  distance, indices = tree.query_balltree_knn(roads_tree, points, 1, True)
  road_names = []
  for d, i in zip(distance, indices):
    if d <= ROAD_THRESHOLD:
      road_names.append(roads_data.iloc[i][cn.ROAD].to_string(index=False).strip())
    else:
      road_names.append("unknown")
  return road_names

def clean_intersection(row):
  if row['road_before'] == row['road_after']:
    res = row['road_before']
  else:
    res = row[cn.ROAD_NAME]
  return res

def clean_classification(df):
  """
  Function that uses logic in order to clean unexpected classification
  :param clusters_df:
  :return:
  """
  # Add the shifted columns to remove the points on intersections
  df["road_before"] = df[cn.ROAD_NAME].shift(1)
  df["road_after"] = df[cn.ROAD_NAME].shift(-1)
  df[cn.ROAD_NAME] = df.apply(clean_intersection, axis=1)
  df = df[df[cn.ROAD_NAME] != 'unknown']
  df = df.drop(["road_before", "road_after"], axis=1)
  return df

def fill_in_missing_road(gps_data, roads_data):
  roads_coords = np.dstack([roads_data.latitude.ravel(), roads_data.longitude.ravel()])[0]
  roads_tree = tree.construct_balltree(roads_coords)
  unlabelled_pings = gps_data[gps_data[cn.ROAD_NAME].isna()]
  # match unlabelled with roads_data
  road_names = assign_road(unlabelled_pings, roads_tree, roads_data)
  assert len(road_names) == len(unlabelled_pings)
  gps_data.loc[gps_data[cn.ROAD_NAME].isna(), cn.ROAD_NAME] = road_names
  gps_data = clean_classification(gps_data)
  return gps_data

def main():
  data_file = config.FILENAMES["all_data_with_road_names"] # file stored locally
  log.info("Reading in data with leg_ids from  from {}".format(data_file))

  data = pd.read_csv(data_file, parse_dates=[cn.EVENT_DTTM])
  legs = data[cn.LEG_ID].unique()

  log.info("Initial number of pings: {}".format(len(data)))
  log.info("{} distinct leg_ids from this dataset".format(len(legs)))

  log.info("Scanning each leg for missing pings...")
  
  legs_with_missing_pings = []
  for leg_id in legs:
    data_sub = data[data[cn.LEG_ID] == leg_id]
    if has_missing_pings(data_sub):
      legs_with_missing_pings.append(leg_id)
  
  log.info("{} legs have two consecutive pings that are >= {} min apart".format(len(legs_with_missing_pings), MISSING_PING_THRESHOLD))

  log.info("Removing legs with missing pings...")
  for leg_id in legs_with_missing_pings:
    data = data[data[cn.LEG_ID] != leg_id]

  log.info("Final number of pings: {}".format(len(data)))
  log.info("{} leg_ids at end of classification".format(data[cn.LEG_ID].nunique()))

  if cn.ROAD_NAME in data.columns:
    log.info("Filling in missing road names...")
    roads_data_file = config.FILENAMES["roads_data"]
    roads_data = pd.read_csv(roads_data_file, dtype={cn.LINK_NAME: str, cn.C_WAY: str, cn.SECTION: str})
    data = fill_in_missing_road(data, roads_data)

  export_path = config.FILENAMES["all_data_with_road_names_cleaned"] 
  log.info("Writing out classified data to {}".format(export_path))
  data.to_csv(export_path, index=False)
  log.info("Preprocessing done")


if __name__ == "__main__":
  main()
