import pandas as pd
import logging
import os
import utils.column_names as cn
import utils.config as config

"""
  1. Check if there are two consecutive pings of more than 5 min away
  2. Construct line segments for every 5 min interval - at least two pings

"""
log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

MISSING_PING_THRESHOLD = 10 # defines a missing ping if two consecutive pings are >= 10 min apart

def has_missing_pings(data):
  total = len(data)
  for i in range(total - 1):
    this_ping = data.iloc[i]
    next_ping = data.iloc[i + 1]
    diff = next_ping[cn.EVENT_DTTM] - this_ping[cn.EVENT_DTTM]
    if diff >= pd.Timedelta(MISSING_PING_THRESHOLD, 'minute'):
      return True
  return False

def main():
  data_file = config.FILENAMES["all_data"] # file stored locally
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

  # Generate the export path for the isotrak file with road labels
  export_path = config.FILENAMES["all_data_cleaned"] 
  log.info("Writing out classified data to {}".format(export_path))
  data.to_csv(export_path, index=False)
  log.info("Preprocessing done")


if __name__ == "__main__":
  main()
