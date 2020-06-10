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

def has_pings_more_than_5min_apart(data):
  total = len(data)
  for i in range(total - 1):
    this_ping = data.iloc[i]
    next_ping = data.iloc[i + 1]
    diff = next_ping[cn.EVENT_DTTM] - this_ping[cn.EVENT_DTTM]
    if diff > pd.Timedelta(5, 'minute'):
      print(this_ping[cn.LEG_ID], ": ", this_ping[cn.EVENT_DTTM])
      return True
  return False

def main():
  data_file = config.FILENAMES["all_data"] # file stored locally
  log.info("Reading in data with leg_ids from  from {}".format(data_file))

  data = pd.read_csv(data_file, parse_dates=[cn.EVENT_DTTM])
  legs = data[cn.LEG_ID].unique()
  log.info("{} distinct leg_ids from this dataset".format(len(legs)))

  log.info("Scanning each leg for pings more than 5min apart...")
  
  legs_with_missing_pings = []
  for leg_id in legs:
    data_sub = data[data[cn.LEG_ID] == leg_id]
    if has_pings_more_than_5min_apart(data_sub):
      legs_with_missing_pings.append(leg_id)
  
  log.info("{} legs have two consecutive pings that are more than 5 min apart".format(len(legs_with_missing_pings)))

if __name__ == "__main__":
  main()
