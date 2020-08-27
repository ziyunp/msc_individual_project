import pandas as pd
import logging
import os
import utils.column_names as cn
import utils.config as config

"""
  Data:
  - 883 unique leg_ids
  - 882 unique duty_ids
  - 19 unique clusters
  - Date range: 20191001 - 20200129
  - 9 pairs RM sites
"""

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def count_distinct_ids(data):
  log.info("{} distinct leg_ids from this dataset".format(data[cn.LEG_ID].nunique()))
  log.info("{} distinct duty_ids from this dataset".format(data[cn.DUTY_ID].nunique()))
  log.info("{} distinct cluster_ids from this dataset".format(data[cn.CLUSTER].nunique()))
  

def main():
  data_file = config.FILENAMES["all_data"] # file stored locally
  log.info("Reading in data with leg_ids from  from {}".format(data_file))

  data = pd.read_csv(data_file)
  count_distinct_ids(data)
  
  log.info("{} pings from this dataset".format(len(data)))
  log.info("Date range: {} - {}".format(min(data[cn.DATE_STR]), max(data[cn.DATE_STR])))


if __name__ == "__main__":
  main()
