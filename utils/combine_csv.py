import os
import glob
import pandas as pd

def main(dir_path, file_path):
  os.chdir(dir_path)
  extension = 'csv'
  all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

  #combine all files in the list
  combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
  #export to csv
  combined_csv.to_csv(file_path, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
  main("../data_to_run_code_on/data_with_labels/20191002-20200130_isotrak_legs_excl_5km_train", "all_training_data_pings_labels_excl_5km.csv")