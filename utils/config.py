# TODO:
# generalise FILENAMES
data_path = "../data_to_run_code_on" 

FILENAMES = {
  "all_data": "{}/all_training_data_pings_labels_incl_5km.csv".format(data_path),
  "all_data_with_road_names": "{}/all_training_data_pings_labels_incl_5km_with_road_names.csv".format(data_path),
  "all_data_cleaned": "{}/all_training_data_pings_labels_incl_5km_cleaned.csv".format(data_path),
  "all_data_with_road_names_cleaned": "{}/all_training_data_pings_labels_incl_5km_with_road_names_cleaned.csv".format(data_path),
  "labelled_data": "{}/20191002-20200130_isotrak_labelled.csv".format(data_path),
  "labelled_subset": "{}/20191002-20200130_subset_isotrak_labelled.csv".format(data_path),
  "test_excl_5km": "{}/20191002-20200130_isotrak_legs_excl_5km_test.csv".format(data_path),
  "roads_data": "{}/roads_data.csv".format(data_path)
} 

CONSTANTS = {
  "earth_radius": 6371 # Radius of earth in kilometers. Use 3956 for miles
}