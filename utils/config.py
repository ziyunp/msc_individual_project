# path to the directory where the data is
data_path = "../data_to_run_code_on" 
heatmap_data = "../data/Heatmap"

FILENAMES = {
  "all_data": "{}/all_training_data_pings_labels_incl_5km.csv".format(data_path),
  "all_data_excl_5km": "{}/all_training_data_pings_labels_excl_5km.csv".format(data_path),
  "all_data_with_road_names": "{}/all_training_data_pings_labels_incl_5km_with_road_names.csv".format(data_path),
  "all_data_cleaned": "{}/all_training_data_pings_labels_incl_5km_cleaned.csv".format(data_path),
  "all_data_excl_5km_cleaned": "{}/all_training_data_pings_labels_excl_5km_cleaned.csv".format(data_path),
  "all_data_clustered": "{}/all_training_data_pings_labels_incl_5km_clustered.csv".format(data_path),
  "test_data": "{}/all_training_data_pings_labels_incl_5km_TEST.csv".format(data_path),
  "test_data_cleaned": "{}/all_training_data_pings_labels_incl_5km_TEST_cleaned.csv".format(data_path),
  "test_data_clustered": "{}/all_training_data_pings_labels_incl_5km_TEST_clustered.csv".format(data_path),
  "all_data_excl_5km_clustered": "{}/all_training_data_pings_labels_excl_5km_clustered.csv".format(data_path),
  "all_data_with_road_names_cleaned": "{}/all_training_data_pings_labels_incl_5km_with_road_names_cleaned.csv".format(data_path),
  "labelled_data": "{}/20191002-20200130_isotrak_labelled.csv".format(data_path),
  "labelled_subset": "{}/20191002-20200130_subset_isotrak_labelled.csv".format(data_path),
  "test_excl_5km": "{}/20191002-20200130_isotrak_legs_excl_5km_test.csv".format(data_path),
  "test_excl_5km_cleaned": "{}/20191002-20200130_isotrak_legs_excl_5km_test_cleaned.csv".format(data_path),
  "dist_mat_RM": "{}/dist_mats_from_hd_labels_train/dist_mat_".format(data_path),
  "roads_data": "{}/roads_data.csv".format(data_path)
} 

DATA_FOR_HEATMAP = {
  "labels": "{}/test_labels/labels_".format(heatmap_data),
  "distances": "{}/distances/distance_matrix_".format(heatmap_data)
}

CONSTANTS = {
  "earth_radius": 6371 # Radius of earth in kilometers. Use 3956 for miles
}
