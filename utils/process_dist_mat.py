"""
  This is a temporary file to get distance matrix from the rtf files given by Alex
"""
import config
import numpy as np
from striprtf.striprtf import rtf_to_text

def read_file(filename):
  raw_data = open(filename).read()
  data = rtf_to_text(raw_data).splitlines()
  data = np.array([line.split(',') for line in data])
  
  # First row is the leg_ids
  leg_ids = data[0, 1:]

  # First col is the cluster labels
  clusters = data[1:, 0]

  # The rest is the distance matrix
  distances = data[1:, 1:]

  return leg_ids, clusters, distances
 

def main(from_to):
  distance_file = config.FILENAMES["dist_mat_RM"] + from_to + "_train.rtf"
  leg_ids, clusters, distances = read_file(distance_file)

  # Rearrange matrix based on our data


if __name__ == "__main__":
  main("nottingham_eastmidsairport")
