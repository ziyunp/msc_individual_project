from sklearn import metrics
import utils.column_names as cn
import sys

def silhouette_score(labels, distance_matrix):
  score = metrics.silhouette_score(distance_matrix, labels, metric='precomputed')
  orig_stdout = sys.stdout
  f = open('silhouette_score.csv', 'w')
  sys.stdout = f
  print(score)
  sys.stdout = orig_stdout
  f.close()

