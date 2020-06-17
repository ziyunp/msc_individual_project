from sklearn import metrics
import utils.column_names as cn

def silhouette_score(labels, distance_matrix):
  return metrics.silhouette_score(distance_matrix, labels, metric='precomputed')
