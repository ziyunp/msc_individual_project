from sklearn import metrics
import utils.column_names as cn

def silhouette_score(labels, distance_matrix):
  return metrics.silhouette_score(distance_matrix, labels, metric='precomputed')

def homogeneity_completeness_v_measure(df):
  return metrics.homogeneity_completeness_v_measure(df[cn.CLUSTER], df[cn.ASSIGNED_CLUSTER])