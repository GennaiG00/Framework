import clusteringAlgorithms.DBSCAN as DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

class ClusteringAlgorithm:
    def __init__(self, data, distance_measure, algorithm_name, *hyperparameters):
        self.data = data
        self.distance_measure = distance_measure
        self.hyperparameters = hyperparameters
        self.algorithmName = algorithm_name

    def fit(self):
        if self.algorithmName == "dbscan" or self.algorithmName == "DBSCAN":
            e = self.hyperparameters[0]
            minPts = self.hyperparameters[1]
            result = DBSCAN.DBSCAN(pd.DataFrame(self.data), self.distance_measure, e = e, minPts = minPts).dbscan()
            return result
        elif self.algorithmName == "kmeans" or self.algorithmName == "KMeans":
            n_clusters = self.hyperparameters[0]
            result = KMeans(n_clusters=n_clusters).fit(self.data)
            return result
        elif self.algorithmName == "agglomerative" or self.algorithmName == "Agglomerative":
            n_clusters = self.hyperparameters[0]
            result = AgglomerativeClustering(n_clusters=n_clusters).fit(self.data)
            return result


