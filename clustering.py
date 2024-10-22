from sklearn.cluster import DBSCAN

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
            result = DBSCAN(eps = e, min_samples = minPts).fit(self.data)
            return result
