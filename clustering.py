
class ClusteringAlgorithm:
    def __init__(self, dataset, distance_measure, algorithm_name, *hyperparameters):
        self.dataset = dataset
        self.distance_measure = distance_measure
        self.hyperparameters = hyperparameters
        self.algorithmName = algorithm_name

    def fit(self):
        from clusteringAlgorithms.DBSCAN import DBSCAN
        if self.algorithmName == "dbscan" or self.algorithmName == "DBSCAN":
            dbscan = DBSCAN.__init__(self.dataset, self.distance_measure, *self.hyperparameters)
            return DBSCAN.dbscan(dbscan)
