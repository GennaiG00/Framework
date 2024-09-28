
class ClusteringAlgorithm:
    def __init__(self, dataset, distance_measure, **hyperparameters):
        self.dataset = dataset
        self.distance_measure = distance_measure
        self.hyperparameters = hyperparameters

    def fit(self):
        raise NotImplementedError("Subclasses should implement this!")
