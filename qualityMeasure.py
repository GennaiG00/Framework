import clusteringAlgorithms.qualityMeasure as qmc
import DimentionalityReductions.qualityMeasure as qmdr
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.preprocessing import LabelEncoder

def measureClusteringTecnique(dataOne, dataTwo, labels):
    dataOne = dataOne.values
    dataTwo = dataTwo.values
    labels = labels.values
    label_encoder = LabelEncoder()
    if isinstance(labels[1], str):
        labels = label_encoder.fit_transform(labels)

    jaccard = qmc.jaccard_similarity(labels, dataTwo)
    silhouette = qmc.silhouette(dataOne, dataTwo)
    return silhouette, jaccard

def measureDimensionalityReduction(nameDr, *hyperparameters):
    if nameDr == '1':
        total_variance = np.sum(np.var(hyperparameters[0], axis=0))
        component_variances = np.var(hyperparameters[1], axis=0)
        explained_variance_ratio = component_variances / total_variance
        total_explained_variance = explained_variance_ratio.sum()
        return total_explained_variance, explained_variance_ratio
    elif nameDr == '2':
        original_distances = pairwise_distances(hyperparameters[0])
        reduced_distances = pairwise_distances(hyperparameters[1])
        return np.corrcoef(original_distances.ravel(), reduced_distances.ravel())[0, 1]
    elif nameDr == '3':
        return qmdr.evaluate_sammon_mapping(hyperparameters[0], hyperparameters[1])


def clusteringPreservation(data, labels):
    return qmc.cluster_preservation(data, labels)

def classPreservation(data):
    return qmdr.class_preservation(data)