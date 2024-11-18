import clusteringAlgorithms.qualityMeasure as qmc
import DimentionalityReductions.qualityMeasure as qmdr
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.preprocessing import LabelEncoder
import Network.quality as q

def measureClusteringTecnique(dataOne, dataTwo, labels):
    silhouette = qmc.silhouette(dataOne, dataTwo)
    dataOne = dataOne.values
    dataTwo = dataTwo.values
    labels = labels.values
    label_encoder = LabelEncoder()
    if isinstance(labels[1], str):
        labels = label_encoder.fit_transform(labels)

    jaccard = qmc.jaccard_similarity(labels, dataTwo)
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

def trustworthinessForDimensionalityReduction(X, X_reduced, n_neighbors=20):
    return qmdr.trustworthiness(X, X_reduced, n_neighbors)

def continuityForDimensionalityReduction(X, X_reduced, n_neighbors=20):
    return qmdr.continuity(X, X_reduced, n_neighbors)

def clusteringPreservation(data, labels, algorithm_name):
    return qmc.cluster_preservation(data, labels,k=50, algorithm_name=algorithm_name)

def classPreservation(data, algorithm_name):
    return qmdr.class_preservation(data, algorithm_name)

def modularity(graph, communities):
    return q.modularity(graph, communities)

def internalDensity(graph, communities):
    return q.internal_density(graph, communities)

def ratioCut(graph, communities):
    return q.ratio_cut(graph, communities)
