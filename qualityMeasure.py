import clusteringAlgorithms.qualityMeasure as qmc
import dimentionalityReductions.qualityMeasure as qmdr
from sklearn.metrics import pairwise_distances
import numpy as np

def measureClusteringTecnique(dataOne, dataTwo, labels):
    dataOne = np.array(dataOne)
    dataTwo = np.array(dataTwo)
    labels = np.array(labels)
    silhouette = qmc.silhouette(dataOne, dataTwo)
    jaccard = qmc.jaccard_similarity(labels, dataTwo)
    return silhouette, jaccard

def measureDimensionalityReduction(nameDr, *hyperparameters):
    if nameDr == 'pca':
        explained_variance = qmdr.evaluate_dimensionality_reduction(hyperparameters[0])
        return explained_variance
    elif nameDr == 'tsne':
        original_distances = pairwise_distances(hyperparameters[0])
        reduced_distances = pairwise_distances(hyperparameters[1])
        return np.corrcoef(original_distances.ravel(), reduced_distances.ravel())[0, 1]

