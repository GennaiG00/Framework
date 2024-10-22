import clusteringAlgorithms.qualityMeasure as qmc
import dimentionalityReductions.qualityMeasure as qmdr
from sklearn.metrics import pairwise_distances
import numpy as np

def measureClusteringTecnique(nameCt, data):
    if nameCt == 'clustering':
        processed_labels = np.where(data == 'noise', -1, data)
        data = processed_labels.astype(int)
        labels = data[:, 2]
        silhouette = qmc.evaluate_clustering(data, labels)
        dbi = qmc.evaluate_clustering_dbi(data, labels)
        return silhouette, dbi

def measureDimensionalityReduction(nameDr, *hyperparameters):
    if nameDr == 'pca':
        explained_variance = qmdr.evaluate_dimensionality_reduction(hyperparameters[0])
        return explained_variance
    elif nameDr == 'tsne':
        original_distances = pairwise_distances(hyperparameters[0])
        reduced_distances = pairwise_distances(hyperparameters[1])
        return np.corrcoef(original_distances.ravel(), reduced_distances.ravel())[0, 1]

