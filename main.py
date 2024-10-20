import clustering
import datasetOperations

if __name__ == '__main__':
    datasetPath = './dataset/blobs.csv'
    features = datasetOperations.Dataset(datasetPath).get_features()
    points = datasetOperations.Dataset(datasetPath).get_points()
    clustering.ClusteringAlgorithm(points, "euclidean", "dbscan", 0.3, 20)
