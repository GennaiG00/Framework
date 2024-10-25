import  clustering as clustering
import datasetOperations
from dimensionalityReduction import dimensionalityReduction
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

def plot_dimentionaly_reduction_matrix(reduction_techniques, dataset_names):
    # Create a 3-column matrix of subplots
    fig, axes = plt.subplots(len(dataset_names), 3, figsize=(15, 15))

    # Ensure `axes` is 2D even if there's only one row (one dataset)
    if len(dataset_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    reduction_techniques[0][0]["labels"] = np.where(reduction_techniques[0][0].values[:,2] == 'UP', 0, 1)
    reduction_techniques[0][1]["labels"] = np.where(reduction_techniques[0][1].values[:,2] == 'UP', 0, 1)
    #reduction_techniques[0][2]["labels"] = np.where(reduction_techniques[0][2].values[:,2] == 'orange', 0, 1)

    reduction_techniques[1][0]["labels"] = np.where(reduction_techniques[1][0].values[:,2] == 'P', 0, 1)
    reduction_techniques[1][1]["labels"] = np.where(reduction_techniques[1][1].values[:,2] == 'P', 0, 1)
    #reduction_techniques[1][2]["labels"] = np.where(reduction_techniques[1][2].values[:,2] == 'orange', 0, 1)

    reduction_techniques[2][0]["labels"] = np.where(reduction_techniques[2][0].values[:,2] == 'orange', 0, 1)
    reduction_techniques[2][1]["labels"] = np.where(reduction_techniques[2][1].values[:,2] == 'orange', 0, 1)
    #reduction_techniques[2][2]["labels"] = np.where(reduction_techniques[2][2].values[:,2] == 'orange', 0, 1)



    for i, dataset_name in enumerate(dataset_names):
        plot_dimensionality_reduction(reduction_techniques[i][0], "PCA", dataset_name, axes[i, 0])

        plot_dimensionality_reduction(reduction_techniques[i][1], "t-SNE", dataset_name, axes[i, 1])

        # DBSCAN Clustering
        #plotClustersDBSCAN(drPca.values, dataset_name, 0.3, 10, "euclidean", clusters[i][2], axes[i, 2])

        # Set titles
        axes[i, 0].set_title(f"PCA - {dataset_name}")
        axes[i, 1].set_title(f"t-SNE - {dataset_name}")
        #axes[i, 2].set_title(f"DBSCAN - {dataset_name}")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

def plot_cluster_matrix(clusters, cluster_values, dataset_names, drPca):
    """
    Plot a matrix of clusters using KMeans, Agglomerative, and DBSCAN.
    """
    # Create a 3-column matrix of subplots
    fig, axes = plt.subplots(len(dataset_names), 3, figsize=(15, 15))

    # Ensure `axes` is 2D even if there's only one row (one dataset)
    if len(dataset_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, dataset_name in enumerate(dataset_names):
        # KMeans Clustering
        plotClustersKMeans(drPca[i].values, dataset_name, clusters[i][0], cluster_values[i][0], axes[i, 0])

        # Agglomerative Clustering
        plotClustersAgglomerative(drPca[i].values, dataset_name, clusters[i][1], axes[i, 1])

        # DBSCAN Clustering
        #plotClustersDBSCAN(drPca.values, dataset_name, 0.3, 10, "euclidean", clusters[i][2], axes[i, 2])

        # Set titles
        axes[i, 0].set_title(f"KMeans - {dataset_name}")
        axes[i, 1].set_title(f"Agglomerative - {dataset_name}")
        #axes[i, 2].set_title(f"DBSCAN - {dataset_name}")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


def plotClustersKMeans(data, dataset_name, kmeans_labels, cluster_centers, ax):
    """
    Plot KMeans clusters.
    """
    points = np.array(data)
    unique_labels = set(kmeans_labels)
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        label_points = points[kmeans_labels == label]
        ax.scatter(label_points[:, 0], label_points[:, 1], color=color, label=f'Cluster {label}')

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plotClustersAgglomerative(data, dataset_name, agglo_labels, ax):
    """
    Plot Agglomerative clusters.
    """
    points = np.array(data)
    unique_labels = set(agglo_labels)
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        label_points = points[agglo_labels == label]
        ax.scatter(label_points[:, 0], label_points[:, 1], color=color, label=f'Cluster {label}')

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plotClustersDBSCAN(data, dataset_name, e, minPts, distance_type, labels, ax):
    """
    Plot DBSCAN clusters.
    """
    points = np.array(data)
    unique_labels = set(labels)
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        label_points = points[labels == label]
        if label == -1:  # Noise points
            ax.scatter(label_points[:, 0], label_points[:, 1], color='black', label='Noise', marker='x')
        else:
            ax.scatter(label_points[:, 0], label_points[:, 1], color=color, label=f'Cluster {label}')

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")



# Function to plot dimensionality reduction scatterplots with labels
def plot_dimensionality_reduction(dr_values, dr_name, dataset_name, axes):
    scatter = axes.scatter(dr_values.iloc[:, 0], dr_values.iloc[:, 1], c=dr_values.iloc[:, 2], cmap='viridis')
    axes.set_title(f"{dr_name} - {dataset_name}")


def preprocess_labels(labels):
    processed_labels = np.where(labels == 'noise', -1, labels)
    return processed_labels.astype(int)




if __name__ == '__main__':
    # Dataset paths
    datasetPathA = './dataset/dataset2.txt'
    da = datasetOperations.Dataset(datasetPathA)

    datasetPathB = './dataset/space_ga.txt'
    db = datasetOperations.Dataset(datasetPathB)

    datasetPathC = './dataset/dataset.txt'
    dc = datasetOperations.Dataset(datasetPathC)

    datasets = [da, db, dc]
    cluster_value = []
    dr_value = []
    clusters = []
    all_drPca = []
    reduction_techniques = []

    # Clustering and DR techniques names
    cluster_names = ['KMeans', 'DBSCAN', 'Agglomerative']
    dr_names = ['PCA', 't-SNE']
    dataset_names = ['Dataset A', 'Dataset B', 'Dataset C']

    for d in datasets:
        result = []
        clusters_result = []
        selected_data = d.replace_missing_values("?")
        labels = None
        d = d.data
        for col in d.columns:
            if d[col].dtype == object:
                try:
                    d[col].astype(float)
                except ValueError:
                    labels_col = col
                    break

        data_original = selected_data.data.drop(selected_data.data.columns[labels_col], axis=1)

        labels = selected_data.data.iloc[:, labels_col].values
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data_original))

        # Dimensionality Reduction
        drPca, pca = dimensionalityReduction(2).reduce(data, "PCA")
        drTsne = pd.DataFrame(dimensionalityReduction(2).reduce(data, "t-SNE"))
        #drSm, e = dimensionalityReduction(2, 20, 0.3, "random").reduce(data.values, "sammonMapping")

        # Clustering
        kmeans = clustering.ClusteringAlgorithm(drPca.values, "euclidean", "kmeans", len(np.unique(labels)))
        tmp = kmeans.fit()
        result.append(tmp.cluster_centers_)
        clusters_result.append(tmp.labels_)

        # dbscan = clustering.ClusteringAlgorithm(drPca.values, "euclidean", "dbscan", 0.3, 10)
        # tmp = dbscan.fit()
        # tmp = pd.DataFrame(tmp, columns=["p1", "p2", "labels"])
        # result.append(tmp[["p1", "p2"]])
        # clusters_result.append(tmp["labels"])

        agglomerative = clustering.ClusteringAlgorithm(drPca.values, "euclidean", "agglomerative", len(np.unique(labels)))
        result.append(agglomerative.fit().children_)
        tmp = agglomerative.fit()
        result.append(tmp.children_)
        clusters_result.append(tmp.labels_)

        # Append labels
        drPca['labels'] = labels
        drTsne['labels'] = labels
        reduction_techniques.append([drPca, drTsne])
        all_drPca.append(drPca)
        # Store DR results
        dr_value.append([drPca, drTsne])
        # drSm['labels'] = labels
        cluster_value.append(result)
        clusters.append(clusters_result)

    #plotClustersDBSCAN(cluster_value[0][1], dataset_names[0], 0.3, 10, "euclidean", clusters[0][1])
    #plot_cluster_matrix(clusters, cluster_value, dataset_names, all_drPca)
    plot_dimentionaly_reduction_matrix(reduction_techniques, dataset_names)












