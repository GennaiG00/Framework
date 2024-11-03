import clustering as clustering
import datasetOperations
from dimensionalityReduction import dimensionalityReduction
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def plot_dimentionaly_reduction_matrix(reduction_techniques, dataset_names):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    if len(dataset_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    reduction_techniques[0][0]["labels"] = np.where(reduction_techniques[0][0]["labels"] == 'UP', 0, 1)
    reduction_techniques[0][1]["labels"] = np.where(reduction_techniques[0][1]["labels"] == 'UP', 0, 1)
    reduction_techniques[0][2]["labels"] = np.where(reduction_techniques[0][2]["labels"] == 'UP', 0, 1)

    reduction_techniques[1][0]["labels"] = np.where(reduction_techniques[1][0]["labels"] == 'P', 0, 1)
    reduction_techniques[1][1]["labels"] = np.where(reduction_techniques[1][1]["labels"] == 'P', 0, 1)
    reduction_techniques[1][2]["labels"] = np.where(reduction_techniques[1][2]["labels"] == 'P', 0, 1)

    reduction_techniques[2][0]["labels"] = np.where(reduction_techniques[2][0]["labels"] == 1, 0, 1)
    reduction_techniques[2][1]["labels"] = np.where(reduction_techniques[2][1]["labels"] == 1, 0, 1)
    reduction_techniques[2][2]["labels"] = np.where(reduction_techniques[2][2]["labels"] == 1, 0, 1)



    for i, dataset_name in enumerate(dataset_names):
        axes[i, 0].scatter(reduction_techniques[i][0].iloc[:, 0], reduction_techniques[i][0].iloc[:, 1], c=reduction_techniques[i][0].iloc[:, 2], cmap='viridis')
        axes[i, 0].set_title(f"PCA - {dataset_name}")

        axes[i, 1].scatter(reduction_techniques[i][1].iloc[:, 0], reduction_techniques[i][1].iloc[:, 1], c=reduction_techniques[i][1].iloc[:, 2], cmap='viridis')
        axes[i, 1].set_title(f"t-SNE - {dataset_name}")

        axes[i, 2].scatter(reduction_techniques[i][2].iloc[:, 0], reduction_techniques[i][1].iloc[:, 2], c=reduction_techniques[i][2].iloc[:, 2], cmap='viridis')
        axes[i, 2].set_title(f"Sammon Mapping - {dataset_name}")

        # Set titles
        axes[i, 0].set_title(f"PCA - {dataset_name}")
        axes[i, 1].set_title(f"t-SNE - {dataset_name}")
        axes[i, 2].set_title(f"Sammon Mapping - {dataset_name}")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

def plot_dimensionality_reduction(dr_values, dr_name, dataset_name, axes):
    axes.scatter(dr_values.iloc[:, 0], dr_values.iloc[:, 1], c=dr_values.iloc[:, 2], cmap='viridis')
    axes.set_title(f"{dr_name} - {dataset_name}")

def plot_cluster_matrix(clusters, dataset_names, drPca):
    # Create a 3-column matrix of subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i, dataset_name in enumerate(dataset_names):
        # KMeans Clustering

        plotClusters(drPca[i].values, clusters[i][0], axes[i, 0])

        # Agglomerative Clustering
        plotClusters(drPca[i].values, clusters[i][1], axes[i, 1])

        # DBSCAN Clustering
        plotClusters(drPca[i].values, clusters[i][2], axes[i, 2])

        # Set titles
        axes[i, 0].set_title(f"KMeans - {dataset_name}")
        axes[i, 1].set_title(f"Agglomerative - {dataset_name}")
        axes[i, 2].set_title(f"DBSCAN - {dataset_name}")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


def plotClusters(data, labels, ax):
    points = np.array(data)
    unique_labels = set(labels)
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        label_points = points[labels == label]
        if label == -1:
            ax.scatter(label_points[:, 0], label_points[:, 1], color='black', label='Noise', marker='x')
        else:
            ax.scatter(label_points[:, 0], label_points[:, 1], color=color, label=f'Cluster {label}')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")



def class_preservation(data, axes):
    k = 50
    max_distance = 0.2
    nn_model = NearestNeighbors(n_neighbors=k, radius=max_distance)
    nn_model.fit(data)
    distances, indices = nn_model.radius_neighbors(data, return_distance=True)
    preservation_scores = []
    for i, neighbors in enumerate(indices):
        class_i = data.iloc[i]['labels']
        same_class_count = sum(data.iloc[neighbor]['labels'] == class_i for neighbor in neighbors[1:k + 1])
        preservation_score = same_class_count / k
        preservation_scores.append(preservation_score)

    cmap = plt.get_cmap('hot')  # Usa una mappa di colori quantitativa (come 'hot')
    sc = axes.scatter(data.iloc[:,0], data.iloc[:,1], c=preservation_scores, cmap=cmap, edgecolor='k')
    plt.colorbar(sc)

    axes.set_xlabel('X')
    axes.set_ylabel('Y')


def cluster_preservation(data, cluster_labels, axes, k=50):
    # Converti i dati in un array numpy per il modello NearestNeighbors
    data_array = np.array(data.iloc[:, :2].values)  # Usa solo le prime due colonne per le coordinate 2D

    # Inizializza NearestNeighbors con il numero desiderato di vicini
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(data_array)
    distances, indices = nn_model.kneighbors(data_array)

    # Calcola la preservation score per ciascun punto
    preservation_scores = []
    for i, neighbors in enumerate(indices):
        # Trova l'etichetta del cluster per il punto corrente
        cluster_i = cluster_labels.iloc[i, 0]
        # Conta quanti dei k vicini appartengono allo stesso cluster
        same_cluster_count = sum(cluster_labels.iloc[neighbor, 0] == cluster_i for neighbor in neighbors[1:])
        preservation_score = same_cluster_count / k
        preservation_scores.append(preservation_score)

    # Visualizza la preservation score su ogni punto con una mappa di colori quantitativa
    cmap = plt.get_cmap('hot')
    sc = axes.scatter(data.iloc[:, 0], data.iloc[:, 1], c=preservation_scores, cmap=cmap, edgecolor='k')
    plt.colorbar(sc, ax=axes)  # Aggiungi una barra colore per mostrare la scala delle preservation scores

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_title("Cluster Preservation Score")




if __name__ == '__main__':
    datasetPathA = './datasets/dataset2.txt'
    da = datasetOperations.Dataset(datasetPathA)
    datasetPathB = './datasets/space_ga.txt'
    db = datasetOperations.Dataset(datasetPathB)

    datasetPathC = './datasets/dataset3.txt'
    dc = datasetOperations.Dataset(datasetPathC)

    datasets = [da, db, dc]
    cluster_value = []
    dr_value = []
    clusters_labels = []
    all_drPca = []
    reduction_techniques = []
    cluster_result = []

    cluster_names = ['KMeans', 'Agglomerative', 'DBSCAN']
    dr_names = ['PCA', 't-SNE', 'Sammon Mapping']
    dataset_names = ['Dataset A', 'Dataset B', 'Dataset C']
    original_labels = []

    for d in datasets:
        result = []
        clusters_resultB = []
        selected_data = d.replace_missing_values("?")
        labels = None
        d = d.data
        for col in d.columns:
            if d[col].dtype == object:  # Controlla se la colonna è di tipo stringa
                try:
                    d[col].astype(float)
                except ValueError:
                    labels_col = col
                    break
            elif d[col].dtype == 'int':  # Controlla se la colonna è di tipo intero
                    labels_col = col

        data_original = selected_data.data.drop(selected_data.data.columns[labels_col], axis=1)

        labels = selected_data.data.iloc[:, labels_col].values
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data_original))

        kmeans = clustering.ClusteringAlgorithm(data.values, "euclidean", "kmeans", len(np.unique(labels)))
        tmp = kmeans.fit()
        clusters_resultB.append(tmp.labels_)
        kmeansValue = pd.DataFrame(tmp.labels_)

        dbscan = clustering.ClusteringAlgorithm(data.values, "euclidean", "dbscan", 0.3, 50)
        tmpData, tmpLabels = dbscan.fit()
        clusters_resultB.append(tmpLabels)
        dbscanValue = pd.DataFrame(tmpLabels)

        agglomerative = clustering.ClusteringAlgorithm(data.values, "euclidean", "agglomerative",len(np.unique(labels)))
        tmp = agglomerative.fit()
        clusters_resultB.append(tmp.labels_)
        agglomertiveValue = pd.DataFrame(tmp.labels_)

        drPca = dimensionalityReduction(2).reduce(data, "PCA")
        drTsne = dimensionalityReduction(2).reduce(data, "t-SNE")
        drSm, e = dimensionalityReduction(2, 20, 0.00001, "random").reduce(data, "sammonMapping")

        original_labels.append(labels)
        drTsne['labels'] = labels
        drSm = pd.DataFrame(drSm, columns=["P1", "P2"])
        drSm['labels'] = labels
        all_drPca.append(drPca)
        reduction_techniques.append([drPca , drTsne, drSm])
        cluster_value.append(result)
        clusters_labels.append([kmeansValue, agglomertiveValue, dbscanValue])
        cluster_result.append(clusters_resultB)

    plot_cluster_matrix(cluster_result, dataset_names, all_drPca)
    for i in range(len(dataset_names)):
        all_drPca[i]['labels'] = original_labels[i]
    plot_dimentionaly_reduction_matrix(reduction_techniques, dataset_names)

    fig, axes = plt.subplots( 3 , 3, figsize=(19, 15))
    for j, rdi in enumerate(reduction_techniques):
        for i in range(3):
            class_preservation(rdi[i], axes[j, i])
            cluster_name = ["PCA", "t-SNE", "Sammon Mapping"][i]
            axes[j, i].set_title(f"{cluster_name} - Dataset {dataset_names[j]}")
    plt.show()

    fig, axes = plt.subplots(3, 3, figsize=(19, 15))
    for j, dataset in enumerate(all_drPca):
        for i, cluster_labels in enumerate(clusters_labels[j]):
            cluster_preservation(all_drPca[j], cluster_labels, axes[j, i], k=50)

            cluster_name = ["KMeans", "Agglomerative", "DBSCAN"][i]
            axes[j, i].set_title(f"{cluster_name} - PCA - Dataset {dataset_names[j]}")

    plt.tight_layout()
    plt.show()

