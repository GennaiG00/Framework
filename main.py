import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn.preprocessing import StandardScaler

import plotFile
import datasetOperations
import clustering
from Network.comunityDetection import CommunityDetection
from Network.Network import networkWithNx
from dimensionalityReduction import dimensionalityReduction
import qualityMeasure
import sys


# def plot_dimentionaly_reduction_matrix(reduction_techniques, dataset_names):
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#
#     if len(dataset_names) == 1:
#         axes = np.expand_dims(axes, axis=0)
#
#     reduction_techniques[0][0]["labels"] = np.where(reduction_techniques[0][0]["labels"] == 'UP', 0, 1)
#     reduction_techniques[0][1]["labels"] = np.where(reduction_techniques[0][1]["labels"] == 'UP', 0, 1)
#     reduction_techniques[0][2]["labels"] = np.where(reduction_techniques[0][2]["labels"] == 'UP', 0, 1)
#
#     reduction_techniques[1][0]["labels"] = np.where(reduction_techniques[1][0]["labels"] == 'P', 0, 1)
#     reduction_techniques[1][1]["labels"] = np.where(reduction_techniques[1][1]["labels"] == 'P', 0, 1)
#     reduction_techniques[1][2]["labels"] = np.where(reduction_techniques[1][2]["labels"] == 'P', 0, 1)
#
#     reduction_techniques[2][0]["labels"] = np.where(reduction_techniques[2][0]["labels"] == 1, 0, 1)
#     reduction_techniques[2][1]["labels"] = np.where(reduction_techniques[2][1]["labels"] == 1, 0, 1)
#     reduction_techniques[2][2]["labels"] = np.where(reduction_techniques[2][2]["labels"] == 1, 0, 1)
#
#     for i, dataset_name in enumerate(dataset_names):
#         axes[i, 0].scatter(reduction_techniques[i][0].iloc[:, 0], reduction_techniques[i][0].iloc[:, 1],
#                            c=reduction_techniques[i][0].iloc[:, 2], cmap='viridis')
#         axes[i, 0].set_title(f"PCA - {dataset_name}")
#
#         axes[i, 1].scatter(reduction_techniques[i][1].iloc[:, 0], reduction_techniques[i][1].iloc[:, 1],
#                            c=reduction_techniques[i][1].iloc[:, 2], cmap='viridis')
#         axes[i, 1].set_title(f"t-SNE - {dataset_name}")
#
#         axes[i, 2].scatter(reduction_techniques[i][2].iloc[:, 0], reduction_techniques[i][1].iloc[:, 2],
#                            c=reduction_techniques[i][2].iloc[:, 2], cmap='viridis')
#         axes[i, 2].set_title(f"Sammon Mapping - {dataset_name}")
#
#         axes[i, 0].set_title(f"PCA - {dataset_name}")
#         axes[i, 1].set_title(f"t-SNE - {dataset_name}")
#         axes[i, 2].set_title(f"Sammon Mapping - {dataset_name}")
#
#     plt.tight_layout()
#     plt.show()


# def plot_cluster_matrix(clusters, dataset_names, drPca):
#     fig, axes = plt.subplots(figsize=(15, 15))
#
#     for i, dataset_name in enumerate(dataset_names):
#         # KMeans Clustering
#
#         plotFile.plot2D(drPca[i].values, clusters[i][0])
#
#         # Agglomerative Clustering
#         plotFile.plot2D(drPca[i].values, clusters[i][1])
#
#         # DBSCAN Clustering
#         plotFile.plot2D(drPca[i].values, clusters[i][2])
#
#         axes.set_title(f"KMeans - {dataset_name}")
#         axes.set_title(f"Agglomerative - {dataset_name}")
#         axes.set_title(f"DBSCAN - {dataset_name}")
#     plt.tight_layout()
#     plt.show()

def frameworkRun():
    while True:
        print("Choose the operation you want to perform...")
        print("1. Clustering and/or dimensionality reduction operations")
        print("2. Network analysis")

        response = input("Enter the number of the operation you want to perform: ").strip()

        if response == '1':
            print("Clustering and/or dimensionality reduction operations")

            while True:
                dataset_files = [f for f in os.listdir('./datasets/') if
                                 os.path.isfile(os.path.join('./datasets/', f)) and not f.startswith(
                                     '.') and not f.endswith('.edges')]  # Esclude i file edgelist

                if dataset_files:
                    print("Available datasets:")
                    for i, file in enumerate(dataset_files, 1):
                        print(f"{i}. {file}")

                    choice = input(
                        f"Enter the number corresponding to the dataset you want to use (1-{len(dataset_files)}): ").strip()

                    try:
                        chosen_file = dataset_files[int(choice) - 1]
                        datasetPath = os.path.join('./datasets/', chosen_file)
                        print(f"Dataset '{chosen_file}' loaded successfully.")
                        d = datasetOperations.Dataset(datasetPath)
                        break
                    except (ValueError, IndexError):
                        print("Invalid choice, please try again.")
                else:
                    print("No datasets found in the 'datasets' folder.")
                    break

            technique = input(
                "Do you want to apply a clustering technique (1) or a dimensionality reduction technique (2)? ").strip().lower()

            if technique == "1":
                print("Choose the clustering technique you want to apply:")
                print("1. KMeans")
                print("2. Agglomerative")
                print("3. DBSCAN")
                response = input("Enter the number of the clustering technique you want to apply: ").strip()

                selected_data = d.replace_missing_values("?")
                d = d.data

                for col in d.columns:
                    if d[col].dtype == object:
                        try:
                            d[col].astype(float)
                        except ValueError:
                            labels_col = col
                            break
                    elif d[col].dtype == 'int':
                        labels_col = col
                labels_original = selected_data.data[labels_col]
                data_original = selected_data.data.drop(selected_data.data.columns[labels_col], axis=1)

                if response == '1':
                    nCluster = int(input("Enter the number of clusters: ").strip().lower())
                    clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "kmeans", nCluster)
                    tmp = clusterTechnique.fit()
                    value = pd.DataFrame(tmp.labels_, columns=['labels'])
                    print("KMeans clustering applied successfully.")
                    print(value)
                elif response == '2':
                    nCluster = int(input("Enter the number of clusters: ").strip().lower())
                    clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean","agglomerative", nCluster)
                    tmp = clusterTechnique.fit()
                    value = pd.DataFrame(tmp.labels_, columns=['labels'])
                    print("Agglomerative clustering applied successfully.")
                    print(value)
                elif response == '3':
                    e = float(input("Enter the value of e (radius from point p to calculate neighbors): ").strip().lower())
                    minPts = int(input("Enter the value of minPts (minimum number of points to be considered part of the cluster): ").strip().lower())
                    clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "dbscan", e, minPts)
                    tmpData, tmpLabels = clusterTechnique.fit()
                    value = pd.DataFrame(tmpLabels.values, columns=['labels'])
                    value['labels'] = value['labels'].replace('noise', -1).astype(int)
                    print("DBSCAN clustering applied successfully.")

                clusteringValidation = input("If you want to perform Clustering Evaluation press 1, otherwise 2: ").strip().lower()
                if clusteringValidation == '1':
                    silhouetteScore, jaccard = qualityMeasure.measureClusteringTecnique(data_original, value, labels_original)
                    print("Silhouette Score: ", silhouetteScore)
                    print("Jaccard Similarity: ", jaccard)
                else:
                    print("Clustering Evaluation not performed.")
                plot = input("If you want to plot the 2D graph press 1, otherwise 2: ").strip().lower()

                if plot == '1':
                    print("Choose the dimensionality reduction technique you want to apply:")
                    print("1. PCA")
                    print("2. t-SNE")
                    print("3. Sammon Mapping")
                    rdTechnique = input("Enter the number of the reduction technique you want to apply: ").strip()
                    if rdTechnique == '1':
                        drResult = dimensionalityReduction(2).reduce(data_original, "PCA")
                        print("Dimensionality reduction with PCA applied successfully.")
                        print(drResult)
                    elif rdTechnique == '2':
                        drResult = dimensionalityReduction(2).reduce(data_original, "t-SNE")
                        print("Dimensionality reduction with t-SNE applied successfully.")
                        print(drResult)
                    elif rdTechnique == '3':
                        nRepetition = int(input("Enter number of repetitions: ").strip())
                        alpha = float(input("Enter the value of alpha: ").strip())
                        drResult, e = dimensionalityReduction(2, nRepetition, alpha, "random").reduce(data_original, "sammonMapping")
                        print("Dimensionality reduction with Sammon Mapping applied successfully.")
                        print(drResult)
                        print("Error: ", e)

                    print("Printing 2D graph...")
                    plotFile.plot2D(drResult, value)
                else:
                    print("Graph not printed.")

                clusteringEvaluetion = input("If you want to perform Clustering Evaluation press 1, otherwise 2: ").strip().lower()

                if clusteringEvaluetion == '1':
                    drResult = dimensionalityReduction(2).reduce(data_original, "PCA")
                    print(drResult)
                    qualityMeasure.clusteringPreservation(drResult, value, response)
                else:
                    print("Clustering Evaluation not performed.")



            elif technique == "2":
                print("Choose the dimensionality reduction technique you want to apply:")
                print("1. PCA")
                print("2. t-SNE")
                print("3. Sammon Mapping")
                rdTechnique = input("Enter the number of the reduction technique you want to apply: ").strip()

                selected_data = d.replace_missing_values("?")
                d = d.data

                for col in d.columns:
                    if d[col].dtype == object:  # Check if the column is of string type
                        try:
                            d[col].astype(float)
                        except ValueError:
                            labels_col = col
                            break
                    elif d[col].dtype == 'int':  # Check if the column is of integer type
                        labels_col = col
                data_original = selected_data.data.drop(selected_data.data.columns[labels_col], axis=1)

                finalDimension = int(input("Enter the final dimension you want to achieve: ").strip())
                if rdTechnique == '1':
                    drResult = dimensionalityReduction(finalDimension).reduce(data_original, "PCA")
                    print("Dimensionality reduction with PCA applied successfully.")
                    print(drResult)
                elif rdTechnique == '2':
                    drResult = dimensionalityReduction(finalDimension).reduce(data_original, "t-SNE")
                    print("Dimensionality reduction with t-SNE applied successfully.")
                    print(drResult)
                elif rdTechnique == '3':
                    alpha = float(input("Enter the scaling parameter: ").strip())
                    itr = int(input("Enter the number of iterations: ").strip())
                    drResult, e = dimensionalityReduction(finalDimension, itr, alpha, "random").reduce(data_original, "sammonMapping")
                    print("Dimensionality reduction with Sammon Mapping applied successfully.")
                    print(drResult)
                    print("Error: ", e)

                plot = input("If you want to plot the 2D graph press 1, otherwise 2: ").strip().lower()
                if plot == '1':
                    print("Printing 2D graph...")
                    plotFile.plot2D(drResult, d[labels_col])

                qualityM = input("If you want to perform Quality Measure press 1, otherwise 2: ").strip().lower()
                if qualityM == '1':
                    if rdTechnique == '1':
                        tot, ex = qualityMeasure.measureDimensionalityReduction(rdTechnique, data_original, drResult)
                        print("Variance per component: ", ex['P1'], ex['P2'])
                        print("Total explained variance: ", tot)
                    elif rdTechnique == '2':
                        corr = qualityMeasure.measureDimensionalityReduction(rdTechnique, data_original, drResult)
                        print("Correlation between original distances and reduced distances: ", corr)
                    elif rdTechnique == '3':
                        stress = qualityMeasure.measureDimensionalityReduction(rdTechnique, data_original, drResult)
                        print("Sammon Stress: ", stress)


                classValidation = input("If you want to perform Class Evaluation press 1, otherwise 2: ").strip().lower()
                if classValidation == '1':
                    data = drResult
                    data['labels'] = d[labels_col]
                    qualityMeasure.classPreservation(data, rdTechnique)
            else:
                print("Invalid option. Please try again.")
                continue
        elif response == '2':
            print("Network analysis")

            while True:
                network_files = [f for f in os.listdir('./datasets/') if
                                 os.path.isfile(os.path.join('./datasets/', f)) and not f.startswith('.')]

                if network_files:
                    print("Available network files:")
                    for i, file in enumerate(network_files, 1):
                        print(f"{i}. {file}")

                    choice = input(
                        f"Enter the number corresponding to the network file you want to use (1-{len(network_files)}): ").strip()

                    try:
                        chosen_file = network_files[int(choice) - 1]
                        datasetPath = os.path.join('./datasets/', chosen_file)
                        print(f"Network file '{chosen_file}' loaded successfully.")
                        G = datasetOperations.Dataset(datasetPath).returnNetwork()
                        break
                    except (ValueError, IndexError):
                        print("Invalid choice, please try again.")
                else:
                    print("No network files found in the 'datasets' folder.")
                    break

            network = networkWithNx(G)
            plot = input("If you want to plot the 2D graph press 1, otherwise 2: ").strip().lower()
            if plot == '1':
                network.plotGraph()
            print("1. PageRank and Betweenness Centrality in the network.")
            print("2. Community Detection.")
            print("3. Operation on graph.")
            analysis = input("Choose the analysis you want to perform: ").strip().lower()
            if analysis == '1':
                alpha = float(input("Enter the damping factor for PageRank: "))
                pagerank = network.computePageRank(alpha)
                print("PageRank: ", pagerank)
                edgeBetweenness = network.edgeBetweeness()
                for edge, centrality in edgeBetweenness.items():
                    print(f"Edge {edge}: {centrality}")
            elif analysis == '2':
                print("1. Fast Newman.")
                print("2. Girvan Newman.")
                communitiesDetectionTechnique = input("Choose the community detection technique you want to apply:")
                if communitiesDetectionTechnique == '1':
                    nCom = int(input("Enter the number of communities: ").strip().lower())
                    detection = CommunityDetection("fastNewman")
                    quality = detection.detection(network.graph, nCom)
                    print("Quality of partitions: ", quality)
                    print("Detected communities: ")
                    plotFile.print_communities(G)
                    plotFile.plot_communities(G)
                elif communitiesDetectionTechnique == '2':
                    detection = CommunityDetection("girvanNewman")
                    communitiesGenerator = detection.detection(network.graph)
                    kCom = int(input(f"Indicate how many communities you want to visualize <= {len(network.getNodes())}: "))
                    if(kCom > len(network.getNodes())):
                        raise Exception("Number of communities exceeds the nodes in the network!")
                    for k in range(0, kCom):
                        level_communities = next(communitiesGenerator)
                        communities = [list(c) for c in level_communities]
                        plotFile.print_communities_GN(G, communities)
            elif analysis == '3':
                print("1. Add a new node")
                print("2. Add a new edge")
                op = input("Choose the operation you want to perform: ").strip().lower()
                if op == '1':
                    node = input("Enter the node you want to add: ")
                    network.addNode(node)
                    print("Node added successfully.")
                    if (input("Press 1 to print all nodes.") == '1'):
                        print(network.getNodes())
                    else:
                        print("Graph not printed.")
                elif op == '2':
                    edge = input("Enter the edge you want to add (e.g. 'node1 node2'): ").split()
                    network.addEdge(edge[0], edge[1])
                    print("Edge added successfully.")
                    if (input("Press 1 to print all edges.") == '1'):
                        print(network.getEdges())
                    else:
                        print("Graph not printed.")

        continueResponse = input("Do you want to perform another operation? (y = yes, n = no): ").strip().lower()
        if continueResponse == 'n':
            print("Exiting the program.")
            sys.exit(0)
        elif continueResponse == 'y':
            continue
        else:
            print("Invalid option. Please try again.")


def clustering_pipeline(datasets, clustering_methods, n_clusters=None, e=None, min_pts=None, plot=True, perform_preservation=True):
    clustering_results = []
    data_original_list = []
    labels_original_list = []

    for dataset_path in datasets:
        print(f"\nProcessing dataset: {dataset_path}")
        d = datasetOperations.Dataset(dataset_path)
        selected_data = d.replace_missing_values("?")
        data = selected_data.data

        for col in data.columns:
            if data[col].dtype == object:
                try:
                    data[col].astype(float)
                except ValueError:
                    labels_col = col
                    break
            elif data[col].dtype == 'int':
                labels_col = col

        labels_original = selected_data.data[labels_col]
        data_original = selected_data.data.drop(selected_data.data.columns[labels_col], axis=1)

        for clustering_method in clustering_methods:
            print(f"\tClustering method: {clustering_method}")

            if clustering_method == 'kmeans':
                clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "kmeans", n_clusters)
            elif clustering_method == 'agglomerative':
                clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "agglomerative",
                                                                  n_clusters)
            elif clustering_method == 'dbscan':
                clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "dbscan", e, min_pts)

            tmp = clusterTechnique.fit()
            if type(tmp) == tuple:
                value = pd.DataFrame(tmp[1].values, columns=['labels'])
                value['labels'] = pd.to_numeric(value['labels'], errors='coerce')
                value['labels'] = value['labels'].replace('noise', -1)
                value['labels'] = value['labels'].fillna(-1).astype(int)
            else:
                value = pd.DataFrame(tmp.labels_, columns=['labels'])

            clustering_results.append(value)
            data_original_list.append(data_original)
            labels_original_list.append(labels_original)

            if plot:
                drResult = dimensionalityReduction(2).reduce(data_original, "PCA")

                plt.figure(figsize=(8, 6))
                plt.scatter(drResult.iloc[:, 0], drResult.iloc[:, 1], c=value['labels'], cmap='viridis')
                plt.title(f'Clustering result ({clustering_method}) after PCA - {dataset_path}')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.show()

    print("\nEvaluating clustering results...")
    for i in range(len(clustering_results)):
        print(f"\nDataset: {datasets[i % len(clustering_methods)]}, Method: {clustering_methods[i % len(clustering_methods)]}")
        s, j = qualityMeasure.measureClusteringTecnique(data_original_list[i], clustering_results[i], labels_original_list[i])
        print("Silhouette Score: ", s)
        print("Jaccard Similarity: ", j)
        if perform_preservation:
            print("\nPerforming Clustering Preservation evaluation...")
            drResult = dimensionalityReduction(2).reduce(data_original_list[i], "PCA")
            qualityMeasure.clusteringPreservation(drResult, clustering_results[i], clustering_methods[i % len(clustering_methods)])


def dimensionality_reduction_pipeline(datasets, reduction_methods, final_dimension, n_repetition=None, alpha=None,  plot=True, perform_preservation=True):
    for dataset_path in datasets:
        print(f"\nProcessing dataset: {dataset_path}")
        # Load dataset
        d = datasetOperations.Dataset(dataset_path)
        selected_data = d.replace_missing_values("?")
        data = selected_data.data

        # Separate labels
        for col in data.columns:
            if data[col].dtype == object:
                try:
                    data[col].astype(float)
                except ValueError:
                    labels_col = col
                    break
            elif data[col].dtype == 'int':
                labels_col = col
        labels_original = selected_data.data[labels_col]
        data_original = selected_data.data.drop(selected_data.data.columns[labels_col], axis=1)

        # Apply dimensionality reduction methods
        for reduction_method in reduction_methods:
            print(f"\nApplying {reduction_method} to dataset {dataset_path}")
            if reduction_method == 'PCA':
                drResult = dimensionalityReduction(final_dimension).reduce(data_original, "PCA")
            elif reduction_method == 't-SNE':
                drResult = dimensionalityReduction(final_dimension).reduce(data_original, "t-SNE")
            elif reduction_method == 'sammonMapping':
                drResult, e = dimensionalityReduction(final_dimension, n_repetition, alpha, "random").reduce(
                    data_original, "sammonMapping")

            # Print the results of dimensionality reduction
            print(f"Dimensionality reduction with {reduction_method} applied successfully.")
            print(drResult)

            if type(labels_original.values[0]) == str:
                labels_original = labels_original.astype('category').cat.codes

            data = drResult
            data['labels'] = labels_original
            if plot:
                # Optionally, plot the results
                plt.figure(figsize=(8, 6))
                plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels_original, cmap='viridis')
                plt.title(f'{reduction_method} result - {dataset_path}')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.show()

            # Perform class preservation if required
            if perform_preservation:
                print("\nPerforming Class Preservation evaluation...")
                qualityMeasure.classPreservation(data, reduction_method)  # Assuming labels_original is the true class


def plot_communities(G, communities, method, file_name):
    """
    Visualizza il grafo con i nodi colorati in base alla comunità.

    Args:
        G (networkx.Graph): Il grafo da visualizzare.
        communities (list): Lista delle comunità, ciascuna contenente nodi.
        method (str): Metodo di rilevamento comunitario utilizzato.
        file_name (str): Nome del file di rete elaborato.
    """
    num_communities = len(communities)
    color_map = plt.get_cmap('tab20')  # Usa 'tab20' per palette discrete
    colors = [color_map(i / max(num_communities - 1, 1)) for i in range(num_communities)]

    # Assegna un colore a ogni comunità
    node_colors = {}
    for i, community in enumerate(communities):
        for node in community:
            node_colors[node] = colors[i]

    # Disegna il grafo
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True,
            node_color=[node_colors.get(node, 'gray') for node in G.nodes()],
            node_size=500, edge_color='gray', font_size=10)
    plt.title(f"{method} - Communities in {file_name}")
    plt.show()

def community_detection_pipeline(network_datasets, n_communities=None):
    for i in range(len(network_datasets)):
        # Carica il grafo
        print(f"\nProcessing network file: {network_datasets[i]}")
        G = datasetOperations.Dataset(network_datasets[i]).returnNetwork()
        network = networkWithNx(G)

        # Metodo Girvan-Newman
        print(f"Running Girvan-Newman on {network_datasets[i]}...")
        detection = CommunityDetection("girvanNewman")
        communitiesGenerator = detection.detection(network.graph)
        communities = next(communitiesGenerator)  # Prendi il primo livello di comunità
        communities = [list(c) for c in communities]
        print(f"Detected communities (Girvan-Newman): {communities}")
        plot_communities(G, communities, "Girvan-Newman", os.path.basename(network_datasets[i]))

        # Metodo Fast Newman
        print(f"Running Fast Newman on {network_datasets[i]}...")
        detection = CommunityDetection("fastNewman")
        quality = detection.detection(network.graph, n_communities[i])
        print("Quality of partitions: ", quality)
        print("Detected communities: ")
        plotFile.print_communities(G)
        plotFile.plot_communities(G)



if __name__ == '__main__':
    action = input("To run the framework manually click 1 or if you want test the code with pipeline click 2: ").strip().lower()
    if action == '1':
        frameworkRun()
    else:
        print("Running clustering pipeline...")
        datasets = ['./datasets/dataset1.txt', './datasets/dataset2.txt', './datasets/dataset3.txt']
        clustering_methods = ['kmeans', 'agglomerative', 'dbscan']
        n_clusters = 2
        e = 0.001
        min_pts = 10
        clustering_pipeline(datasets, clustering_methods, n_clusters=n_clusters, e=e, min_pts=min_pts, plot=True, perform_preservation=True)
        dimensionality_reduction_pipeline(datasets, ['PCA', 't-SNE', 'sammonMapping'], final_dimension=2, n_repetition=2, alpha=0.85, plot=True, perform_preservation=True)
        networkDataset = ['./datasets/karate.edgelist', './datasets/les_miserables.edgelist', './datasets/three_communities.edgelist']
        n_communities = [2,5,3]
        community_detection_pipeline(networkDataset, n_communities=n_communities)
