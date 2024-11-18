import csv
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


def clustering_pipeline(datasets, clustering_methods, n_clusters=None, e=None, min_pts=None, plot=True,
                        perform_preservation=True):
    results_list = []

    # To track the best results
    best_results = {
        'best_silhouette': {'dataset': None, 'method': None, 'score': -float('inf')},
        'best_jaccard': {'dataset': None, 'method': None, 'score': -float('inf')},
        'best_preservation': {'dataset': None, 'method': None, 'score': -float('inf')}
    }

    for dataset_path in datasets:
        clustering_results = []
        data_original_list = []
        labels_original_list = []
        print(f"\nProcessing dataset: {dataset_path}")
        d = datasetOperations.Dataset(dataset_path)
        selected_data = d.replace_missing_values("?")
        data = selected_data.data

        # Identify the label column
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

        # Loop over clustering methods
        for clustering_method in clustering_methods:
            print(f"\tClustering method: {clustering_method}")

            if clustering_method == 'kmeans':
                clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "kmeans",
                                                                  n_clusters)
            elif clustering_method == 'agglomerative':
                clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "agglomerative",
                                                                  n_clusters)
            elif clustering_method == 'dbscan':
                clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "dbscan", e,
                                                                  min_pts)

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
                plotFile.plot2D(drResult, value)

        print("\nEvaluating clustering results...")
        for i in range(len(clustering_results)):
            method = clustering_methods[i % len(clustering_methods)]
            print(f"\nDataset: {dataset_path}, Method: {method}")

            s, j = qualityMeasure.measureClusteringTecnique(data_original_list[i], clustering_results[i],
                                                            labels_original_list[i])
            print("Silhouette Score: ", s)
            print("Jaccard Similarity: ", j)

            if s > best_results['best_silhouette']['score']:
                best_results['best_silhouette'] = {'dataset': dataset_path, 'method': method, 'score': s}

            if j > best_results['best_jaccard']['score']:
                best_results['best_jaccard'] = {'dataset': dataset_path, 'method': method, 'score': j}

            value = 0
            if perform_preservation:
                print("\nPerforming Clustering Preservation evaluation...")
                drResult = dimensionalityReduction(2).reduce(data_original_list[i], "PCA")
                value = qualityMeasure.clusteringPreservation(drResult, clustering_results[i], method)
                print("Clustering Preservation Score: ", value)

                if value > best_results['best_preservation']['score']:
                    best_results['best_preservation'] = {'dataset': dataset_path, 'method': method, 'score': value}

            # Append the results to the list
            results_list.append({
                'dataset': dataset_path,
                'method': method,
                'silhouette_score': s,
                'jaccard_similarity': j,
                'clustering_preservation': value
            })

    # Convert the list to a DataFrame and save to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('clustering_quality_measures.csv', index=False)
    print("Quality measures saved to 'clustering_quality_measures.csv'")

    return best_results


def dimensionality_reduction_pipeline(datasets, reduction_methods, final_dimension, n_repetition=None, alpha=None, plot=True, perform_preservation=True):
    results_list = []
    best_results_per_dataset = []

    global_best_trustworthiness = -float('inf')
    global_best_continuity = -float('inf')
    global_best_class_preservation = -float('inf')

    global_best_trustworthiness_method = ''
    global_best_continuity_method = ''
    global_best_class_preservation_method = ''

    global_best_trustworthiness_dataset = ''
    global_best_continuity_dataset = ''
    global_best_class_preservation_dataset = ''

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

        best_trustworthiness = -float('inf')
        best_continuity = -float('inf')
        best_class_preservation = -float('inf')
        best_method_trustworthiness = ''
        best_method_continuity = ''
        best_method_class_preservation = ''

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

            print(f"Dimensionality reduction with {reduction_method} applied successfully.")
            print(drResult)

            if type(labels_original.values[0]) == str:
                labels_original = labels_original.astype('category').cat.codes

            if plot:
                plotFile.plot2D(drResult, labels_original)

            # Calculate trustworthiness and continuity for the reduced data
            trustworthiness_score = qualityMeasure.trustworthinessForDimensionalityReduction(data_original.values, drResult.values)
            continuity_score = qualityMeasure.continuityForDimensionalityReduction(data_original.values, drResult.values)

            print(f"Trustworthiness for {reduction_method}: {trustworthiness_score:.4f}")
            print(f"Continuity for {reduction_method}: {continuity_score:.4f}")

            value = 0
            if perform_preservation:
                print("\nPerforming Class Preservation evaluation...")
                drData = drResult.copy()
                drData['labels'] = labels_original
                value = qualityMeasure.classPreservation(drData, reduction_method)
                print(f"Class Preservation for {reduction_method}: {value:.4f}")

            # Save the results
            results_list.append({
                'dataset': dataset_path,
                'reduction_method': reduction_method,
                'trustworthiness': trustworthiness_score,
                'continuity': continuity_score,
                'class_preservation': value
            })

            if trustworthiness_score > best_trustworthiness:
                best_trustworthiness = trustworthiness_score
                best_method_trustworthiness = reduction_method

            if continuity_score > best_continuity:
                best_continuity = continuity_score
                best_method_continuity = reduction_method

            if value > best_class_preservation:
                best_class_preservation = value
                best_method_class_preservation = reduction_method

            if trustworthiness_score > global_best_trustworthiness:
                global_best_trustworthiness = trustworthiness_score
                global_best_trustworthiness_method = reduction_method
                global_best_trustworthiness_dataset = dataset_path

            if continuity_score > global_best_continuity:
                global_best_continuity = continuity_score
                global_best_continuity_method = reduction_method
                global_best_continuity_dataset = dataset_path

            if value > global_best_class_preservation:
                global_best_class_preservation = value
                global_best_class_preservation_method = reduction_method
                global_best_class_preservation_dataset = dataset_path

        # Append the best results for the current dataset
        best_results_per_dataset.append({
            'dataset': dataset_path,
            'best_trustworthiness_method': best_method_trustworthiness,
            'best_trustworthiness': best_trustworthiness,
            'best_continuity_method': best_method_continuity,
            'best_continuity': best_continuity,
            'best_class_preservation_method': best_method_class_preservation,
            'best_class_preservation': best_class_preservation
        })

    # Convert the results list to DataFrame and save to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('dimensionality_reduction_quality_measures.csv', index=False)
    print("Quality measures saved to 'dimensionality_reduction_quality_measures.csv'")

    # Return the best global results
    return {
        'best_results_per_dataset': best_results_per_dataset,
        'global_best_trustworthiness': global_best_trustworthiness,
        'global_best_trustworthiness_method': global_best_trustworthiness_method,
        'global_best_trustworthiness_dataset': global_best_trustworthiness_dataset,
        'global_best_continuity': global_best_continuity,
        'global_best_continuity_method': global_best_continuity_method,
        'global_best_continuity_dataset': global_best_continuity_dataset,
        'global_best_class_preservation': global_best_class_preservation,
        'global_best_class_preservation_method': global_best_class_preservation_method,
        'global_best_class_preservation_dataset': global_best_class_preservation_dataset
    }





def community_detection_pipeline(network_datasets, n_communities=None):
    results = []
    output_csv = "community_detection_results.csv"

    for i in range(len(network_datasets)):
        print(f"\nProcessing network file: {network_datasets[i]}")
        G = datasetOperations.Dataset(network_datasets[i]).returnNetwork()
        network = networkWithNx(G)

        print(f"Running Girvan-Newman on {network_datasets[i]}...")
        detection = CommunityDetection("girvanNewman")
        communitiesGenerator = detection.detection(network.graph)
        communities_gn = next(communitiesGenerator)  # Prendi il primo livello di comunità
        communities_gn = [list(c) for c in communities_gn]
        print(f"Detected communities (Girvan-Newman): {communities_gn}")
        detection.plot_communities(G, communities_gn, "Girvan-Newman", os.path.basename(network_datasets[i]))

        modularity_gn = qualityMeasure.modularity(network.graph, communities_gn)
        density_gn = qualityMeasure.internalDensity(network.graph, communities_gn)
        ratio_cut_gn = qualityMeasure.ratioCut(network.graph, communities_gn)
        print(f"Quality Metrics (Girvan-Newman):")
        print(f"  Modularity: {modularity_gn:.4f}")
        print(f"  Internal Density: {density_gn:.4f}")
        print(f"  Ratio Cut: {ratio_cut_gn:.4f}")

        results.append({
            "Dataset": os.path.basename(network_datasets[i]),
            "Algorithm": "Girvan-Newman",
            "Modularity": modularity_gn,
            "Internal Density": density_gn,
            "Ratio Cut": ratio_cut_gn,
            "Communities": communities_gn
        })

        print(f"Running Fast Newman on {network_datasets[i]}...")
        detection = CommunityDetection("fastNewman")
        quality_fn, communities_fn = detection.detection(network.graph, n_communities[i])

        print("Detected communities (Fast Newman):", communities_fn)
        plotFile.print_communities(G)
        plotFile.plot_communities(G)

        modularity_fn = qualityMeasure.modularity(network.graph, communities_fn)
        density_fn = qualityMeasure.internalDensity(network.graph, communities_fn)
        ratio_cut_fn = qualityMeasure.ratioCut(network.graph, communities_fn)
        print(f"Quality Metrics (Fast Newman):")
        print(f"  Modularity: {modularity_fn:.4f}")
        print(f"  Internal Density: {density_fn:.4f}")
        print(f"  Ratio Cut: {ratio_cut_fn:.4f}")

        results.append({
            "Dataset": os.path.basename(network_datasets[i]),
            "Algorithm": "Fast Newman",
            "Modularity": modularity_fn,
            "Internal Density": density_fn,
            "Ratio Cut": ratio_cut_fn,
            "Communities": communities_fn
        })

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Dataset", "Algorithm", "Modularity", "Internal Density", "Ratio Cut",
                                                  "Communities"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"\nResults saved to {output_csv}")

    return results


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
        best_results_cluster = clustering_pipeline(datasets, clustering_methods, n_clusters=n_clusters, e=e, min_pts=min_pts, plot=True, perform_preservation=True)

        best_results_dr = dimensionality_reduction_pipeline(datasets, ['PCA', 't-SNE', 'sammonMapping'], final_dimension=2, n_repetition=2, alpha=0.85, plot=True, perform_preservation=True)

        networkDataset = ['./datasets/karate.edgelist', './datasets/les_miserables.edgelist', './datasets/three_communities.edgelist']
        n_communities = [2,5,3]
        results = community_detection_pipeline(networkDataset, n_communities=n_communities)


        print("----------------------------------Dimensional clustering analysis----------------------------------")


        print("Best Results:")
        print("-----------------")
        print(
            f"Best Silhouette Score: \n  Dataset: {best_results_cluster['best_silhouette']['dataset']}\n  Method: {best_results_cluster['best_silhouette']['method']}\n  Score: {best_results_cluster['best_silhouette']['score']:.4f}\n")
        print(
            f"Best Jaccard Similarity: \n  Dataset: {best_results_cluster['best_jaccard']['dataset']}\n  Method: {best_results_cluster['best_jaccard']['method']}\n  Score: {best_results_cluster['best_jaccard']['score']:.4f}\n")
        print(
            f"Best Clustering Preservation Score: \n  Dataset: {best_results_cluster['best_preservation']['dataset']}\n  Method: {best_results_cluster['best_preservation']['method']}\n  Score: {best_results_cluster['best_preservation']['score']:.4f}\n")


        print("----------------------------------Dimensional reduction quality analysis----------------------------------")

        print("Best Results Across All Datasets:")
        print("-----------------")

        print(
            f"Best Trustworthiness Score: \n"
            f"  Dataset: {best_results_dr['global_best_trustworthiness_dataset']}\n"
            f"  Method: {best_results_dr['global_best_trustworthiness_method']}\n"
            f"  Score: {best_results_dr['global_best_trustworthiness']:.4f}\n"
        )

        print(
            f"Best Continuity Score: \n"
            f"  Dataset: {best_results_dr['global_best_continuity_dataset']}\n"
            f"  Method: {best_results_dr['global_best_continuity_method']}\n"
            f"  Score: {best_results_dr['global_best_continuity']:.4f}\n"
        )

        print(
            f"Best Class Preservation Score: \n"
            f"  Dataset: {best_results_dr['global_best_class_preservation_dataset']}\n"
            f"  Method: {best_results_dr['global_best_class_preservation_method']}\n"
            f"  Score: {best_results_dr['global_best_class_preservation']:.4f}\n"
        )

        print("----------------------------------Community detection quality analysis----------------------------------")

        best_result_network = max(results, key=lambda x: x["Modularity"])
        print(f"\nBest result based on modularity:")
        print(f"  Dataset: {best_result_network['Dataset']}")
        print(f"  Algorithm: {best_result_network['Algorithm']}")
        print(f"  Modularity: {best_result_network['Modularity']:.4f}")
        print(f"  Communities: {best_result_network['Communities']}")

        best_result_network = max(results, key=lambda x: x["Internal Density"])
        print(f"\nBest result based on internal density:")
        print(f"  Dataset: {best_result_network['Dataset']}")
        print(f"  Algorithm: {best_result_network['Algorithm']}")
        print(f"  Internal Density: {best_result_network['Internal Density']:.4f}")
        print(f"  Communities: {best_result_network['Communities']}")

        best_result_network = max(results, key=lambda x: x["Ratio Cut"])
        print(f"\nBest result based on ratio cut:")
        print(f"  Dataset: {best_result_network['Dataset']}")
        print(f"  Algorithm: {best_result_network['Algorithm']}")
        print(f"  Ratio Cut: {best_result_network['Ratio Cut']:.4f}")
        print(f"  Communities: {best_result_network['Communities']}")