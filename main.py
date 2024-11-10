import os
import pandas as pd
import plotFile
import datasetOperations
import clustering
from Network.comunityDetection import CommunityDetection
from Network.Network import netoworkWithNx
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
                    qualityMeasure.clusteringPreservation(drResult, value)
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
                    qualityMeasure.classPreservation(data)
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

            network = netoworkWithNx(G)
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


if __name__ == '__main__':

    frameworkRun()
