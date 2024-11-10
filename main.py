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
    while True:  # Infinite loop to allow the user to repeat operations
        print("Choose the operation you want to perform...")
        print("1. Clustering and/or dimensionality reduction operations")
        print("2. Network analysis")

        response = input("Enter the number of the operation you want to perform: ").strip()

        if response == '1':
            print("Clustering and/or dimensionality reduction operations")
            datasetPath = input("Enter the dataset name including the extension: ").strip()
            d = datasetOperations.Dataset("./datasets/" + datasetPath)
            print("Dataset loaded successfully.")
            technique = input("Do you want to apply a clustering technique (1) or a dimensionality reduction technique (2)? ").strip().lower()

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
                    value = pd.DataFrame(tmp.labels_)
                    print("KMeans clustering applied successfully.")
                    print(value)
                elif response == '2':
                    nCluster = int(input("Enter the number of clusters: ").strip().lower())
                    clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean","agglomerative", nCluster)
                    tmp = clusterTechnique.fit()
                    value = pd.DataFrame(tmp.labels_)
                    print("Agglomerative clustering applied successfully.")
                    print(value)
                elif response == '3':
                    e = float(input("Enter the value of e (radius from point p to calculate neighbors): ").strip().lower())
                    minPts = int(input("Enter the value of minPts (minimum number of points to be considered part of the cluster): ").strip().lower())
                    clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "dbscan", e, minPts)
                    tmpData, tmpLabels = clusterTechnique.fit()
                    value = pd.DataFrame(tmpLabels)
                    print("DBSCAN clustering applied successfully.")
                    print(value)

                clusteringValidation = input("If you want to perform Clustering Evaluation press 1, otherwise 2: ").strip().lower()
                if clusteringValidation == '1':
                    silhouetteScore, jaccard = qualityMeasure.measureClusteringTecnique(data_original, value, labels_original)
                    print("Silhouette Score: ", silhouetteScore)
                    print("Jaccard Similarity: ", jaccard)
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
                        nRepetition = int(input("Enter the final dimension you want to obtain: ").strip())
                        alpha = int(input("Enter the value of alpha: ").strip())
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
            datasetPath = input("Enter the name of the file to load the network: ").strip()
            G = datasetOperations.Dataset('./datasets/' + datasetPath).returnNetwork()
            print("Network loaded successfully.")
            network = netoworkWithNx(G)
            plot = input("If you want to plot the 2D graph press 1, otherwise 2: ").strip().lower()
            if plot == '1':
                network.plotGraph()
            print("1. PageRank and Betweenness Centrality in the network.")
            print("2. Community Detection.")
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
