from re import error
import pandas as pd
import plotFile
import numpy as np
import datasetOperations
import clustering
import qualityMeasure
from Network.comunityDetection import CommunityDetection
from Network.networkWithNx import netoworkWithNx
from dimensionalityReduction import dimensionalityReduction
import sys

def frameworkRun():
    while True:  # Ciclo infinito per consentire all'utente di ripetere le operazioni
        print("Scegli l'operazione che vuoi svolgere...")
        print("1. Per fare operazioni di clustering e/o riduzione della dimensionalità")
        print("2. Analisi della rete")

        risposta = input("Inserisci il numero dell'operazione che vuoi svolgere: ").strip()

        if risposta == '1':
            print("Operazioni di clustering e/o riduzione della dimensionalità")
            datasetPath = input("Inserisci il nome del dataset compreso di estensione: ").strip()
            d = datasetOperations.Dataset("./datasets/" + datasetPath)
            print("Dataset caricato con successo.")
            tecnica = input("Vuoi applicare una tecnica di clusterizzazione (1) o una tecnica di riduzione della dimensionalità (2)? ").strip().lower()

            if tecnica == "1":
                print("Scegli la tecnica di clusterizzazione che vuoi applicare:")
                print("1. KMeans")
                print("2. Agglomerative")
                print("3. DBSCAN")
                risposta = input("Inserisci il numero della tecnica di clusterizzazione che vuoi applicare: ").strip()

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
                    elif d[col].dtype == 'int':
                        labels_col = col
                labels_original = selected_data.data[labels_col]
                data_original = selected_data.data.drop(selected_data.data.columns[labels_col], axis=1)

                if risposta == '1':
                    nCluster = int(input("Inserisci il numero di cluster. ").strip().lower())
                    clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "kmeans", nCluster)
                    tmp = clusterTechnique.fit()
                    value = pd.DataFrame(tmp.labels_)
                    print("Clusterizzazione con KMeans applicata con successo.")
                    print(value)
                elif risposta == '2':
                    nCluster = int(input("Inserisci il numero di cluster.").strip().lower())
                    clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean","agglomerative", nCluster)
                    tmp = clusterTechnique.fit()
                    value = pd.DataFrame(tmp.labels_)
                    print("Clusterizzazione con Agglomerative applicata con successo.")
                    print(value)
                elif risposta == '3':
                    e = float(input("Inserisci il valore di e(raggio dal punto p per calcolare i vicini). ").strip().lower())
                    minPts = int(input("Inserisci il valore di minPts(il numero minimo di punti per essere considerato elemento del cluster). ").strip().lower())
                    clusterTechnique = clustering.ClusteringAlgorithm(data_original.values, "euclidean", "dbscan", e, minPts)
                    tmpData, tmpLabels = clusterTechnique.fit()
                    value = pd.DataFrame(tmpLabels)
                    print("Clusterizzazione con DBSCAN applicata con successo.")
                    print(value)

                clusteringValidation = input("Se vuoi effettuare la Clustering Evaluation premi 1 altrimenti 2: ").strip().lower()
                if clusteringValidation == '1':
                    silhouetteScore, jaccard = qualityMeasure.measureClusteringTecnique(data_original, value, labels_original)
                    print("Silhouette Score: ", silhouetteScore)
                    print("Jaccard Similarity: ", jaccard)
                plot = input("Se vuoi stampare il grafico in 2D premi 1 altrimenti 2? ").strip().lower()

                if plot == '1':
                    print("Scegli la tecnica di riduzione della dimensionalità che vuoi applicare:")
                    print("1. PCA")
                    print("2. t-SNE")
                    print("3. Sammon Mapping")
                    rdTechnique = input("Inserisci il numero della tecnica di riduzione che vuoi applicare: ").strip()
                    finalDimension = int(input("Inserisci la dimensione finale che vuoi ottenere: ").strip())
                    if rdTechnique == '1':
                        drResult = dimensionalityReduction(finalDimension).reduce(data_original, "PCA")
                        print("Riduzione della dimensionalità con PCA applicata con successo.")
                        print(drResult)
                    elif rdTechnique == '2':
                        drResult = dimensionalityReduction(finalDimension).reduce(data_original, "t-SNE")
                        print("Riduzione della dimensionalità con t-SNE applicata con successo.")
                        print(drResult)
                    elif rdTechnique == '3':
                        nRepetition = int(input("Inserisci la dimensione finale che vuoi ottenere: ").strip())
                        alpha = int(input("Inserisci il valore di alpha: ").strip())
                        drResult, e = dimensionalityReduction(finalDimension, nRepetition, alpha, "random").reduce(data_original,
                                                                                               "sammonMapping")
                        print("Riduzione della dimensionalità con Sammon Mapping applicata con successo.")
                        print(drResult)
                        print("Errore: ", e)

                    print("Stampa grafico in 2D...")
                    plotFile.plot2D(drResult, value)
                elif plot == '2':
                    continue
                else:
                    print("Opzione non valida. Riprova.")


            elif tecnica == "2":
                print("Scegli la tecnica di riduzione della dimensionalità che vuoi applicare:")
                print("1. PCA")
                print("2. t-SNE")
                print("3. Sammon Mapping")
                rdTechnique = input("Inserisci il numero della tecnica di riduzione che vuoi applicare: ").strip()

                selected_data = d.replace_missing_values("?")
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

                finalDimension = int(input("Inserisci la dimensione finale che vuoi ottenere: ").strip())
                if rdTechnique == '1':
                    drResult = dimensionalityReduction(finalDimension).reduce(data_original, "PCA")
                    print("Riduzione della dimensionalità con PCA applicata con successo.")
                    print(drResult)
                elif rdTechnique == '2':
                    drResult = dimensionalityReduction(finalDimension).reduce(data_original, "t-SNE")
                    print("Riduzione della dimensionalità con t-SNE applicata con successo.")
                    print(drResult)
                elif rdTechnique == '3':
                    nRepetition = int(input("Inserisci la dimensione finale che vuoi ottenere: ").strip())
                    alpha = int(input("Inserisci il valore di alpha: ").strip())
                    drResult, e = dimensionalityReduction(finalDimension, nRepetition, alpha, "random").reduce(
                        data_original,
                        "sammonMapping")
                    print("Riduzione della dimensionalità con Sammon Mapping applicata con successo.")
                    print(drResult)
                    print("Errore: ", e)

                plot = input("Se vuoi stampare il grafico in 2D premi 1 altrimenti 2? ").strip().lower()
                if plot == '1':
                    print("Stampa grafico in 2D...")
                    plotFile.plot2D(drResult, d[labels_col])
                elif plot == '2':
                    continue
                else:
                    print("Opzione non valida. Riprova.")
                    continue
            else:
                print("Opzione non valida. Riprova.")
                continue
        elif risposta == '2':
            print("Analisi della rete")
            datasetPath = input("Inserisci il nome del file per caricare la rete: ").strip()
            G = datasetOperations.Dataset('./datasets/' + datasetPath).returnNetwork()
            print("Rete caricata con successo.")
            network = netoworkWithNx(G)
            plot = input("Se vuoi stampare il grafico in 2D premi 1 altrimenti 2? ").strip().lower()
            if plot == '1':
                network.plotGraph()
            print("1. PageRank e Betweenness Centrality nella rete.")
            print("2. Community Detection.")
            analysis = input("Scegli l'analisi che vuoi fare: ").strip().lower()
            if analysis == '1':
                alpha = float(input("Inserisci il damping factor per il PageRank: "))
                pagerank = network.computePageRank(alpha)
                print("PageRank: ", pagerank)
                edgeBetweenness = network.edgeBetweeness()
                for edge, centrality in edgeBetweenness.items():
                    print(f"Arco {edge}: {centrality}")
            elif analysis == '2':
                print("1. Fast Newman.")
                print("2. Girvan Newman.")
                communitiesDetectionTechnique = input("Scegli la tecnica di community detection che vuoi applicare:")
                if communitiesDetectionTechnique == '1':
                    nCom = int(input("Inserisci il numero di comunità: ").strip().lower())
                    detection = CommunityDetection("fastNewman")
                    quality = detection.detection(network.graph, nCom)
                    print("Qualità delle partizioni: ", quality)
                    print("Comunità rilevate: ")
                    plotFile.print_communities(G)
                    plotFile.plot_communities(G)
                elif communitiesDetectionTechnique == '2':
                    detection = CommunityDetection("girvanNewman")
                    communitiesGenerator = detection.detection(network.graph)
                    kCom = int(input(f"Indica quante comunità vuoi visualizzare <= {len(network.getNodes())}: "))
                    if(kCom > len(network.getNodes())):
                        raise Exception("Numero delle comunità maggiore dei nodi nella rete!")
                    for k in range(0, kCom):
                        level_communities = next(communitiesGenerator)
                        communities = [list(c) for c in level_communities]
                        plotFile.print_communities_GN(G, communities)

        risposta_continuare = input("Vuoi fare un'altra operazione? (s = sì, n = no): ").strip().lower()
        if risposta_continuare == 'n':
            print("Uscita dal programma.")
            sys.exit(0)
        elif risposta_continuare == 's':
            continue
        else:
            print("Opzione non valida. Riprova.")


if __name__ == '__main__':

    frameworkRun()



