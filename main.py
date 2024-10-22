import pandas as pd

import datasetOperations
from plotFile import plotClusters
import qualityMeasure as qm
import clusteringAlgorithms.DBSCAN as DBSCAN
from dimensionalityReduction import dimensionalityReduction
import numpy as np

def preprocess_labels(labels):
    # Sostituisce 'noise' con -1 nei label
    processed_labels = np.where(labels == 'noise', -1, labels)
    return processed_labels.astype(int)

if __name__ == '__main__':
    datasetPath = './dataset/household_power_consumption.txt'
    d = datasetOperations.Dataset(datasetPath)

    # Chain replace_missing_values and select_columns
    selected_data = (d.replace_missing_values("?"))
    labels = selected_data.data["Date"]
    selected_data = d.select_columns(["Global_active_power", "Global_reactive_power", "Voltage",
                                      "Global_intensity", "Sub_metering_1", "Sub_metering_2",
                                      "Sub_metering_3"])
    #data = dataset.select_columns(["duration", "credit_amount", "installment_commitment", "age"])
    #data = dataset.encode_columns(data)
    #data, pca = dimensionalityReduction(selected_data.values, n_components=2).reduce("PCA")
    data, e = dimensionalityReduction(2, 1, 0.3, "random").reduce(selected_data.values, "sammonMapping")
    data = pd.DataFrame(data, columns=['P1', 'P2'])
    dbscan = DBSCAN.DBSCAN(data, "euclidean" , 0.01, 10)
    result = dbscan.dbscan()
    plotClusters(result)
    print(qm.measureClusteringTecnique("clustering", result))
    #print(qm.measureDimensionalityReduction("pca", pca))






