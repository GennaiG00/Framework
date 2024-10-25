import numpy as np
import pandas as pd
from tqdm import tqdm
import distance

class DBSCAN:
    def __init__(self, data, distanceMeasure, e, minPts):
        self.dist = distance.Distance(distanceMeasure)
        self.distanceMeasure = distanceMeasure
        self.e = e
        self.minPts = minPts
        self.data = data
        self.data["labels"] = "undefined"
        print("DBSCAN object created")

    def rangeQuery(self, pointIndex):
        neighbors = []
        point = self.data[pointIndex][:-1]
        for i, row in enumerate(self.data):
            if i != pointIndex:
                neighborPoint = row[:-1]
                dTmp = self.dist.calculate_distance(point, neighborPoint)
                if dTmp <= self.e:
                    neighbors.append(i)
        return neighbors

    def dbscan(self):
        label = 0
        self.data = np.array(self.data)

        # Aggiungi tqdm per il ciclo principale con la descrizione "DBSCAN Progress"
        for pointIndex in tqdm(range(len(self.data)), desc="DBSCAN Progress"):
            if self.data[pointIndex][-1] == "undefined":
                neighbors = self.rangeQuery(pointIndex)
                if len(neighbors) < self.minPts:
                    self.data[pointIndex][-1] = "noise"
                else:
                    label += 1
                    self.data[pointIndex][-1] = str(label)

                    # Aggiungi tqdm per tracciare il progresso del ciclo while
                    with tqdm(total=len(neighbors), desc=f"Expanding Cluster {label}", leave=False) as pbar:
                        while neighbors:
                            i = neighbors.pop(0)
                            if self.data[i][-1] == "noise":
                                self.data[i][-1] = str(label)
                            if self.data[i][-1] == "undefined":
                                self.data[i][-1] = str(label)
                                new_neighbors = self.rangeQuery(i)
                                if len(new_neighbors) >= self.minPts:
                                    neighbors.extend([x for x in new_neighbors if x not in neighbors])
                            pbar.update(1)  # Aggiorna la barra di progresso nel ciclo while

        return self.data
