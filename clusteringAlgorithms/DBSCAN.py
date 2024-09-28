import tqdm
import distance

def rangeQuery(data, pointIndex, e, distanceFunction):
    neighbors = []
    point = data[pointIndex][:-1]
    for i, row in enumerate(data):
        if i != pointIndex:
            neighborPoint = row[:-1]
            dTmp = distance.distance(point, neighborPoint, distanceFunction)
            if dTmp <= e:
                neighbors.append(i)
    return neighbors

def dbscan(data, e, minPts, distanceFunction):
    label = 0
    for pointIndex in tqdm(range(len(data)), desc="DBSCAN Progress"):
        if data[pointIndex][-1] == "undefined":
            neighbors = rangeQuery(data, pointIndex, e, distanceFunction)
            if len(neighbors) < minPts:
                data[pointIndex][-1] = "noise"
            else:
                label += 1
                data[pointIndex][-1] = str(label)
                while neighbors:
                    i = neighbors.pop(0)
                    if data[i][-1] == "noise":
                        data[i][-1] = str(label)
                    if data[i][-1] == "undefined":
                        data[i][-1] = str(label)
                        new_neighbors = rangeQuery(data, i, e, distanceFunction)
                        if len(new_neighbors) >= minPts:
                            neighbors.extend([x for x in new_neighbors if x not in neighbors])
