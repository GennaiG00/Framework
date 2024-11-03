import csv
import numpy as np
from tqdm import tqdm

def readFromFile(fileName):
    with open(fileName, mode='r') as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]
    return data

def writePointsToFile(fileName, data):
    with open(fileName, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        for point in data:
            writer.writerow(point)

def euclideanDistance(pointOne, pointTwo):
    if len(pointOne) == len(pointTwo):
        tot = 0
        for i in range(len(pointOne)):
            tot += np.abs(pointOne[i] - pointTwo[i]) ** 2
        return tot ** (1 / 2)

def startSammonMapping(originalD, y, c, mf=0.3):
    n = y.shape[0]
    dimension = y.shape[1]
    updateD = np.array([[euclideanDistance(y[i], y[j]) for j in range(n)] for i in range(n)])
    yNew = np.copy(y)
    for p in range(n):
        for q in range(dimension):
            g = gradient(y, p, q, originalD, updateD, c)
            h = hessian(y, p, q, originalD, updateD, c)
            if h != 0:
                delta = g / abs(h)
            else:
                delta = 0
            yNew[p][q] -= mf * delta
    E = np.sum(((originalD - updateD) ** 2) * updateD) / c
    return yNew, E

def gradient(y, p, q, originalD, updateD, c):
    accum = 0
    for j in range(y.shape[0]):
        if j != p and originalD[p][j] > 0 and updateD[p][j] > 0:
            a = (originalD[p][j] - updateD[p][j]) / (originalD[p][j] * updateD[p][j])
            b = y[p][q] - y[j][q]
            accum += a * b
    return (-2 / c) * accum

def hessian(y, p, q, originalD, updateD, c):
    accum = 0
    for j in range(y.shape[0]):
        if j != p and updateD[p][j] > 0 and originalD[p][j] > 0:
            a = 1 / (originalD[p][j] * updateD[p][j])
            b = originalD[p][j] - updateD[p][j]
            e = ((y[p][q] - y[j][q]) ** 2) / updateD[p][j]
            d = 1 + ((originalD[p][j] - updateD[p][j]) / updateD[p][j])
            accum += a * (b - e * d)
    return (-2 / c) * accum

def sammonMapping( data, dimension=2, iter=20, mf=0.3, initialization="random"):
    rnd = np.random.default_rng()
    if initialization == "random":
        y = rnd.random((len(data), dimension))
    elif initialization == "orthogonal":
        cov_matrix = np.cov(data, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        sortedIndices = np.argsort(eigvals)[::-1]
        principalComponents = eigvecs[:, sortedIndices[:dimension]]
        y = np.dot(data, principalComponents)
    c = np.sum(1 / (data[data != 0]))

    originalD = np.zeros((data.shape[0], data.shape[0]))
    for i in tqdm(range(data.shape[0]), desc="Calcolo distanze originali"):
        for j in range(i + 1, data.shape[0]):
            dist = euclideanDistance(data[i], data[j])
            originalD[i][j] = dist
            originalD[j][i] = dist

    E = []
    for itr in tqdm(range(iter), desc="Esecuzione Sammon Mapping"):
        yNew, e = startSammonMapping(originalD, y, c, mf=mf)
        E.append(e)
        y = np.copy(yNew)
    return y, E

