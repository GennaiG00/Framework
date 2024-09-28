import numpy as np

def minkowskiDistance(pointOne, pointTwo, p):
    if len(pointOne) == len(pointTwo):
        tot = 0
        for i in range(len(pointOne)):
            tot += np.abs(pointOne[i] - pointTwo[i]) ** p
        return tot ** (1 / p)
