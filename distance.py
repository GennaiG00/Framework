from distanceAlgorithms.minkowski import minkowskiDistance


class Distance:
    def __init__(self, distanceFunction, **otherParameters):
        self.distanceFunction = distanceFunction
        self.otherParameters = otherParameters

    def calculate_distance(self, pointOne, pointTwo):
        if self.distanceFunction == 'manhattan':
            minkowskiDistance(pointOne, pointTwo, 1)
        elif self.distanceFunction == 'euclidean':
            minkowskiDistance(pointOne, pointTwo, 2)
        elif self.distanceFunction == 'minkowski':
            minkowskiDistance(pointOne, pointTwo, self.otherParameters)
        else:
            raise Exception('Distance function not supported')
