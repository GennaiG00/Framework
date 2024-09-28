from distanceAlgorithms.minkowski import minkowskiDistance


class Distance:
    def calculate_distance(self, pointOne, pointTwo, distanceFunction, p=1):
        if distanceFunction == 'manhattan':
            minkowskiDistance(pointOne, pointTwo, 1)
        elif distanceFunction == 'euclidean':
            minkowskiDistance(pointOne, pointTwo, 2)
        elif distanceFunction == 'minkowski':
            minkowskiDistance(pointOne, pointTwo, p)
        else:
            raise Exception('Distance function not supported')
