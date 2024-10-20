from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

class qualityMeasure:
    def __init__(self, typeOfMeasure, x, y):
        self.typeOfMeasure = typeOfMeasure
        self.xTrue = x
        self.yPred = y

    def measure(self):
        if self.typeOfMeasure == 'clustering':
            homogeneity = homogeneity_score(self.xTrue, self.yPred)
            completeness = completeness_score(self.xTrue, self.yPred)
            v_measure = v_measure_score(self.xTrue, self.yPred)
            return homogeneity, completeness, v_measure