import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, data):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data

    def get_points(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.values.tolist()
        elif isinstance(self.data, np.ndarray):
            return self.data.tolist()
        elif isinstance(self.data, list):
            return self.data
        else:
            raise ValueError("Unsupported data format")

    def get_features(self):
        if isinstance(self.data, pd.DataFrame):
            return list(self.data.columns)
        elif isinstance(self.data, np.ndarray):
            return [f"Feature_{i}" for i in range(self.data.shape[1])]
        elif isinstance(self.data, list):
            if isinstance(self.data[0], list):
                return [f"Feature_{i}" for i in range(len(self.data[0]))]
            else:
                raise ValueError("Data is not properly structured (list of lists expected)")
        else:
            raise ValueError("Unsupported data format")
