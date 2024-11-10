import json
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
import networkx as nx

class Dataset:
    def __init__(self, path):
        self.path = path
        if path.endswith('.csv'):
            self._load_csv(path)
        elif path.endswith('.arff'):
            self._load_arff(path)
        elif path.endswith('.txt'):
            self._load_txt(path)
        elif path.endswith('.json'):
            self._load_json(path)
        elif path.endswith('.edgelist'):
            self._load_edgelist(path)
        else:
            raise ValueError("Unsupported file format! Please provide a .csv, .arff, .txt, or .json file.")

    def _load_csv(self, path):
        self.data = pd.read_csv(path)

    def _load_arff(self, path):
        data, meta = arff.loadarff(path)
        self.data = pd.DataFrame(data)

    def _load_txt(self, path):
        try:
            self.data = pd.read_csv(path, delimiter=',', encoding='utf-8', header=None)
        except:
            raise ValueError("Error reading .txt file. Ensure the file is properly formatted.")

    def _load_json(self, path):
        with open(path, 'r') as file:
            self.data = json.load(file)

    def get_nodes(self):
        return self.data['nodes']

    def get_edges(self):
        if 'links' in self.data:
            return self.data['links']
        else:
            return self.data['edges']


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

    def print_head(self, n=5):
        if isinstance(self.data, pd.DataFrame):
            print(self.data.head(n))
        elif isinstance(self.data, list):
            for row in self.data[:n]:
                print(row)
        else:
            raise ValueError("Unsupported data format for printing head")

    def replace_missing_values(self, replacement=1):
        self.data = self.data.replace('?', replacement)

        return self

    def select_columns(self, columns):
        if not all(col in self.data.columns for col in columns):
            raise ValueError("One or more specified columns do not exist in the datasets.")
        return self.data[columns]

    def encode_columns(self, df):
        df = df.copy()
        le = LabelEncoder()
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = le.fit_transform(df[column])
        return df

    def _load_edgelist(self, filename):
        self.data = nx.read_edgelist(filename)

    def returnNetwork(self):
        return self.data