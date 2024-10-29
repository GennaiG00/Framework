import pandas as pd
import numpy as np
from scipy.io import arff
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(self, path):
        self.path = path
        if path.endswith('.csv'):
            self._load_csv(path)
        elif path.endswith('.arff'):
            self._load_arff(path)
        elif path.endswith('.txt'):
            self._load_txt(path)
        elif path.endswith('.mat'):
            self._load_mat_(path)
        elif path.endswith('.names'):
            self._load_names(path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .arff, or .txt file.")

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

    def _load_mat_(self, path):
        try:
            self.data = sio.loadmat(path)
        except:
            raise ValueError("Error reading .mat file. Ensure the file is properly formatted.")

    def _load_names(self, path):
        try:
            self.data = pd.read_csv(path, delimiter=';', encoding='utf-8')
        except:
            raise ValueError("Error reading .names file. Ensure the file is properly formatted.")

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

    # New method to print the head of the dataset
    def print_head(self, n=5):
        if isinstance(self.data, pd.DataFrame):
            print(self.data.head(n))
        elif isinstance(self.data, list):
            # For list data, print the first `n` rows
            for row in self.data[:n]:
                print(row)
        else:
            raise ValueError("Unsupported data format for printing head")

    def replace_missing_values(self, replacement=1):
        """
        Replaces all occurrences of '?' in the dataset with the provided replacement value (np.nan).
        Then drops rows with any missing values.
        """
        # Replace '?' with np.nan
        self.data = self.data.replace('?', replacement)

        return self

    def select_columns(self, columns):
        """
        Select specific columns from the dataset.
        """
        if not all(col in self.data.columns for col in columns):
            raise ValueError("One or more specified columns do not exist in the dataset.")
        return self.data[columns]

    def encode_columns(self, df):
        """
        Encode categorical columns in the dataframe.
        """
        df = df.copy()  # Create a copy of the DataFrame
        le = LabelEncoder()
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = le.fit_transform(df[column])
        return df
