
# Framework for Clustering, Dimensionality Reduction, and Network Analysis

## Introduction
This is a Python-based framework for performing various operations such as clustering, dimensionality reduction, and network analysis. You can choose between these operations and execute them step by step. It supports techniques like KMeans, DBSCAN, PCA, t-SNE, and Sammon Mapping, as well as community detection and graph analysis operations.

## Requirements
To run this framework, ensure you have the following dependencies installed:
- Python 3.x
- Pandas
- Scikit-learn
- NetworkX
- Matplotlib (for plotting)

You can install the necessary packages using:

```bash
pip install pandas scikit-learn networkx matplotlib
```

## Structure of the Framework
The framework contains several modules:
- `plotFile`: For visualizing clustering results and network graphs.
- `datasetOperations`: Handles dataset loading and preprocessing.
- `clustering`: Implements clustering algorithms like KMeans, DBSCAN, and Agglomerative.
- `Network`: Contains classes for performing network analysis like community detection and centrality measures.
- `dimensionalityReduction`: Implements dimensionality reduction techniques like PCA, t-SNE, and Sammon Mapping.
- `qualityMeasure`: Used for evaluating clustering and dimensionality reduction results.

## How to Run the Framework

1. **How start the code**:
  To run the code, just go to the terminal and run it:
   ```bash
   python3 main.py
   ```
   After that you can follow the instructions in the terminal.


2. **Right choose of dataset**
    An important recommendation is that the terminal will print all available datasets found in the "datasets" folder. However, for Network operations, you must use the files with the ".edgelist" extension, while for other operations, you should use those with the ".txt" extension. This is important because the datasets are formatted differently. Of course, if additional datasets with the same structure are added, the system will work in the same way. This approach was implemented to avoid the need to manually type the dataset names, reducing the risk of errors. By listing the available datasets automatically, it ensures that the correct file is selected without the user needing to remember or input the exact name.


1. **Clustering and Dimensionality Reduction Operations**:
   - Choose between clustering (KMeans, Agglomerative, DBSCAN) or dimensionality reduction (PCA, t-SNE, Sammon Mapping).
   - Load a dataset, select the technique, and perform clustering or dimensionality reduction.
   - Optionally, evaluate clustering results using Silhouette Score and Jaccard Similarity.
   - Visualize the 2D graph using dimensionality reduction techniques.

2. **Network Analysis**:
   - Choose a network file from the datasets.
   - Perform network analysis including PageRank, Betweenness Centrality, and Community Detection (Fast Newman, Girvan-Newman).
   - Add nodes or edges to the network as needed.
   - Visualize the network graph and communities.

## Example Usage

After running the script, you will be prompted with various options. Here is an example:

1. Choose an operation:
   - Option 1: Clustering and/or dimensionality reduction operations
   - Option 2: Network analysis

2. Choose the technique:
   - Option 1: Clustering (KMeans, Agglomerative, DBSCAN)
   - Option 2: Dimensionality reduction (PCA, t-SNE, Sammon Mapping)

3. Choose a network operation:
   - Option 1: PageRank and Betweenness Centrality
   - Option 2: Community Detection
   - Option 3: Graph operations

## Acknowledgments
- Uses the `sklearn` library for clustering and dimensionality reduction.
- Uses `NetworkX` for network analysis and visualization.
