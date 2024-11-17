
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


## Pipeline Operations

### Dimensionality Reduction Pipeline
The **`dimensionality_reduction_pipeline`** function allows applying dimensionality reduction techniques to multiple datasets sequentially, supporting methods such as PCA, t-SNE, and Sammon Mapping. It also provides options to evaluate class preservation performance and visualize results.

#### Key Steps:
1. **Dataset Loading**:
   - Datasets are loaded from the provided list, and missing values are replaced automatically.
2. **Label Separation**:
   - The label (target) column is automatically identified and separated from the data.
3. **Dimensionality Reduction**:
   - For each selected method, dimensionality reduction is applied, and the results are saved and printed.
4. **Performance Evaluation**:
   - If enabled, class preservation is evaluated using the `classPreservation` method.
5. **Visualization**:
   - Results are visualized in a 2D plot, where points are colored based on the original labels.

### Community Detection Pipeline
The **`community_detection_pipeline`** function processes multiple network files sequentially to detect communities using both Girvan-Newman and Fast Newman methods. The pipeline generates community visualizations and prints the detected communities for each method.

#### Key Steps:
1. **Network File Processing**:
   - Each network file in the provided list is loaded and processed as a graph.
2. **Girvan-Newman Method**:
   - Communities are detected iteratively, and the first level of communities is extracted and visualized.
3. **Fast Newman Method**:
   - Communities are detected based on the specified number of clusters, and the quality of the partitions is computed and displayed.
4. **Visualization**:
   - Nodes are colored according to their community, and the graph is displayed for both methods.

#### Example
```python
datasets = ['./datasets/dataset1.txt', './datasets/dataset2.txt', './datasets/dataset3.txt']
dimensionality_reduction_pipeline(
    datasets, 
    ['PCA', 't-SNE', 'sammonMapping'], 
    final_dimension=2, 
    n_repetition=2, 
    alpha=0.85, 
    plot=True, 
    perform_preservation=True
)

network_files = ['./datasets/karate.edgelist', './datasets/les_miserables.edgelist', './datasets/three_communities.edgelist']
n_communities = [2, 5, 3]
community_detection_pipeline(network_files, n_communities=n_communities)
```
