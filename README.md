# Dimensionality Reduction and Clustering Visualization Assignment 4

This program performs dimensionality reduction on datasets and visualizes the results using various clustering techniques. The supported dimensionality reduction techniques include PCA, t-SNE, and Sammon Mapping, while clustering techniques include KMeans, Agglomerative Clustering, and DBSCAN.

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `scikit-learn`
  - Additional modules: `clustering`, `datasetOperations`, and `dimensionalityReduction` (ensure these are available in your project)

## Getting Started

1. **Clone the repository** (if applicable) or ensure that you have the program files available in a directory.

2. **Prepare your datasets**:
   - Place your datasets in the `datasets` folder. The program is currently set up to read from three datasets: `dataset2.txt`, `space_ga.txt`, and `dataset3.txt`. You may modify these file names as needed.

3. **Run the program**:
   - Open your terminal and navigate to the directory where the program files are located.
   - Execute the following command:

   ```bash
   python3 main.py
   ```
## Running the Program

This command will execute the program using the pre-defined datasets without requiring any additional input parameters.

## Output

The program will produce the following visualizations:

- Results of clustering for each dataset using various clustering techniques.
- Visualizations of dimensionality reduction for each dataset.
- Scores indicating the preservation of class labels and clusters after the application of dimensionality reduction techniques.
