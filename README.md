# Topological Analysis of High-Dimensional Data Structures

## Overview

This repository provides a unified framework for **Topological Data Analysis (TDA)** of high-dimensional datasets. It supports two core implementations:

1. **Correlation Matrix Analysis** — for network/biological data  
2. **Geometric Structure Analysis** — for point cloud data such as tori or manifolds

The toolkit utilizes **persistent homology**, **Vietoris-Rips complexes**, and the **mapper algorithm** to extract, quantify, and visualize topological features of complex datasets.

---

## 🔧 Features

- **Persistent Homology Analysis**:
  - Vietoris-Rips filtration using Ripser and GUDHI
  - Compute persistence diagrams and Betti curves
  - Automatic scale selection and thresholding

- **Mapper Graph Construction**:
  - UMAP-based filtering and DBSCAN clustering
  - Interactive mapper graphs from high-dimensional data
  - Validation via Betti number comparison

- **Geometric and Network Data Compatibility**:
  - Works on both correlation matrices and geometric point clouds
  - Modular functions for reuse and adaptation

---

## 📐 Mathematical Foundations

### 1. Persistent Homology

Given a point cloud $X = \{x_1, ..., x_n\} \subset \mathbb{R}^d$:

- **Vietoris–Rips Complex** at scale $t$:
  $$
  K_t = \left\{ \sigma \subseteq X \mid \text{diam}(\sigma) \leq t \right\}, \quad \text{diam}(\sigma) = \max_{x,y \in \sigma} d(x,y)
  $$

- **Persistence Diagram**:
  Tracks birth-death pairs of homological features $H_k(K_t)$:
  $$
  H_k(K_t) = \bigoplus_i \mathbb{F}^{(b_i,d_i]}, \quad \beta_k(t) = \text{rank}(H_k(K_t))
  $$

### 2. Mapper Graphs

Given a filter function $f: X \to \mathbb{R}^d$:

- **Cover**:
  $$
  \mathcal{U} = \{U_i\}_{i=1}^N \text{ where } \bigcup_i U_i \supseteq f(X)
  $$

- **Clustering in Pullbacks**:
  $$
  C_{i,j} = \text{connected components}(f^{-1}(U_i)) \quad \text{(e.g., via DBSCAN)}
  $$

- **Graph Construction**:
  Vertices represent clusters; edges are shared points:
  $$
  (C_{i,j}, C_{k,l}) \in E \iff C_{i,j} \cap C_{k,l} \neq \emptyset
  $$



## 📦 Installation

### Requirements

- Operatin System : Windows OS
- Python 3.7+
- Required packages:
  ```
  numpy
  pandas
  matplotlib
  scipy
  networkx
  ripser
  umap-learn
  gudhi
  scikit-learn
  gtda
  ```

- You can install the required Python packages individually using pip:
  ```python
  # Core libraries
  pip install pandas
  pip install numpy

  # TDA and topology tools
  pip install ripser
  pip install gudhi
  pip install giotto-tda

  # Dimensionality reduction
  pip install umap-learn

  # Clustering and distance tools
  pip install scikit-learn
  pip install scipy

  # Graph and network tools
  pip install networkx

  # Visualization
  pip install matplotlib
  ```


## 🚀 How to Usage

### 1. Correlation Matrix Analysis
#### Case 1 : Using Jupyter NoteBook
Step 1 : Import the `Correlation.ipnb` file in Jupyter Notebook.

Step 2 : Run the First line od codes to load all the data and wait for it to finish.

Step 3 : Upload your correlation.csv file and then Use the code :
```python
corr_matrix_path = "torsion_correlation.csv" # paste the link for your correlation.csv file
k = 0.99  # Example value for k
cutoff_betti = 4  # Example cutoff Betti value
results_df_protein = process_correlation_matrix(corr_matrix_path, k, cutoff_betti)
```
Step 4 : All the process will automatically load but to see the final result we can use the following code and run it to get the final table result.
``` python
results_df_protein
````


#### Case 2 : Using Python programmer
Step 1 : Open the Python compiler and copy the whole code from `correlation_matrix.py` from PythonCode folder in our github.

Step 2 : Then copy the `Exapmle_correlation_compiler.py` by changing the correlation matrix that you want to upload and follow other code and run the file similar code give below :

```python
corr_matrix_path = "torsion_correlation.csv" # paste the link for your correlation.csv file
k = 0.99  # Example value for k
cutoff_betti = 4  # Example cutoff Betti value
results_df_protein = process_correlation_matrix(corr_matrix_path, k, cutoff_betti)
```
Step 3 : All the process will automatically load but to see the final result we can use the following code and run it to get the final table result.
``` python
print(results_df_protein)
````



### 2. Geometric Point Cloud (any 2D or 3D dataset)
### 1. Correlation Matrix Analysis
#### Case 1 : Using Jupyter NoteBook
Step 1 : Import the `Coordinate.ipnb` file in Jupyter Notebook.

Step 2 : Run the First line od codes to load all the data and wait for it to finish.

Step 3 : Upload your coordinate.csv file or load any random 2D & 3D sample and then Use the code :
```python
xyz_points = xyz_points_normalized # load the point cloud data
k = 0.25  # Example value for k
cutoff_betti = 1 # Example cutoff Betti value
results_df_torus = process_structural_data(xyz_points, k, cutoff_betti)
```
Step 4 : All the process will automatically load but to see the final result we can use the following code and run it to get the final table result.
``` python
results_df_torus
````

#### Case 2 : Using Python programmer
Step 1 : Open the Python compiler and copy the whole code from `Step1_Coordinate_data.py` from PythonCode folder in our github.

Step 2 : Then copy the `Step2_loading_running_code.py` by changing the correlation matrix that you want to upload and follow other code and run the file similar code give below :

```python
corr_matrix_path = "torsion_correlation.csv" # paste the link for your correlation.csv file
k = 0.99  # Example value for k
cutoff_betti = 4  # Example cutoff Betti value
results_df_protein = process_correlation_matrix(corr_matrix_path, k, cutoff_betti)
```
Step 3 : All the process will automatically load but to see the final result we can use the following code and run it to get the final table result.
``` python
print(results_df_torus)
````

## 📊 Methodology

### Pipeline

1. **Preprocessing**
   - Normalize data
   - Compute distance or correlation matrices
   - Apply dimensionality reduction (UMAP)

2. **Persistent Homology**
   - Vietoris-Rips filtration
   - Compute persistence diagrams $(b_i, d_i)$
   - Calculate Betti curves $\beta_k(t)$

3. **Mapper Graph Construction**
   - Cover filter space with overlapping intervals
   - Cluster in preimages of covers
   - Construct graph with overlapping clusters

4. **Parameter Optimization**
   - Optimal filtration scale:
     $$
     \epsilon^* = \arg\max_{\epsilon} \left( \sum_{(b,d)} \mathbb{I}_{[b,d]}(\epsilon) \right)
     $$
   - Number of intervals:
     $$
     N(\epsilon, \alpha) = \left\lfloor \frac{L - \epsilon}{\epsilon(1 - \alpha / 100)} \right\rfloor + 1
     $$

5. **Validation**
   - Betti number agreement:
     $$
     |\beta_1^{\text{persistence}} - \beta_1^{\text{mapper}}| \leq \tau
     $$

---


## 📈 Output

### Example Table

| %overlap | N value | Betti_1_1 | Betti_1_2 | epsilon_1 | epsilon_2 |
|----------|---------|-----------|-----------|-----------|-----------|
|   20     |   15    |     2     |     1     |  0.4521   |  0.3876   |
|   25     |   18    |     2     |     2     |  0.4521   |  0.4213   |

### Visual Outputs
1. Persistence Diagram of the data
2. Betti curve
3. Mapper Nerve (Graph)
4. Simplicial Complex.

---

# Most Persistent Betti-1 Calculator

This repository provides a Python implementation for identifying the **most persistent Betti-1** feature in topological data using persistent homology via the `ripser` library.

The code supports input in the form of:
- A **correlation matrix** (converted to a distance matrix), or
- **Coordinate data** (Euclidean space).

---

## 📊 Methodology


The algorithm follows these steps:

1. **Import Data**: Load either a correlation matrix or coordinate data.
2. **Preprocess**:
   - If using a correlation matrix, convert it to a distance matrix.
   - If using coordinates, proceed directly.
3. **Persistent Homology**: Use the `ripser` package to compute persistent homology and plot the persistence diagram.
4. **Betti Curve**: Track the evolution of Betti-1 features over a range of filtration radii.
5. **Interval Partitioning**: Identify intervals where Betti-1 = 1 and compute their lifespans.
6. **Noise Reduction**: Retain only the top 20% longest intervals to reduce redundant noise.
7. **Persistence Extraction**: Select the longest interval and define:
   - **K\***: Most persistent Betti-1 range
   - **ε (epsilon)**: Mean of maximum and minimum lifespans
8. **Output**: Print the (K\*, ε) pair for further use.
---

## 🚀 How to Use Most Persistent Mapper Algorithm

Step 1 , Step 2, Step 3 are same as the above code for loading the data and running the code 

Step 4 : we can now seperately run the code for finding the most persistent betti 1 characher using the follow code:
Case 1 : For geometrical and manifold data
```python
xyz_points = xyz_points_normalized
most_persistenet_structural(xyz_points)
```
Case 2 : For correlation matrix
```python
corr_matrix_path = "clinical_corr_2.csv"
most_persistenet_corr(corr_matrix_path)
```

## Seperate code using most perisstent algorithm is give in Moest_Persistent folder with jupyter notebook file.

---













## 📚 References

1. Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence barcodes. Journal of Applied and Computational Topology, 5(3), 391-423.
2. Saul, Nathaniel and Tralie, Chris. (2019). Scikit-TDA: Topological Data Analysis for Python. Zenodo.
3. Tauzin, G., Lupo, U., Tunstall, L., Pérez, J. B., Caorsi, M., Medina-Mardones, A. M., ... & Hess, K. (2021). giotto-tda:: A topological data analysis toolkit for machine learning and data exploration. Journal of Machine Learning Research, 22(39), 1-6.
4. Maria, C., Boissonnat, J. D., Glisse, M., & Yvinec, M. (2014). The gudhi library: Simplicial complexes and persistent homology. In Mathematical Software–ICMS 2014: 4th International Congress, Seoul, South Korea, August 5-9, 2014. Proceedings 4 (pp. 167-174). Springer Berlin Heidelberg.
5. Caorsi, M., Reinauer, R., & Berkouk, N. (2022). giotto-deep: A python package for topological deep learning. Journal of Open Source Software, 7(79).


---

