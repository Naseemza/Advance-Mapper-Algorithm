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




### 2. Geometric Point Cloud (e.g., Torus)

```python
from analysis import process_geometric_structure, generate_torus_points

xyz_points = generate_torus_points(n_points=5000)
results = process_geometric_structure(xyz_points, k=0.25, cutoff_betti=2)
```

---

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

## 🌐 Example: Torus Geometry

### Torus Parametrization

```python
def torus(u, v, R=3, r=1):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.column_stack((x, y, z))
```

### Expected Topology

- $\beta_0 = 1$ — 1 connected component  
- $\beta_1 = 2$ — 2 independent loops  
- $\beta_2 = 1$ — void enclosed by the torus

---

## 📈 Output

### Example Table

| %overlap | N value | Betti_1_1 | Betti_1_2 | epsilon_1 | epsilon_2 |
|----------|---------|-----------|-----------|-----------|-----------|
|   20     |   15    |     2     |     1     |  0.4521   |  0.3876   |
|   25     |   18    |     2     |     2     |  0.4521   |  0.4213   |

### Visual Outputs

#### Biological Data
- `images/bio_persistence.png`  
- `images/bio_mapper.png`

#### Torus Data
- `images/torus_3d.png`  
- `images/torus_persistence.png`

---

## 📚 References

1. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology*  
2. Carlsson, G. (2009). *Topology and Data*  
3. Singh, G., et al. (2007). *Topological Methods for the Analysis of High Dimensional Data*  
4. McInnes, L., et al. (2018). *UMAP*  
5. Chazal, F., & Michel, B. (2017). *Introduction to TDA*

---

## 🤝 Contributing

Contributions welcome! Please fork the repo, create a branch, and submit a pull request.  
Bug reports and feature requests via [issues](https://github.com/yourusername/topological-analysis/issues) are appreciated.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Your Name**  
[Your Email]  
[Your Institution]
```

---

### ✅ Next Steps:

1. Save the content above as `README.md` in your project root directory.
2. Replace `yourusername`, `Your Name`, `Your Email`, and `Your Institution` with your actual details.
3. Ensure that referenced image files (e.g., `images/bio_persistence.png`) exist in the `images/` folder in your repo.

Would you like help generating a matching `requirements.txt` or an example Jupyter notebook to go with this?

















