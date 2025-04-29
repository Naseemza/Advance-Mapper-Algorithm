# Advance-Mapper-Algorithm
# Topological Data Analysis Toolkit

## Overview
This repository provides tools for topological analysis of both correlation matrices and experimental 3D point cloud data. The toolkit includes:

1. **Correlation Matrix Analysis**: Processes correlation matrices to extract topological features
2. **Experimental Data Analysis**: Analyzes 3D point clouds (including synthetic shapes like torus)

Key features include persistent homology calculations, Vietoris-Rips complex construction, and mapper graph visualization.

## Features
- **For Correlation Matrices**:
  - Converts correlation matrices to distance matrices
  - Computes persistent homology using Ripser
  - Visualizes persistence diagrams and Betti curves
  - Implements Mapper algorithm with UMAP projections

- **For Experimental Data**:
  - Processes 3D point clouds (including synthetic shapes)
  - Normalizes and samples data
  - Constructs Vietoris-Rips complexes
  - Analyzes topological features through Betti numbers
  - Provides 3D visualizations

- **Common Features**:
  - Interactive visualizations of simplicial complexes
  - Automatic parameter optimization
  - Topological consistency validation

## Requirements
- Python 3.7+
- Required packages:  ``` pandas, numpy, matplotlib, scipy, networkx, ripser, umap-learn, gudhi, scikit-learn, gtda (giotto-tda), mpl_toolkits```


## Installation
```bash
git clone https://github.com/Naseemza/Advance-Mapper-Algorithm.git
cd tda-toolkit
pip install -r requirements.txt
```

























