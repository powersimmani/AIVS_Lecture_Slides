# Lecture 18 - Part 2: Time Series and Graph Clustering

## Overview
This part covers specialized clustering techniques for time series data using DTW and graph data using spectral methods and GNNs.

## Key Topics

### 1. Characteristics of Time Series Data
- **Temporal ordering**: Sequence matters
- **Variable length**: Different series lengths
- **Warping**: Similar patterns at different speeds
- **Noise**: Sensor artifacts, missing values

**Challenge**: Standard distance metrics don't capture similarity

### 2. DTW (Dynamic Time Warping)
**Problem**: Euclidean distance fails for misaligned series
**Solution**: Allow flexible alignment

**Algorithm**:
- Find optimal alignment between two series
- Allow one-to-many matching
- Minimize total distance along warping path

**Formula**:
```
DTW(X, Y) = min_π Σ d(x_i, y_π(i))
```

**Computed via dynamic programming**: O(n×m)

### 3. DTW-based Clustering
**Approach**: Use DTW as distance metric for clustering

```python
from tslearn.clustering import TimeSeriesKMeans

model = TimeSeriesKMeans(n_clusters=3, metric="dtw")
labels = model.fit_predict(time_series_data)
```

**Limitations**:
- Computationally expensive
- O(n² × m²) for n series of length m

### 4. K-Shape Algorithm
**Idea**: Use cross-correlation for similarity
```
SBD(x, y) = 1 - max_s(CC_w(x, y_s)) / (||x|| × ||y||)
```

**Benefits**:
- Scale and shift invariant
- Faster than DTW
- Shape-based similarity

### 5. Subsequence Clustering
**Problem**: Find patterns within long series
**Approach**: Extract and cluster subsequences

**Motif Discovery**: Find recurring patterns
**Shapelet Learning**: Find discriminative subsequences

### 6. Introduction to Graph Data
**Components**:
- Nodes (vertices): Entities
- Edges: Relationships
- Attributes: Node/edge features

**Examples**:
- Social networks
- Molecules
- Citation networks
- Knowledge graphs

### 7. Spectral Clustering
**Approach**: Use graph Laplacian eigenvalues

**Process**:
1. Construct affinity matrix A
2. Compute Laplacian L = D - A
3. Find smallest k eigenvectors
4. Cluster in eigenspace (K-means)

```python
from sklearn.cluster import SpectralClustering

clustering = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
labels = clustering.fit_predict(X)
```

### 8. Community Detection (Louvain)
**Goal**: Find densely connected groups in networks
**Modularity**:
```
Q = (1/2m) Σ (A_ij - k_i*k_j/2m) δ(c_i, c_j)
```

**Louvain Algorithm**:
1. Greedily assign nodes to maximize modularity
2. Build new network with communities as nodes
3. Repeat until no improvement

**Benefits**: Fast, scalable, no need to specify k

### 9. GNN Fundamentals
**Graph Neural Networks**: Neural networks for graphs

**Message Passing**:
```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

**GCN Layer**:
```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

### 10. GNN-based Clustering
**Deep Graph Infomax**:
- Self-supervised graph learning
- Maximize mutual information
- Learn node embeddings

**Process**:
1. Train GNN with unsupervised objective
2. Extract node embeddings
3. Cluster embeddings with K-means

```python
# After GNN training
embeddings = gnn_encoder(graph)
labels = KMeans(n_clusters=k).fit_predict(embeddings)
```

## Important Takeaways
1. DTW handles temporal misalignment in time series
2. K-Shape provides efficient shape-based clustering
3. Spectral clustering uses graph eigenspace
4. Louvain is efficient for community detection
5. GNNs learn representations through message passing
6. Combining GNNs with clustering enables deep graph clustering
7. Domain-specific methods outperform generic approaches

