# Lecture 17 - Part 3: Advanced Clustering and Anomaly Detection

## Overview
This part covers advanced clustering methods, cluster evaluation, and introduction to anomaly detection techniques.

## Key Topics

### 1. Mean Shift
**Concept**: Find density modes via gradient ascent
**Algorithm**:
1. For each point, compute mean of points within bandwidth
2. Shift point to that mean
3. Repeat until convergence
4. Points converging to same mode form cluster

**Benefits**:
- No need to specify K
- Finds arbitrary shapes
- Automatically determines cluster count

**Limitation**: Bandwidth selection critical

### 2. Gaussian Mixture Models (GMM)
**Model**: Data from mixture of Gaussians
```
P(x) = Σₖ πₖ N(x; μₖ, Σₖ)
```
- πₖ: Mixture weights
- μₖ: Cluster means
- Σₖ: Covariance matrices

**Training**: Expectation-Maximization (EM)
- E-step: Compute soft assignments
- M-step: Update parameters

**Benefits**:
- Soft clustering (probabilities)
- Elliptical clusters
- Handles overlapping clusters

### 3. Cluster Evaluation Metrics
**Internal Metrics** (no labels):
- **Silhouette Score**:
  ```
  s = (b - a) / max(a, b)
  ```
  - a: Mean intra-cluster distance
  - b: Mean nearest-cluster distance
  - Range: [-1, 1], higher better

- **Davies-Bouldin Index**: Lower better
- **Calinski-Harabasz**: Higher better

**External Metrics** (with labels):
- **Normalized Mutual Information (NMI)**
- **Adjusted Rand Index (ARI)**
- **Homogeneity, Completeness, V-measure**

### 4. Choosing Number of Clusters
**Elbow Method**: Plot inertia vs K
```python
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
plt.plot(range(1, 10), inertias)
```

**Silhouette Analysis**: Choose K with highest score

**Gap Statistic**: Compare to null reference

### 5. Anomaly Detection Overview
**Definition**: Find data points that differ significantly
**Types**:
- Point anomalies: Single outlier
- Contextual: Outlier in specific context
- Collective: Group of related anomalies

**Applications**:
- Fraud detection
- Network intrusion
- Manufacturing defects
- Medical diagnosis

### 6. Statistical Methods
**Z-Score**:
```
z = (x - μ) / σ
Outlier if |z| > 3
```

**IQR Method**:
```
Outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
```

**Mahalanobis Distance**:
```
D = √((x-μ)ᵀ Σ⁻¹ (x-μ))
```
- Accounts for correlations

### 7. Isolation Forest
**Key Insight**: Anomalies are easier to isolate

**Algorithm**:
1. Build trees with random splits
2. Measure path length to isolate each point
3. Anomalies have shorter paths

**Properties**:
- Works with high dimensions
- No distance calculation
- Fast training and inference

```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1)
predictions = clf.fit_predict(X)  # -1 for anomaly
```

### 8. Hands-on Implementation
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
print(f"Silhouette: {silhouette_score(X, labels):.3f}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# GMM
gmm = GaussianMixture(n_components=3)
labels = gmm.fit_predict(X)
probs = gmm.predict_proba(X)  # Soft assignments
```

## Important Takeaways
1. Mean Shift automatically finds cluster count
2. GMM provides soft probabilistic assignments
3. Silhouette score is the most common internal metric
4. Elbow method helps choose K for K-Means
5. Anomaly detection identifies unusual patterns
6. Isolation Forest is efficient for high-dimensional data
7. Scikit-learn provides unified interface for all methods

