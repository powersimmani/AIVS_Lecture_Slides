# Lecture 17 - Part 1: Unsupervised Learning and Clustering

## Overview
This part introduces unsupervised learning concepts and covers fundamental clustering algorithms including K-Means, hierarchical clustering, and DBSCAN.

## Key Topics

### 1. What is Unsupervised Learning?
- **Definition**: Learning patterns without labeled data
- **Goal**: Discover hidden structure in data
- **Types**:
  - Clustering: Group similar data
  - Dimensionality Reduction: Compress features
  - Anomaly Detection: Find outliers
  - Association: Find relationships

### 2. Supervised vs Unsupervised vs Semi-supervised
| Type | Labels | Goal | Examples |
|------|--------|------|----------|
| Supervised | All | Predict y from x | Classification, Regression |
| Unsupervised | None | Find structure | Clustering, PCA |
| Semi-supervised | Some | Leverage unlabeled | Self-training |

### 3. Applications of Unsupervised Learning
- **Customer Segmentation**: Group customers by behavior
- **Anomaly Detection**: Fraud, intrusion detection
- **Topic Modeling**: Document clustering
- **Feature Learning**: Representation learning
- **Data Compression**: Reduce dimensionality

### 4. Key Challenges and Evaluation
**Challenges**:
- No ground truth to validate
- Choosing number of clusters
- Handling high dimensions
- Interpreting results

**Evaluation Metrics**:
- **Internal**: Silhouette score, Davies-Bouldin
- **External** (if labels available): NMI, ARI

### 5. Data Preprocessing and Scaling
**Important**: Clustering sensitive to scale

**Methods**:
- **StandardScaler**: z = (x - μ) / σ
- **MinMaxScaler**: x' = (x - min) / (max - min)
- **RobustScaler**: Uses median and IQR

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

### 6. K-Means Algorithm
**Objective**: Minimize within-cluster variance
```
J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

**Algorithm**:
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as cluster means
4. Repeat until convergence

**Complexity**: O(n × K × iterations × d)

### 7. K-Means++ Initialization
**Problem**: Random init can give poor results
**Solution**: Smart initialization

**Algorithm**:
1. Choose first centroid randomly
2. For each point, compute distance to nearest centroid
3. Choose next centroid with probability ∝ distance²
4. Repeat until K centroids

**Benefit**: Better starting point, faster convergence

### 8. Hierarchical Clustering
**Two Approaches**:
- **Agglomerative** (bottom-up): Start with points, merge
- **Divisive** (top-down): Start with all, split

**Linkage Methods**:
- Single: min distance between clusters
- Complete: max distance
- Average: mean distance
- Ward: minimize variance increase

**Dendrogram**: Tree visualization of merging process

### 9. DBSCAN - Density-Based
**Parameters**:
- ε (eps): Neighborhood radius
- minPts: Minimum points for core point

**Point Types**:
- **Core**: ≥ minPts in ε neighborhood
- **Border**: In ε of core, but not core itself
- **Noise**: Neither core nor border

**Benefits**:
- Finds arbitrary shapes
- Identifies outliers
- No need to specify K
- Robust to noise

**Limitation**: Sensitive to ε, struggles with varying density

## Important Takeaways
1. Unsupervised learning finds patterns without labels
2. Data scaling is critical for distance-based clustering
3. K-Means is simple but assumes spherical clusters
4. K-Means++ provides better initialization
5. Hierarchical clustering provides dendrogram visualization
6. DBSCAN finds arbitrary shapes and outliers
7. No single best algorithm - choose based on data

