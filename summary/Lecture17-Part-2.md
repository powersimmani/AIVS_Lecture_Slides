# Lecture 17 - Part 2: Dimensionality Reduction

## Overview
This part covers dimensionality reduction techniques from linear methods like PCA to non-linear methods like t-SNE, UMAP, and autoencoders.

## Key Topics

### 1. Curse of Dimensionality
**Problems with High Dimensions**:
- Distance concentration: All points equidistant
- Sparse data: Volume grows exponentially
- Overfitting: Too many features
- Computation: Expensive algorithms

**Solution**: Reduce dimensions while preserving structure

### 2. PCA Principles
**Goal**: Find directions of maximum variance
**Method**:
1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto top-k components

**Formula**:
```
X_reduced = X @ V[:, :k]
```
- V: Matrix of eigenvectors
- k: Number of components

### 3. PCA Implementation
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Explained variance
print(pca.explained_variance_ratio_)
```

**Choosing k**: Keep 95% variance typically

### 4. Kernel PCA
**Problem**: PCA only captures linear relationships
**Solution**: Kernel trick for non-linear PCA

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf')
X_reduced = kpca.fit_transform(X)
```

**Kernels**: RBF, polynomial, sigmoid

### 5. t-SNE Algorithm
**t-Distributed Stochastic Neighbor Embedding**:
- Non-linear, focused on visualization
- Preserves local structure

**Process**:
1. Compute pairwise similarities in high-D (Gaussian)
2. Compute similarities in low-D (t-distribution)
3. Minimize KL divergence between distributions

**Hyperparameters**:
- **Perplexity**: Effective number of neighbors (5-50)
- **Learning rate**: Typically 200
- **Iterations**: 1000+

**Limitations**:
- Slow for large data
- Non-deterministic
- Don't interpret distances globally

### 6. UMAP
**Uniform Manifold Approximation and Projection**:
- Based on topological data analysis
- Faster than t-SNE
- Better preservation of global structure

```python
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
X_reduced = reducer.fit_transform(X)
```

**Advantages over t-SNE**:
- Faster (O(N) vs O(N²))
- More scalable
- Better global structure
- Can embed new points

### 7. Autoencoder Dimensionality Reduction
**Architecture**:
```
Input → Encoder → Latent (bottleneck) → Decoder → Reconstruction
```

**Training**: Minimize reconstruction error
```
L = ||x - decoder(encoder(x))||²
```

**Benefits**:
- Non-linear
- Learnable transformation
- Can handle complex data
- Generalizes to new points

### 8. VAE Basics
**Variational Autoencoder**:
- Probabilistic latent space
- Generates new samples

**Key Differences from AE**:
- Latent is distribution, not point
- Regularized latent space
- Can sample and generate

**Loss**:
```
L = Reconstruction + KL(q(z|x) || p(z))
```

### 9. Comparison of Methods
| Method | Type | Global | Speed | New Points |
|--------|------|--------|-------|------------|
| PCA | Linear | Yes | Fast | Yes |
| t-SNE | Non-linear | No | Slow | No |
| UMAP | Non-linear | Better | Fast | Yes |
| Autoencoder | Non-linear | Variable | Medium | Yes |

## Important Takeaways
1. High dimensionality causes multiple problems
2. PCA is the go-to linear method
3. Kernel PCA handles non-linear data
4. t-SNE excels at visualization but is slow
5. UMAP is faster and preserves more global structure
6. Autoencoders provide flexible non-linear reduction
7. Choose method based on task (visualization vs preprocessing)

