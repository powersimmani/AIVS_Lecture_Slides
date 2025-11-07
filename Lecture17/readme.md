# Lecture 17: Clustering and Unsupervised Learning Fundamentals

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Total Slides:** 32  
**Lecture Duration:** Approximately 3 hours  
**Difficulty Level:** Intermediate to Advanced  
**Course Type:** Machine Learning / Data Science

This comprehensive lecture provides an in-depth exploration of unsupervised learning techniques, with particular emphasis on clustering algorithms, dimensionality reduction methods, and anomaly detection. The course combines theoretical foundations with practical implementations using scikit-learn.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand and differentiate** between supervised, unsupervised, and semi-supervised learning paradigms, and identify appropriate use cases for each approach
2. **Implement and evaluate** multiple clustering algorithms (K-Means, DBSCAN, Hierarchical, GMM) with appropriate initialization strategies and evaluation metrics
3. **Apply dimensionality reduction techniques** (PCA, t-SNE, UMAP, Autoencoders) to high-dimensional data while understanding the curse of dimensionality
4. **Detect anomalies** using statistical methods and machine learning approaches, particularly Isolation Forest
5. **Design complete ML pipelines** incorporating data preprocessing, scaling, clustering/dimensionality reduction, and model evaluation using scikit-learn

---

## 📚 Lecture Structure

### Part 1/4: Unsupervised Learning Overview

**Slides:** 3-8 (6 slides)

#### Topics Covered:

**Slide 3: Part Introduction**
- Overview of unsupervised learning fundamentals
- Roadmap for the first section

**Slide 4: What is Unsupervised Learning?**
- **Comparison with Supervised Learning:**
  - Supervised: Labeled data (X, y), explicit targets, learning from examples
  - Unsupervised: Unlabeled data only, no targets, discover hidden patterns
- **Three Main Categories:**
  - Clustering: Grouping similar data points
  - Dimensionality Reduction: Reducing feature space complexity
  - Anomaly Detection: Identifying unusual patterns

**Slide 5: Supervised vs Unsupervised vs Semi-supervised**
- **Supervised Learning:**
  - Data type: Fully labeled (X, y)
  - Goal: Predict y from X
  - Characteristics: High accuracy, expensive labeling, requires domain expertise
  - Labeling cost: 90% (highest)
- **Unsupervised Learning:**
  - Data type: Only X (unlabeled)
  - Goal: Find structure in X
  - Characteristics: No labels needed, discover patterns, highly scalable, cost-effective
  - Labeling cost: 10% (lowest)
- **Semi-supervised Learning:**
  - Data type: Small labeled + large unlabeled
  - Goal: Leverage both data types
  - Characteristics: Best of both worlds, balanced approach, practical solution, often best results
  - Labeling cost: 40% (medium)

**Slide 6: Applications of Unsupervised Learning**
- **Customer Segmentation:** Group customers by behavior patterns
- **Anomaly Detection:** Fraud detection, intrusion detection
- **Recommendation Systems:** Find similar items without ratings
- **Image Compression:** Reduce dimensionality efficiently
- **Market Basket Analysis:** Discover product associations
- **Document Clustering:** Organize articles and papers
- **Biological Analysis:** Gene expression patterns
- **Feature Engineering:** Create better features for models

**Slide 7: Key Challenges and Evaluation Methods**
- **Key Challenge:** No ground truth available, cannot compute accuracy
- **Evaluation Question:** What makes a 'good' clustering?
- **Internal Metrics:**
  - Silhouette Score
  - Davies-Bouldin Index
  - Inertia (WCSS)
- **External Metrics (when labels available):**
  - ARI (Adjusted Rand Index)
  - NMI (Normalized Mutual Information)
  - Purity
- **Validation Methods:**
  - Visual inspection: Plot clusters in 2D/3D
  - Domain validation: Do clusters make business sense?
  - Stability analysis: Consistent results across runs?
  - Interpretability: Can we explain patterns?

**Slide 8: Data Preprocessing and Scaling**
- **Critical Importance:** Feature scaling is critical for distance-based algorithms
- **Scaling Methods:**
  - **StandardScaler:** z = (x - μ) / σ, zero mean, unit variance
  - **MinMaxScaler:** x' = (x - min) / (max - min), scale to [0, 1]
  - **RobustScaler:** x' = (x - median) / IQR, robust to outliers
- **Preprocessing Pipeline:**
  1. Missing values: Imputation before clustering
  2. Outlier treatment: Remove or cap extremes
  3. Feature selection: Remove irrelevant features
  4. Dimensionality: Handle curse of dimensions
- **Calculation Examples:** Complete numerical examples for each scaler with data [10, 20, 30, 40, 50]

#### Key Concepts:
- Unsupervised learning discovers patterns without labeled data
- Three main paradigms: supervised (labeled), unsupervised (unlabeled), semi-supervised (mixed)
- Feature scaling is mandatory for distance-based algorithms
- Evaluation requires internal metrics, external validation, and domain knowledge
- Cost-accuracy tradeoff between labeling effort and model performance

#### Learning Outcomes:
- Distinguish between learning paradigms and select appropriate approaches
- Apply proper preprocessing and scaling techniques
- Evaluate clustering quality using multiple metrics
- Understand practical applications across different domains

---

### Part 2/4: Clustering Algorithms

**Slides:** 9-17 (9 slides)

#### Topics Covered:

**Slide 9: Part Introduction**
- Overview of clustering algorithms
- Topics: Problem definition, K-Means, initialization, hierarchical, DBSCAN, Mean Shift, GMM, evaluation

**Slide 10: Clustering Problem Definition**
- **Core Objective:**
  - Partition n observations into k groups
  - Maximize intra-cluster similarity, minimize inter-cluster similarity
- **Distance Metrics:** Euclidean, Manhattan, Cosine, Mahalanobis
- **Types of Clustering:**
  - **Hard Clustering:** Binary membership, each point belongs to exactly one cluster
  - **Soft Clustering:** Fuzzy membership, probability distribution over clusters
  - **Hierarchical:** Nested clusters forming tree structure (dendrogram)
  - **Density-Based:** Clusters as high-density regions, arbitrary shapes
- **Model-Based Approach:** Data generated from mixture of probability distributions

**Slide 11: K-Means Algorithm**
- **Algorithm Steps:**
  1. Initialize k centroids randomly
  2. Assignment: Assign each point to nearest centroid
  3. Update: Recalculate centroids as cluster means
  4. Repeat until convergence (centroids don't change)
- **Characteristics:**
  - Time complexity: O(n × k × i × d) where n=samples, k=clusters, i=iterations, d=dimensions
  - Convergence: Guaranteed but to local optimum
  - Assumptions: Spherical clusters, similar sizes
- **Choosing K:**
  - Elbow method: Plot inertia vs k
  - Silhouette analysis: Measure cluster quality
  - Domain knowledge: Business requirements

**Slide 12: K-Means++ Initialization**
- **Algorithm:**
  1. First centroid: Choose randomly from data points
  2. Subsequent centroids: Probability ∝ D(x)² (squared distance to nearest centroid)
  3. Result: Spreads initial centroids far apart
- **Benefits:**
  - Faster convergence: Significantly reduces iterations
  - Better quality: Large gains in clustering quality
  - Provable guarantee: O(log k) approximation to optimal
  - Widely adopted: Default in scikit-learn
- **Trade-off:** Small initialization overhead → Large gains in quality and speed

**Slide 13: Hierarchical Clustering**
- **Two Approaches:**
  - **Agglomerative (Bottom-up):** Start with individual points, merge closest pairs
  - **Divisive (Top-down):** Start with all points, recursively split
- **Linkage Methods:**
  - Single: Minimum distance between clusters
  - Complete: Maximum distance between clusters
  - Average: Average distance between all pairs
  - Ward: Minimize within-cluster variance
- **Advantages:**
  - No need to specify k in advance
  - Dendrogram visualization
  - Hierarchical structure revealed
- **Disadvantages:**
  - Computational complexity: O(n³) for naïve, O(n² log n) optimized
  - Cannot undo merges/splits
  - Sensitive to noise and outliers

**Slide 14: DBSCAN - Density-Based**
- **Key Parameters:**
  - ε (epsilon): Neighborhood radius
  - MinPts: Minimum points to form dense region
- **Point Classifications:**
  - Core points: ≥ MinPts within ε
  - Border points: In neighborhood of core, but < MinPts
  - Noise points: Neither core nor border
- **Advantages:**
  - Discovers arbitrary-shaped clusters
  - No need to specify number of clusters
  - Robust to outliers (marks them as noise)
  - Works well with spatial data
- **Disadvantages:**
  - Sensitive to ε and MinPts selection
  - Struggles with varying density clusters
  - Not suitable for high-dimensional data (curse of dimensionality)

**Slide 15: Mean Shift**
- **Core Concept:** Iteratively shift points toward mode (peak) of density
- **Algorithm:**
  1. For each point, compute mean of neighbors within bandwidth
  2. Shift point to computed mean
  3. Repeat until convergence
  4. Points converging to same mode form a cluster
- **Characteristics:**
  - Automatically determines number of clusters
  - Finds arbitrary-shaped clusters
  - Single parameter: bandwidth (window size)
  - Can be slow on large datasets

**Slide 16: Gaussian Mixture Models (GMM)**
- **Model:** Data generated from mixture of k Gaussian distributions
- **Components:** Each Gaussian has mean (μ), covariance (Σ), weight (π)
- **EM Algorithm:**
  - **E-step (Expectation):** Assign probabilities to clusters
  - **M-step (Maximization):** Update parameters (μ, Σ, π)
  - Iterate until convergence
- **Comparison with K-Means:**
  - K-means: Hard assignment, spherical clusters
  - GMM: Soft assignment, elliptical clusters
- **Advantages:**
  - Probability distribution over clusters
  - Models elliptical clusters
  - Captures uncertainty
- **Model Selection:** BIC (Bayesian Information Criterion), AIC (Akaike Information Criterion)
- **Limitation:** Assumes Gaussian distribution

**Slide 17: Cluster Evaluation Metrics**
- **Internal Metrics (no labels needed):**
  - **Silhouette Score:** Range [-1, 1], measures separation and cohesion, higher is better
  - **Davies-Bouldin Index:** Range [0, ∞), ratio of within to between cluster distances, lower is better
  - **Calinski-Harabasz Index:** Range [0, ∞), ratio of between to within variance, higher is better
  - **Inertia (WCSS):** Within-cluster sum of squares, minimize for K-means
- **External Metrics (labels available):**
  - **Adjusted Rand Index (ARI):** Range [-1, 1], similarity between clusterings, closer to 1 is better
  - **Normalized Mutual Information (NMI):** Range [0, 1], mutual information between clusterings, closer to 1 is better
  - **Fowlkes-Mallows Index:** Range [0, 1], geometric mean of precision and recall, closer to 1 is better
- **Additional Validation:**
  - Visual methods: Silhouette plots, cluster distributions
  - Domain validation: Do clusters make business sense? (MOST IMPORTANT)
- **Interactive Tool:** Clustering Visualizer at https://clustering-visualizer.web.app/

#### Key Concepts:
- K-Means is fast and scalable but assumes spherical clusters and requires k specification
- K-Means++ initialization dramatically improves convergence and quality
- DBSCAN handles arbitrary shapes and noise but struggles with varying densities
- GMM provides probabilistic soft clustering with elliptical cluster modeling
- Evaluation combines internal metrics, external validation, and critical domain knowledge

#### Learning Outcomes:
- Implement and compare multiple clustering algorithms
- Select appropriate algorithms based on data characteristics
- Optimize clustering through proper initialization and parameter tuning
- Evaluate clustering quality using multiple complementary metrics
- Interpret and validate clustering results in business context

---

### Part 3/4: Dimensionality Reduction

**Slides:** 18-25 (8 slides)

#### Topics Covered:

**Slide 18: Part Introduction**
- Overview of dimensionality reduction techniques
- Topics: Curse of dimensionality, PCA, Kernel PCA, t-SNE, UMAP, Autoencoders, VAE

**Slide 19: Curse of Dimensionality**
- **Problems in High Dimensions:**
  - **Distance Concentration:** All points become equidistant
  - **Volume Concentration:** Data occupies tiny fraction of space
  - **Exponential Growth:** Required samples grow exponentially with features
- **Consequences:**
  - Nearest neighbors become meaningless
  - Visualization impossible beyond 3D
  - Computational cost increases (memory & time)
- **Solution:** Dimensionality reduction preserves structure in lower dimensions while maintaining essential information

**Slide 20: PCA Principles and Implementation**
- **Principal Component Analysis:**
  - Linear transformation to orthogonal components
  - Components ordered by variance explained
  - First k components capture most variance
- **Steps:**
  1. Standardize data (mean=0, variance=1)
  2. Compute covariance matrix
  3. Calculate eigenvectors and eigenvalues
  4. Sort by eigenvalues (variance explained)
  5. Project data onto top k eigenvectors
- **Interpretation:**
  - PC1 captures maximum variance
  - PC2 captures maximum remaining variance (orthogonal to PC1)
  - Cumulative variance: Typically 80-95% with fewer components
- **Advantages:**
  - Fast and efficient
  - Mathematically interpretable
  - No hyperparameters
- **Limitations:**
  - Assumes linearity
  - Sensitive to scaling
  - May not preserve local structure

**Slide 21: Kernel PCA**
- **Extension of PCA:** Non-linear dimensionality reduction
- **Key Idea:** Map data to higher-dimensional space using kernel trick, then apply PCA
- **Common Kernels:**
  - RBF (Radial Basis Function): Gaussian similarity
  - Polynomial: Captures polynomial relationships
  - Sigmoid: Neural network-like transformation
- **Advantages:**
  - Captures non-linear relationships
  - Flexible through kernel selection
  - No explicit feature mapping needed (kernel trick)
- **Disadvantages:**
  - More computationally expensive than PCA
  - Kernel selection can be challenging
  - Less interpretable than linear PCA

**Slide 22: t-SNE Algorithm**
- **t-Distributed Stochastic Neighbor Embedding**
- **Purpose:** Visualization of high-dimensional data in 2D/3D
- **Key Concepts:**
  - Preserves local structure (neighborhoods)
  - Converts distances to probabilities
  - Student-t distribution in low dimensions
- **Parameters:**
  - **Perplexity:** Balance between local and global structure (5-50)
  - **Learning rate:** Step size for optimization (10-1000)
  - **Iterations:** Typically 1000-5000 for convergence
- **Characteristics:**
  - Excellent for visualization
  - Stochastic: Different runs give different results
  - Slow on large datasets (O(n²))
  - Not for dimensionality reduction to >3 dimensions
- **Use Cases:**
  - Exploratory data analysis
  - Cluster visualization
  - Understanding data structure

**Slide 23: UMAP**
- **Uniform Manifold Approximation and Projection**
- **Advantages over t-SNE:**
  - Faster: Scales to larger datasets
  - Better global structure preservation
  - Can reduce to any number of dimensions
  - More consistent results across runs
  - Allows supervised dimension reduction
- **Key Parameters:**
  - **n_neighbors:** Local neighborhood size (2-100)
  - **min_dist:** Minimum distance in embedding (0.0-1.0)
  - **metric:** Distance metric (Euclidean, cosine, etc.)
- **Use Cases:**
  - Both visualization and preprocessing
  - Large-scale data analysis
  - When global structure matters
  - General-purpose dimensionality reduction

**Slide 24: Autoencoder Dimensionality Reduction**
- **Architecture:**
  - **Input:** High-dimensional data
  - **Encoder:** Compress to latent space (e.g., 8D → 4D)
  - **Bottleneck:** Low-dimensional representation (latent vector)
  - **Decoder:** Reconstruct from latent (e.g., 4D → 8D)
  - **Output:** Reconstructed data
- **Training:** Minimize reconstruction error (MSE)
- **Calculation Example (8D → 4D):**
  - Input: [2.5, 1.8, 3.2, 0.9, 1.5, 2.1, 0.7, 3.8]
  - Encoder: z = max(0, W₁x + b₁) with ReLU
  - Latent: [1.42, 0.88, 2.15, 0.53] (compressed 50%)
  - Decoder: x̂ = σ(W₂z + b₂) with Sigmoid
  - Output: [2.48, 1.79, 3.18, 0.91, 1.52, 2.08, 0.72, 3.75]
- **Variants:**
  - Denoising: Add noise to input, train to recover clean version
  - Sparse: Enforce sparsity in latent representation
  - Contractive: Penalize sensitivity to input perturbations
- **Applications:**
  - Image compression
  - Feature learning
  - Anomaly detection
- **Advantage:** Non-linear transformation can learn complex manifolds

**Slide 25: VAE Basics**
- **Variational Autoencoder:** Probabilistic generative model
- **Architecture:**
  - Input → Encoder → Latent space → Decoder → Output
- **Key Difference from Autoencoder:**
  - Learns distribution in latent space (not just point)
  - Latent variables: μ (mean) and σ² (variance)
  - Sampling using reparameterization trick
- **Loss Function:**
  - **Reconstruction Loss:** Minimize reconstruction error
  - **KL Divergence:** Regularize to Gaussian prior N(0, I)
  - Total Loss = Reconstruction Loss + KL Divergence
- **Key Concepts:**
  - **Probabilistic:** Distribution over latent space
  - **Reparameterization:** Enables backpropagation through sampling
  - **Generation:** Sample from latent → generate new data
- **Applications:**
  - Image generation
  - Data augmentation
  - Semi-supervised learning

#### Key Concepts:
- High-dimensional data suffers from distance concentration and exponential sample requirements
- PCA provides fast linear dimensionality reduction with interpretable components
- t-SNE and UMAP excel at visualization with t-SNE preserving local structure
- Autoencoders learn non-linear compressions through neural networks
- VAEs add probabilistic structure enabling generative capabilities

#### Learning Outcomes:
- Diagnose and mitigate curse of dimensionality effects
- Apply PCA for interpretable linear dimensionality reduction
- Use t-SNE and UMAP for effective data visualization
- Implement autoencoder-based dimensionality reduction
- Understand generative modeling with VAEs

---

### Part 4/4: Basic Anomaly Detection

**Slides:** 26-30 (5 slides)

#### Topics Covered:

**Slide 26: Part Introduction**
- Overview of anomaly detection methods
- Topics: Overview, statistical methods, Isolation Forest, hands-on implementation, summary

**Slide 27: Anomaly Detection Overview**
- **Definition:** Identifying data points that deviate significantly from normal behavior
- **Types of Anomalies:**
  - **Point anomalies:** Individual instances that are anomalous
  - **Contextual anomalies:** Anomalous in specific context
  - **Collective anomalies:** Collection of instances that are anomalous
- **Applications:**
  - Fraud detection: Credit card fraud, insurance claims
  - Network security: Intrusion detection, DDoS attacks
  - Healthcare: Disease outbreak, abnormal vitals
  - Manufacturing: Defect detection, quality control
  - System monitoring: Server failures, performance issues
- **Challenges:**
  - Imbalanced data: Anomalies are rare
  - Defining "normal": What constitutes normal behavior?
  - Evolving patterns: Normal behavior changes over time
  - High dimensionality: Curse of dimensionality applies

**Slide 28: Statistical Methods**
- **Assumption:** Data follows a known distribution (usually Gaussian)
- **Methods:**
  - **Z-Score Method:**
    - Formula: |z| > threshold (e.g., ±3σ)
    - Points beyond threshold are anomalies
    - Example: Student test scores [85, 90, 88, 92, 87, 89, 150]
      - Mean = 94.4, Std Dev = 22.9
      - Z-score(150) = 2.43 → Detected as outlier
  - **Box Plot Method:**
    - Formula: Q1 - 1.5×IQR, Q3 + 1.5×IQR
    - Points beyond 1.5×IQR from quartiles
    - Example: Employee salaries [300, 320, 310, 330, 340, 325, 800] (×$10k)
      - Q1 = 310, Q3 = 340, IQR = 30
      - Upper fence = 385 → $8M detected as outlier
  - **Mahalanobis Distance:**
    - Formula: D² = (x-μ)ᵀΣ⁻¹(x-μ)
    - For multi-feature data, accounts for correlations
    - Example: Customer purchases (amount, frequency)
      - Normal: ($500k, 10 times)
      - Suspicious: ($2M, 2 times)
      - Mahalanobis distance = 5.8 → Abnormal (D² > 4)
- **Strengths:**
  - Simple and interpretable
  - Fast computation
  - Well-understood theory
  - Easy to implement
- **Limitations:**
  - Assumes specific distribution
  - Sensitive to outliers in training
  - May not work for complex patterns
  - Requires distribution knowledge
- **Works Well For:** Low-dimensional data with known distribution
- **Preprocessing Tip:** Remove known anomalies before fitting distribution

**Slide 29: Isolation Forest**
- **Key Insight:** Anomalies are "few and different", easier to isolate
- **Algorithm:**
  1. Randomly select feature and split value
  2. Recursively partition data into tree
  3. Anomalies require fewer splits (shorter paths)
  4. Build ensemble of trees (forest)
  5. Compute anomaly score based on average path length
- **Characteristics:**
  - Ensemble method: Multiple random trees
  - No distance calculations needed
  - Linear time complexity: O(n log n)
  - Memory efficient
- **Parameters:**
  - **n_estimators:** Number of trees (100-200 typical)
  - **max_samples:** Samples per tree (256 default)
  - **contamination:** Expected proportion of anomalies (0.1 = 10%)
- **Advantages:**
  - Scalable to large datasets
  - Handles high-dimensional data well
  - No assumptions about data distribution
  - Efficient in both time and memory
- **Disadvantages:**
  - Less interpretable than statistical methods
  - May struggle with uniformly distributed anomalies
  - Sensitive to contamination parameter

**Slide 30: Hands-on Implementation with scikit-learn**
- **Key Imports:**
  - `from sklearn.cluster import KMeans`
  - `from sklearn.cluster import DBSCAN`
  - `from sklearn.decomposition import PCA`
  - `from sklearn.ensemble import IsolationForest`
- **ML Pipeline:**
  1. StandardScaler
  2. Dimensionality Reduction
  3. Clustering / Anomaly Detection
- **Evaluation Metrics:**
  - `silhouette_score`
  - `davies_bouldin_score`
  - `calinski_harabasz_score`
- **Visualization Tools:**
  - matplotlib: For plotting results
  - seaborn: For enhanced visualizations
- **Best Practices:**
  - Cross-validation for robustness
  - Careful parameter tuning
  - Domain knowledge validation
  - Iterative experimentation

#### Key Concepts:
- Anomaly detection identifies rare, unusual patterns in data
- Statistical methods work well for low-dimensional Gaussian data
- Isolation Forest efficiently handles high-dimensional non-Gaussian data
- Proper preprocessing and scaling are critical for all methods
- Domain validation is essential for anomaly detection success

#### Learning Outcomes:
- Identify appropriate anomaly detection methods for different data types
- Implement statistical methods for low-dimensional data
- Apply Isolation Forest to high-dimensional datasets
- Build complete scikit-learn pipelines for anomaly detection
- Validate and interpret anomaly detection results

---

### Concluding Slides

**Slides:** 31-32 (2 slides)

**Slide 31: Summary and Next Lecture Preview**
- **Today's Summary:**
  - **Clustering:** K-Means, DBSCAN, Hierarchical, GMM - each with trade-offs
  - **Dimensionality Reduction:** PCA, t-SNE, UMAP - visualization and compression
  - **Anomaly Detection:** Statistical methods, Isolation Forest - identify outliers
  - **Key Considerations:** Scaling, evaluation metrics, interpretability
- **Next Lecture Topics:**
  - Deep Autoencoders
  - Generative Adversarial Networks (GANs)
  - Self-Supervised Learning
- **Practice Assignment:** Apply these algorithms to real-world datasets

**Slide 32: Thank You**
- Instructor contact information and closing

---

## 🛠️ Prerequisites

### Knowledge Prerequisites:
1. **Mathematics:**
   - Linear algebra: Vectors, matrices, eigenvalues/eigenvectors
   - Probability and statistics: Distributions, variance, covariance
   - Calculus: Gradients, optimization basics
   - Distance metrics: Euclidean, Manhattan, cosine similarity

2. **Machine Learning:**
   - Supervised learning fundamentals
   - Train/test split, cross-validation
   - Model evaluation metrics
   - Basic neural network concepts (for autoencoder sections)

3. **Programming:**
   - Python proficiency (intermediate level)
   - NumPy for array operations
   - Pandas for data manipulation
   - Basic object-oriented programming

### Software Requirements:
```bash
# Core libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Optional but recommended
jupyter>=1.0.0
scipy>=1.7.0
plotly>=5.0.0  # for interactive visualizations

# For deep learning sections (Autoencoders, VAE)
tensorflow>=2.8.0  # or pytorch>=1.10.0
```

### Recommended Reading:
- "Pattern Recognition and Machine Learning" by Christopher Bishop (Chapters 9, 12)
- "The Elements of Statistical Learning" by Hastie et al. (Chapter 14)
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/clustering.html

---

## 💻 Hands-on Components

### 1. **Data Preprocessing Pipeline**
**Objective:** Implement complete preprocessing workflow

**Tasks:**
- Load a real-world dataset with mixed feature scales
- Handle missing values using appropriate imputation strategies
- Apply StandardScaler, MinMaxScaler, and RobustScaler
- Compare results and document which scaler works best for the data

**Expected Outputs:**
- Preprocessed dataset ready for clustering
- Comparison report with visualizations showing scaling effects
- Recommendations for scaler selection based on data characteristics

**Estimated Time:** 30 minutes

---

### 2. **Clustering Comparison Study**
**Objective:** Compare multiple clustering algorithms on the same dataset

**Tasks:**
- Implement K-Means with K-Means++ initialization
- Implement DBSCAN with parameter tuning (ε, MinPts)
- Implement Hierarchical clustering with different linkage methods
- Implement GMM with optimal number of components
- Compare results using internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Visualize clusters using PCA reduction to 2D

**Dataset Suggestions:**
- Iris dataset (simple, 4D, well-separated clusters)
- Mall customer segmentation (real-world business data)
- Wholesale customers dataset (6D, mixed cluster shapes)

**Expected Outputs:**
- Comparison table with metrics for each algorithm
- 2D visualization of clustering results for each method
- Written analysis of which algorithm performs best and why
- Parameter sensitivity analysis for DBSCAN

**Estimated Time:** 90 minutes

---

### 3. **Dimensionality Reduction Visualization**
**Objective:** Apply and compare dimensionality reduction techniques

**Tasks:**
- Load high-dimensional dataset (e.g., MNIST digits, >50 dimensions)
- Apply PCA: Plot explained variance, reduce to 2D, visualize
- Apply t-SNE: Tune perplexity parameter, visualize
- Apply UMAP: Compare with t-SNE, note differences
- Measure runtime for each method
- Document preservation of local vs global structure

**Expected Outputs:**
- Side-by-side visualizations of PCA, t-SNE, and UMAP results
- Scree plot showing PCA explained variance
- Performance comparison table (runtime, cluster separation)
- Analysis of when to use each technique

**Estimated Time:** 60 minutes

---

### 4. **Autoencoder Implementation**
**Objective:** Build and train an autoencoder for dimensionality reduction

**Tasks:**
- Design encoder-decoder architecture (e.g., 784 → 128 → 32 → 128 → 784 for MNIST)
- Implement in TensorFlow/Keras or PyTorch
- Train with reconstruction loss (MSE)
- Visualize original vs reconstructed images
- Extract and visualize latent representations
- Compare with PCA on same data

**Expected Outputs:**
- Trained autoencoder model
- Learning curves (loss over epochs)
- Visualization of latent space in 2D (if bottleneck is 2D)
- Reconstruction quality comparison
- Comparison with PCA in terms of reconstruction error

**Estimated Time:** 90 minutes

---

### 5. **Anomaly Detection System**
**Objective:** Build a complete anomaly detection pipeline

**Tasks:**
- Implement statistical methods (Z-score, Box plot)
- Implement Isolation Forest
- Generate synthetic dataset with known anomalies
- Evaluate detection performance (precision, recall, F1-score)
- Tune contamination parameter
- Visualize anomalies in 2D using dimensionality reduction

**Dataset Options:**
- Credit card fraud dataset
- Network intrusion detection dataset (KDD Cup)
- Synthetic data with controlled anomaly rate

**Expected Outputs:**
- Complete pipeline from raw data to anomaly detection
- Performance metrics for each method
- Confusion matrix and ROC curve
- Visualization of detected anomalies
- Recommendations for deployment

**Estimated Time:** 75 minutes

---

### 6. **Complete ML Pipeline Project**
**Objective:** Integrate all components into production-ready pipeline

**Tasks:**
- Build scikit-learn pipeline with preprocessing, dimensionality reduction, and clustering
- Implement model persistence (save/load models)
- Create evaluation dashboard with multiple metrics
- Document hyperparameter choices
- Perform cross-validation experiments
- Generate comprehensive report

**Expected Outputs:**
- Production-ready pipeline code
- Saved model files
- Interactive visualization dashboard
- Technical documentation
- Performance report with recommendations

**Estimated Time:** 120 minutes

---

## 📖 Additional Resources

### Official Documentation:
1. **Scikit-learn Clustering:** https://scikit-learn.org/stable/modules/clustering.html
2. **Scikit-learn Decomposition:** https://scikit-learn.org/stable/modules/decomposition.html
3. **Scikit-learn Preprocessing:** https://scikit-learn.org/stable/modules/preprocessing.html
4. **Isolation Forest Documentation:** https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

### Interactive Tools:
- **Clustering Visualizer:** https://clustering-visualizer.web.app/ (test K-means, DBSCAN, Hierarchical, MeanShift)
- **Seeing Theory:** https://seeing-theory.brown.edu/ (probability and statistics visualization)
- **Distill.pub:** https://distill.pub/2016/misread-tsne/ (understanding t-SNE)

### Research Papers:
1. **K-Means++:** Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"
2. **DBSCAN:** Ester, M., et al. (1996). "A density-based algorithm for discovering clusters"
3. **t-SNE:** van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE"
4. **UMAP:** McInnes, L., et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"
5. **Isolation Forest:** Liu, F. T., et al. (2008). "Isolation Forest"

### Video Lectures:
- **StatQuest:** Josh Starmer's videos on PCA, K-Means, Hierarchical Clustering
- **3Blue1Brown:** "Neural networks" series (for understanding autoencoders)
- **Coursera:** Andrew Ng's Machine Learning course (Unsupervised Learning module)

### Books:
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron (Chapters 8-9)
- "Python Data Science Handbook" by Jake VanderPlas (Chapter 5)
- "Introduction to Statistical Learning" by James et al. (Chapter 10)

### Datasets for Practice:
1. **UCI Machine Learning Repository:** https://archive.ics.uci.edu/ml/index.php
2. **Kaggle Datasets:** https://www.kaggle.com/datasets (search "clustering", "anomaly detection")
3. **Scikit-learn built-in datasets:** `sklearn.datasets` module
4. **Real-world business data:** Retail transaction data, customer demographics

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Sequential Learning (Recommended):**
   - Review slides 1-8 (Part 1) → Complete preprocessing exercises
   - Review slides 9-17 (Part 2) → Implement clustering algorithms
   - Review slides 18-25 (Part 3) → Apply dimensionality reduction
   - Review slides 26-30 (Part 4) → Build anomaly detection system
   - Complete integrated project combining all components

2. **Topic-Based Learning:**
   - Focus on specific algorithm (e.g., K-Means) → Deep dive with implementation
   - Move to related topics (K-Means++ → GMM)
   - Compare and contrast different approaches

3. **Project-First Approach:**
   - Start with hands-on project
   - Reference slides when encountering concepts
   - Iterate between theory and practice

### For Classroom Instruction:
1. **3-Hour Lecture Format:**
   - Hour 1: Parts 1-2 (Overview + Clustering) with live K-Means demo
   - Hour 2: Part 3 (Dimensionality Reduction) with PCA/t-SNE visualization
   - Hour 3: Part 4 (Anomaly Detection) + hands-on lab session

2. **Workshop Format (6 hours):**
   - Session 1 (1.5h): Theory + Clustering implementation
   - Session 2 (1.5h): Dimensionality reduction + visualization
   - Session 3 (1.5h): Anomaly detection + pipeline integration
   - Session 4 (1.5h): Final project + presentations

3. **Flipped Classroom:**
   - Pre-class: Students review slides and watch supplementary videos
   - In-class: Focus on hands-on implementation and troubleshooting
   - Post-class: Complete comprehensive project

### Using the Slideshow:
- **HTML Slideshow:** Open `lecture17_slideshow.html` in web browser
- **Navigation:**
  - Arrow keys (←/→) or spacebar to navigate
  - Home/End keys to jump to first/last slide
  - Progress bar shows completion percentage
  - Slide counter shows current position
- **Individual Slides:** All slides available as separate HTML files for focused review

---

## 📊 Assessment Suggestions

### 1. **Quiz Questions** (Comprehension Check)

**Part 1 - Fundamentals:**
- Q: What is the key difference between supervised and unsupervised learning?
- Q: Why is feature scaling critical for distance-based algorithms? Give an example.
- Q: Name three internal clustering evaluation metrics and explain when to use each.

**Part 2 - Clustering:**
- Q: Explain how K-Means++ initialization improves over random initialization.
- Q: What are the advantages and disadvantages of DBSCAN compared to K-Means?
- Q: When would you choose GMM over K-Means? Provide a concrete scenario.

**Part 3 - Dimensionality Reduction:**
- Q: Describe the curse of dimensionality and its three main problems.
- Q: Compare PCA and t-SNE: when would you use each technique?
- Q: How does a VAE differ from a standard autoencoder?

**Part 4 - Anomaly Detection:**
- Q: Explain the key insight behind Isolation Forest.
- Q: What are the limitations of statistical methods for anomaly detection?
- Q: Why is domain validation the most important evaluation method?

---

### 2. **Programming Assignments**

**Assignment 1: Clustering Analysis (25 points)**
- Implement K-Means, DBSCAN, and Hierarchical clustering (9 points)
- Evaluate using multiple metrics (6 points)
- Visualize results with proper labels and legends (5 points)
- Write analysis comparing algorithms (5 points)

**Assignment 2: Dimensionality Reduction Pipeline (25 points)**
- Apply PCA, t-SNE, and UMAP to high-dimensional data (9 points)
- Create side-by-side visualizations (6 points)
- Measure and report computational performance (5 points)
- Analyze quality of dimensionality reduction (5 points)

**Assignment 3: Anomaly Detection System (25 points)**
- Implement statistical methods and Isolation Forest (10 points)
- Evaluate on labeled anomaly dataset (8 points)
- Create visualization dashboard (4 points)
- Document findings and recommendations (3 points)

---

### 3. **Final Project** (100 points)

**Project Requirements:**
- Choose real-world dataset (e.g., customer segmentation, fraud detection)
- Complete data preprocessing pipeline (20 points)
- Apply at least 3 clustering algorithms with comparison (25 points)
- Perform dimensionality reduction for visualization (15 points)
- Implement anomaly detection if applicable (15 points)
- Create comprehensive report with:
  - Problem statement and motivation (5 points)
  - Methodology and implementation details (10 points)
  - Results with visualizations (5 points)
  - Conclusions and business recommendations (5 points)

**Deliverables:**
- Jupyter notebook with complete code
- Written report (4-6 pages)
- 10-minute presentation
- Code repository on GitHub

---

### 4. **Evaluation Rubric**

| Criteria | Excellent (90-100%) | Good (75-89%) | Satisfactory (60-74%) | Needs Improvement (<60%) |
|----------|---------------------|---------------|----------------------|--------------------------|
| **Technical Implementation** | All algorithms correctly implemented, efficient code, proper error handling | Most algorithms correct, some efficiency issues | Basic implementation works but has bugs or inefficiencies | Major implementation errors, doesn't run |
| **Data Preprocessing** | Comprehensive preprocessing, appropriate scaling, handles edge cases | Good preprocessing, correct scaling, minor issues | Basic preprocessing, some scaling errors | Missing or incorrect preprocessing |
| **Evaluation & Analysis** | Uses multiple metrics, thorough comparison, insightful analysis | Uses appropriate metrics, good comparison, reasonable analysis | Uses some metrics, basic comparison, surface-level analysis | Inadequate evaluation, poor or missing analysis |
| **Visualization** | Clear, publication-quality plots with proper labels and legends | Good visualizations, mostly clear, proper labels | Basic plots that convey information but could be improved | Poor or missing visualizations |
| **Documentation** | Excellent comments, clear explanations, comprehensive report | Good documentation, clear report, minor gaps | Adequate documentation, report covers basics | Poor or missing documentation |
| **Code Quality** | Clean, modular, follows PEP 8, reusable functions | Good structure, mostly follows conventions | Basic structure, some organization issues | Poorly organized, hard to read |

---

### 5. **Practical Exam** (Optional)

**Format:** 90-minute hands-on coding exam

**Sample Questions:**
1. Given a dataset, implement K-Means clustering and plot the elbow curve (30 min)
2. Apply PCA to reduce dimensions to 2D and visualize the result (20 min)
3. Implement Isolation Forest to detect anomalies in provided dataset (25 min)
4. Short answer questions on algorithm selection and parameter tuning (15 min)

**Grading:** 
- Code functionality: 60%
- Code quality and efficiency: 20%
- Visualization and interpretation: 20%

---

## 📝 Notes for Implementation

### Technical Requirements:

**1. Computational Resources:**
- **Minimum:** 8GB RAM, dual-core processor
- **Recommended:** 16GB RAM, quad-core processor, GPU for deep learning sections
- **Cloud alternatives:** Google Colab (free GPU), Kaggle Kernels, AWS SageMaker

**2. Development Environment:**
```bash
# Option 1: Anaconda (recommended for beginners)
conda create -n unsupervised python=3.9
conda activate unsupervised
conda install scikit-learn matplotlib seaborn jupyter
conda install tensorflow  # for autoencoder sections

# Option 2: pip + virtualenv
python -m venv unsupervised_env
source unsupervised_env/bin/activate  # On Windows: unsupervised_env\Scripts\activate
pip install -r requirements.txt
```

**3. Jupyter Notebook Setup:**
```python
# Essential imports for all notebooks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Configure visualization
%matplotlib inline
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

---

### Implementation Tips:

**1. Clustering Best Practices:**
```python
# Always scale data before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use K-Means++ initialization (default in scikit-learn)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)

# For DBSCAN, estimate eps using k-distance graph
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1], axis=0)
plt.plot(distances)  # Look for "elbow" to choose eps
```

**2. Dimensionality Reduction Guidelines:**
```python
# PCA: Determine number of components
pca = PCA()
pca.fit(X_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1  # 95% variance

# t-SNE: Be patient, it's slow
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)  # Can take several minutes

# UMAP: Faster alternative
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
```

**3. Anomaly Detection Pipeline:**
```python
from sklearn.ensemble import IsolationForest

# Pipeline with preprocessing
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', IsolationForest(contamination=0.1, random_state=42))
])

# Fit and predict
pipe.fit(X_train)
predictions = pipe.predict(X_test)  # -1 for anomalies, 1 for normal
```

---

### Common Pitfalls and Solutions:

**1. Convergence Issues:**
- **Problem:** K-Means doesn't converge or gives poor results
- **Solution:** Increase `max_iter`, try different `n_init` values, ensure proper scaling

**2. Memory Errors with Large Datasets:**
- **Problem:** Out of memory with t-SNE or hierarchical clustering
- **Solution:** Use MiniBatchKMeans, sample data for visualization, use UMAP instead

**3. DBSCAN Finding No Clusters:**
- **Problem:** All points classified as noise
- **Solution:** Increase `eps` parameter, decrease `min_samples`, verify scaling

**4. Autoencoder Not Learning:**
- **Problem:** Reconstruction loss doesn't decrease
- **Solution:** Adjust learning rate, check data normalization, increase model capacity

---

### Performance Optimization:

**1. Speed Up K-Means:**
```python
# Use MiniBatchKMeans for large datasets
from sklearn.cluster import MiniBatchKMeans
kmeans_fast = MiniBatchKMeans(n_clusters=8, batch_size=100, random_state=42)
```

**2. Parallel Processing:**
```python
# Use n_jobs parameter for parallelization
kmeans = KMeans(n_clusters=5, n_jobs=-1)  # Use all CPU cores
pca = PCA(n_components=2, n_jobs=-1)
```

**3. Memory Management:**
```python
# Process in chunks for very large datasets
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=2, batch_size=100)
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    ipca.partial_fit(chunk)
```

---

### Debugging Strategies:

**1. Visualization at Each Step:**
```python
# Always visualize intermediate results
def plot_clusters(X, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Cluster')
    plt.title('Clustering Results')
    plt.show()
```

**2. Sanity Checks:**
```python
# Check data properties
print(f"Data shape: {X.shape}")
print(f"Missing values: {np.isnan(X).sum()}")
print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Data mean: {X.mean():.2f}, std: {X.std():.2f}")
```

**3. Logging and Timing:**
```python
import time
from datetime import datetime

start = time.time()
kmeans.fit(X_scaled)
elapsed = time.time() - start
print(f"Training completed in {elapsed:.2f} seconds")
```

---

### Dataset Preparation:

**Recommended Datasets by Difficulty:**

**Beginner:**
- Iris (150 samples, 4 features, 3 clear clusters)
- Wine (178 samples, 13 features, 3 wine types)
- Breast Cancer Wisconsin (569 samples, 30 features, binary classification → unsupervised)

**Intermediate:**
- Mall Customers (200 samples, 5 features, customer segmentation)
- Credit Card (30,000 samples, 23 features, fraud detection)
- MNIST digits (70,000 images, 784 features, visualization)

**Advanced:**
- KDD Cup 99 (4.9M samples, 41 features, intrusion detection)
- ImageNet subset (100k images, high-dimensional, clustering/compression)
- Text datasets (TF-IDF vectors, document clustering)

---

## 🙏 Credits

**Lecture Developed By:**  
Ho-min Park  
Email: homin.park@ghent.ac.kr | powersimmani@gmail.com

**Institution:** Ghent University (assumed based on email domain)

**Course Materials:**
- 32 interactive HTML slides with visualizations
- Comprehensive coverage of unsupervised learning fundamentals
- Integrated hands-on components with scikit-learn
- Real-world examples and applications

**Interactive Tools:**
- Clustering Visualizer: https://clustering-visualizer.web.app/
  - Test K-means, DBSCAN, Hierarchical clustering, and Mean Shift
  - Interactive parameter tuning
  - Real-time visualization

**Acknowledgments:**
- Scikit-learn development team for excellent documentation
- Andrew Ng for foundational ML course materials
- Research papers authors for algorithmic innovations
- Open-source community for tools and libraries

---

**License:** These materials are intended for educational purposes. Please contact the instructor for permission regarding redistribution or commercial use.

**Last Updated:** November 2025

**Version:** 1.0

---

## 📌 Quick Reference Card

### Algorithm Selection Guide:

| Use Case | Recommended Algorithm | Key Parameters |
|----------|----------------------|----------------|
| Spherical clusters, known k | K-Means | `n_clusters`, `init='k-means++'` |
| Arbitrary shapes, unknown k | DBSCAN | `eps`, `min_samples` |
| Hierarchical relationships | Hierarchical | `linkage`, `n_clusters` |
| Probabilistic soft clustering | GMM | `n_components`, `covariance_type` |
| Linear dimensionality reduction | PCA | `n_components` (or explained variance) |
| Visualization (local structure) | t-SNE | `perplexity`, `n_iter` |
| Visualization (fast, scalable) | UMAP | `n_neighbors`, `min_dist` |
| Non-linear compression | Autoencoder | Architecture, epochs |
| Anomaly detection (low-dim) | Statistical | Threshold (e.g., 3σ) |
| Anomaly detection (high-dim) | Isolation Forest | `contamination`, `n_estimators` |

### Common Parameters:

- **random_state:** Always set for reproducibility (e.g., `random_state=42`)
- **n_jobs:** Use `-1` for parallel processing on all CPU cores
- **verbose:** Set to `1` or `2` for training progress monitoring

### Evaluation Metrics Quick Reference:

**Clustering:**
- Silhouette Score: [-1, 1], higher is better
- Davies-Bouldin Index: [0, ∞), lower is better
- Calinski-Harabasz: [0, ∞), higher is better

**Anomaly Detection:**
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)

---

**End of README**