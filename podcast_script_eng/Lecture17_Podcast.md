# Lecture 17 Podcast: Unsupervised Learning - Clustering and Dimensionality Reduction

## Episode Information
- **Topic**: Unsupervised Learning Concepts, Clustering Algorithms, Dimensionality Reduction, Anomaly Detection
- **Estimated Time**: 15 minutes
- **Target Audience**: Students studying machine learning, those working with unlabeled data

---

## Script

**[Intro - 0:00]**

**Host A**: Hello! AI Vision Systems Podcast. Today we're covering unsupervised learning - finding patterns in data without labels!

**Host B**: Yes! We've seen a lot of supervised learning like classification and regression, but in reality, unlabeled data is much more common.

**Host A**: What can you do without labels?

**Host B**: More than you think! Group similar data together, visualize high-dimensional data, or find anomalous data. We'll cover all of this today!

---

**[Section 1: What is Unsupervised Learning? - 1:30]**

**Host A**: Define unsupervised learning.

**Host B**: When you have only input data X without ground truth labels y, finding hidden structure in the data. If supervised learning is "this is a cat" while training, unsupervised learning is "group similar things together on your own."

**Host A**: What specific types are there?

**Host B**: Four main types! Clustering groups similar data, dimensionality reduction compresses features, anomaly detection finds outliers, and association finds relationships.

**Host A**: When is it used?

**Host B**: Customer segmentation divides customers by purchase patterns, fraud detection finds abnormal transactions, gene data analysis finds similar genetic patterns. Useful when there are no labels or labeling is expensive!

---

**[Section 2: K-Means Clustering - 3:30]**

**Host A**: Tell me about the most basic clustering algorithm.

**Host B**: K-Means! The idea is really intuitive. Set K centroids and assign each data point to the nearest centroid.

**Host A**: Explain the algorithm step by step.

**Host B**: Four steps! First, randomly initialize K centroids. Second, assign each point to the nearest centroid. Third, update centroids to the mean of each cluster. Fourth, repeat until centroids stop moving!

**Host A**: Does it always work well?

**Host B**: It's sensitive to initialization! Unlucky starts give poor results. That's why K-Means++ emerged, initializing centroids far apart. First centroid random, next centroids probabilistically selected far from existing ones.

**Host A**: How do you choose K?

**Host B**: Elbow Method is common! Graph cluster within-variance (inertia) while increasing K from 1. The "elbow" where decrease rate sharply drops is appropriate K.

---

**[Section 3: Other Clustering Algorithms - 5:30]**

**Host A**: Does K-Means have limitations?

**Host B**: Yes! It only finds spherical clusters well. Can't find elongated or complex shapes. And you must pre-set K. Also sensitive to outliers.

**Host A**: Alternatives?

**Host B**: There's DBSCAN! Density-Based Spatial Clustering. Views high-density regions as clusters, treats low-density points as noise (outliers).

**Host A**: How does it work?

**Host B**: Two parameters. epsilon is neighbor radius, minPts is minimum neighbors needed to be a core point. Connect core points and their neighbors to form clusters. No need to set K, finds complex shapes!

**Host A**: There's also hierarchical clustering?

**Host B**: Yes! Hierarchical Clustering. Agglomerative starts with each point and merges close ones. Can visualize as a dendrogram tree structure, getting different cluster numbers depending on where you cut.

---

**[Section 4: Curse of Dimensionality and PCA - 7:30]**

**Host A**: Now let's move to dimensionality reduction.

**Host B**: First, why it's needed! Curse of Dimensionality - as dimensions increase, problems arise. Data becomes sparse, distances become meaningless, computation slows.

**Host A**: Why do distances become meaningless?

**Host B**: In high dimensions, distances between all points become similar! Counter-intuitive. In 1000 dimensions, the distance difference between nearest and farthest points is almost zero. So algorithms like K-NN don't work well.

**Host A**: Is PCA the solution?

**Host B**: The most basic linear dimensionality reduction! Principal Component Analysis. Finds the direction of maximum variance in data and projects onto those axes.

**Host A**: How do you compute it?

**Host B**: Find eigenvectors of the covariance matrix! The eigenvector corresponding to the largest eigenvalue is the first principal component. Project data onto top k principal components to reduce to k dimensions. Usually choose k that preserves 95% variance.

---

**[Section 5: t-SNE and UMAP - 9:30]**

**Host A**: What are PCA's limitations?

**Host B**: Captures only linear relationships! PCA doesn't work well if data is distributed in curved form. So nonlinear methods are needed.

**Host A**: t-SNE is famous?

**Host B**: Yes! t-Distributed Stochastic Neighbor Embedding. Keeps points close in high dimensions also close in low dimensions. Really good for visualization!

**Host A**: How does it work?

**Host B**: Computes pairwise similarity in high dimensions with Gaussian, in low dimensions with t-distribution. Optimizes low-dimensional coordinates to minimize KL Divergence between the two distributions.

**Host A**: Downsides?

**Host B**: Slow! O(n^2) complexity makes it hard with large data. And results vary each time. Tends to preserve local rather than global structure, so distances between far clusters are meaningless.

**Host A**: How about UMAP?

**Host B**: An alternative to t-SNE! Much faster (O(n)), preserves global structure better. Can embed new points. Recently UMAP is used more than t-SNE!

---

**[Section 6: Autoencoders and Dimensionality Reduction - 11:00]**

**Host A**: You can also do dimensionality reduction with deep learning?

**Host B**: Autoencoders! Train to compress input then restore it. Input → Encoder → Latent vector (bottleneck) → Decoder → Output. Train so output equals input.

**Host A**: How is it different from PCA?

**Host B**: PCA is linear transformation, but autoencoders have nonlinear activation functions so can capture complex relationships! And you can design the architecture as you want.

**Host A**: There's also VAE?

**Host B**: Variational Autoencoder! Models latent space as probability distributions. While regular autoencoders output point latent vectors, VAE outputs mean and variance, sampling from that distribution. This also enables generating new samples!

---

**[Section 7: Cluster Evaluation - 12:30]**

**Host A**: How do you know if clustering is good?

**Host B**: Difficult problem! No labels. Use Internal Metrics. Silhouette Score is representative.

**Host A**: What's Silhouette Score?

**Host B**: For each point, compute within-cluster average distance a and nearest other cluster average distance b. (b-a)/max(a,b) is that point's silhouette value. Between -1 and 1, closer to 1 is better!

**Host A**: Other metrics?

**Host B**: Davies-Bouldin Index, lower is better. If you have actual labels, can use external metrics. NMI (Normalized Mutual Information), ARI (Adjusted Rand Index), etc.

---

**[Section 8: Anomaly Detection - 13:30]**

**Host A**: Anomaly detection is also unsupervised learning?

**Host B**: Yes! Finding outliers that differ from most data. Used in fraud detection, network intrusion detection, manufacturing defect detection.

**Host A**: What methods exist?

**Host B**: Statistical methods include Z-score. Points beyond 3 standard deviations from mean are outliers. IQR method: below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are outliers.

**Host A**: Machine learning methods?

**Host B**: Isolation Forest is popular! The idea is interesting. Outliers are easy to isolate. If you randomly select features and split, outliers get isolated quickly with short tree paths.

**Host A**: Other methods too?

**Host B**: One-Class SVM creates a boundary around normal data. Local Outlier Factor (LOF) compares local density. Autoencoders learn to reconstruct normal data, and high reconstruction error means outlier!

---

**[Outro - 14:30]**

**Host A**: Let's summarize today's content.

**Host B**: First, unsupervised learning finds data structure without labels. Clustering, dimensionality reduction, and anomaly detection are representative!

**Host A**: Second, K-Means is simple but suited for spherical clusters, while DBSCAN is robust to complex shapes and outliers.

**Host B**: Third, PCA is basic linear dimensionality reduction, while t-SNE and UMAP are nonlinear methods good for visualization.

**Host A**: Fourth, evaluate cluster quality with Silhouette Score and detect outliers with methods like Isolation Forest!

**Host B**: Unsupervised learning is really useful for data exploration and preprocessing. We'll be back with interesting topics next time!

**Host A**: Thank you!

---

## Key Keywords
- Unsupervised Learning, Supervised Learning
- Clustering: K-Means, K-Means++, DBSCAN, Hierarchical
- Curse of Dimensionality
- Dimensionality Reduction: PCA, Kernel PCA
- Visualization: t-SNE, UMAP
- Autoencoder, VAE (Variational Autoencoder)
- Cluster Evaluation: Silhouette Score, Elbow Method
- Anomaly Detection: Z-score, IQR, Isolation Forest, LOF
- Gaussian Mixture Model (GMM)
