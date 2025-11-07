# Lecture 18: Advanced Unsupervised Learning

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Institution:** Ghent University

**Course Details:**
- Total Slides: 32
- Duration: Approximately 3-4 hours (lecture + hands-on)
- Level: Advanced Graduate/Senior Undergraduate
- Prerequisites: Machine Learning, Deep Learning, Python Programming

This lecture provides a comprehensive exploration of modern unsupervised learning techniques, covering self-supervised learning, time series clustering, graph clustering, and advanced applications. The course integrates theoretical foundations with practical implementations across four major domains.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand and implement self-supervised learning methods** including contrastive learning frameworks (SimCLR, MoCo, BYOL) for visual representation learning without labels
2. **Apply specialized clustering techniques to time series data** using Dynamic Time Warping (DTW), K-Shape algorithm, and subsequence clustering methods
3. **Perform graph clustering and community detection** using spectral methods, Louvain algorithm, and Graph Neural Networks (GCN, Deep Graph Infomax)
4. **Design and deploy deep clustering solutions** for multi-modal data and large-scale applications across various industries
5. **Critically evaluate clustering approaches** by comparing classical and deep learning methods on real-world datasets

---

## 📚 Lecture Structure

### Part 1/4: Self-Supervised Learning

**Topics Covered:**

#### Slide 4: Lecture Overview and Recap
- Comparison of three learning paradigms:
  - **Supervised Learning**: Requires labeled datasets with manual annotation (high cost)
  - **Unsupervised Learning**: Works with unlabeled data to find patterns
  - **Self-Supervised Learning**: Automatically generates labels from data structure (modern bridge approach)
- Evolution from traditional to modern unsupervised methods

#### Slide 5: What is Self-Supervised Learning?
- Four-stage workflow:
  1. Start with massive unlabeled datasets
  2. Design pretext tasks that generate pseudo-labels automatically
  3. Train models to learn useful feature representations
  4. Apply learned features to downstream tasks
- Pretext task examples:
  - **Image Rotation**: Predict rotation angle (0°, 90°, 180°, 270°)
  - **Colorization**: Predict colors from grayscale images
  - **BERT (NLP)**: Masked language modeling
  - **GPT (NLP)**: Next token prediction
- Key advantage: Scales to massive datasets without human annotation

#### Slide 6: Contrastive Learning Principles
- Core idea: Similar items should be close in embedding space, dissimilar items far apart
- Three key components:
  - **Positive Pairs**: Augmented versions of the same data point
  - **Negative Pairs**: Different data points in the batch
  - **InfoNCE Loss**: Maximizes agreement of positive pairs while pushing negatives away
- Temperature parameter (τ) controls distribution sharpness
- Visual representation of embedding space with anchor, positive, and negative samples

#### Slide 7: SimCLR Algorithm
- Simple Framework for Contrastive Learning of visual Representations
- Architecture pipeline:
  1. Input image undergoes two different augmentations (crop + color + blur)
  2. Both augmented views pass through shared ResNet encoder
  3. Projection head (MLP) maps to contrastive space
  4. NT-Xent Loss maximizes agreement between positive pairs
- Key implementation details:
  - Large batch sizes: 4096-8192 samples
  - Strong data augmentation is critical
  - State-of-the-art results on ImageNet without labels

#### Slide 8: MoCo and BYOL
- **MoCo (Momentum Contrast)**:
  - Uses queue-based architecture with large dictionary of negative samples
  - Momentum encoder provides consistent representations over time
  - Maintains queue of negative samples for efficient contrastive learning
  - Excellent ImageNet performance
- **BYOL (Bootstrap Your Own Latent)**:
  - Revolutionary: No negative pairs needed
  - Asymmetric architecture with online and target networks
  - Predictor network prevents representation collapse
  - Stop-gradient operation is critical
  - Matches performance without requiring large batch sizes

#### Slide 9: Self-Supervised Pretraining
- Three-phase workflow:
  1. **Pretrain**: Learn representations from large unlabeled datasets
  2. **Fine-tune**: Adapt to small labeled datasets for specific tasks
  3. **Deploy**: Apply to downstream applications
- Key benefits:
  - Cross-domain transfer learning capability
  - Dramatically reduces labeled data requirements
  - Superior performance in low-data regimes
- Applications across modalities:
  - Medical imaging (limited labeled data)
  - Satellite imagery analysis
  - Text, audio, and video domains
- Performance curve shows self-supervised pretraining + fine-tuning outperforms supervised learning from scratch, especially with limited labels

#### Slide 10: Hands-on: Image Representation Learning
- Practical implementation session
- Students implement contrastive learning pipeline
- Dataset: CIFAR-10 or ImageNet subset
- Framework: PyTorch with standard augmentation libraries

**Key Concepts:**
1. **Pretext Tasks**: Self-generated supervision signals from data structure
2. **Contrastive Learning**: Learning by comparing positive and negative pairs
3. **Momentum Encoders**: Slowly updated networks for stable representations
4. **Transfer Learning**: Leveraging pretrained models for downstream tasks

**Learning Outcomes:**
- Implement SimCLR and understand its components
- Design effective data augmentation strategies
- Apply self-supervised models to new domains
- Evaluate representation quality using downstream tasks

---

### Part 2/4: Time Series Clustering

**Topics Covered:**

#### Slide 12: Characteristics of Time Series Data
- **Definition**: Sequential data with temporal dependencies and ordering
- Key challenges in time series clustering:
  - **Variable lengths**: Different time durations require alignment
  - **Sampling rates**: Inconsistent frequencies across sources
  - **Phase shifts**: Time warping between similar patterns
  - **Noise**: Random fluctuations and outliers obscure patterns
  - **Euclidean distance inadequacy**: Simple distance metrics fail to capture temporal alignment
- Visual examples demonstrating:
  - Trend and seasonality patterns
  - Phase shifts (time warping effects)
  - Noisy signals
  - Series of different lengths
- Application domains: Finance (stock prices), IoT sensors (device monitoring), Healthcare (patient vitals)
- Key considerations: Seasonality detection, trend analysis, outlier handling, alignment-invariant metrics

#### Slide 13: DTW (Dynamic Time Warping) Principles
- **Core Problem**: Traditional Euclidean distance fails for time-shifted series
- DTW algorithm:
  - Finds optimal alignment between two time series
  - Dynamic programming approach: O(n×m) complexity
  - Constructs cost matrix and finds minimum warping path
  - Allows elastic matching of patterns at different speeds
- Alignment strategies:
  - One-to-many mapping of time points
  - Handles speed variations and local distortions
  - Warping path constraints prevent pathological alignments
- Visual demonstration:
  - Two similar patterns with phase shift
  - DTW aligns peaks and valleys correctly
  - Warping path shows optimal correspondence
- Applications: Speech recognition, gesture recognition, financial pattern matching

#### Slide 14: DTW-based Clustering
- Integration with clustering algorithms:
  - **Hierarchical Clustering**: Use DTW as distance metric in agglomerative clustering
  - **K-means variant**: DTW Barycentric Averaging for centroid computation
  - **DBSCAN**: Density-based clustering with DTW distances
- Computational considerations:
  - DTW matrix: O(N² × n²) for N series of length n
  - Memory requirements: Can exceed RAM for large datasets
  - Optimization techniques:
    - FastDTW: Approximation with O(n) complexity
    - Lower bounding techniques (LB_Keogh)
    - Multiscale approach
- Implementation with Python libraries:
  - `tslearn`: Comprehensive time series ML toolkit
  - `dtaidistance`: Optimized DTW implementations
  - `scipy.spatial.distance`: Custom metric integration

#### Slide 15: K-Shape Algorithm
- **Innovation**: Shape-based distance for time series clustering
- Key advantages over DTW:
  - Faster computation: O(n log n) per comparison using FFT
  - Scale and shift invariant
  - Normalized cross-correlation based
- Algorithm components:
  1. **Shape-Based Distance (SBD)**: Measures shape similarity independent of amplitude
  2. **Shape Extraction**: Computes cluster centroid as shape average
  3. **Iterative refinement**: Alternates between assignment and centroid update
- Mathematical foundation:
  - Cross-correlation in frequency domain
  - Z-normalization for scale invariance
  - Optimal shift alignment
- Performance characteristics:
  - Converges faster than DTW-based methods
  - Better scalability to large datasets
  - Comparable or superior clustering quality on shape-based tasks
- Implementation: Available in `tslearn.clustering.KShape`

#### Slide 16: Subsequence Clustering
- **Objective**: Find recurring patterns in long time series
- Four-stage pipeline:
  1. **Long Time Series Input**: Continuous data stream
  2. **Sliding Window**: Extract fixed-length subsequences
  3. **Matrix Profile**: Efficient motif discovery technique
  4. **Cluster Patterns**: Group similar motifs
- Key challenges:
  - **Trivial Matches**: Preventing self-matches and near-duplicates
  - **Overlapping Patterns**: Adjacent subsequences are often too similar
- Solution methods:
  - **Matrix Profile**: Efficiently computes nearest neighbor distances for all subsequences
  - **STOMP Algorithm**: Scalable Time series Ordered-search Matrix Profile
    - O(n²) complexity instead of O(n³)
    - Anytime algorithm: Returns improving results over time
    - Exact computation (not approximate)
- Applications:
  - Anomaly detection in sensor data
  - Pattern mining in financial time series
  - Motif discovery for recurring behaviors
  - IoT sensor analysis
- Example use case: Identify repeated behaviors in IoT sensor data, such as recurring activity patterns, operational cycles, or anomalous sequences in manufacturing or healthcare monitoring systems

#### Slide 17: Hands-on: Stock/Sensor Data Clustering
- Seven-step implementation workflow:
  1. **Dataset Selection**: Yahoo Finance stock data or UCI sensor datasets
  2. **Preprocessing**: 
     - Z-score normalization for scale invariance
     - Interpolation for handling missing values and resampling
  3. **DTW Distance Implementation**: Custom Python/NumPy distance function
  4. **Hierarchical Clustering**: Apply DTW linkage using `scipy.cluster`
  5. **K-Shape Comparison**: Alternative algorithm using `tslearn` library
  6. **Visualization**: Plot clusters and centroids with `matplotlib`
  7. **Interpretation**: Identify similar behaviors and validate results
- Data sources:
  - Yahoo Finance API for stock price data
  - UCI Machine Learning Repository sensor datasets
- Key question: Which stocks/sensors behave similarly over time?
- Expected outcomes: Students can cluster financial instruments or sensor readings based on temporal patterns

**Key Concepts:**
1. **Dynamic Time Warping**: Flexible alignment metric for time series comparison
2. **Shape-Based Distance**: Scale and shift invariant similarity measure
3. **Matrix Profile**: Efficient all-pairs similarity computation for subsequence analysis
4. **Temporal Invariance**: Handling phase shifts and speed variations in pattern matching

**Learning Outcomes:**
- Select appropriate distance metrics for time series data
- Implement DTW-based clustering pipelines
- Apply K-Shape algorithm for efficient clustering
- Identify recurring patterns using Matrix Profile and subsequence clustering
- Evaluate clustering quality for temporal data

---

### Part 3/4: Graph Clustering

**Topics Covered:**

#### Slide 19: Introduction to Graph Data
- **Graph Representation**: Nodes (entities) and edges (relationships)
- Graph types:
  - **Undirected**: Symmetric relationships (social networks, collaboration networks)
  - **Directed**: Asymmetric relationships (web links, citation networks, workflow)
  - **Weighted**: Edge weights represent strength/cost (transportation networks, influence)
  - **Attributed**: Nodes/edges have features (social profiles, transaction amounts)
- Graph clustering objective: Partition nodes into communities
- Key properties to consider:
  - **Modularity**: Measure of community structure quality
  - **Density**: Edges within vs. between clusters
  - **Conductance**: Ratio of cut edges to community size
- Application domains:
  - Social networks (friend groups, influence communities)
  - Biological networks (protein interactions, gene regulatory networks)
  - Transportation networks (traffic flow optimization)
  - Recommendation systems (user-item collaborative filtering)

#### Slide 20: Spectral Clustering
- **Mathematical Foundation**: Graph Laplacian matrix analysis
- Three-step algorithm:
  1. **Construct graph Laplacian**: L = D - A
     - D: Degree matrix (diagonal)
     - A: Adjacency matrix
  2. **Eigendecomposition**: Compute k smallest eigenvectors
     - These eigenvectors embed nodes in Euclidean space
     - Spectral gap indicates number of clusters
  3. **K-means clustering**: Apply to eigenspace embeddings
- Types of Laplacian matrices:
  - **Unnormalized**: L = D - A
  - **Symmetric normalized**: L_sym = I - D^(-1/2) A D^(-1/2)
  - **Random walk normalized**: L_rw = I - D^(-1) A
- Advantages:
  - Theoretically grounded (graph cut optimization)
  - Works well for convex cluster shapes
  - Handles non-linear cluster boundaries
- Limitations:
  - Computational cost: O(n³) for eigendecomposition
  - Requires knowing number of clusters k
  - Sensitive to graph construction parameters
- Implementation: `sklearn.cluster.SpectralClustering`

#### Slide 21: Community Detection (Louvain)
- **Louvain Method**: Hierarchical modularity optimization
- Modularity metric: Q = (edges within communities - expected edges) / total edges
- Two-phase algorithm:
  1. **Local optimization**: 
     - Each node initially in own community
     - Iteratively move nodes to maximize modularity gain
     - Greedy local moves until convergence
  2. **Community aggregation**: 
     - Aggregate communities into super-nodes
     - Create new network with communities as nodes
     - Repeat until no modularity improvement
- Key advantages:
  - Fast: O(n log n) complexity
  - Scales to millions of nodes
  - Hierarchical community structure
  - No need to specify number of clusters
- Resolution limit: May miss small communities in large graphs
- Variants:
  - **Leiden algorithm**: Improved quality guarantees
  - **Louvain with multiresolution**: Adjustable resolution parameter
- Applications:
  - Social network analysis (finding friend groups)
  - Biological network modules
  - Web community detection
- Implementation: `python-louvain` or `networkx.community.louvain_communities`

#### Slide 22: GNN Fundamentals
- **Graph Neural Networks**: Deep learning on graph-structured data
- Message passing framework:
  1. **Aggregate**: Collect features from neighbors
  2. **Combine**: Merge neighbor information with node's own features
  3. **Update**: Generate new node representation
- Common GNN architectures:
  - **GCN (Graph Convolutional Network)**: 
    - H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    - Symmetric normalization
    - Spectral convolution approximation
  - **GraphSAGE**: 
    - Sampling-based aggregation
    - Inductive learning capability
    - Multiple aggregator functions (mean, max, LSTM)
  - **GAT (Graph Attention Network)**: 
    - Attention-weighted aggregation
    - Different weights for different neighbors
    - Multi-head attention mechanism
- Key concepts:
  - **Receptive field**: K-hop neighborhood after K layers
  - **Over-smoothing**: Too many layers make node features indistinguishable
  - **Heterophily**: Not all neighbors are similar
- Applications beyond clustering:
  - Node classification
  - Link prediction
  - Graph classification
  - Recommendation systems

#### Slide 23: Clustering with GCN
- **End-to-End Trainable Graph Clustering**
- Architecture pipeline:
  1. **Input Graph**: Nodes with features + adjacency matrix
  2. **GCN Encoder**: Multi-layer graph convolutions extract representations
  3. **Node Embeddings**: Low-dimensional representations in latent space
  4. **Clustering**: Assign nodes to clusters (soft or hard assignment)
- Loss function components:
  - **Reconstruction Loss**: L_recon = ||A - Â||²
    - Preserves graph structure
    - Encourages similar nodes to be close
  - **Clustering Loss**: L_cluster = KL(P||Q)
    - P: Target distribution (sharpened)
    - Q: Soft cluster assignments
    - Encourages confident predictions
  - **Combined objective**: L_total = L_recon + λ·L_cluster
- Key features:
  - Supervised or unsupervised training possible
  - Captures both local (neighbors) and global (multi-hop) structure
  - Joint optimization of encoder and clustering objectives
- Advantages over classical methods:
  - Learns task-specific representations
  - Outperforms spectral clustering on benchmarks
  - End-to-end differentiable
  - Better performance on attributed graphs
- Implementation frameworks:
  - PyTorch Geometric
  - DGL (Deep Graph Library)
  - Spektral (Keras-based)

#### Slide 24: Deep Graph Infomax (DGI)
- **Self-Supervised Graph Representation Learning**
- Contrastive learning objective: Maximize mutual information between local and global representations
- Architecture:
  - **Positive Branch**:
    1. Original graph with real structure
    2. GNN encoder extracts node features
    3. Local node embeddings (patch representations)
  - **Negative Branch**:
    1. Corrupted graph (shuffled node features)
    2. Same GNN encoder
    3. Global graph summary (readout function)
- Mutual information maximization:
  - Discriminator learns to distinguish positive pairs (local-global from real graph) from negative pairs (local-global from corrupted graph)
  - Node-level and graph-level representations learned jointly
- Key advantages:
  - **Self-supervised**: Learns from graph structure alone
  - **No labels needed**: Unsupervised training approach
  - **Transfer learning**: Pretrained representations apply to downstream tasks
- Training procedure:
  1. Sample positive and negative graph pairs
  2. Encode both through shared GNN
  3. Compute readout for graph-level representation
  4. Train discriminator with binary cross-entropy
- Applications:
  - Node classification (after pretraining)
  - Link prediction
  - Graph classification
  - Clustering (use learned embeddings)
- Performance: Competitive with supervised methods on many benchmarks

#### Slide 25: Hands-on: Social Network Analysis
- Seven-step implementation workflow:
  1. **Load Dataset**: Zachary's Karate Club or Facebook ego-networks
  2. **Load Graph Data**: Import and preprocess using NetworkX
  3. **Spectral Clustering** (Classical):
     - Eigendecomposition-based method
     - Implementation: `scikit-learn`
  4. **Louvain Method** (Classical):
     - Community detection via modularity optimization
     - Implementation: `python-louvain`
  5. **Train GCN** (Deep Learning):
     - Graph neural network approach
     - Implementation: PyTorch Geometric
  6. **Visualize Communities**: Plot detected clusters with matplotlib
  7. **Compare Results**: Analyze Spectral vs Louvain vs GCN performance
- Datasets:
  - **Zachary's Karate Club**: 34 nodes, 78 edges (classic benchmark)
  - **Facebook Ego-networks**: Larger scale social network data
- Evaluation metrics:
  - Modularity score
  - Normalized Mutual Information (if ground truth available)
  - Visual coherence of communities
- Expected insights: Students compare classical graph algorithms with modern deep learning approaches

**Key Concepts:**
1. **Spectral Methods**: Eigendecomposition-based graph partitioning
2. **Modularity Optimization**: Quality measure for community detection
3. **Graph Neural Networks**: Deep learning architectures for graph data
4. **Self-Supervised Graph Learning**: Contrastive methods for representation learning

**Learning Outcomes:**
- Construct and analyze graph representations of relational data
- Apply spectral clustering to partition graph nodes
- Implement Louvain algorithm for community detection
- Train GNN models for graph clustering tasks
- Compare classical and deep learning graph clustering methods
- Visualize and interpret graph communities

---

### Part 4/4: Advanced Topics and Applications

**Topics Covered:**

#### Slide 27: Deep Clustering (DeepCluster, SwAV)
- **DeepCluster** (2018):
  - Iterative approach:
    1. Train CNN with current cluster assignments
    2. Update cluster assignments using k-means on learned features
    3. Repeat until convergence
  - Prevents trivial solutions using:
    - Empty cluster reassignment
    - Uniform cluster size constraints
  - First to show deep learning can work without labels for clustering
  - Performance: 91.2% on ImageNet using AlexNet architecture
- **SwAV - Swapped Assignment Views** (2020):
  - Online clustering without k-means:
    - Uses Sinkhorn-Knopp algorithm for optimal transport
    - Swapped prediction: predict cluster of one view from another
  - Multi-crop strategy:
    - 2 standard crops (160×160) + 6 small crops (96×96)
    - Efficient training with asymmetric crops
  - Prototype-based learning:
    - Learns cluster prototypes during training
    - Swapped assignment prevents collapse
  - Performance: 75.3% top-1 accuracy on ImageNet (self-supervised)
- Comparison:
  - DeepCluster requires explicit k-means step (offline)
  - SwAV integrates clustering into training (online)
  - SwAV is more memory efficient and faster
  - Both enable self-supervised representation learning
- Applications:
  - Image clustering without labels
  - Pretraining for transfer learning
  - Discovering visual categories automatically

#### Slide 28: Multi-modal Clustering
- **Challenge**: Different feature spaces and scales across modalities
- Three clustering approaches:
  1. **Early Fusion**:
     - Concatenate features from different modalities
     - Apply clustering to combined feature vector
     - Pros: Simple, preserves all information
     - Cons: Different scales, high dimensionality, assumes linear relationships
  2. **Late Fusion**:
     - Cluster each modality separately
     - Combine clustering results (voting, consensus clustering)
     - Pros: Respects modality-specific structure
     - Cons: May miss cross-modal patterns
  3. **Deep Multi-modal**:
     - Learn shared representation across modalities
     - Neural network projects both modalities to common space
     - Pros: Learns optimal fusion, captures cross-modal patterns
     - Cons: Requires more data, complex training
- Visual example workflow:
  - Image + Text → Shared Embedding Space → Joint Clustering
- Modern approach: **CLIP-style contrastive learning**
  - Contrastive multi-modal learning
  - Align image and text embeddings
  - Self-supervised training on image-caption pairs
- Applications:
  - **Video Analysis**: Visual + audio modalities
  - **Medical Imaging**: Scans + clinical reports
  - **E-commerce**: Product images + descriptions
  - **Social media**: Posts + user metadata
- Implementation considerations:
  - Modality alignment
  - Handling missing modalities
  - Balancing modality contributions

#### Slide 29: Large-Scale Clustering
- **Challenges**:
  - Billions of samples (e.g., web-scale data)
  - High dimensionality (thousands of features)
  - Memory constraints (distance matrix doesn't fit in RAM)
  - Computational time (quadratic or cubic complexity)
- Scalability techniques:
  1. **Mini-batch K-means**:
     - Process streaming data in small batches
     - Update centroids incrementally
     - O(n) memory, suitable for online learning
     - Implementation: `scikit-learn.cluster.MiniBatchKMeans`
  2. **Approximate Nearest Neighbors**:
     - Fast similarity search using indexing structures
     - FAISS (Facebook AI Similarity Search) library
     - Achieves sub-linear query time
     - Tradeoff: Approximate results for speed
  3. **Hierarchical Clustering**:
     - Multi-level clustering for scalability
     - Coarsen graph at each level
     - Parallel processing possible
     - Implementation: `scipy.cluster.hierarchy`
  4. **Distributed Computing**:
     - Parallel processing across clusters
     - MapReduce paradigm
     - Apache Spark MLlib for distributed clustering
     - Scales to petabyte-scale datasets
  5. **GPU Acceleration**:
     - Fast distance computations on GPU
     - CUDA-accelerated implementations
     - cuML library (RAPIDS ecosystem)
     - 10-50x speedup for large datasets
  6. **Sampling Methods**:
     - Representative subset selection
     - CoreSets: Weighted samples preserving clustering structure
     - Reduces problem size while maintaining quality
- **Critical Trade-off**: Speed vs Accuracy
  - Approximate methods trade accuracy for scalability
  - Need to validate approximation quality
  - Choose based on application requirements
- Performance considerations:
  - Mini-batch k-means: 100x faster than standard k-means
  - FAISS: Billion-scale similarity search in milliseconds
  - GPU clustering: 50x speedup on appropriate hardware

#### Slide 30: Real-world Applications (by Industry)
- **E-commerce** 🛒:
  - Customer segmentation (behavioral clusters for marketing)
  - Product recommendations (collaborative filtering clusters)
- **Healthcare** 🏥:
  - Patient stratification (treatment response groups)
  - Disease subtyping (identifying patient phenotypes)
- **Finance** 💰:
  - Fraud detection (anomalous transaction patterns)
  - Portfolio clustering (correlated asset groups)
- **Manufacturing** 🏭:
  - Anomaly detection (equipment failure patterns)
  - Predictive maintenance (sensor data clustering)
- **Social Media** 📱:
  - Community detection (user interest groups)
  - Content recommendation (similar content clustering)
- **Genomics** 🧬:
  - Gene expression clustering (co-expressed gene modules)
  - Cell type discovery (single-cell RNA-seq clustering)
- **Transportation** 🚗:
  - Traffic pattern analysis (congestion detection)
  - Route optimization (similar journey clustering)
- Industry-specific considerations:
  - Healthcare: Privacy constraints (HIPAA compliance)
  - Finance: Real-time requirements (millisecond latency)
  - E-commerce: Scalability (millions of users)
  - Genomics: High dimensionality (20,000+ genes)
- Success metrics vary by industry:
  - E-commerce: Conversion rate improvement
  - Healthcare: Clinical outcome improvement
  - Finance: Fraud detection rate
  - Manufacturing: Downtime reduction

**Key Concepts:**
1. **Deep Clustering**: End-to-end learned clustering with neural networks
2. **Multi-modal Fusion**: Integrating heterogeneous data sources
3. **Scalability Techniques**: Methods for billion-scale clustering
4. **Industry Applications**: Domain-specific clustering challenges and solutions

**Learning Outcomes:**
- Implement deep clustering algorithms (DeepCluster, SwAV)
- Design multi-modal clustering pipelines
- Apply scalability techniques for large datasets
- Evaluate clustering solutions in real-world contexts
- Select appropriate methods based on industry requirements
- Deploy clustering models in production environments

---

## 🛠️ Prerequisites

### Required Knowledge
- **Machine Learning Fundamentals**:
  - Supervised vs. unsupervised learning
  - Clustering algorithms (K-means, hierarchical clustering)
  - Model evaluation metrics
  - Cross-validation techniques
- **Deep Learning**:
  - Neural network architectures (CNN, fully connected)
  - Backpropagation and optimization (SGD, Adam)
  - Regularization techniques (dropout, batch normalization)
  - Transfer learning concepts
- **Mathematics**:
  - Linear algebra (matrix operations, eigenvalues/eigenvectors)
  - Calculus (gradients, optimization)
  - Probability and statistics (distributions, hypothesis testing)
  - Graph theory basics (nodes, edges, adjacency matrices)
- **Programming**:
  - Python proficiency (intermediate level)
  - Experience with Jupyter notebooks
  - Familiarity with NumPy arrays and operations
  - Basic command line usage

### Software Requirements
- **Python Environment**:
  - Python 3.8 or higher
  - Jupyter Notebook or JupyterLab
  - Conda or virtualenv for environment management
- **Core Libraries**:
  - NumPy >= 1.20 (numerical computing)
  - Pandas >= 1.3 (data manipulation)
  - Matplotlib >= 3.4 (visualization)
  - Scikit-learn >= 1.0 (classical ML algorithms)
- **Deep Learning Frameworks**:
  - PyTorch >= 1.10 (primary framework)
  - TorchVision >= 0.11 (image transformations)
  - PyTorch Geometric >= 2.0 (graph neural networks)
- **Specialized Libraries**:
  - tslearn >= 0.5 (time series clustering)
  - NetworkX >= 2.6 (graph algorithms)
  - python-louvain >= 0.15 (community detection)
  - FAISS >= 1.7 (approximate nearest neighbors)
- **Optional but Recommended**:
  - TensorBoard (training visualization)
  - Weights & Biases (experiment tracking)
  - Plotly (interactive visualizations)
  - Seaborn (statistical plots)

### Hardware Recommendations
- **Minimum**: 
  - 8GB RAM
  - 2-core CPU
  - 10GB free disk space
- **Recommended**: 
  - 16GB+ RAM
  - 4-core+ CPU
  - NVIDIA GPU with 6GB+ VRAM (for deep learning sections)
  - 50GB free disk space
- **Optimal for large-scale experiments**:
  - 32GB+ RAM
  - 8-core+ CPU
  - NVIDIA GPU with 12GB+ VRAM (RTX 3080 or better)
  - SSD storage

### Installation Guide
```bash
# Create conda environment
conda create -n advanced-unsupervised python=3.9
conda activate advanced-unsupervised

# Install core packages
conda install numpy pandas matplotlib scikit-learn jupyter

# Install PyTorch (CPU version)
conda install pytorch torchvision -c pytorch

# Install PyTorch (GPU version - CUDA 11.3)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Install specialized libraries
pip install torch-geometric
pip install tslearn
pip install networkx python-louvain
pip install faiss-cpu  # or faiss-gpu for GPU version

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

---

## 💻 Hands-on Components

### Hands-on 1: Image Representation Learning (Part 1)
**Objective**: Implement self-supervised contrastive learning for visual representations

**Dataset**: CIFAR-10 or ImageNet subset (10-100 classes)

**Implementation Steps**:
1. **Data Augmentation Pipeline**:
   - Random crop and resize
   - Color jittering (brightness, contrast, saturation, hue)
   - Random horizontal flip
   - Gaussian blur
   - Normalization
2. **SimCLR Architecture**:
   - ResNet-18 or ResNet-50 backbone
   - Projection head (2-layer MLP): 2048 → 2048 → 128
   - NT-Xent loss with temperature τ=0.5
3. **Training Configuration**:
   - Batch size: 256-512 (or as large as GPU memory allows)
   - Optimizer: Adam or LARS
   - Learning rate: 0.001 with cosine decay
   - Epochs: 100-200
4. **Evaluation**:
   - Freeze encoder, train linear classifier on labels
   - Compare with supervised baseline
   - Visualize learned representations using t-SNE/UMAP

**Expected Outcomes**:
- Understand impact of augmentation choices
- Achieve 80-90% accuracy on CIFAR-10 (linear evaluation)
- Visualize clustering in learned embedding space

**Code Template**:
```python
import torch
import torch.nn as nn
from torchvision import transforms, models

class SimCLR(nn.Module):
    def __init__(self, base_encoder='resnet50', projection_dim=128):
        super().__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()  # Remove classification head
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return z

# Data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

**Estimated Time**: 2-3 hours (including training)

---

### Hands-on 2: Stock/Sensor Data Clustering (Part 2)
**Objective**: Apply time series clustering to financial or sensor data

**Dataset Options**:
- Yahoo Finance: 20-50 stock tickers (2+ years of daily prices)
- UCI Repository: Human Activity Recognition or Power Consumption datasets

**Implementation Steps**:
1. **Data Collection & Preprocessing**:
   - Download historical stock prices (Open, High, Low, Close, Volume)
   - Handle missing values (forward fill, interpolation)
   - Normalize each series (z-score or min-max)
   - Resample to consistent frequency
2. **DTW Distance Implementation**:
   - Implement basic DTW with dynamic programming
   - Use `tslearn.metrics.dtw` for optimized version
   - Compute distance matrix (NxN for N time series)
3. **Hierarchical Clustering**:
   - Linkage methods: complete, average, ward
   - Dendrogram visualization
   - Cut tree at optimal height (elbow method, silhouette score)
4. **K-Shape Clustering**:
   - Use `tslearn.clustering.KShape`
   - Compare with DTW-based results
   - Visualize cluster centroids (shape patterns)
5. **Evaluation**:
   - Silhouette score
   - Davies-Bouldin index
   - Visual inspection of cluster coherence
   - Industry sector consistency (for stocks)

**Expected Outcomes**:
- Identify groups of stocks with similar behavior (e.g., tech stocks, utilities)
- Understand when DTW outperforms Euclidean distance
- Compare K-Shape efficiency vs. DTW accuracy

**Code Template**:
```python
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from scipy.cluster.hierarchy import dendrogram, linkage
import yfinance as yf

# Download stock data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Close']

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# DTW-based hierarchical clustering
from tslearn.metrics import cdist_dtw
distance_matrix = cdist_dtw(data_normalized.T)
Z = linkage(distance_matrix, method='complete')
dendrogram(Z, labels=tickers)

# K-Shape clustering
ks = TimeSeriesKMeans(n_clusters=3, metric="softdtw", verbose=True)
labels = ks.fit_predict(data_normalized.T)
```

**Estimated Time**: 2-3 hours

---

### Hands-on 3: Social Network Analysis (Part 3)
**Objective**: Compare graph clustering methods on social network data

**Dataset Options**:
- Zachary's Karate Club (34 nodes, built-in to NetworkX)
- Facebook Ego-networks (4,039 nodes, available from SNAP)
- Twitch social networks (community detection challenge)

**Implementation Steps**:
1. **Data Loading**:
   - Load graph structure with NetworkX
   - Visualize initial network layout
   - Compute basic statistics (degree distribution, clustering coefficient)
2. **Classical Method 1 - Spectral Clustering**:
   - Compute graph Laplacian
   - Find k smallest eigenvectors
   - Apply k-means to embedding
   - Implementation: `sklearn.cluster.SpectralClustering`
3. **Classical Method 2 - Louvain**:
   - Modularity optimization
   - Hierarchical community detection
   - Implementation: `community.best_partition`
4. **Deep Learning - GCN Clustering**:
   - Build GCN with PyTorch Geometric
   - Train with reconstruction + clustering loss
   - Extract node embeddings
5. **Comparison**:
   - Modularity scores
   - Normalized Mutual Information (if ground truth available)
   - Running time comparison
   - Visualization: color nodes by cluster

**Expected Outcomes**:
- Understand when spectral methods work well (balanced clusters)
- See Louvain's speed advantage on large graphs
- Observe GCN's superior performance on attributed graphs

**Code Template**:
```python
import networkx as nx
from sklearn.cluster import SpectralClustering
import community as community_louvain
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

# Load Karate Club
G = nx.karate_club_graph()

# Spectral Clustering
sc = SpectralClustering(n_clusters=2, affinity='precomputed')
labels_spectral = sc.fit_predict(nx.adjacency_matrix(G).todense())

# Louvain
partition = community_louvain.best_partition(G)
labels_louvain = list(partition.values())

# GCN Implementation
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_clusters):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_clusters)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Train GCN and extract clusters
```

**Estimated Time**: 2-3 hours

---

### Integration Project (Optional Extended Exercise)
**Objective**: Combine multiple techniques for a comprehensive clustering pipeline

**Scenario**: Multi-modal social media analysis
- Cluster users based on:
  - Social graph structure (who follows whom)
  - Image posting patterns (visual content)
  - Temporal activity patterns (when they post)

**Skills Integrated**:
- Graph clustering (Part 3) for social connections
- Self-supervised learning (Part 1) for image embeddings
- Time series clustering (Part 2) for activity patterns
- Multi-modal fusion (Part 4) to combine representations

**Estimated Time**: 5-8 hours

---

## 📖 Additional Resources

### Research Papers

**Self-Supervised Learning**:
1. Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR). ICML.
2. He et al. (2020). "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo). CVPR.
3. Grill et al. (2020). "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" (BYOL). NeurIPS.
4. Caron et al. (2020). "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments" (SwAV). NeurIPS.

**Time Series Clustering**:
1. Sakoe & Chiba (1978). "Dynamic programming algorithm optimization for spoken word recognition". IEEE Transactions on Acoustics, Speech, and Signal Processing.
2. Paparrizos & Gravano (2015). "k-Shape: Efficient and Accurate Clustering of Time Series". SIGMOD.
3. Yeh et al. (2016). "Matrix Profile I: All Pairs Similarity Joins for Time Series". IEEE ICDM.

**Graph Clustering**:
1. Von Luxburg (2007). "A tutorial on spectral clustering". Statistics and Computing.
2. Blondel et al. (2008). "Fast unfolding of communities in large networks" (Louvain). Journal of Statistical Mechanics.
3. Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks". ICLR.
4. Veličković et al. (2019). "Deep Graph Infomax". ICLR.

**Deep Clustering**:
1. Caron et al. (2018). "Deep Clustering for Unsupervised Learning of Visual Features" (DeepCluster). ECCV.
2. Xie et al. (2016). "Unsupervised Deep Embedding for Clustering Analysis" (DEC). ICML.

### Online Courses & Tutorials
- Stanford CS224W: Machine Learning with Graphs
- DeepLearning.AI: Unsupervised Learning, Recommenders, Reinforcement Learning
- Fast.ai: Practical Deep Learning for Coders
- PyTorch Geometric Tutorials: https://pytorch-geometric.readthedocs.io/

### Books
- "Deep Learning" by Goodfellow, Bengio, and Courville (2016) - Chapters 14, 20
- "Pattern Recognition and Machine Learning" by Bishop (2006) - Chapters 9, 12
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Géron (2019)
- "Graph Representation Learning" by Hamilton (2020)

### Software Documentation
- PyTorch: https://pytorch.org/docs/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- tslearn: https://tslearn.readthedocs.io/
- NetworkX: https://networkx.org/documentation/
- Scikit-learn Clustering: https://scikit-learn.org/stable/modules/clustering.html

### Datasets
- **Image**: ImageNet, CIFAR-10/100, STL-10
- **Time Series**: UCR Time Series Archive, Yahoo Finance API
- **Graph**: SNAP (Stanford Network Analysis Project), Karate Club, Cora, CiteSeer
- **Multi-modal**: COCO (images + captions), AudioSet (audio + video)

### Community & Forums
- PyTorch Forums: https://discuss.pytorch.org/
- Reddit: r/MachineLearning, r/learnmachinelearning
- Stack Overflow: [pytorch], [graph-theory], [clustering] tags
- Papers With Code: https://paperswithcode.com/task/clustering

---

## 🔗 How to Use These Materials

### For Self-Study
1. **Sequential Learning Path** (Recommended):
   - Start with Part 1 if new to self-supervised learning
   - Progress through Parts 2-4 in order
   - Each part builds on previous concepts
   - Complete hands-on exercises before moving to next part

2. **Topic-Focused Path**:
   - Jump directly to relevant part based on your interests:
     - Computer vision → Part 1 (Self-Supervised Learning)
     - Time series analysis → Part 2 (Time Series Clustering)
     - Network analysis → Part 3 (Graph Clustering)
     - Production deployment → Part 4 (Advanced Topics)

3. **Practice Recommendations**:
   - Implement algorithms from scratch before using libraries
   - Experiment with hyperparameters
   - Apply methods to your own datasets
   - Compare results with different approaches

### For Instructors
1. **Course Integration**:
   - Use as 3-4 week module in graduate ML course
   - Combine lectures with weekly labs
   - Assign hands-on exercises as homework

2. **Lecture Delivery**:
   - Each part = 1 lecture (90-120 minutes)
   - Pause for questions after each major concept
   - Live coding demonstrations encouraged
   - Use provided visualization slides

3. **Assessment Integration**:
   - Weekly quizzes on theoretical concepts
   - Hands-on assignments due after each part
   - Final project combining multiple techniques
   - Peer review of implementations

### Presentation Instructions
1. **Opening the Slideshow**:
   - Open `lecture18_slideshow.html` in a web browser
   - Full-screen mode recommended (F11 on most browsers)
   - Works best in Chrome or Firefox

2. **Navigation**:
   - Arrow keys: Previous/Next slide
   - Space bar: Next slide
   - Home/End: First/Last slide
   - Progress bar shows lecture completion

3. **Individual Slides**:
   - Each HTML file is a standalone slide
   - Can be opened independently for focused discussion
   - Print-friendly for handouts

### Code Execution
1. **Setting Up Environment**:
   ```bash
   # Clone/download materials
   git clone [repository_url]
   cd lecture18-advanced-unsupervised
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Launch Jupyter
   jupyter notebook
   ```

2. **Running Notebooks**:
   - Open notebooks in `notebooks/` directory
   - Execute cells sequentially
   - Modify hyperparameters to experiment
   - Save your results for comparison

3. **GPU Acceleration**:
   ```python
   # Check GPU availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   
   # Use GPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```

---

## 📊 Assessment Suggestions

### Theoretical Understanding (30%)
**Quiz Questions** (Multiple choice + Short answer):
1. Explain the difference between self-supervised and unsupervised learning
2. Why does Euclidean distance fail for time series with phase shifts?
3. What is the modularity metric in graph clustering?
4. Compare early vs. late fusion in multi-modal clustering
5. Describe the message passing mechanism in GNNs

**Grading Rubric**:
- Conceptual accuracy: 50%
- Clarity of explanation: 30%
- Use of examples: 20%

### Implementation Skills (40%)
**Hands-on Assignments**:

**Assignment 1** (Part 1): SimCLR Implementation
- Implement data augmentation pipeline (10 points)
- Build SimCLR architecture (15 points)
- Train model and report results (15 points)
- Bonus: Experiment with different augmentations (5 points)

**Assignment 2** (Part 2): Time Series Clustering
- Preprocess financial/sensor data (10 points)
- Implement DTW distance (15 points)
- Apply hierarchical and K-Shape clustering (10 points)
- Compare and analyze results (10 points)

**Assignment 3** (Part 3): Graph Clustering
- Load and visualize network (5 points)
- Implement spectral clustering (10 points)
- Apply Louvain algorithm (10 points)
- Train GCN model (15 points)
- Compare all three methods (5 points)

**Grading Rubric**:
- Code correctness: 40%
- Code efficiency and style: 20%
- Results quality: 30%
- Documentation: 10%

### Final Project (30%)
**Project Requirements**:
- Choose a real-world dataset
- Apply at least 2 clustering methods from different parts
- Compare methods quantitatively and qualitatively
- Write 5-8 page report or create video presentation

**Project Ideas**:
1. Customer segmentation for e-commerce (multi-modal: purchase history + product views)
2. Disease subtyping from medical records (time series: patient vitals over time)
3. Community detection in citation networks (graph: paper citations + content)
4. Traffic pattern analysis (time series: sensor data + spatial graph)

**Grading Rubric**:
- Problem formulation: 15%
- Technical implementation: 35%
- Experimental evaluation: 25%
- Presentation/report quality: 15%
- Creativity and insights: 10%

### Evaluation Timeline
- **Week 1**: Part 1 Quiz + Assignment 1 assigned
- **Week 2**: Assignment 1 due, Part 2 Quiz + Assignment 2 assigned
- **Week 3**: Assignment 2 due, Part 3 Quiz + Assignment 3 assigned
- **Week 4**: Assignment 3 due, Final project assigned
- **Week 5-6**: Work on final project
- **Week 7**: Final project due + presentations

---

## 📝 Notes for Implementation

### Computational Requirements by Section

**Part 1: Self-Supervised Learning**
- Training time: 2-8 hours on single GPU (depends on dataset size)
- Memory: 6-12GB GPU VRAM for batch size 256-512
- Storage: 10-50GB (ImageNet subset requires more)
- Recommendation: Start with CIFAR-10 (163MB) before scaling to ImageNet

**Part 2: Time Series Clustering**
- Training time: Minutes to hours (DTW scales quadratically with N)
- Memory: O(N²) for distance matrix (~4GB for 1000 series)
- Optimization: Use FastDTW or lower bounding for large N
- Recommendation: Subsample large datasets (e.g., 100-500 time series)

**Part 3: Graph Clustering**
- Training time: Seconds (classical) to hours (GNN training)
- Memory: Depends on graph size (adjacency matrix or edge list)
- Karate Club: < 1MB, trivial computation
- Large graphs (>100K nodes): Require sampling or distributed computing
- Recommendation: Use Louvain for graphs >10K nodes

**Part 4: Advanced Topics**
- DeepCluster/SwAV: Similar to Part 1 (hours on GPU)
- Multi-modal: 1.5-2x time of single modality
- Large-scale: Requires distributed setup or approximations

### Common Issues & Solutions

**Issue 1**: Out of Memory (OOM) errors during training
- **Solution**: Reduce batch size, use gradient accumulation, or mixed-precision training
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for data in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Issue 2**: DTW distance matrix too large for memory
- **Solution**: Use chunked computation or approximate methods
```python
from tslearn.metrics import dtw
# Compute distance matrix in chunks
chunk_size = 100
for i in range(0, N, chunk_size):
    for j in range(0, N, chunk_size):
        dist_chunk = cdist_dtw(X[i:i+chunk_size], X[j:j+chunk_size])
        # Process or save chunk
```

**Issue 3**: GNN training doesn't converge
- **Solution**: Check learning rate, add layer normalization, use residual connections
```python
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)  # Add normalization
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

**Issue 4**: Slow data loading bottleneck
- **Solution**: Use DataLoader with multiple workers and pin memory
```python
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset, 
    batch_size=256,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### Performance Optimization Tips

1. **Mixed Precision Training** (1.5-3x speedup):
```python
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
    loss = criterion(output, target)
```

2. **Data Preprocessing**:
   - Precompute augmentations for static datasets
   - Cache processed data to disk
   - Use efficient data formats (HDF5, Parquet)

3. **Model Optimization**:
   - Use torch.compile() in PyTorch 2.0+
   - Profile with PyTorch Profiler to find bottlenecks
   - Batch operations when possible

4. **Distributed Training**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)
```

### Reproducibility Checklist
- [ ] Set random seeds (Python, NumPy, PyTorch)
- [ ] Document library versions in requirements.txt
- [ ] Save model checkpoints with optimizer state
- [ ] Log hyperparameters and metrics
- [ ] Version control code and configurations
- [ ] Document hardware specifications

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 🙏 Credits

**Instructor**: Ho-min Park, PhD  
**Affiliation**: Ghent University  
**Contact**: 
- Email: homin.park@ghent.ac.kr
- Email: powersimmani@gmail.com

**Course Information**:
- Course: Lecture 18 - Advanced Unsupervised Learning
- Level: Graduate/Advanced Undergraduate
- Department: Computer Science / Data Science

**Acknowledgments**:
This lecture material synthesizes concepts from foundational papers in self-supervised learning, time series analysis, graph neural networks, and deep clustering. Special thanks to the research communities in computer vision, time series mining, and graph representation learning for their pioneering work.

**License**: Educational use permitted with attribution

**Last Updated**: 2024

---

## 📌 Quick Reference Card

### Key Algorithms Summary

| Method | Type | Use Case | Complexity | Library |
|--------|------|----------|------------|---------|
| SimCLR | Self-supervised | Image clustering | O(n) per epoch | PyTorch |
| DTW | Distance | Time series alignment | O(n²) | tslearn |
| K-Shape | Clustering | Time series shapes | O(n log n) | tslearn |
| Spectral | Graph | Balanced cuts | O(n³) | sklearn |
| Louvain | Graph | Community detection | O(n log n) | python-louvain |
| GCN | Deep learning | Attributed graphs | O(E) per epoch | PyG |
| DeepCluster | Deep learning | End-to-end clustering | O(n) per epoch | PyTorch |

### Command Cheatsheet

```bash
# Environment setup
conda create -n unsupervised python=3.9
conda activate unsupervised
pip install -r requirements.txt

# Run specific hands-on
jupyter notebook notebooks/hands_on_1_simclr.ipynb

# View slides
firefox lecture18_slideshow.html

# GPU check
python -c "import torch; print(torch.cuda.is_available())"

# Run training
python scripts/train_simclr.py --config configs/simclr_cifar10.yaml
```

### Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| OOM Error | Reduce batch size or use gradient accumulation |
| Slow training | Enable mixed precision (AMP) |
| Poor clustering | Try different distance metrics or increase clusters |
| GNN not converging | Lower learning rate, add normalization |
| Import errors | Check requirements.txt versions |

---

**Document Version**: 1.0  
**Total Slides**: 32  
**Estimated Study Time**: 12-16 hours (including hands-on practice)