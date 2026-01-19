# Lecture 07 - Part 3: Learning-based Representations

## Overview
This part covers how deep learning automatically learns feature representations, transfer learning strategies, and multimodal fusion approaches.

## Key Topics

### 1. Concept of Representation Learning
- **Definition**: Automatically learning useful features from data
- **Traditional vs Deep Learning**:
  - Traditional: Manual feature engineering
  - Deep Learning: End-to-end feature learning
- **Hierarchy of Features**:
  - Low-level: Edges, colors, textures
  - Mid-level: Parts, patterns
  - High-level: Objects, concepts
- **Benefits**: Task-adaptive, often superior to hand-crafted features

### 2. Word Embeddings (Word2Vec, GloVe)
- **Word2Vec**:
  - Words as dense vectors (50-300 dimensions)
  - Two architectures: CBOW, Skip-gram
  - Captures semantic relationships
  - king - man + woman ≈ queen
- **GloVe (Global Vectors)**:
  - Matrix factorization approach
  - Uses co-occurrence statistics
  - Combines global and local context
- **Properties**:
  - Similar words have similar vectors
  - Vector arithmetic captures analogies
  - Pre-trained embeddings available
- **Modern**: Contextual embeddings (BERT, GPT)

### 3. CNN-based Image Features
- **Learned Hierarchy**:
  - Conv1: Edges, corners
  - Conv2: Textures, patterns
  - Conv3: Parts
  - Conv4-5: Objects, scenes
- **Feature Extraction**:
  - Use pre-trained CNN (VGG, ResNet, etc.)
  - Extract features from intermediate layers
  - Pool or flatten for fixed-size vector
- **Common Approach**:
  ```python
  model = ResNet50(weights='imagenet', include_top=False)
  features = model.predict(images)
  ```
- **Transfer Learning**: Features transfer across tasks

### 4. RNN-based Sequence Features
- **Recurrent Neural Networks**:
  - Process sequences step by step
  - Hidden state carries context
  - Variable length input handling
- **LSTM/GRU**: Handle long-range dependencies
- **Sequence Representation**:
  - Last hidden state
  - Mean/max pooling of all states
  - Attention-weighted combination
- **Applications**: Text, audio, time series
- **Modern**: Transformers largely replace RNNs

### 5. Autoencoders and Latent Representations
- **Autoencoder Architecture**:
  - Encoder: Input → Compressed latent space
  - Decoder: Latent space → Reconstructed input
  - Trained to minimize reconstruction error
- **Latent Space**: Learned compressed representation
- **Variants**:
  - **VAE (Variational)**: Probabilistic, smooth latent space
  - **Denoising**: Robust to noise
  - **Sparse**: Encourages sparsity
- **Applications**: Dimensionality reduction, anomaly detection, generation

### 6. Transfer Learning Strategies
- **Concept**: Leverage knowledge from source to target task
- **When to Use**:
  - Limited target data
  - Related source task available
  - Source domain has more data
- **Approaches**:
  - Feature extraction: Use frozen pre-trained features
  - Fine-tuning: Update pre-trained weights
  - Progressive: Gradually unfreeze layers
- **Benefits**: Faster training, better performance, less data needed

### 7. Fine-tuning vs Feature Extraction
- **Feature Extraction**:
  - Freeze pre-trained model
  - Only train new classifier head
  - Fast, works with limited data
  - Less risk of overfitting
- **Fine-tuning**:
  - Update some/all pre-trained weights
  - Better adaptation to target task
  - Requires more data
  - Risk of forgetting pre-trained knowledge
- **Strategy Selection**:
  | Target Data | Similarity | Approach |
  |-------------|------------|----------|
  | Small       | Similar    | Feature extraction |
  | Small       | Different  | Fine-tune last layers |
  | Large       | Similar    | Fine-tune more layers |
  | Large       | Different  | Fine-tune all or retrain |

### 8. Domain Adaptation
- **Problem**: Source and target have different distributions
- **Types**:
  - Covariate shift: Input distribution differs
  - Label shift: Output distribution differs
  - Concept drift: P(y|x) changes
- **Techniques**:
  - Domain adversarial training (DANN)
  - Feature alignment
  - Self-training on target domain
- **Unsupervised DA**: No labels in target domain

### 9. Multimodal Fusion
- **Fusion Strategies**:
  - **Early Fusion**: Combine raw features before model
  - **Late Fusion**: Combine predictions from separate models
  - **Hybrid Fusion**: Intermediate combination
- **Attention-based Fusion**: Learn which modality to focus on
- **Cross-modal Learning**: Learn relationships between modalities
- **Challenges**:
  - Modality alignment
  - Missing modalities
  - Different scales/representations
- **Applications**: VQA, video understanding, multimodal sentiment

## Important Takeaways
1. Deep learning automatically learns hierarchical features
2. Word embeddings capture semantic relationships in dense vectors
3. CNN features transfer well across vision tasks
4. Autoencoders learn useful compressed representations
5. Transfer learning enables training with limited data
6. Choose fine-tuning vs feature extraction based on data size and similarity
7. Multimodal fusion combines complementary information from different modalities
