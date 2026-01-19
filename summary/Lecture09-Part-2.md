# Lecture 09 - Part 2: Normalization Techniques

## Overview
This part covers various normalization techniques used in deep learning to stabilize training and improve model performance, from batch normalization to specialized variants for different architectures.

## Key Topics

### 1. Internal Covariate Shift
- **Definition**: Change in distribution of layer inputs during training
- **Problem**:
  - Each layer must adapt to changing input distribution
  - Slows down training
  - Requires careful learning rate selection
- **Cause**: Updates to previous layers change current layer's input distribution
- **Solution**: Normalization techniques

### 2. Batch Normalization (BatchNorm)
- **Formula**:
  ```
  x̂ = (x - μ_B) / √(σ²_B + ε)
  y = γx̂ + β
  ```
  - μ_B: Batch mean
  - σ²_B: Batch variance
  - γ, β: Learnable parameters (scale and shift)
  - ε: Small constant for stability
- **Training vs Inference**:
  - Training: Use batch statistics
  - Inference: Use running/moving average
- **Benefits**:
  - Enables higher learning rates
  - Acts as regularizer
  - Reduces sensitivity to initialization
  - Faster convergence
- **Placement**: After linear/conv layer, before activation (or after, debated)
- **Limitations**:
  - Depends on batch size
  - Problematic for small batches
  - Issues with RNNs

### 3. Layer Normalization (LayerNorm)
- **Formula**:
  ```
  x̂ = (x - μ_L) / √(σ²_L + ε)
  y = γx̂ + β
  ```
  - Normalize across features (not batch)
- **Key Difference**: Normalizes across channels/features for each sample
- **Benefits**:
  - Independent of batch size
  - Works with batch size 1
  - Suitable for RNNs and Transformers
- **Use Cases**: NLP, sequence models, Transformers
- **Standard in**: BERT, GPT, all modern Transformers

### 4. Instance Normalization
- **Formula**: Normalize each channel of each sample independently
  ```
  x̂ = (x - μ_i) / √(σ²_i + ε)
  ```
- **Scope**: Per sample, per channel
- **Use Cases**:
  - Style transfer
  - Image generation
  - GANs
- **Property**: Removes instance-specific contrast information
- **Not Suitable For**: Classification (removes discriminative info)

### 5. Group Normalization
- **Concept**: Divide channels into groups, normalize within groups
- **Formula**:
  ```
  x̂ = (x - μ_g) / √(σ²_g + ε)
  ```
- **Hyperparameter**: Number of groups (typically 32 channels per group)
- **Properties**:
  - Independent of batch size
  - Compromise between LayerNorm and InstanceNorm
  - Works for very small batches
- **Benefits**: Good for detection/segmentation with small batch sizes
- **Use Case**: Object detection, video models

### 6. Weight Normalization
- **Formula**:
  ```
  w = g × (v / ||v||)
  ```
  - g: Scalar magnitude (learnable)
  - v: Direction vector (learnable)
  - ||v||: L2 norm of v
- **Key Idea**: Decouple magnitude and direction of weights
- **Benefits**:
  - Faster convergence than unnormalized
  - Works with any batch size
  - Simpler than BatchNorm
- **Comparison**: No running statistics needed
- **Use Case**: When BatchNorm is impractical

### 7. Spectral Normalization
- **Formula**:
  ```
  W_SN = W / σ(W)
  ```
  - σ(W): Largest singular value of W
- **Purpose**: Ensures Lipschitz continuity
- **Computation**: Power iteration method (efficient)
- **Benefits**:
  - Stabilizes GAN training
  - Controls discriminator capacity
  - Prevents mode collapse
- **Critical For**: Generative Adversarial Networks (GANs)

### 8. Comparison of Normalization Techniques
| Method | Normalizes Over | Batch Dependent | Best For |
|--------|-----------------|-----------------|----------|
| BatchNorm | Batch | Yes | CNNs (large batch) |
| LayerNorm | Features | No | Transformers, RNNs |
| InstanceNorm | Single instance | No | Style transfer |
| GroupNorm | Channel groups | No | Detection (small batch) |
| WeightNorm | Weights | No | General |
| SpectralNorm | Weights | No | GANs |

### 9. When to Use Which Normalization
- **CNNs with Large Batches**: BatchNorm (default)
- **CNNs with Small Batches**: GroupNorm
- **Transformers/NLP**: LayerNorm (essential)
- **RNNs/LSTMs**: LayerNorm
- **Style Transfer**: InstanceNorm
- **GANs (Discriminator)**: SpectralNorm
- **General Alternative**: WeightNorm
- **When in Doubt**: Start with BatchNorm or LayerNorm

## Important Takeaways
1. Normalization addresses internal covariate shift and stabilizes training
2. BatchNorm is the standard for CNNs but requires reasonable batch sizes
3. LayerNorm is essential for Transformers and works with any batch size
4. InstanceNorm is specialized for style transfer applications
5. GroupNorm provides BatchNorm benefits without batch dependency
6. Spectral Normalization is critical for stable GAN training
7. Choice of normalization depends on architecture and batch size
8. Normalization often acts as implicit regularization

