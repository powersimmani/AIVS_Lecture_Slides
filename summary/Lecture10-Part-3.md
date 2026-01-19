# Lecture 10 - Part 3: Advanced Architecture Patterns

## Overview
This part covers advanced neural network architecture patterns including skip connections, dense connections, bottleneck designs, and modern efficient architectures.

## Key Topics

### 1. Skip Connections (ResNet)
- **Core Formula**:
  ```
  y = F(x) + x
  ```
  - F(x): Residual function (conv layers)
  - x: Identity shortcut
- **Gradient Flow**:
  ```
  ∂y/∂x = ∂F(x)/∂x + 1
  ```
  - Gradient always has component of 1
  - "Gradient highway" prevents vanishing
- **Key Insight**: Learn residual F(x) = H(x) - x, easier than H(x) directly
- **ResNet Variants**:
  | Model | Layers | Parameters | Block Type |
  |-------|--------|------------|------------|
  | ResNet-18 | 18 | 11.7M | Basic |
  | ResNet-34 | 34 | 21.8M | Basic |
  | ResNet-50 | 50 | 25.6M | Bottleneck |
  | ResNet-101 | 101 | 44.5M | Bottleneck |
  | ResNet-152 | 152 | 60.2M | Bottleneck |

### 2. Dense Connections (DenseNet)
- **Formula**:
  ```
  xₗ = Hₗ([x₀, x₁, ..., xₗ₋₁])
  ```
  - Each layer receives all previous feature maps
  - Concatenation (not addition)
- **Growth Rate (k)**:
  - Number of channels added per layer
  - Typical k = 12, 24, 32
- **Transition Layers**:
  - 1×1 conv + 2×2 avg pool
  - Reduces feature map size
  - Compression factor θ (typically 0.5)
- **Benefits**:
  - Maximum feature reuse
  - Strong gradient flow
  - Parameter efficient
  - Fewer parameters than ResNet for same accuracy

### 3. Bottleneck Architecture
- **Pattern**: 1×1 reduce → 3×3 process → 1×1 expand
- **ResNet Bottleneck**:
  ```
  Input (256ch) → 1×1 Conv (64ch) → 3×3 Conv (64ch) → 1×1 Conv (256ch)
  ```
- **Benefits**:
  - Reduces computation in 3×3 conv
  - More non-linearities
  - Deeper with same parameters
- **Comparison**:
  - Basic block: 3×3 → 3×3 (two 3×3 convs)
  - Bottleneck: 1×1 → 3×3 → 1×1 (compression)
- **Used In**: ResNet-50+, Inception, EfficientNet

### 4. The Role of 1×1 Convolution
- **Functions**:
  1. **Channel Reduction**: Reduce dimensions before expensive operations
  2. **Channel Expansion**: Increase feature channels
  3. **Channel Mixing**: Learn cross-channel correlations
  4. **Adding Non-linearity**: With activation function
- **Network-in-Network Concept**: 1×1 conv as mini neural network
- **Computational Savings**:
  - Without: 256 × 256 × 3 × 3 = 589K ops
  - With 1×1: 256 × 64 × 1 × 1 + 64 × 64 × 3 × 3 + ... = much less
- **Ubiquitous**: Used in almost all modern architectures

### 5. Inception Module
- **Concept**: Multi-scale parallel processing
- **Structure**:
  ```
  Input → [1×1 conv]
        → [1×1 conv → 3×3 conv]  → Concatenate → Output
        → [1×1 conv → 5×5 conv]
        → [3×3 pool → 1×1 conv]
  ```
- **Benefits**:
  - Captures features at multiple scales
  - Efficient through 1×1 reductions
  - Automatic scale selection
- **Evolution**:
  - Inception v1 (GoogLeNet)
  - Inception v2/v3: Factorized convolutions
  - Inception v4: Combined with ResNet
  - Inception-ResNet: Hybrid architecture

### 6. Depthwise Separable Convolution
- **Standard Convolution**: C_in × C_out × K × K parameters
- **Depthwise Separable**:
  1. Depthwise: One filter per input channel
  2. Pointwise: 1×1 conv to combine
- **Parameters**:
  - Standard: C_in × C_out × K²
  - Separable: C_in × K² + C_in × C_out
- **Computational Savings**: 8-9× fewer operations
- **Used In**: MobileNet, EfficientNet, Xception
- **Trade-off**: Slight accuracy drop for major efficiency gain

### 7. Neural Architecture Search (NAS)
- **Concept**: Automatically design neural network architectures
- **Search Space**:
  - Number of layers
  - Layer types (conv, pool, etc.)
  - Connections
  - Hyperparameters
- **Search Strategies**:
  - Reinforcement Learning
  - Evolutionary algorithms
  - Gradient-based (DARTS)
  - One-shot methods
- **Notable Results**:
  - NASNet: Outperformed hand-designed
  - EfficientNet: State-of-the-art efficiency
  - AmoebaNet: Evolutionary approach
- **Cost**: Originally very expensive (1000s GPU hours)
- **Modern**: More efficient methods available

### 8. Model Compression Techniques
- **Pruning**:
  - Remove unimportant weights/neurons
  - Structured vs unstructured
  - Can achieve 10× compression
- **Quantization**:
  - Reduce precision (FP32 → INT8)
  - 4× smaller, 2-4× faster
  - Minimal accuracy loss
- **Knowledge Distillation**:
  - Train small model to mimic large model
  - Transfer dark knowledge
  - Compress while preserving accuracy
- **Low-rank Factorization**:
  - Decompose weight matrices
  - Reduce parameters
- **Weight Sharing**:
  - Multiple connections share same weight
  - HashNets, deep compression

### 9. Practical Design Guidelines
- **Architecture Selection**:
  | Task | Recommended |
  |------|-------------|
  | Image Classification | ResNet, EfficientNet |
  | Object Detection | ResNet + FPN |
  | Mobile/Edge | MobileNet, EfficientNet-Lite |
  | NLP | Transformer |
- **Key Components**:
  - Skip connections (essential for depth)
  - Batch/Layer normalization
  - Proper activation (ReLU/GELU)
  - Global average pooling (vs FC)
- **Regularization**:
  - Dropout (FC layers)
  - DropPath (residual paths)
  - Data augmentation
- **Rule of Thumb**: Start with proven architecture, then customize

## Important Takeaways
1. Skip connections are essential for training very deep networks
2. Dense connections maximize feature reuse and gradient flow
3. Bottleneck design reduces computation while maintaining capacity
4. 1×1 convolutions are crucial for efficient channel manipulation
5. Inception modules capture multi-scale information efficiently
6. Depthwise separable convolutions provide 8-9× efficiency gains
7. NAS can discover architectures better than human-designed ones
8. Model compression enables deployment on resource-constrained devices
9. Modern architectures combine multiple patterns (ResNet + Inception + Separable)

