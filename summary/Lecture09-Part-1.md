# Lecture 09 - Part 1: Initialization Strategies

## Overview
This part covers weight initialization strategies for neural networks, explaining why proper initialization is crucial and presenting various methods from simple random initialization to modern techniques.

## Key Topics

### 1. Why is Initialization Important?
- **Critical for Training**: Initialization affects:
  - Convergence speed
  - Final model quality
  - Training stability
- **Poor Initialization Leads To**:
  - Vanishing/exploding gradients
  - Slow or failed training
  - Suboptimal solutions
- **Goal**: Start weights in a region where gradients flow well

### 2. Problems with Zero Initialization
- **Symmetry Problem**:
  - All neurons learn the same thing
  - Identical gradients for all weights
  - Network cannot learn diverse features
- **Result**: Effectively reduces network to single neuron per layer
- **Solution**: Break symmetry with random initialization

### 3. Random Initialization and Breaking Symmetry
- **Simple Random**:
  - Initialize from small random values
  - Typically uniform or normal distribution
- **Breaking Symmetry**:
  - Each neuron starts differently
  - Learns different features
  - Enables network to represent complex functions
- **Challenge**: Scale of random values matters critically

### 4. Gradient Vanishing/Exploding
- **Vanishing Gradients**:
  - Gradients become exponentially small
  - Early layers stop learning
  - Caused by: small weights, saturating activations
- **Exploding Gradients**:
  - Gradients become exponentially large
  - Unstable training, NaN values
  - Caused by: large weights
- **Mathematical View**:
  ```
  ∂L/∂W₁ = ∂L/∂Wₙ × Wₙ × Wₙ₋₁ × ... × W₂
  ```
  - Product of many terms < 1 → vanishing
  - Product of many terms > 1 → exploding

### 5. Xavier/Glorot Initialization
- **Formula**:
  ```
  W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]
  ```
  or
  ```
  W ~ N(0, 2/(n_in + n_out))
  ```
  - n_in: Number of input units
  - n_out: Number of output units
- **Derivation**: Maintains variance of activations and gradients
- **Best For**: Linear activations, tanh, sigmoid
- **Goal**: Var(output) ≈ Var(input)

### 6. He Initialization (for ReLU)
- **Formula**:
  ```
  W ~ N(0, 2/n_in)
  ```
  or
  ```
  W ~ U[-√(6/n_in), √(6/n_in)]
  ```
- **Why Different from Xavier?**:
  - ReLU zeros out ~half the values
  - Needs larger initial weights
  - Factor of 2 compensates for ReLU
- **Best For**: ReLU and variants (Leaky ReLU, PReLU)
- **Default Choice**: For modern CNN architectures

### 7. LSUV Initialization
- **Layer-Sequential Unit-Variance**:
  1. Initialize with orthogonal matrices
  2. For each layer, forward mini-batch
  3. Scale weights to achieve unit variance
  4. Repeat for next layer
- **Properties**:
  - Data-driven initialization
  - Ensures unit variance at each layer
  - Works regardless of activation function
- **Benefits**:
  - Better than fixed formulas for some networks
  - Adapts to actual data distribution
- **Limitation**: Requires forward pass, more complex

### 8. Leveraging Pre-trained Weights
- **Transfer Learning**:
  - Use weights from model trained on related task
  - Fine-tune for new task
- **Benefits**:
  - Often better than random initialization
  - Faster convergence
  - Better generalization
- **Common Sources**:
  - ImageNet pre-trained CNNs
  - Language model weights (BERT, GPT)
- **When to Use**:
  - Limited training data
  - Related source domain
  - Standard architectures available

### 9. Comparison of Initialization Strategies
| Method | Best For | Variance | Notes |
|--------|----------|----------|-------|
| Zero | Never | 0 | Breaks network |
| Small Random | Simple cases | Fixed | May not scale well |
| Xavier/Glorot | Tanh, Sigmoid | 2/(n_in+n_out) | Standard for linear |
| He/Kaiming | ReLU family | 2/n_in | CNN default |
| LSUV | Any | Data-driven | Adaptive |
| Pre-trained | Transfer | N/A | Often best |

## Important Takeaways
1. Never initialize weights to zero - it breaks symmetry
2. Random initialization must be carefully scaled
3. Xavier initialization is optimal for tanh/sigmoid activations
4. He initialization is essential for ReLU networks
5. LSUV provides data-adaptive initialization
6. Pre-trained weights often provide the best starting point
7. Proper initialization enables training of very deep networks
8. Most frameworks default to appropriate initialization per layer type

