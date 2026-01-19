# Lecture 08 - Part 2: Optimization Algorithms

## Overview
This part covers gradient-based optimization algorithms used to train neural networks, from basic gradient descent to modern adaptive methods.

## Key Topics

### 1. Gradient Descent Review
- **Basic Principle**:
  ```
  θ = θ - η·∇L(θ)
  ```
  - θ: Parameters
  - η: Learning rate
  - ∇L: Gradient of loss
- **Goal**: Find parameters that minimize loss function
- **Challenges**:
  - Learning rate selection
  - Local minima and saddle points
  - Computational cost

### 2. Batch vs Mini-Batch vs Stochastic
- **Batch Gradient Descent**:
  - Uses entire dataset per update
  - Accurate gradient estimate
  - Slow, memory intensive
- **Stochastic Gradient Descent (SGD)**:
  - One sample per update
  - Noisy but fast
  - Can escape local minima
- **Mini-Batch Gradient Descent**:
  - Batch size typically 32-256
  - Balance of accuracy and speed
  - Most commonly used in practice
- **Trade-offs**:
  | Method | Gradient Quality | Speed | Memory |
  |--------|-----------------|-------|--------|
  | Batch | Accurate | Slow | High |
  | Mini-Batch | Good | Fast | Medium |
  | Stochastic | Noisy | Fastest | Low |

### 3. Momentum Method
- **Formula**:
  ```
  v = β·v - η·∇L(θ)
  θ = θ + v
  ```
  - β: Momentum coefficient (typically 0.9)
  - v: Velocity (accumulated gradient)
- **Benefits**:
  - Accelerates convergence
  - Dampens oscillations
  - Escapes shallow local minima
- **Analogy**: Ball rolling downhill with inertia

### 4. Nesterov Accelerated Gradient (NAG)
- **Formula**:
  ```
  v = β·v - η·∇L(θ + β·v)
  θ = θ + v
  ```
- **Key Difference**: Look-ahead gradient computation
- **Benefits**:
  - Faster convergence than standard momentum
  - Better at slowing down near minimum
  - More responsive to gradient changes
- **Intuition**: "Look before you leap"

### 5. AdaGrad
- **Formula**:
  ```
  G = G + (∇L)²
  θ = θ - η·∇L / (√G + ε)
  ```
  - G: Accumulated squared gradients
  - ε: Small constant for numerical stability
- **Properties**:
  - Adapts learning rate per parameter
  - Large gradients → smaller updates
  - Good for sparse features
- **Limitation**: Learning rate can become too small

### 6. RMSprop
- **Formula**:
  ```
  G = β·G + (1-β)·(∇L)²
  θ = θ - η·∇L / (√G + ε)
  ```
  - β: Decay rate (typically 0.9)
- **Improvement over AdaGrad**:
  - Exponential moving average of squared gradients
  - Prevents learning rate from vanishing
  - Maintains effective learning throughout training
- **Applications**: Works well for RNNs and non-stationary problems

### 7. Adam and AdamW
- **Adam (Adaptive Moment Estimation)**:
  ```
  m = β₁·m + (1-β₁)·∇L     (first moment)
  v = β₂·v + (1-β₂)·(∇L)²   (second moment)
  m̂ = m / (1-β₁ᵗ)          (bias correction)
  v̂ = v / (1-β₂ᵗ)
  θ = θ - η·m̂ / (√v̂ + ε)
  ```
  - Default: β₁=0.9, β₂=0.999, ε=1e-8
- **Benefits**:
  - Combines momentum and adaptive learning rates
  - Works well with default hyperparameters
  - Efficient and widely applicable
- **AdamW**:
  - Decoupled weight decay regularization
  - Better generalization than Adam
  - Formula: θ = θ - η·(m̂/(√v̂+ε) + λ·θ)

### 8. Second-Order Optimization - L-BFGS
- **Concept**: Uses Hessian (curvature) information
- **Newton's Method**:
  ```
  θ = θ - H⁻¹·∇L
  ```
  - H: Hessian matrix
- **L-BFGS**: Limited-memory approximation
  - Approximates inverse Hessian
  - Memory efficient
  - Faster convergence in some cases
- **When to Use**:
  - Small to medium models
  - Full batch training
  - Convex or near-convex problems

### 9. Comparison of Optimization Algorithms
| Algorithm | Adaptive LR | Momentum | Memory | Best For |
|-----------|------------|----------|--------|----------|
| SGD | No | No | Low | Simple problems |
| SGD+Momentum | No | Yes | Low | General training |
| AdaGrad | Yes | No | Medium | Sparse data |
| RMSprop | Yes | No | Medium | RNNs |
| Adam | Yes | Yes | Medium | Default choice |
| AdamW | Yes | Yes | Medium | Regularization |
| L-BFGS | N/A | N/A | Higher | Small models |

## Important Takeaways
1. Mini-batch SGD is the foundation of deep learning optimization
2. Momentum accelerates training and helps escape local minima
3. NAG improves on momentum with look-ahead gradient
4. Adaptive methods (AdaGrad, RMSprop, Adam) adjust learning rates per parameter
5. Adam is the most popular choice due to robust performance
6. AdamW provides better regularization than standard Adam
7. Second-order methods can be faster but have higher memory requirements
8. No single optimizer is best for all problems - experimentation is key

