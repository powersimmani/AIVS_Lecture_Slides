# Lecture 10 - Part 2: Modern Activation Functions

## Overview
This part covers the evolution of activation functions from sigmoid to modern variants, explaining how better activations enabled training of deep neural networks.

## Key Topics

### 1. The ReLU Revolution
- **Definition**:
  ```
  ReLU(x) = max(0, x)
  ```
- **Historical Impact**: AlexNet (2012) popularized ReLU
- **Advantages**:
  - Simple computation
  - No vanishing gradient for x > 0
  - Sparse activation (biological plausibility)
  - Faster convergence (6× faster than tanh)
- **Limitations**:
  - Dead ReLU problem (neurons stop learning)
  - Not zero-centered
  - Unbounded activation (can explode)
- **Default Choice**: Still the most widely used activation

### 2. Leaky ReLU and PReLU
- **Leaky ReLU**:
  ```
  f(x) = x if x > 0, else α×x
  ```
  - α typically 0.01 (fixed)
  - Allows small gradient for negative inputs
  - Prevents dead neurons
- **PReLU (Parametric ReLU)**:
  ```
  f(x) = x if x > 0, else α×x
  ```
  - α is learnable per channel
  - Can adapt to data
  - Slightly more parameters
- **Comparison**:
  | Variant | Negative Slope | Learnable |
  |---------|---------------|-----------|
  | ReLU | 0 | No |
  | Leaky | 0.01 | No |
  | PReLU | Variable | Yes |

### 3. ELU and SELU
- **ELU (Exponential Linear Unit)**:
  ```
  f(x) = x if x > 0, else α(e^x - 1)
  ```
  - Smooth for negative values
  - Pushes mean toward zero
  - α typically 1.0
- **SELU (Scaled ELU)**:
  ```
  f(x) = λ × (x if x > 0, else α(e^x - 1))
  ```
  - λ ≈ 1.0507, α ≈ 1.6733 (specific values)
  - Self-normalizing property
  - Maintains mean≈0, variance≈1
- **SELU Requirements**:
  - Specific initialization (LeCun normal)
  - Alpha Dropout instead of regular dropout
  - Fully connected networks primarily

### 4. Swish and GELU
- **Swish**:
  ```
  f(x) = x × σ(βx)
  ```
  - σ: Sigmoid function
  - β: Learnable or fixed (typically β=1)
  - Smooth, non-monotonic
- **GELU (Gaussian Error Linear Unit)**:
  ```
  f(x) = x × Φ(x)
  ```
  - Φ: Standard Gaussian CDF
  - Approximation: x × σ(1.702x)
- **Properties**:
  - Smooth everywhere
  - Non-monotonic (slight dip)
  - State-of-the-art in Transformers
- **Used In**: BERT, GPT, Vision Transformers
- **Modern Standard**: GELU for Transformers

### 5. Activation Function Selection Guide
| Use Case | Recommended | Notes |
|----------|-------------|-------|
| CNNs (general) | ReLU | Simple, effective |
| CNNs (deep) | Leaky ReLU | Prevents dead neurons |
| Transformers | GELU | State-of-the-art |
| Self-normalizing | SELU | Specific setup required |
| Hidden layers | ReLU family | Most common |
| Output (regression) | Linear | Unbounded output |
| Output (binary) | Sigmoid | Probability [0,1] |
| Output (multi-class) | Softmax | Probability distribution |

### 6. Dead ReLU Problem
- **Definition**: Neurons that output zero for all inputs
- **Cause**:
  - Large negative bias
  - Large learning rate spike
  - Unfortunate initialization
- **Statistics**: Up to 40%+ neurons can become dead
- **Symptoms**:
  - Decreasing number of active neurons
  - Training plateau
  - Reduced model capacity
- **Solutions**:
  - Use Leaky ReLU or PReLU
  - Proper initialization (He init)
  - Lower learning rate
  - Batch normalization

### 7. Gradient Flow Analysis
- **Ideal Properties**:
  - Gradient ≈ 1 for positive inputs
  - Non-zero gradient for negative inputs
  - Smooth transitions
- **Comparison**:
  | Activation | Gradient (x>0) | Gradient (x<0) |
  |------------|----------------|----------------|
  | Sigmoid | 0-0.25 | 0-0.25 |
  | Tanh | 0-1 | 0-1 |
  | ReLU | 1 | 0 |
  | Leaky ReLU | 1 | 0.01 |
  | GELU | ≈1 | Small positive |
- **Best Gradient Flow**: Leaky ReLU, GELU, Swish

### 8. Layer-wise Activation Patterns
- **Different Activations per Layer**:
  - Early layers: Simple activations (ReLU)
  - Middle layers: May benefit from variations
  - Final layers: Task-specific (softmax, sigmoid)
- **Modern Practice**:
  - Usually same activation throughout
  - Except for output layer
  - GELU uniformly for Transformers
- **Experimental Approaches**:
  - Learned activation functions
  - Neural Architecture Search (NAS)
  - Mixture of activations

## Important Takeaways
1. ReLU revolutionized deep learning by enabling very deep networks
2. Dead ReLU problem can be mitigated with Leaky/PReLU variants
3. GELU is the modern standard for Transformer architectures
4. SELU enables self-normalizing networks but requires specific setup
5. Output activation must match the task (softmax, sigmoid, linear)
6. Gradient flow analysis helps understand training dynamics
7. Simple ReLU is still excellent for most CNN applications
8. Modern research continues to develop new activation functions

