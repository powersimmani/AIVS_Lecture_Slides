# Lecture 05 - Part 2: MLP Structure and Forward Propagation

## Overview
This part covers the architecture of multi-layer perceptrons (MLPs), the mathematics of forward propagation, and the various activation functions used in neural networks.

## Key Topics

### 1. Single-layer Perceptron Review
- **Model**: ŷ = σ(wᵀx + b)
- **Components**:
  - Input: x ∈ ℝⁿ
  - Weights: w ∈ ℝⁿ
  - Bias: b ∈ ℝ
  - Activation: σ (sigmoid, step, etc.)
- **Computation**:
  1. Linear combination: z = Σwᵢxᵢ + b
  2. Activation: ŷ = σ(z)
- **Limitation**: Only linear decision boundaries

### 2. Multi-layer Perceptron Architecture
- **Structure**: Input → Hidden → ... → Hidden → Output
- **Notation** (L-layer network):
  - Layer 0: Input (n⁰ features)
  - Layers 1 to L-1: Hidden layers
  - Layer L: Output layer
- **Forward Equation**:
  ```
  h⁽ˡ⁾ = σ(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)
  ```
- **Example 5-layer Network**:
  - Input: 784 neurons (28×28 image)
  - Hidden 1: 256 neurons
  - Hidden 2: 128 neurons
  - Hidden 3: 64 neurons
  - Output: 10 neurons (10 classes)

### 3. Weights and Biases
- **Weight Matrix**: W⁽ˡ⁾ ∈ ℝᵐˣⁿ
  - m: Number of neurons in layer l
  - n: Number of neurons in layer l-1
  - Wᵢⱼ: Connection from neuron j in layer l-1 to neuron i in layer l
- **Bias Vector**: b⁽ˡ⁾ ∈ ℝᵐ
  - One bias per neuron in layer l
- **Total Parameters**: Σₗ(nˡ × nˡ⁻¹ + nˡ)
- **Initialization**: Critical for training
  - Xavier/Glorot: Good for tanh/sigmoid
  - He: Good for ReLU

### 4. Forward Propagation Algorithm
- **Algorithm**:
  ```
  1. a⁽⁰⁾ = x                          # Input
  2. For l = 1 to L:
     z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾          # Linear
     a⁽ˡ⁾ = σ⁽ˡ⁾(z⁽ˡ⁾)                 # Activation
  3. ŷ = a⁽ᴸ⁾                          # Output
  ```
- **Numerical Example** (2-layer, inputs [1.0, 0.5]):
  - z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾ = [0.8, 1.2]
  - a⁽¹⁾ = σ(z⁽¹⁾) = [0.69, 0.77]
  - z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾ = [1.1]
  - ŷ = σ(z⁽²⁾) = [0.75]

### 5. Activation Functions - Sigmoid and Tanh
- **Sigmoid**:
  ```
  σ(z) = 1 / (1 + e⁻ᶻ)
  ```
  - Range: (0, 1)
  - Output interpretable as probability
  - **Issue**: Vanishing gradients at extremes
- **Tanh**:
  ```
  tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
  ```
  - Range: (-1, 1)
  - Zero-centered (often preferred over sigmoid)
  - **Issue**: Still suffers from vanishing gradients
- **When to Use**:
  - Sigmoid: Output layer for binary classification
  - Tanh: Hidden layers (historically), zero-centered

### 6. ReLU and Its Variants
- **ReLU (Rectified Linear Unit)**:
  ```
  ReLU(z) = max(0, z)
  ```
  - Simple and efficient
  - No vanishing gradient for z > 0
  - **Issue**: "Dying ReLU" - neurons can stop learning
- **Leaky ReLU**:
  ```
  LeakyReLU(z) = max(αz, z), α ≈ 0.01
  ```
  - Small gradient for z < 0
  - Prevents dying ReLU
- **ELU (Exponential Linear Unit)**:
  ```
  ELU(z) = z if z > 0, else α(eᶻ - 1)
  ```
  - Smooth everywhere
  - Negative values push mean toward zero
- **Comparison**: ReLU is default choice; use variants if dying ReLU is an issue

### 7. Output Layer Design (Regression vs Classification)
- **Regression**:
  - Activation: Linear (identity)
  - Loss: MSE = (1/n)Σ(yᵢ - ŷᵢ)²
  - Output: Single neuron for each target
- **Binary Classification**:
  - Activation: Sigmoid
  - Loss: Binary Cross-Entropy
  - Output: Single neuron, probability P(y=1)
- **Multiclass Classification**:
  - Activation: Softmax
  - Loss: Categorical Cross-Entropy
  - Output: K neurons (one per class), probabilities sum to 1
- **Key Principle**: Output activation + loss function must match task

### 8. Network Capacity and Complexity
- **Model Capacity**: Ability to fit various functions
- **Factors**:
  - Number of layers (depth)
  - Neurons per layer (width)
  - Activation functions
- **Underfitting**: Capacity too low
  - Both training and validation error high
  - Solution: Increase capacity
- **Overfitting**: Capacity too high
  - Training error low, validation error high
  - Solution: Regularization, dropout, early stopping
- **Optimal**: Balance between underfitting and overfitting
- **Rule of Thumb**: Start simple, increase complexity as needed

## Important Takeaways
1. MLPs consist of input, hidden, and output layers connected by weights
2. Forward propagation computes output through sequential layer transformations
3. ReLU is the default activation for hidden layers (fast, no vanishing gradient)
4. Output layer design depends on task (linear/sigmoid/softmax)
5. Network capacity must balance fitting ability with generalization
6. Proper weight initialization is crucial for training deep networks
7. Number of parameters grows with width and depth
