# Lecture 05 - Part 3: Backpropagation and Learning

## Overview
This part covers how neural networks learn through backpropagation, the algorithm that computes gradients for all parameters by applying the chain rule through the computational graph.

## Key Topics

### 1. Loss Function Definition
- **Purpose**: Quantify prediction error
- **Regression Loss - MSE**:
  ```
  L = (1/n) Σ(yᵢ - ŷᵢ)²
  ```
- **Classification Loss - Cross-Entropy**:
  ```
  L = -(1/n) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
  ```
- **Multiclass - Categorical Cross-Entropy**:
  ```
  L = -(1/n) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)
  ```
- **Loss Landscape**: Surface over parameter space; goal is to find minimum

### 2. Chain Rule
- **Single Variable**: d/dx[f(g(x))] = f'(g(x)) · g'(x)
- **Multiple Variables**: Gradients flow through composition
- **Numerical Example**:
  - f(x) = (x + 2)²
  - Let u = x + 2, f(u) = u²
  - df/dx = df/du × du/dx = 2u × 1 = 2(x + 2)
- **For Networks**: Apply chain rule through all layers
- **Key Insight**: Gradients multiply through layers (can vanish or explode)

### 3. Backpropagation Algorithm Derivation
- **Goal**: Compute ∂L/∂W⁽ˡ⁾ and ∂L/∂b⁽ˡ⁾ for all layers
- **Backward Pass**:
  1. Compute output error: δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾
  2. Propagate error backward: δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
  3. Compute gradients: ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
- **Complete Numerical Example** (2-2-1 network):
  - Forward: Compute all z and a values
  - Backward: Compute δ for each layer
  - Gradients: Compute ∂L/∂W and ∂L/∂b
- **Efficiency**: Reuses intermediate values, avoids redundant computation

### 4. Computational Graph
- **Definition**: Directed graph showing computation flow
- **Nodes**: Operations (add, multiply, activation)
- **Edges**: Data flow (tensors)
- **Example**: z = w·x + b
  - Node 1: Multiply (w, x) → wx
  - Node 2: Add (wx, b) → z
- **Forward Pass**: Traverse graph from inputs to outputs
- **Backward Pass**: Traverse from output to inputs, applying chain rule
- **Benefits**: Automatic gradient computation, memory optimization

### 5. Gradient Calculation Example
- **Function**: y = σ(wx + b)
- **Step-by-step Backward**:
  1. ∂L/∂y (from loss)
  2. ∂y/∂z = σ(z)(1 - σ(z)) (sigmoid derivative)
  3. ∂z/∂w = x
  4. ∂z/∂b = 1
  5. ∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w
  6. ∂L/∂b = ∂L/∂y × ∂y/∂z × ∂z/∂b
- **Result**: Gradients for all parameters

### 6. Automatic Differentiation (Autograd)
- **Three Approaches**:
  - **Manual**: Hand-derive gradients (error-prone)
  - **Numerical**: Finite differences (slow, inaccurate)
  - **Automatic**: Build computation graph, apply chain rule
- **PyTorch Example**:
  ```python
  x = torch.tensor([1.0], requires_grad=True)
  y = x ** 2 + 2 * x
  y.backward()
  print(x.grad)  # dy/dx = 2x + 2 = 4.0
  ```
- **Benefits**:
  - Exact gradients (unlike numerical)
  - Automatic (unlike manual)
  - Efficient memory use

### 7. Mini-batch Gradient Descent
- **Batch Gradient Descent**:
  - Use entire dataset
  - Stable but slow
  - Memory intensive
- **Stochastic Gradient Descent (SGD)**:
  - Single sample per update
  - Fast but noisy
  - May not converge smoothly
- **Mini-batch (Best of Both)**:
  - Use subset (32-1024 samples)
  - Balances speed and stability
  - Enables GPU parallelization
- **Comparison**:
  | Method | Batch Size | Speed | Stability |
  |--------|-----------|-------|-----------|
  | Batch  | N         | Slow  | High      |
  | SGD    | 1         | Fast  | Low       |
  | Mini-batch | 32-1024 | Medium | Medium |

### 8. Implementation Tips and Debugging
- **Data Tips**:
  - Normalize/standardize inputs
  - Shuffle training data
  - Use data augmentation
- **Training Tips**:
  - Start with small learning rate
  - Monitor loss and metrics
  - Use validation set
  - Early stopping
- **Debugging**:
  - Gradient checking (compare to numerical gradients)
  - Overfit small batch first
  - Visualize loss curves
  - Check for NaN/Inf values
- **Hyperparameters**:
  - Learning rate (most important)
  - Batch size
  - Network architecture
  - Regularization strength

### 9. PyTorch/TensorFlow Hands-on
- **PyTorch Example**:
  ```python
  import torch
  import torch.nn as nn

  model = nn.Sequential(
      nn.Linear(784, 256),
      nn.ReLU(),
      nn.Linear(256, 10)
  )

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  # Training loop
  for epoch in range(epochs):
      output = model(x_train)
      loss = criterion(output, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  ```
- **TensorFlow/Keras Example**:
  ```python
  import tensorflow as tf

  model = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy')
  model.fit(x_train, y_train, epochs=10)
  ```

## Important Takeaways
1. Backpropagation efficiently computes gradients using the chain rule
2. Computational graphs enable automatic differentiation
3. Mini-batch gradient descent balances efficiency and stability
4. Modern frameworks (PyTorch, TensorFlow) handle backprop automatically
5. Proper initialization, normalization, and monitoring are crucial
6. Start simple: overfit a small batch before scaling up
7. Gradient checking helps verify implementations
