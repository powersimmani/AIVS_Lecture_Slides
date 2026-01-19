# Lecture 08 - Part 1: Loss Function Design

## Overview
This part covers the fundamentals of loss functions, their role in machine learning, and various loss functions for different tasks including regression, classification, and metric learning.

## Key Topics

### 1. Role and Importance of Loss Functions
- **Definition**: Quantifies the difference between predictions and actual values
- **Purpose**: Guides model optimization during training
- **Properties**:
  - Differentiable (for gradient-based optimization)
  - Task-appropriate
  - Well-scaled
- **Impact**: Choice of loss function affects convergence and model behavior

### 2. Regression Losses - MSE, MAE, Huber
- **MSE (Mean Squared Error)**:
  ```
  L = (1/n) Σ(y - ŷ)²
  ```
  - Penalizes large errors heavily
  - Sensitive to outliers
  - Smooth gradient everywhere
- **MAE (Mean Absolute Error)**:
  ```
  L = (1/n) Σ|y - ŷ|
  ```
  - Robust to outliers
  - Non-smooth at zero
- **Huber Loss**:
  ```
  L = 0.5(y-ŷ)² if |y-ŷ| ≤ δ, else δ|y-ŷ| - 0.5δ²
  ```
  - Combines MSE and MAE benefits
  - Smooth and robust
  - δ controls the transition point

### 3. Classification Loss - Cross-Entropy
- **Binary Cross-Entropy**:
  ```
  L = -[y·log(p) + (1-y)·log(1-p)]
  ```
- **Categorical Cross-Entropy**:
  ```
  L = -Σ yᵢ·log(pᵢ)
  ```
- **Properties**:
  - Works with probability outputs (softmax/sigmoid)
  - Heavily penalizes confident wrong predictions
  - Standard for classification tasks

### 4. Hinge Loss and SVM
- **Formula**:
  ```
  L = max(0, 1 - y·f(x))
  ```
- **Properties**:
  - Used in Support Vector Machines
  - Encourages margin maximization
  - Creates sparse solutions
  - Non-differentiable at hinge point
- **Squared Hinge**: Smoothed variant

### 5. Focal Loss - Class Imbalance
- **Formula**:
  ```
  FL = -αₜ(1-pₜ)^γ · log(pₜ)
  ```
- **Purpose**: Address class imbalance
- **Parameters**:
  - α: Class weighting factor
  - γ: Focusing parameter (typically 2)
- **Behavior**: Down-weights easy examples, focuses on hard ones
- **Applications**: Object detection (RetinaNet)

### 6. Contrastive Loss
- **Formula**:
  ```
  L = (1-Y)·D² + Y·max(0, m-D)²
  ```
  - Y: 0 for similar pairs, 1 for dissimilar
  - D: Distance between embeddings
  - m: Margin parameter
- **Purpose**: Learn similarity/dissimilarity
- **Applications**: Face verification, image retrieval, siamese networks

### 7. Triplet Loss
- **Formula**:
  ```
  L = max(0, d(a,p) - d(a,n) + margin)
  ```
  - a: Anchor sample
  - p: Positive sample (same class)
  - n: Negative sample (different class)
- **Goal**: Push anchor-positive closer, anchor-negative farther
- **Mining Strategies**: Hard, semi-hard, easy negatives
- **Applications**: Face recognition (FaceNet), person re-identification

### 8. Regularization Terms
- **L1 Regularization (Lasso)**:
  ```
  R = λ·Σ|wᵢ|
  ```
  - Encourages sparsity
  - Feature selection effect
- **L2 Regularization (Ridge/Weight Decay)**:
  ```
  R = λ·Σwᵢ²
  ```
  - Prevents large weights
  - Smooth penalty
- **Elastic Net (L1 + L2)**:
  ```
  R = λ₁·Σ|wᵢ| + λ₂·Σwᵢ²
  ```
  - Combines benefits of both

### 9. Custom Loss Function Design
- **Principles**:
  - Match loss to task requirements
  - Ensure differentiability
  - Consider gradient behavior
  - Balance multiple objectives
- **Techniques**:
  - Weighted combinations
  - Task-specific penalties
  - Curriculum learning losses
- **Examples**: Perceptual loss, adversarial loss, reconstruction loss

## Important Takeaways
1. MSE is standard for regression; Huber is robust to outliers
2. Cross-entropy is the go-to loss for classification
3. Focal loss addresses class imbalance by focusing on hard examples
4. Contrastive and triplet losses are essential for metric learning
5. Regularization terms prevent overfitting and encourage desired properties
6. Loss function choice significantly impacts model training and final performance
7. Custom losses can be designed by combining existing losses for specific tasks

