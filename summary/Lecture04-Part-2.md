# Lecture 04 - Part 2: Transition to Classification

## Overview
This part bridges the gap between regression and classification, introducing fundamental classification concepts, the perceptron algorithm, and the mathematical foundations leading to logistic regression.

## Key Topics

### 1. Regression vs Classification Problems
- **Regression**: Predict continuous values
  - Output: y ∈ ℝ (any real number)
  - Examples: House prices, temperature, stock prices
- **Classification**: Predict discrete categories
  - Output: y ∈ {0, 1} or {class₁, class₂, ...}
  - Examples: Spam detection, disease diagnosis, image recognition
- **Key Difference**: Output space and loss function
- **Evaluation Metrics**:
  - Regression: MSE, RMSE, R²
  - Classification: Accuracy, precision, recall, F1, AUC

### 2. Linear Classifier Concepts
- **Decision Function**: f(x) = wᵀx + b
- **Classification Rule**: ŷ = sign(f(x))
  - f(x) > 0 → Class 1
  - f(x) < 0 → Class 0
- **Decision Boundary**: The hyperplane where f(x) = 0
- **Linear Separability**: Classes can be perfectly separated by a hyperplane
- **Weight Vector**: w is perpendicular to decision boundary
- **Bias**: b shifts the boundary from origin

### 3. Perceptron Algorithm
- **Model**: ŷ = sign(wᵀx + b)
- **Learning Rule** (update on misclassification):
  ```
  if yᵢ(wᵀxᵢ + b) ≤ 0:
      w = w + ηyᵢxᵢ
      b = b + ηyᵢ
  ```
- **Algorithm**:
  1. Initialize w and b
  2. For each misclassified point
  3. Update weights toward correct classification
  4. Repeat until convergence
- **Convergence**: Guaranteed only for linearly separable data
- **Limitation**: No convergence guarantee for non-separable data
- **Historical**: Foundational algorithm (Rosenblatt, 1958)

### 4. Decision Boundaries and Linear Separability
- **Linear Decision Boundary**: Hyperplane wᵀx + b = 0
- **Properties**:
  - In 2D: a line
  - In 3D: a plane
  - In nD: a hyperplane
- **Linear Separability**: Exists a hyperplane perfectly separating classes
- **XOR Problem**: Classic example of non-linearly separable data
- **Margin**: Distance from boundary to nearest points
- **Support Vectors**: Points closest to the boundary (SVM concept preview)

### 5. Why Linear Regression Fails for Classification
- **Problems**:
  - Output unbounded (can be < 0 or > 1)
  - Sensitive to outliers far from boundary
  - Doesn't model probability properly
  - Decision boundary affected by all points, not just boundary region
- **Example**: Adding outlier shifts boundary significantly
- **Need**: Function that outputs values in [0, 1] interpretable as probability

### 6. Odds and Log Odds
- **Probability**: P(Y=1|X) = p
- **Odds**: p / (1-p)
  - Ratio of success to failure probability
  - Range: [0, ∞)
- **Log Odds (Logit)**: log(p / (1-p))
  - Range: (-∞, ∞)
  - Linear in parameters
- **Key Insight**: Log odds can be modeled as linear function:
  ```
  log(p / (1-p)) = wᵀx + b
  ```
- **Inverse**: Solving for p gives the logistic function

### 7. Introduction to Sigmoid Function
- **Logistic (Sigmoid) Function**:
  ```
  σ(z) = 1 / (1 + e⁻ᶻ)
  ```
- **Properties**:
  - Range: (0, 1) - perfect for probability
  - Monotonically increasing
  - σ(0) = 0.5
  - σ(-z) = 1 - σ(z)
- **Derivative**: σ'(z) = σ(z)(1 - σ(z)) - convenient for optimization
- **Interpretation**: Maps any real number to probability

### 8. Properties of Logistic Function
- **Mathematical Properties**:
  - Smooth and differentiable everywhere
  - Asymptotes at 0 and 1
  - S-shaped curve (sigmoid)
- **Gradient**: Maximum at z=0, approaches 0 at extremes
- **Inverse (Logit)**: z = log(p / (1-p))
- **Connection to Linear**:
  - Input: z = wᵀx + b (linear combination)
  - Output: p = σ(z) (probability)
- **Decision Boundary**: p = 0.5 when wᵀx + b = 0

## Important Takeaways
1. Classification predicts discrete categories rather than continuous values
2. Linear classifiers use hyperplanes to separate classes
3. The perceptron is a foundational but limited classification algorithm
4. Linear regression fails for classification due to unbounded outputs
5. The logit transformation converts probabilities to a linear-friendly scale
6. The sigmoid function maps any real number to a probability in (0, 1)
7. These concepts build the foundation for logistic regression
