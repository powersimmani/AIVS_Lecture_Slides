# Lecture 04 - Part 3: Completing Logistic Regression

## Overview
This part completes the development of logistic regression, covering the model definition, maximum likelihood estimation, loss functions, gradient descent optimization, and extensions to multiclass classification.

## Key Topics

### 1. Defining the Logistic Regression Model
- **Model**:
  ```
  P(Y=1|X) = σ(wᵀx + b) = 1 / (1 + e⁻⁽ʷᵀˣ⁺ᵇ⁾)
  ```
- **Components**:
  - Linear part: z = wᵀx + b
  - Non-linear transformation: p = σ(z)
- **Decision Rule**: Predict class 1 if P(Y=1|X) > 0.5
- **Decision Boundary**: wᵀx + b = 0 (where p = 0.5)
- **Interpretation**: Log odds increase linearly with features

### 2. Maximum Likelihood Estimation (MLE)
- **Likelihood**: Probability of observed data given parameters
- **For Single Sample**:
  ```
  P(y|x; w) = p^y × (1-p)^(1-y)
  ```
  - If y=1: P = p
  - If y=0: P = 1-p
- **Full Dataset Likelihood**:
  ```
  L(w) = Π pᵢ^yᵢ × (1-pᵢ)^(1-yᵢ)
  ```
- **Log-Likelihood** (easier to optimize):
  ```
  ℓ(w) = Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
  ```
- **Goal**: Find w that maximizes log-likelihood

### 3. Binary Cross-Entropy Loss
- **Loss Function** (negative log-likelihood):
  ```
  L = -1/n Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
  ```
- **Properties**:
  - Convex function (single global minimum)
  - Penalizes confident wrong predictions heavily
  - Range: [0, ∞)
- **Interpretation**:
  - When y=1: Loss = -log(p) → wants p close to 1
  - When y=0: Loss = -log(1-p) → wants p close to 0
- **Information Theory**: Cross-entropy between true and predicted distributions

### 4. Applying Gradient Descent
- **Gradient**:
  ```
  ∂L/∂w = 1/n Σ (pᵢ - yᵢ) xᵢ
  ```
- **Update Rule**:
  ```
  w = w - η × ∂L/∂w
  ```
- **No Closed-form Solution**: Unlike linear regression, must iterate
- **Convergence**: Guaranteed for convex cross-entropy loss
- **Variants**:
  - Batch: Use all samples
  - Stochastic (SGD): One sample at a time
  - Mini-batch: Small subsets
- **Learning Rate**: Critical hyperparameter (too high: diverge, too low: slow)

### 5. Multiclass - One-vs-Rest Strategy
- **Problem**: Extend binary to K classes
- **One-vs-Rest (OvR)**:
  - Train K binary classifiers
  - Each: Class k vs all others
  - Predict: Class with highest probability
- **Training**: K separate logistic regression models
- **Prediction**:
  ```
  ŷ = argmax P(Y=k|X)
  ```
- **Limitation**: Probabilities don't sum to 1

### 6. Softmax Regression
- **Direct Multiclass Extension**
- **Softmax Function**:
  ```
  P(Y=k|X) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
  ```
- **Properties**:
  - Probabilities sum to 1
  - Generalizes sigmoid (K=2 reduces to sigmoid)
  - Each class has own weight vector
- **Output**: Probability distribution over K classes
- **Implementation**:
  ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression(multi_class='multinomial')
  ```

### 7. Categorical Cross-Entropy
- **Loss for Multiclass**:
  ```
  L = -1/n Σᵢ Σₖ yᵢₖ log(pᵢₖ)
  ```
  - yᵢₖ: 1 if sample i belongs to class k (one-hot encoding)
  - pᵢₖ: Predicted probability for class k
- **Equivalence**: Reduces to binary cross-entropy for K=2
- **Gradient**: Similar form, computed for each class weight
- **Used In**: Neural network output layers with softmax

### 8. Regularized Logistic Regression
- **Add Penalty** (like linear regression):
  ```
  L = CrossEntropy + λ × Regularization
  ```
- **L2 (Ridge)**:
  ```
  L = -Σ[y log(p) + (1-y) log(1-p)] + λΣwⱼ²
  ```
- **L1 (Lasso)**: Sparse coefficients, feature selection
- **Elastic Net**: Combination of L1 and L2
- **Benefits**:
  - Prevents overfitting
  - Improves generalization
  - Handles correlated features
- **Implementation**:
  ```python
  model = LogisticRegression(penalty='l2', C=1.0)  # C = 1/λ
  ```

### 9. Real-World Cases and Implementation
- **Common Applications**:
  - Spam detection
  - Medical diagnosis
  - Credit risk assessment
  - Click-through rate prediction
- **Practical Considerations**:
  - Feature scaling (standardization)
  - Handling class imbalance (class weights)
  - Threshold tuning (not always 0.5)
- **scikit-learn Implementation**:
  ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression(
      penalty='l2',
      C=1.0,
      solver='lbfgs',
      max_iter=1000
  )
  model.fit(X_train, y_train)
  probs = model.predict_proba(X_test)
  ```

## Important Takeaways
1. Logistic regression models probability using the sigmoid function
2. MLE provides the theoretical foundation; cross-entropy is the loss
3. Gradient descent is required since there's no closed-form solution
4. Softmax extends logistic regression to multiple classes naturally
5. Regularization is crucial for preventing overfitting
6. Logistic regression is interpretable: coefficients show feature importance
7. Despite "regression" in the name, it's fundamentally a classification method
