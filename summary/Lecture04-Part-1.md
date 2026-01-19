# Lecture 04 - Part 1: Advanced Linear Regression

## Overview
This part covers advanced topics in linear regression including polynomial extensions, regularization techniques, and the limitations that motivate the transition to classification methods.

## Key Topics

### 1. Review and Connection to Previous Lecture
- **Recap**: Linear regression minimizes MSE: L = Σ(yᵢ - ŷᵢ)²
- **Normal Equation**: β = (XᵀX)⁻¹Xᵀy
- **Limitations**: Only captures linear relationships
- **Goal**: Extend to handle non-linear patterns and prevent overfitting

### 2. Revisiting Linear Regression Assumptions
- **Key Assumptions**:
  - Linearity in parameters (not necessarily in features)
  - Independence of residuals
  - Homoscedasticity (constant variance)
  - Normality of residuals (for inference)
  - No multicollinearity
- **Violations and Solutions**:
  - Non-linearity → Feature transformation
  - Heteroscedasticity → Weighted least squares
  - Multicollinearity → Regularization or feature selection

### 3. Polynomial Regression and Basis Expansion
- **Idea**: Create non-linear features from original features
- **Polynomial Features**: x → [1, x, x², x³, ...]
- **Model**: y = β₀ + β₁x + β₂x² + β₃x³ + ... + ε
- **Still Linear**: Linear in parameters (β), non-linear in input (x)
- **Basis Functions**: φ(x) = [φ₁(x), φ₂(x), ...] - any transformation
- **Risk**: Higher-degree polynomials can overfit
- **Implementation**:
  ```python
  from sklearn.preprocessing import PolynomialFeatures
  poly = PolynomialFeatures(degree=3)
  X_poly = poly.fit_transform(X)
  ```

### 4. Ridge Regression (L2 Regularization)
- **Problem**: Large coefficients lead to overfitting
- **Solution**: Add penalty on coefficient magnitude
- **Loss Function**:
  ```
  L = Σ(yᵢ - ŷᵢ)² + λΣβⱼ²
  ```
- **Closed-form Solution**:
  ```
  β = (XᵀX + λI)⁻¹Xᵀy
  ```
- **Effect**:
  - Shrinks coefficients toward zero
  - Never exactly zero (keeps all features)
  - λ controls regularization strength
- **Benefits**: Handles multicollinearity, reduces variance
- **Implementation**:
  ```python
  from sklearn.linear_model import Ridge
  model = Ridge(alpha=1.0).fit(X, y)
  ```

### 5. Lasso Regression (L1 Regularization)
- **Loss Function**:
  ```
  L = Σ(yᵢ - ŷᵢ)² + λΣ|βⱼ|
  ```
- **Key Property**: Can shrink coefficients exactly to zero
- **Sparse Solutions**: Performs automatic feature selection
- **No Closed-form**: Requires iterative optimization (coordinate descent)
- **Comparison with Ridge**:
  - L1 (Lasso): Sparse, feature selection
  - L2 (Ridge): Dense, coefficient shrinkage
- **Implementation**:
  ```python
  from sklearn.linear_model import Lasso
  model = Lasso(alpha=0.1).fit(X, y)
  ```

### 6. Elastic Net
- **Combines L1 and L2**: Best of both worlds
- **Loss Function**:
  ```
  L = Σ(yᵢ - ŷᵢ)² + λ₁Σ|βⱼ| + λ₂Σβⱼ²
  ```
- **Hyperparameters**:
  - α: Overall regularization strength
  - l1_ratio: Mix between L1 and L2 (0 = Ridge, 1 = Lasso)
- **Benefits**:
  - Feature selection (from L1)
  - Stability with correlated features (from L2)
- **When to Use**: Many correlated features, want sparse solution
- **Implementation**:
  ```python
  from sklearn.linear_model import ElasticNet
  model = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X, y)
  ```

### 7. Feature Selection and Importance
- **Methods**:
  - **Filter**: Statistical tests (correlation, mutual information)
  - **Wrapper**: Search through feature subsets (RFE, forward/backward)
  - **Embedded**: Regularization-based (Lasso)
- **Coefficient Magnitude**: |βⱼ| indicates feature importance (after standardization)
- **Recursive Feature Elimination (RFE)**:
  ```python
  from sklearn.feature_selection import RFE
  selector = RFE(estimator, n_features_to_select=5)
  selector.fit(X, y)
  ```
- **Considerations**:
  - Standardize features before comparing coefficients
  - Correlated features share importance

### 8. Limitations of Linear Regression
- **Cannot Handle**:
  - Non-linear decision boundaries naturally
  - Assumes continuous output (not suitable for classification)
  - Sensitive to outliers without robust variants
  - Complex patterns require extensive feature engineering
- **Why Classification Needed**:
  - Output is categorical (yes/no, class A/B/C)
  - Linear regression outputs can exceed [0,1] range
  - Probabilistic interpretation needed
- **Transition**: These limitations motivate logistic regression

## Important Takeaways
1. Polynomial features allow non-linear relationships within linear framework
2. Regularization prevents overfitting by penalizing large coefficients
3. Ridge (L2) shrinks all coefficients; Lasso (L1) can zero them out
4. Elastic Net combines benefits of both L1 and L2 regularization
5. Feature selection improves interpretability and can reduce overfitting
6. Cross-validation is essential for choosing regularization strength
7. Linear regression limitations lead naturally to classification methods
