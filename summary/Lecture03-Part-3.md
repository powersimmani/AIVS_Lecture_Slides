# Lecture 03 - Part 3: Linear Regression Model

## Overview
This part covers the theory and practice of linear regression, from problem formulation through mathematical derivation to implementation and diagnostics.

## Key Topics

### 1. Linear Regression Problem Definition
- **Goal**: Predict continuous target y from features X
- **Assumption**: Linear relationship between features and target
- **Model**: y = Xβ + ε
  - X: Feature matrix (n × p)
  - β: Weight vector (p × 1)
  - ε: Error term (residuals)
- **Applications**: House price prediction, sales forecasting, trend analysis

### 2. Simple Linear Regression Model
- **Model**: y = β₀ + β₁x + ε
  - β₀: Intercept (y-value when x = 0)
  - β₁: Slope (change in y per unit change in x)
- **Interpretation**:
  - β₁ > 0: Positive relationship
  - β₁ < 0: Negative relationship
  - β₁ = 0: No linear relationship
- **Goal**: Find β₀ and β₁ that minimize prediction error

### 3. Deriving the Least Squares Method
- **Objective**: Minimize Sum of Squared Residuals (SSR)
  ```
  L(β) = Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - β₀ - β₁xᵢ)²
  ```
- **Derivation**:
  - Take partial derivatives: ∂L/∂β₀ = 0, ∂L/∂β₁ = 0
  - Solve system of equations
- **Solution (Simple)**:
  ```
  β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
  β₀ = ȳ - β₁x̄
  ```
- **Why Squared Error?**: Differentiable, penalizes large errors, Gaussian noise assumption

### 4. Normal Equation Solution
- **Matrix Form**: y = Xβ + ε
- **Loss**: L(β) = ||y - Xβ||²
- **Normal Equation**:
  ```
  β = (XᵀX)⁻¹Xᵀy
  ```
- **Derivation**:
  - ∇L = -2Xᵀ(y - Xβ) = 0
  - Xᵀy = XᵀXβ
  - β = (XᵀX)⁻¹Xᵀy
- **Considerations**:
  - Requires XᵀX to be invertible
  - O(p³) complexity for inversion
  - Numerical stability issues with ill-conditioned matrices

### 5. Geometric Interpretation
- **Projection**: ŷ = Xβ is the projection of y onto the column space of X
- **Residuals**: e = y - ŷ is orthogonal to column space
- **Hat Matrix**: H = X(XᵀX)⁻¹Xᵀ, where ŷ = Hy
- **Orthogonality**: Xᵀe = Xᵀ(y - ŷ) = 0
- **Visual**: Best fit line/plane minimizes perpendicular distance to data points
- **Insight**: Linear regression finds the closest point to y in the span of X

### 6. Multiple Linear Regression Extension
- **Model**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
- **Matrix Form**: y = Xβ where X includes intercept column
- **Interpretation**: Each βⱼ represents change in y per unit change in xⱼ, holding other variables constant
- **Assumptions**:
  - No perfect multicollinearity
  - Sufficient observations (n > p)
- **Multicollinearity**: High correlation between predictors causes unstable estimates

### 7. Model Assumptions and Diagnostics
- **Key Assumptions (LINE)**:
  - **L**inearity: Linear relationship between X and y
  - **I**ndependence: Residuals are independent
  - **N**ormality: Residuals are normally distributed
  - **E**qual variance (Homoscedasticity): Constant residual variance
- **Diagnostic Plots**:
  - Residuals vs Fitted: Check linearity and homoscedasticity
  - Q-Q Plot: Check normality of residuals
  - Scale-Location: Check homoscedasticity
  - Residuals vs Leverage: Identify influential points
- **Metrics**:
  - R² (Coefficient of Determination): Proportion of variance explained
  - Adjusted R²: Accounts for number of predictors
  - RMSE, MAE: Prediction error measures

### 8. Python Implementation and Practice
- **Using scikit-learn**:
  ```python
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  ```
- **From scratch**:
  ```python
  # Normal equation
  beta = np.linalg.inv(X.T @ X) @ X.T @ y
  ```
- **Using statsmodels** (for statistical summaries):
  ```python
  import statsmodels.api as sm
  model = sm.OLS(y, X).fit()
  print(model.summary())
  ```
- **Evaluation**:
  ```python
  from sklearn.metrics import r2_score, mean_squared_error
  r2 = r2_score(y_true, y_pred)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  ```

## Important Takeaways
1. Linear regression minimizes squared error between predictions and targets
2. The normal equation provides a closed-form solution
3. Geometric interpretation: regression projects y onto the column space of X
4. Checking assumptions is crucial for valid inference
5. R² alone doesn't indicate model quality - check residual plots
6. Multiple regression requires attention to multicollinearity
7. Use train/test splits to evaluate generalization performance
