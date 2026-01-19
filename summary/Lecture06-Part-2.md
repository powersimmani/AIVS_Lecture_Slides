# Lecture 06 - Part 2: Regression Evaluation Metrics

## Overview
This part covers evaluation metrics specifically designed for regression tasks, where the goal is to predict continuous values.

## Key Topics

### 1. MSE, RMSE, MAE
- **MSE (Mean Squared Error)**:
  ```
  MSE = (1/n) Σ(yᵢ - ŷᵢ)²
  ```
  - Squares emphasize large errors
  - Same units as target squared
  - Sensitive to outliers
- **RMSE (Root Mean Squared Error)**:
  ```
  RMSE = √MSE
  ```
  - Same units as target
  - Interpretable as "average error magnitude"
  - Most commonly used
- **MAE (Mean Absolute Error)**:
  ```
  MAE = (1/n) Σ|yᵢ - ŷᵢ|
  ```
  - More robust to outliers
  - Linear penalty for errors
  - When to use: Outliers shouldn't dominate

### 2. R² (Coefficient of Determination) and Adjusted R²
- **R² (R-squared)**:
  ```
  R² = 1 - (SS_res / SS_tot)
      = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
  ```
  - Range: (-∞, 1], typically [0, 1]
  - Interpretation: Proportion of variance explained
  - R² = 1: Perfect prediction
  - R² = 0: Model = mean prediction
  - R² < 0: Worse than predicting mean
- **Adjusted R²**:
  ```
  R²_adj = 1 - (1 - R²)(n - 1)/(n - p - 1)
  ```
  - Penalizes adding features
  - p: Number of predictors
  - Better for model comparison
- **Limitation**: High R² doesn't mean good model (check residuals)

### 3. MAPE and SMAPE
- **MAPE (Mean Absolute Percentage Error)**:
  ```
  MAPE = (100/n) Σ|yᵢ - ŷᵢ|/|yᵢ|
  ```
  - Percentage error (scale-independent)
  - Problem: Undefined when yᵢ = 0
  - Problem: Asymmetric (over vs underprediction)
- **SMAPE (Symmetric MAPE)**:
  ```
  SMAPE = (100/n) Σ|yᵢ - ŷᵢ|/((|yᵢ| + |ŷᵢ|)/2)
  ```
  - Bounded [0%, 200%]
  - Symmetric for over/underprediction
  - Still issues with zeros
- **When to Use**: When relative error matters more than absolute

### 4. Residual Analysis and Diagnostics
- **Residuals**: eᵢ = yᵢ - ŷᵢ
- **Key Diagnostic Plots**:
  - **Residuals vs Fitted**: Check linearity, homoscedasticity
    - Ideal: Random scatter around zero
    - Pattern: Indicates missing non-linearity
  - **Q-Q Plot**: Check normality
    - Ideal: Points on diagonal line
    - Deviation: Non-normal residuals
  - **Scale-Location**: Check variance
    - Ideal: Horizontal band
    - Funnel: Heteroscedasticity
  - **Residuals vs Leverage**: Identify influential points
- **What to Look For**:
  - Random scatter (no patterns)
  - Constant variance (homoscedasticity)
  - Normal distribution
  - No influential outliers

## Important Takeaways
1. RMSE is the most common regression metric (same units as target)
2. MAE is more robust to outliers than MSE/RMSE
3. R² measures proportion of variance explained, not prediction quality
4. Adjusted R² is better for comparing models with different feature counts
5. MAPE provides scale-independent percentage error
6. Residual analysis is essential for validating model assumptions
7. Multiple metrics together give a complete picture
