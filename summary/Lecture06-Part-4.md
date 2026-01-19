# Lecture 06 - Part 4: Model Validation Techniques

## Overview
This part covers advanced techniques for model validation, including cross-validation variants and strategies for hyperparameter tuning and model selection.

## Key Topics

### 1. K-fold Cross Validation
- **Process**:
  1. Split data into K equal folds
  2. For each fold i:
     - Train on K-1 folds
     - Validate on fold i
  3. Average scores across all K folds
- **Common Values**: K = 5 or 10
- **Benefits**:
  - Uses all data for both training and validation
  - More robust estimate than single split
  - Reduces variance of performance estimate
- **Implementation**:
  ```python
  from sklearn.model_selection import cross_val_score
  scores = cross_val_score(model, X, y, cv=5)
  print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
  ```

### 2. Stratified K-fold
- **Purpose**: Preserve class distribution in each fold
- **When to Use**:
  - Classification tasks
  - Imbalanced datasets
  - Small datasets
- **Process**: Same as K-fold but each fold has same class proportions
- **Implementation**:
  ```python
  from sklearn.model_selection import StratifiedKFold
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  for train_idx, val_idx in skf.split(X, y):
      # Train and evaluate
  ```
- **Default in scikit-learn**: Used automatically for classification

### 3. Leave-One-Out Cross Validation (LOOCV)
- **Process**: K-fold where K = n (number of samples)
- **Each Iteration**: Train on n-1, validate on 1 sample
- **Pros**:
  - Maximum training data per iteration
  - Deterministic (no random splits)
  - Useful for very small datasets
- **Cons**:
  - Computationally expensive (n model trainings)
  - High variance in estimate
- **When to Use**: Small datasets (< 100 samples)
- **Implementation**:
  ```python
  from sklearn.model_selection import LeaveOneOut
  loo = LeaveOneOut()
  ```

### 4. Bootstrapping
- **Process**:
  - Sample n points WITH replacement
  - ~63.2% unique samples (in-bag)
  - ~36.8% out-of-bag (OOB)
  - Evaluate on OOB samples
- **Benefits**:
  - Confidence intervals for metrics
  - Can estimate variance
  - Works with any sample size
- **Bootstrap 0.632 Estimator**:
  ```
  Error = 0.632 × OOB_error + 0.368 × Train_error
  ```
- **Use Cases**: Uncertainty estimation, ensemble methods

### 5. Hyperparameter Tuning
- **Grid Search**:
  - Try all combinations of parameter values
  - Exhaustive but expensive
  ```python
  from sklearn.model_selection import GridSearchCV
  param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1]}
  grid = GridSearchCV(SVC(), param_grid, cv=5)
  ```
- **Random Search**:
  - Sample random parameter combinations
  - Often more efficient than grid search
  ```python
  from sklearn.model_selection import RandomizedSearchCV
  ```
- **Bayesian Optimization**:
  - Uses surrogate model to guide search
  - More efficient for expensive evaluations
  - Libraries: Optuna, hyperopt, scikit-optimize
- **Key**: Always use cross-validation within search

### 6. Model Selection Strategy
- **Nested Cross-Validation**:
  - Outer loop: Estimate generalization
  - Inner loop: Select best hyperparameters
  - Avoids optimistic bias from using same data for selection and evaluation
- **Complete Workflow**:
  1. Split into train/test
  2. Use CV on train for hyperparameter tuning
  3. Retrain best model on all training data
  4. Final evaluation on test set (once!)
- **Comparing Models**:
  - Use same CV splits for fair comparison
  - Statistical tests for significance (paired t-test, Wilcoxon)
  - Consider variance, not just mean performance
- **Final Model**:
  - After selection, retrain on all available data
  - Deploy and monitor performance

## Important Takeaways
1. K-fold CV provides more robust performance estimates than single split
2. Stratified K-fold is essential for imbalanced classification
3. LOOCV is best for very small datasets despite computational cost
4. Bootstrapping enables confidence intervals for metrics
5. Random search is often more efficient than grid search
6. Nested CV prevents optimistic bias in hyperparameter tuning
7. Always report mean ± standard deviation from CV
