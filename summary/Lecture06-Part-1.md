# Lecture 06 - Part 1: Importance and Fundamentals of Evaluation

## Overview
This part establishes why model evaluation is critical in machine learning and covers fundamental concepts including data splitting, overfitting, bias-variance tradeoff, and data leakage prevention.

## Key Topics

### 1. Why is Evaluation Important?
- **Key Reasons**:
  - Measure model performance objectively
  - Compare different models fairly
  - Detect overfitting and underfitting
  - Guide model selection and improvement
- **Without Proper Evaluation**:
  - Overconfident predictions
  - Poor generalization
  - Deployment failures
- **Goal**: Estimate how model will perform on unseen data

### 2. Train vs Validation vs Test
- **Training Set** (~60-80%):
  - Used to train/fit the model
  - Model sees and learns from this data
- **Validation Set** (~10-20%):
  - Used for hyperparameter tuning
  - Used for model selection
  - Can be used multiple times during development
- **Test Set** (~10-20%):
  - Final evaluation only
  - Used ONCE at the end
  - Provides unbiased performance estimate
- **Key Rule**: Never use test set for any decisions during development

### 3. Overfitting and Underfitting
- **Underfitting** (High Bias):
  - Model too simple for data
  - High error on both train and test
  - Signs: Training loss doesn't decrease
  - Solution: Increase model capacity
- **Overfitting** (High Variance):
  - Model memorizes training data
  - Low train error, high test error
  - Signs: Large gap between train/test performance
  - Solution: Regularization, more data, simpler model
- **Optimal Fit**: Low error on both train and test

### 4. Bias-Variance Tradeoff
- **Bias**: Error from wrong assumptions
  - High bias → underfitting
  - Model can't capture patterns
- **Variance**: Error from sensitivity to training data
  - High variance → overfitting
  - Model captures noise
- **Total Error** = Bias² + Variance + Irreducible Error
- **Tradeoff**: Decreasing bias often increases variance and vice versa
- **Sweet Spot**: Minimize total error, not just one component

### 5. Data Splitting Strategies
- **Random Split**: Simple but may not preserve distributions
- **Common Ratios**:
  - 80/20 (train/test)
  - 70/15/15 (train/val/test)
  - 60/20/20 (train/val/test)
- **Considerations**:
  - Dataset size
  - Class distribution
  - Data dependencies
- **Implementation**:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
  ```

### 6. Stratified Sampling
- **Problem**: Random split may create imbalanced subsets
- **Solution**: Preserve class proportions in each split
- **When to Use**:
  - Imbalanced classification
  - Small datasets
  - Rare classes
- **Implementation**:
  ```python
  X_train, X_test = train_test_split(X, y, stratify=y, test_size=0.2)
  ```
- **Result**: Each subset has same class distribution as original

### 7. Time Series Data Splitting
- **Problem**: Random splitting violates temporal ordering
- **Future Information Leakage**: Training on future to predict past
- **Correct Approaches**:
  - **Walk-forward**: Train on past, predict next period
  - **Expanding window**: Grow training set over time
  - **Rolling window**: Fixed-size moving window
- **Key Rule**: Always train on past, test on future
- **Never**: Use random shuffle for time series

### 8. Preventing Data Leakage
- **Data Leakage**: Training data contains information from test
- **Common Causes**:
  - Preprocessing before split (e.g., normalization on full data)
  - Features derived from target
  - Time-based leakage
  - Duplicates across splits
- **Prevention**:
  - Split first, preprocess second
  - Use pipelines to ensure proper flow
  - Check for data duplicates
  - Be cautious with time-related features
- **Detection**: Suspiciously high test performance

## Important Takeaways
1. Never use test data during model development
2. Overfitting means memorizing; underfitting means not learning
3. Bias-variance tradeoff is fundamental to model selection
4. Stratified sampling preserves class distributions
5. Time series requires temporal ordering in splits
6. Data leakage invalidates evaluation results
7. Proper splitting is the foundation of reliable evaluation
