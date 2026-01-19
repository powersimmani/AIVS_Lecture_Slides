# Lecture 06 - Part 3: Classification Evaluation Metrics

## Overview
This part covers evaluation metrics for classification tasks, from basic accuracy to sophisticated threshold-independent metrics like ROC and PR curves.

## Key Topics

### 1. Confusion Matrix
- **Structure** (Binary Classification):
  |                | Predicted Positive | Predicted Negative |
  |----------------|-------------------|-------------------|
  | Actual Positive| TP (True Positive) | FN (False Negative)|
  | Actual Negative| FP (False Positive)| TN (True Negative) |
- **Key Values**:
  - **TP**: Correctly predicted positives
  - **TN**: Correctly predicted negatives
  - **FP**: Type I error (false alarm)
  - **FN**: Type II error (missed detection)
- **Foundation**: All classification metrics derived from this

### 2. Accuracy and Its Limitations
- **Accuracy**:
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```
  - Proportion of correct predictions
  - Intuitive and simple
- **Limitations**:
  - Misleading with imbalanced data
  - Example: 99% negative class → predicting all negative = 99% accuracy
  - Doesn't distinguish between error types
- **When NOT to Use**: Imbalanced datasets, different error costs

### 3. Precision and Recall
- **Precision** (Positive Predictive Value):
  ```
  Precision = TP / (TP + FP)
  ```
  - "Of those predicted positive, how many are actually positive?"
  - High precision → few false positives
  - Important when FP is costly (spam filtering)
- **Recall** (Sensitivity, True Positive Rate):
  ```
  Recall = TP / (TP + FN)
  ```
  - "Of all actual positives, how many did we catch?"
  - High recall → few false negatives
  - Important when FN is costly (disease detection)
- **Tradeoff**: Increasing one typically decreases the other

### 4. F1 Score and Fβ
- **F1 Score** (Harmonic Mean):
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```
  - Balances precision and recall
  - Range: [0, 1]
  - F1 = 1: Perfect precision and recall
- **Fβ Score** (Weighted):
  ```
  Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
  ```
  - β > 1: Emphasize recall
  - β < 1: Emphasize precision
  - F0.5: Precision twice as important
  - F2: Recall twice as important

### 5. ROC Curve and AUC
- **ROC Curve**:
  - X-axis: False Positive Rate (FPR) = FP/(FP+TN)
  - Y-axis: True Positive Rate (TPR) = TP/(TP+FN) = Recall
  - Each point is a different threshold
- **Key Points**:
  - (0, 0): Threshold = 1 (predict all negative)
  - (1, 1): Threshold = 0 (predict all positive)
  - Diagonal: Random classifier
- **AUC (Area Under Curve)**:
  - Range: [0, 1]
  - AUC = 0.5: Random
  - AUC = 1.0: Perfect
  - Interpretation: Probability that random positive ranks higher than random negative
- **Best for**: Balanced datasets, overall discrimination ability

### 6. Precision-Recall Curve
- **PR Curve**:
  - X-axis: Recall
  - Y-axis: Precision
  - Each point is a different threshold
- **AP (Average Precision)**:
  - Area under PR curve
  - Summary metric
- **When to Use Over ROC**:
  - Imbalanced datasets
  - When positive class is more important
  - When true negatives are abundant
- **Baseline**: Horizontal line at positive class proportion

### 7. Multi-class Metrics
- **Extension Strategies**:
  - **Macro Average**: Compute metric for each class, average
    - Treats all classes equally
  - **Micro Average**: Aggregate TP, FP, FN globally
    - Dominated by frequent classes
  - **Weighted Average**: Weight by class frequency
- **Confusion Matrix**: K×K matrix for K classes
- **One-vs-Rest**: Treat each class as binary problem
- **Example**:
  ```python
  from sklearn.metrics import classification_report
  print(classification_report(y_true, y_pred))
  ```

## Important Takeaways
1. Confusion matrix is the foundation of classification metrics
2. Accuracy is misleading for imbalanced data
3. Precision-recall tradeoff must be considered for the application
4. F1 score balances precision and recall
5. ROC AUC measures overall discrimination ability
6. PR curves are better for imbalanced datasets
7. Multi-class: choose averaging method based on goal
