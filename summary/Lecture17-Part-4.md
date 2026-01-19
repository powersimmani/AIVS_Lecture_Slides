# Lecture 17 - Part 4: Basic Anomaly Detection

## Overview
This part covers anomaly detection fundamentals and basic methods for identifying outliers in data.

## Key Topics

### 1. Anomaly Detection Overview
**Definition**: Finding data points that differ significantly from the majority

**Types of Anomalies**:
- **Point anomalies**: Individual outliers
- **Contextual anomalies**: Anomalous in specific context
- **Collective anomalies**: Group of points anomalous together

**Applications**:
- Fraud detection
- Network intrusion
- Manufacturing defects
- Medical diagnosis

### 2. Statistical Methods
**Z-Score**:
```python
z = (x - μ) / σ
anomaly if |z| > threshold  # typically 2 or 3
```
- Assumes normal distribution
- Simple and fast
- Sensitive to outliers in training

**IQR (Interquartile Range)**:
```python
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```
- Robust to outliers
- Works for skewed distributions

**Mahalanobis Distance**:
```
D = √((x-μ)ᵀ Σ⁻¹ (x-μ))
```
- Accounts for correlations
- Multivariate generalization

### 3. Isolation Forest
**Key Idea**: Anomalies are easier to isolate

**Algorithm**:
1. Randomly select feature
2. Randomly select split value
3. Recursively partition data
4. Anomaly score = average path length

**Intuition**:
- Normal points: Deep in tree (many splits needed)
- Anomalies: Shallow (isolated quickly)

```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1, random_state=42)
predictions = clf.fit_predict(X)
# -1 = anomaly, 1 = normal
```

### 4. Other Methods Overview
**One-Class SVM**:
- Learn boundary around normal data
- Anything outside = anomaly

**Local Outlier Factor (LOF)**:
- Density-based
- Compare local density to neighbors
- Low relative density = anomaly

**Autoencoders**:
- Train to reconstruct normal data
- High reconstruction error = anomaly

### 5. Hands-on with scikit-learn
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Isolation Forest
iso = IsolationForest(contamination=0.1)
iso_pred = iso.fit_predict(X)

# LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_pred = lof.fit_predict(X)

# One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', nu=0.1)
svm_pred = oc_svm.fit_predict(X)
```

### 6. Evaluation Challenges
**The Label Problem**:
- Often no labels for anomalies
- Or very few labeled anomalies

**Metrics When Labels Available**:
- Precision, Recall, F1
- AUC-ROC
- Precision @ k

**When No Labels**:
- Domain expert review
- Business metrics
- Stability analysis

## Important Takeaways
1. Anomalies are data points that differ from majority
2. Statistical methods work for simple cases
3. Isolation Forest uses path length intuition
4. LOF detects density-based outliers
5. Choose method based on data characteristics
6. Evaluation is challenging without labels

