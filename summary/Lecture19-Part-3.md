# Lecture 19 - Part 3: Feature Importance Methodologies

## Overview
This part covers methods for measuring feature importance including permutation importance, PDP, ICE, and ALE.

## Key Topics

### 1. Permutation Importance
**Concept**: Measure importance by shuffling feature values

**Process**:
1. Train model, measure baseline performance
2. Randomly shuffle one feature
3. Measure performance drop
4. Drop = feature importance
5. Repeat for all features

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
importance = result.importances_mean
```

**Benefits**: Model-agnostic, considers feature interactions

### 2. Drop-Column Importance
**Concept**: Retrain without each feature

**Process**:
1. Train model on all features
2. For each feature: remove it, retrain, measure drop
3. Drop = importance

**Disadvantage**: Computationally expensive (retrain for each feature)

### 3. Partial Dependence Plots (PDP)
**Goal**: Show marginal effect of feature on prediction

**Formula**:
```
PD(xₛ) = E_xc[f(xₛ, xc)] ≈ (1/n) Σ f(xₛ, xc^(i))
```
- Average prediction over all other feature values

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(model, X, features=[0, 1])
plt.show()
```

**Interpretation**: How prediction changes as feature varies

### 4. Individual Conditional Expectation (ICE)
**Concept**: PDP for individual instances

**Difference from PDP**:
- PDP: Average across all instances
- ICE: One line per instance

**Benefits**: Shows heterogeneity, interactions

```python
PartialDependenceDisplay.from_estimator(
    model, X, features=[0], kind='both'  # PDP + ICE
)
```

### 5. Accumulated Local Effects (ALE)
**Problem with PDP**: Assumes feature independence

**Solution**: ALE uses conditional distributions

**Process**: Compute local effects, accumulate

**Benefits**:
- Handles correlated features
- Faster than PDP
- More accurate marginal effects

### 6. Feature Interaction Analysis
**H-statistic**: Measure interaction strength
```
H²_jk = Σ[PD_jk - PD_j - PD_k]² / Σ[PD_jk]²
```

**SHAP Interaction Values**: Decompose interactions
- Second-order Shapley values
- Pairwise feature effects

## Important Takeaways
1. Permutation importance is model-agnostic and considers interactions
2. PDP shows marginal feature effects
3. ICE reveals individual-level heterogeneity
4. ALE handles correlated features better than PDP
5. Interaction analysis reveals feature relationships
6. Choose method based on data and analysis goals

