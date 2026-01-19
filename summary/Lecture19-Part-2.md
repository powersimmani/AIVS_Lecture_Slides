# Lecture 19 - Part 2: Feature Importance and Model-Agnostic Methods

## Overview
This part covers feature importance methods and model-agnostic explanation techniques including permutation importance, PDP, and LIME.

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
PD(x_s) = E_x_c[f(x_s, x_c)] ≈ (1/n) Σ f(x_s, x_c^(i))
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

### 7. Surrogate Models
**Concept**: Train interpretable model to mimic black box

**Process**:
1. Get predictions from black box model
2. Train interpretable model (decision tree, linear) on predictions
3. Interpret the surrogate

**Global Surrogate**: Approximate entire model
**Local Surrogate**: Approximate around specific point (LIME)

### 8. Introduction to LIME
**Local Interpretable Model-agnostic Explanations**

**Process**:
1. Generate perturbed samples around instance
2. Get black box predictions for perturbations
3. Weight by proximity to original instance
4. Fit linear model on weighted samples
5. Linear coefficients = local explanation

**Formula**:
```
explanation = argmin_g L(f, g, π_x) + Ω(g)
```
- f: Black box model
- g: Interpretable model
- π_x: Proximity measure
- Ω: Complexity penalty

### 9. LIME for Different Data Types
**Tabular Data**: Perturb feature values
**Text**: Remove words
**Images**: Superpixel perturbations

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train, feature_names=names)
explanation = explainer.explain_instance(x, model.predict_proba)
explanation.show_in_notebook()
```

### 10. Anchor Explanations
**Concept**: Find sufficient conditions for prediction

**Format**: IF (conditions) THEN prediction WITH high precision

**Benefits**:
- Coverage: How many instances covered
- Precision: How accurate within anchor
- Easy to understand rules

## Important Takeaways
1. Permutation importance is model-agnostic and considers interactions
2. PDP shows marginal feature effects
3. ICE reveals individual-level heterogeneity
4. ALE handles correlated features better than PDP
5. LIME provides local linear explanations
6. Surrogate models approximate complex models
7. Anchors provide rule-based explanations

