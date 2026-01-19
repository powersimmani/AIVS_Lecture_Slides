# Lecture 20 - Part 1: SHAP Theory and Fundamentals

## Overview
This part covers SHAP (SHapley Additive exPlanations), its game-theoretic foundations, and how it provides unified feature attribution.

## Key Topics

### 1. Game Theory-Based Explainability
**Idea**: Features are "players" contributing to prediction
**From**: Cooperative game theory (Shapley values)
**Goal**: Fairly distribute prediction among features

### 2. Cooperative Game Theory Basics
**Setup**:
- Players: Features
- Coalition: Subset of features
- Payoff: Model prediction with coalition

**Question**: How to fairly attribute total payoff to players?

### 3. Shapley Values Mathematical Definition
**Formula**:
```
φᵢ = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)! / |N|!] × [v(S∪{i}) - v(S)]
```

- φᵢ: Shapley value for feature i
- S: Coalition without feature i
- v(S): Value function (prediction with features S)
- N: All features

**Interpretation**: Average marginal contribution across all orderings

### 4. Core Ideas of SHAP
**SHAP = Shapley + Additive**:
```
f(x) = φ₀ + Σᵢ φᵢ
```
- φ₀: Base value (average prediction)
- φᵢ: SHAP value for feature i
- Sum equals prediction

**Key Properties**:
- **Efficiency**: SHAP values sum to prediction - base
- **Symmetry**: Equal contribution → equal value
- **Dummy**: No contribution → zero value
- **Additivity**: Consistent across models

### 5. SHAP vs LIME Comparison
| Aspect | SHAP | LIME |
|--------|------|------|
| Foundation | Game theory | Local linear approx |
| Consistency | Guaranteed | Not guaranteed |
| Computation | Often expensive | Fast |
| Global view | Natural aggregation | No |

### 6. Interpreting SHAP Values
**Positive SHAP**: Feature pushes prediction higher
**Negative SHAP**: Feature pushes prediction lower
**Magnitude**: Importance of feature for this prediction

**Example**:
```
Base value: $200,000 (average house price)
Bedrooms = 4: +$30,000
Location = downtown: +$50,000
Age = 50 years: -$20,000
Prediction: $260,000
```

### 7. Mathematical Properties of SHAP
**Local Accuracy**:
```
f(x) = g(x') = φ₀ + Σᵢ φᵢx'ᵢ
```

**Missingness**: Missing features have zero attribution

**Consistency**: If feature contributes more, SHAP value increases

**These properties make SHAP theoretically grounded**

### 8. Hands-on: First SHAP Analysis
```python
import shap

# Create explainer
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values
shap_values = explainer(X_test)

# Visualize for single prediction
shap.plots.waterfall(shap_values[0])

# Global summary
shap.plots.summary(shap_values, X_test)
```

### 9. SHAP Value Computation Challenge
**Exact Computation**: O(2^n) - exponential in features
**Solutions**:
- KernelSHAP: Sampling-based approximation
- TreeSHAP: Exact for tree models O(TLD²)
- DeepSHAP: For deep learning
- LinearSHAP: Exact for linear models

## Important Takeaways
1. SHAP is based on Shapley values from game theory
2. SHAP values satisfy important theoretical properties
3. Values sum to difference between prediction and base value
4. Positive values increase prediction, negative decrease
5. SHAP provides both local and global interpretations
6. Multiple efficient algorithms for different model types
7. SHAP is the gold standard for feature attribution

