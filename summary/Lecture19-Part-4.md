# Lecture 19 - Part 4: Model-Agnostic Methods

## Overview
This part covers model-agnostic explanation methods including surrogate models, LIME, and anchor explanations.

## Key Topics

### 1. Surrogate Models
**Concept**: Train interpretable model to mimic black box

**Process**:
1. Get predictions from black box model
2. Train interpretable model (decision tree, linear) on predictions
3. Interpret the surrogate

**Global Surrogate**: Approximate entire model
**Local Surrogate**: Approximate around specific point (LIME)

### 2. Introduction to LIME
**Local Interpretable Model-agnostic Explanations**

**Process**:
1. Generate perturbed samples around instance
2. Get black box predictions for perturbations
3. Weight by proximity to original instance
4. Fit linear model on weighted samples
5. Linear coefficients = local explanation

**Formula**:
```
explanation = argmin_g L(f, g, πₓ) + Ω(g)
```
- f: Black box model
- g: Interpretable model
- πₓ: Proximity measure
- Ω: Complexity penalty

### 3. LIME for Different Data Types
**Tabular Data**: Perturb feature values
**Text**: Remove words
**Images**: Superpixel perturbations

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train, feature_names=names)
explanation = explainer.explain_instance(x, model.predict_proba)
explanation.show_in_notebook()
```

### 4. LIME Advanced Topics
**Perturbation Strategy**:
- Continuous: Normal around original
- Categorical: Sample from training distribution
- Text: Word removal
- Images: Superpixel on/off

**Weighting Function**:
```
πₓ(z) = exp(-D(x,z)² / σ²)
```
- Closer samples weighted more

**Model Selection**:
- Ridge regression common
- Decision stump for simpler explanations

### 5. Anchor Explanations
**Concept**: Find sufficient conditions for prediction

**Format**: IF (conditions) THEN prediction WITH high precision

**Example**:
```
IF age > 30 AND income > 50k
THEN approved (95% precision)
```

**Benefits**:
- **Coverage**: How many instances covered
- **Precision**: How accurate within anchor
- Easy to understand rules

### 6. Practical Guidelines and Best Practices
**Choosing Method**:
| Situation | Recommended |
|-----------|-------------|
| Single prediction | LIME, SHAP |
| Feature effects | PDP, ALE |
| Feature importance | Permutation |
| Rule-based | Anchors |
| Deep learning | Gradient methods, SHAP |

**Validation**:
- Check explanations make sense
- Compare multiple methods
- Domain expert review

## Important Takeaways
1. LIME provides local linear explanations
2. Perturbation-based methods are model-agnostic
3. Surrogate models approximate complex models
4. Anchors provide rule-based explanations
5. Different methods suit different needs
6. Always validate explanations with domain knowledge

