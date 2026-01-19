# Lecture 20 - Part 2: SHAP Implementation and Deep Learning XAI

## Overview
This part covers SHAP implementation methods, visualization techniques, and XAI methods for deep learning models.

## Key Topics

### 1. KernelSHAP
**Approach**: Weighted linear regression approximation

**Process**:
1. Sample coalitions (feature subsets)
2. Get model predictions for each
3. Weight by Shapley kernel
4. Solve weighted regression

**Complexity**: O(2^n) → O(k) with sampling

```python
explainer = shap.KernelExplainer(model.predict, X_background)
shap_values = explainer.shap_values(X_test)
```

### 2. TreeSHAP
**Exact SHAP for tree models**: Random Forest, XGBoost, LightGBM

**Benefits**:
- Polynomial time O(TLD²)
- Exact values (no approximation)
- Very fast for tree ensembles

```python
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)
```

### 3. DeepSHAP
**Combines DeepLIFT with Shapley values**

**Process**: Backpropagate through network, attributing to inputs

```python
explainer = shap.DeepExplainer(model, X_background)
shap_values = explainer.shap_values(X_test)
```

### 4. SHAP Visualizations
**Waterfall Plot**: Single prediction breakdown
```python
shap.plots.waterfall(shap_values[0])
```

**Force Plot**: Horizontal breakdown
```python
shap.plots.force(shap_values[0])
```

**Summary Plot**: Global feature importance + distributions
```python
shap.plots.summary(shap_values, X)
```

**Dependence Plot**: Feature effect with interactions
```python
shap.plots.scatter(shap_values[:, "feature_name"])
```

### 5. SHAP Interaction Values
**Second-order Shapley values**: Measure pairwise interactions

```python
shap_interaction = explainer.shap_interaction_values(X)
# Shape: (n_samples, n_features, n_features)
```

**Interpretation**: How features jointly affect prediction

### 6. Deep Learning XAI Methods
**Gradient-based**:
- Saliency Maps: |∂y/∂x|
- Integrated Gradients: Accumulate gradients along path
- SmoothGrad: Average over noisy inputs

**Attention-based**:
- Attention weights as explanation
- Limitation: Attention ≠ explanation always

**Perturbation-based**:
- Occlusion: Mask regions, observe change
- LIME for images

### 7. Gradient-based Methods
**Vanilla Gradient**:
```
saliency = |∂f(x)/∂x|
```

**Integrated Gradients**:
```
IG_i = (x_i - x'_i) × ∫₀¹ (∂f(x' + α(x-x'))/∂x_i) dα
```
- x': Baseline (e.g., black image)
- Satisfies completeness axiom

### 8. CAM-family Methods
**CAM (Class Activation Mapping)**:
- Requires global average pooling
- Limited architecture

**Grad-CAM**: Generalized CAM
```
L_c = ReLU(Σₖ αₖᶜ Aᵏ)
αₖᶜ = (1/Z) Σᵢⱼ (∂yᶜ/∂Aᵏᵢⱼ)
```

**Grad-CAM++**: Improved weighting
**Score-CAM**: Gradient-free variant

```python
from pytorch_grad_cam import GradCAM

cam = GradCAM(model=model, target_layers=[layer])
grayscale_cam = cam(input_tensor=input_tensor)
```

### 9. Concept-based Explanations
**TCAV (Testing with Concept Activation Vectors)**:
- Define human concepts (e.g., "stripes")
- Find direction in activation space
- Measure concept influence on prediction

**Benefits**: Human-understandable concepts, not just pixels

### 10. Future of XAI
**Challenges**:
- Faithfulness: Do explanations reflect true reasoning?
- Human studies: Are explanations actually helpful?
- Scalability: Explaining billion-parameter models

**Trends**:
- Built-in interpretability
- Concept-based reasoning
- Multi-modal explanations
- Causality integration

## Important Takeaways
1. TreeSHAP is exact and fast for tree models
2. DeepSHAP adapts SHAP for neural networks
3. Multiple visualization types for different insights
4. Interaction values reveal feature relationships
5. Gradient methods provide pixel-level attribution
6. Grad-CAM shows where CNN "looks"
7. Concept-based methods provide higher-level explanations
8. XAI research continues to evolve rapidly

