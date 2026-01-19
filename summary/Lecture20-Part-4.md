# Lecture 20 - Part 4: Advanced Deep Learning XAI Techniques

## Overview
This part covers XAI methods specific to deep learning including gradient-based methods, CAM-family, and concept-based explanations.

## Key Topics

### 1. Attention Mechanisms as Explanations
**Concept**: Use attention weights to explain

**Attention Visualization**:
- Which input parts model attends to
- For transformers: attention maps

**Limitations**:
- Attention ≠ explanation always
- Multiple heads complicate interpretation
- Attention may not reflect true importance

### 2. Gradient-based Methods
**Vanilla Gradient (Saliency Maps)**:
```
saliency = |∂f(x)/∂x|
```
- Magnitude of input gradients
- Shows input sensitivity

**Gradient × Input**:
```
attribution = x × ∂f(x)/∂x
```
- Scale by input values
- Often clearer results

**SmoothGrad**:
```
SmoothGrad = (1/n) Σ ∂f(x + noise)/∂x
```
- Average over noisy inputs
- Reduces noise in saliency

### 3. Integrated Gradients
**Formula**:
```
IG_i = (x_i - x'_i) × ∫₀¹ (∂f(x' + α(x-x'))/∂x_i) dα
```
- x': Baseline (e.g., black image, zero embedding)
- Accumulate gradients along path from baseline

**Properties**:
- Satisfies completeness axiom
- Attributions sum to prediction difference

**Implementation**:
```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(input, baseline, target=class_idx)
```

### 4. CAM-family Methods
**CAM (Class Activation Mapping)**:
- Requires global average pooling
- Limited architecture compatibility

**Grad-CAM**: Generalized CAM
```
L_c = ReLU(Σₖ αₖᶜ Aᵏ)
αₖᶜ = (1/Z) Σᵢⱼ (∂yᶜ/∂Aᵏᵢⱼ)
```
- Works with any CNN
- Uses gradients to weight feature maps

**Grad-CAM++**: Improved weighting
**Score-CAM**: Gradient-free variant

```python
from pytorch_grad_cam import GradCAM

cam = GradCAM(model=model, target_layers=[layer])
grayscale_cam = cam(input_tensor=input_tensor)
```

### 5. Concept-based Explanations
**TCAV (Testing with Concept Activation Vectors)**:
- Define human concepts (e.g., "stripes", "wheels")
- Find direction in activation space
- Measure concept influence on prediction

**Process**:
1. Collect concept examples
2. Train linear classifier in activation space
3. Get concept direction (CAV)
4. Measure sensitivity to concept

**Benefits**: Human-understandable concepts, not just pixels

### 6. Future of XAI
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
1. Gradient methods provide pixel-level attribution
2. Integrated gradients have theoretical guarantees
3. Grad-CAM shows where CNN "looks"
4. Concept-based methods provide higher-level explanations
5. No single method is perfect - use multiple
6. XAI research continues to evolve rapidly

