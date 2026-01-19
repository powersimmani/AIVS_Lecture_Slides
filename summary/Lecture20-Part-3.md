# Lecture 20 - Part 3: SHAP Visualization and Analysis

## Overview
This part covers SHAP visualization techniques for interpreting model predictions at local and global levels.

## Key Topics

### 1. Waterfall Plot
**Purpose**: Explain single prediction breakdown

**Components**:
- Base value (average prediction)
- Each feature's contribution
- Final prediction

```python
shap.plots.waterfall(shap_values[0])
```

**Reading the Plot**:
- Red bars: Push prediction higher
- Blue bars: Push prediction lower
- Starting from base, add/subtract to get final

### 2. Force Plot and Decision Plot
**Force Plot**: Horizontal breakdown
```python
shap.plots.force(shap_values[0])
```
- Single instance visualization
- Interactive in notebooks

**Decision Plot**: Track cumulative effect
```python
shap.plots.decision(shap_values)
```
- Shows path from base to prediction
- Good for many instances

### 3. Summary Plot and Dependence Plot
**Summary Plot**: Global feature importance + distributions
```python
shap.plots.summary(shap_values, X)
```
- Beeswarm: Each dot is an instance
- X-axis: SHAP value
- Color: Feature value (high/low)
- Shows importance AND effect direction

**Bar Summary**:
```python
shap.plots.bar(shap_values)
```
- Mean absolute SHAP value per feature
- Overall feature importance

### 4. Dependence Plot
**Purpose**: Feature effect with interactions
```python
shap.plots.scatter(shap_values[:, "feature_name"])
```

**Components**:
- X-axis: Feature value
- Y-axis: SHAP value
- Color: Interaction feature (auto-detected)

**Interpretation**:
- Shape shows main effect
- Color variation shows interactions

### 5. SHAP for Time Series
**Challenges**:
- Temporal dependencies
- Sequential features
- Lagged effects

**Approaches**:
- Treat time steps as features
- Rolling window features
- Deep SHAP for RNNs

### 6. SHAP for Text and Images
**Text**:
```python
explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(texts)
shap.plots.text(shap_values[0])
```
- Word-level contributions
- Highlight important words

**Images**:
```python
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(images)
shap.image_plot(shap_values, images)
```
- Pixel-level importance
- Overlay on original image

## Important Takeaways
1. Waterfall plots explain individual predictions
2. Summary plots show global patterns
3. Dependence plots reveal feature effects and interactions
4. Multiple visualization types for different insights
5. SHAP extends to text and images
6. Combine local and global views for full understanding

