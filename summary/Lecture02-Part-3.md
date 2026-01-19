# Lecture 02 - Part 3: Advanced Visualization for ML

## Overview
This part covers specialized visualization techniques for machine learning workflows, including exploratory data analysis, model evaluation, interpretation methods, and dashboard design.

## Key Topics

### 1. Visualization Strategy for EDA
- **EDA Workflow**:
  1. **Overview**: Dataset shape, types, missing values
  2. **Univariate**: Distribution of each feature
  3. **Bivariate**: Relationships between pairs
  4. **Multivariate**: Complex interactions
- **Key Questions to Answer**:
  - What is the target distribution?
  - Which features have high correlation?
  - Are there outliers or anomalies?
  - Is the data balanced?
- **Tools**: pandas-profiling, sweetviz, D-Tale for automated EDA

### 2. Feature Distribution and Outlier Detection
- **Distribution Analysis**:
  - Histograms/KDE for continuous features
  - Bar charts for categorical features
  - Check for: Skewness, normality, multimodality
- **Outlier Detection Visualization**:
  - Boxplots: IQR-based outliers
  - Scatter plots: Visual identification
  - Z-score plots: Statistical threshold
- **Handling Outliers**: Visualize before/after transformation
- **Class Imbalance**: Visualize target distribution, consider resampling

### 3. Visualizing Feature Interactions
- **Methods**:
  - Pair plots: All pairwise scatter plots
  - Interaction effects: Joint effects of two features
  - Contour plots: Response surface visualization
  - Conditional plots: Fix one feature, vary others
- **Key Insights**:
  - Color-encode target variable
  - Reveal discriminative patterns
  - Identify non-linear interactions
  - Feature crosses for new feature space
- **Best Practices**: Use stratification, separate by categories

### 4. Dimensionality Reduction Visualization (PCA, t-SNE, UMAP)
- **PCA (Principal Component Analysis)**:
  - Linear projection
  - Preserves global structure
  - Fast computation
  - Scree plot for variance explained
  - Interpretable components
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
  - Non-linear projection
  - Focuses on local structure
  - Perplexity parameter (5-50 typical)
  - Slow for large datasets
  - Not deterministic
- **UMAP (Uniform Manifold Approximation and Projection)**:
  - Non-linear projection
  - Preserves both local and global structure
  - Faster than t-SNE
  - Better for large datasets
- **Use Cases**: Cluster visualization, data exploration, embedding analysis

### 5. Model Performance Visualization - Learning Curves
- **Training vs Validation Curves**:
  - X-axis: Training iterations/epochs or dataset size
  - Y-axis: Loss or accuracy
- **Diagnosing Problems**:
  - **Underfitting (High Bias)**: Both curves plateau at high error
  - **Overfitting (High Variance)**: Training good, validation poor (gap)
  - **Good Fit**: Both curves converge to low error
- **Bias-Variance Tradeoff**: Visualize as function of model complexity
- **Actions**: Underfitting → more capacity; Overfitting → regularization

### 6. Classification Model Evaluation - Confusion Matrix, ROC, PR
- **Confusion Matrix**:
  - True Positives (TP), False Positives (FP)
  - False Negatives (FN), True Negatives (TN)
  - Derived metrics: Precision, Recall, F1, Accuracy
  - Heatmap visualization
- **ROC Curve (Receiver Operating Characteristic)**:
  - X-axis: False Positive Rate (FPR)
  - Y-axis: True Positive Rate (TPR/Recall)
  - AUC (Area Under Curve): Model quality metric
  - Good for: Balanced datasets, threshold selection
- **Precision-Recall Curve**:
  - X-axis: Recall
  - Y-axis: Precision
  - AP (Average Precision): Summary metric
  - Better for: Imbalanced datasets
- **When to Use**: ROC for balanced, PR for imbalanced datasets

### 7. Regression Model Evaluation - Residuals and QQ Plot
- **Residual Plots**:
  - Residuals vs Fitted values
  - Check for: Homoscedasticity, patterns, non-linearity
  - Ideal: Random scatter around zero
- **Q-Q Plot (Quantile-Quantile)**:
  - Compare residual distribution to normal distribution
  - Points on diagonal = normality
  - Deviations indicate non-normality
- **Scale-Location Plot**: Check for heteroscedasticity
- **Predicted vs Actual**: 45-degree line indicates perfect prediction
- **Metrics**: R², RMSE, MAE visualized as distributions

### 8. Model Interpretation - SHAP, LIME, Attention
- **SHAP (SHapley Additive exPlanations)**:
  - Game-theory based
  - Global and local explanations
  - Summary plot: Feature importance
  - Dependence plots: Feature effects
  - Waterfall plots: Individual predictions
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Model-agnostic
  - Local perturbation-based
  - Explains individual predictions
  - Creates interpretable local approximation
- **Attention Visualization**:
  - Built into transformer models
  - Attention heatmaps
  - Useful for NLP and Vision
  - Shows what model "focuses on"
- **Comparison**: SHAP for global+local, LIME for quick local, Attention for deep learning

### 9. Dashboard Design and Storytelling
- **Dashboard Layout Principles**:
  - KPIs at top
  - Information hierarchy (most important first)
  - Consistent visual language
  - Interactive filters
- **Data Storytelling Framework**:
  - **Setup**: Context and background
  - **Conflict**: Problem or insight discovered
  - **Resolution**: Recommendations and actions
- **Tools**: Streamlit, Dash, Tableau, Power BI
- **Best Practices**:
  - Clear titles and labels
  - Appropriate chart types
  - Consistent color scheme
  - Guide viewer attention

## Important Takeaways
1. EDA visualization is the foundation of any ML project
2. Dimensionality reduction (PCA, t-SNE, UMAP) helps visualize high-dimensional data
3. Learning curves diagnose model problems (underfitting vs overfitting)
4. Choose ROC for balanced, PR curves for imbalanced classification tasks
5. Residual analysis is essential for regression model validation
6. SHAP and LIME make black-box models interpretable
7. Effective dashboards tell a story and guide decision-making
