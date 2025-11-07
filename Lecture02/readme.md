# Lecture 2: Data Visualization

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com

This lecture covers the fundamentals of data visualization, essential chart types, and advanced visualization techniques for machine learning applications. Learn both theoretical principles of effective visualization and practical implementation strategies.

---

## 🎯 Learning Objectives

1. **Applying fundamental visualization principles** - Visual encoding hierarchy, Gestalt principles, Data-ink ratio
2. **Selecting appropriate chart types** - Charts for 1D, 2D, categorical, temporal, geographic data
3. **Designing accessible visualizations** - Color theory, colorblindness considerations, typography
4. **Visualizing ML workflows** - EDA, feature analysis, model performance evaluation
5. **Communicating insights effectively** - Dashboard design, data storytelling

---

## 📚 Key Topics

**Visualization fundamentals**
- **Visual encoding effectiveness ranking**:
  1. Position (100%) - most accurate
  2. Length (85%)
  3. Angle (70%)
  4. Area (55%)
  5. Color (40%)
  6. Shape (28%)
  7. Texture (18%)
- **Gestalt principles**: Proximity, Similarity, Continuity, Closure, Figure-Ground
- **Data-Ink Ratio**: Remove unnecessary elements, clean minimal design
- **Color theory**: Colorblind-safe palettes, avoid red-green combinations
- **Good vs Bad visualization**:
  - ❌ 3D pie charts, dual y-axes, truncated axes, excessive colors
  - ✅ 2D bar charts, normalized charts, zero-baseline axes, intentional color use

**Chart types**
- **1D data**: Histogram (distribution), Box plot (quartiles, outliers), Violin plot (density)
- **2D data**: Scatter plot (correlation, clusters), Line chart (trends), Heatmap (correlation matrix)
- **Categorical data**: Bar chart (comparison), Pie chart (proportions, use sparingly), Stacked bar (parts and whole)
- **Time series**: Time series plot (trends, seasonality), Moving average (noise reduction), Area chart (cumulative)
- **Geographic data**: Choropleth map (regional values), Point map (location and size), Heat map (density)

**ML visualization**
- **Exploratory Data Analysis (EDA)**: Distribution, outliers, missing values, correlation analysis
- **Feature engineering visualization**: Feature importance (bar chart), correlation (heatmap), distribution comparison (violin plot)
- **Model training monitoring**: Loss curves (train/val), Metric tracking (accuracy, F1), Learning rate schedule
- **Model performance evaluation**: Confusion matrix, ROC curve (TPR vs FPR), Precision-Recall curve, Residual plot
- **Model interpretation**: Feature importance, SHAP values, Attention weights, Activation maps

**Advanced visualization**
- **Dashboard design principles**: Clear hierarchy, consistent layout, appropriate chart selection, interactive elements
- **Data storytelling**: Set context, highlight key insights, clear conclusions, call to action
- **Interactive visualization**: Zoom, Pan, Filter, Tooltips, Linked views
- **Tools**: Matplotlib (basic plotting), Seaborn (statistical), Plotly (interactive), Tableau/Power BI (dashboards)

---

## 💡 Key Concepts

- Understand and apply visual encoding hierarchy
- Leverage Gestalt principles for visual perception
- Maximize data-ink ratio
- Select chart types matching data structure
- Bar charts for comparison, line charts for trends
- Use pie charts sparingly
- Understand data through EDA
- Monitor training process
- Clearly evaluate model performance
- Enhance interpretability
- Provide integrated view through dashboards
- Communicate insights through storytelling

---

## 🛠️ Prerequisites

- Basic Python programming
- Understanding of previous lecture content
- Basic machine learning concepts

---

## 📖 Additional Resources

For detailed code examples, practice materials, and slides, please refer to the original lecture files.
Lecture materials: HTML-based interactive slides provided
