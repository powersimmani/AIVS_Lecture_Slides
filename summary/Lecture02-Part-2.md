# Lecture 02 - Part 2: Mastering Basic Charts

## Overview
This part covers essential chart types for different data scenarios, from single-variable distributions to multi-dimensional data and geographic visualizations.

## Key Topics

### 1. 1D Data - Histogram and KDE
- **Histogram**:
  - Displays distribution of continuous data
  - Bin width affects interpretation
  - Shows frequency/count in each bin
  - Good for: Identifying distribution shape, outliers, modality
- **KDE (Kernel Density Estimation)**:
  - Smooth continuous estimate of distribution
  - Bandwidth controls smoothness
  - No binning artifacts
  - Better for: Comparing distributions, probability estimation
- **Combined Use**: Overlay KDE on histogram for best insights
- **Considerations**: Choose appropriate number of bins; Sturges' rule, Scott's rule

### 2. 2D Relationships - Scatter Plot and Bubble Chart
- **Scatter Plot**:
  - Shows relationship between two continuous variables
  - Reveals: Correlation, clusters, outliers
  - Add: Trend lines, confidence intervals
  - Overplotting solutions: Transparency, jittering, 2D histograms
- **Bubble Chart**:
  - Scatter plot with size as third dimension
  - Size encodes additional quantitative variable
  - Color can add fourth dimension (categorical)
  - Caution: Area perception is less accurate than position
- **Best Practices**: Label axes, include zero when appropriate, add regression line

### 3. Categorical Data - Bar Chart and Pie Chart
- **Bar Chart**:
  - Best for comparing categories
  - Horizontal bars for many categories or long labels
  - Grouped bars for comparing across groups
  - Stacked bars for part-to-whole relationships
  - Always start y-axis at zero
- **Pie Chart**:
  - Shows part-to-whole relationships
  - Limited to ~5-7 slices maximum
  - Difficult to compare similar-sized slices
  - **Alternative**: Bar chart or treemap often better
- **When to Use What**: Bar for comparison, pie only for simple proportions

### 4. Distribution Comparison - Boxplot and Violin Plot
- **Boxplot**:
  - Five-number summary: min, Q1, median, Q3, max
  - Shows outliers as individual points
  - Good for comparing multiple groups
  - Compact representation
- **Violin Plot**:
  - Combines boxplot with KDE
  - Shows full distribution shape
  - Better for multimodal distributions
  - Takes more space than boxplot
- **Variations**: Swarm plots, strip plots, rain cloud plots
- **Use Cases**: Boxplot for quick comparison, violin for detailed distribution shape

### 5. Time Series - Line Graph and Area Chart
- **Line Graph**:
  - Shows trends over time
  - Connect data points to show continuity
  - Multiple lines for comparison
  - Highlight: Trends, seasonality, anomalies
- **Area Chart**:
  - Line graph with filled area below
  - Stacked area for part-to-whole over time
  - Emphasizes magnitude
  - Caution: Can obscure overlapping series
- **Best Practices**:
  - Time on x-axis (left to right)
  - Consistent time intervals
  - Highlight important events
  - Use aspect ratio that reveals patterns (banking to 45°)

### 6. Correlation - Heatmap and Correlation Matrix
- **Correlation Matrix**:
  - Shows pairwise correlations between variables
  - Values range from -1 to +1
  - Essential for feature selection in ML
- **Heatmap**:
  - Color-encodes values in a matrix
  - Uses: Correlation, confusion matrices, temporal patterns
  - Color scales: Sequential (one direction), diverging (positive/negative)
- **Best Practices**:
  - Use diverging colormap centered at zero
  - Cluster similar variables
  - Annotate cells with values
  - Consider hierarchical clustering for ordering

### 7. Multidimensional - Parallel Coordinates and Radar Chart
- **Parallel Coordinates**:
  - Each variable is a vertical axis
  - Lines connect values for each observation
  - Good for: High-dimensional data, identifying clusters
  - Interactive filtering enhances utility
- **Radar Chart (Spider Chart)**:
  - Variables on radial axes
  - Shows profile/pattern of each observation
  - Good for: Comparing few items across many variables
  - Limitation: Axis order affects perception
- **Alternatives**: Scatterplot matrix (SPLOM), dimensionality reduction plots

### 8. Geographic Data - Choropleth and Bubble Map
- **Choropleth Map**:
  - Colors regions based on data values
  - Good for: Regional comparisons, density data
  - Caution: Large areas visually dominate
  - Consider: Per capita normalization
- **Bubble Map**:
  - Circles placed at geographic locations
  - Size encodes data value
  - Better for point data (cities, events)
  - Avoids area-bias of choropleth
- **Tools**: Folium, Plotly, Mapbox, GeoPandas
- **Projections**: Choose appropriate map projection for region

## Important Takeaways
1. Match chart type to data type and analysis goal
2. Histograms and boxplots are essential for understanding distributions
3. Scatter plots reveal relationships; add trend lines for clarity
4. Time series visualization requires appropriate aspect ratios
5. Correlation heatmaps are crucial for feature selection in ML
6. Geographic visualizations require careful consideration of projections and normalization
7. Always consider your audience and message when choosing chart types
