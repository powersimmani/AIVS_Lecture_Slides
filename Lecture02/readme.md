# Lecture 2: Data Visualization

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Total Slides:** 31  
**Duration:** Approximately 3-4 hours  
**Level:** Intermediate  
**Course Type:** Theoretical + Practical

This comprehensive lecture covers the fundamentals of data visualization, essential chart types, and advanced visualization techniques for machine learning applications. Students will learn both the theoretical principles of effective visualization and practical implementation strategies.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Apply fundamental visualization principles** including visual encoding hierarchy, Gestalt principles, and data-ink ratio to create effective visualizations
2. **Select and implement appropriate chart types** for different data structures (1D, 2D, categorical, temporal, geographic)
3. **Design accessible visualizations** considering color theory, color blindness, typography, and layout principles
4. **Visualize machine learning workflows** including EDA strategies, feature analysis, model performance evaluation, and interpretation
5. **Communicate insights effectively** through dashboard design and data storytelling techniques

---

## 📚 Lecture Structure

### **Part 1: Visualization Fundamentals** (Slides 3-11)

#### Topics Covered:

1. **Importance and Goals of Data Visualization** (Slide 4)
   - Data → Visualization → Insights → Decisions workflow
   - Transforming complex unstructured information into actionable insights
   - Visual representation as a bridge between raw data and decision-making

2. **Visual Encoding Principles** (Slide 5)
   - Effectiveness ranking from most to least accurate:
     1. Position (100% effectiveness) - most accurate for quantitative data
     2. Length (85%) - effective for comparing magnitudes
     3. Angle (70%) - useful for proportions but less accurate
     4. Area (55%) - can be misleading, use with caution
     5. Color (40%) - excellent for categories and gradients
     6. Shape (28%) - best for distinguishing categories
     7. Texture (18%) - secondary encoding, limited perceptual range

3. **Gestalt Principles and Perception** (Slide 6)
   - **Proximity:** Objects close together are perceived as grouped
   - **Similarity:** Similar elements are grouped together
   - **Continuity:** Eyes follow smooth paths naturally
   - **Closure:** Mind completes incomplete shapes
   - **Figure-Ground:** Distinction between objects and background
   - **Common Fate:** Elements moving together are perceived as related
   - **Symmetry:** Balanced compositions are more pleasing
   - **Application:** Guide viewer attention intentionally

4. **Color Theory and Color Blindness Accessibility** (Slide 7)
   - Colorblind-safe palette design
   - Avoiding red-green combinations
   - Using color with redundant encodings (shape, texture)

5. **Typography and Layout** (Slide 8)
   - Font selection and hierarchy
   - Whitespace utilization
   - Alignment and grid systems

6. **Information Density and Data-Ink Ratio** (Slide 9)
   - **Before (Bad):** Low ratio ~30%
     - Heavy borders and shadows
     - Excessive gridlines
     - 3D effects and gradients
     - Redundant decorations
   - **After (Good):** High ratio ~90%
     - Clean, minimal design
     - No unnecessary elements
     - Focus on actual data
     - Every element serves a purpose
   - Key principles:
     - Remove chartjunk and decorations
     - Eliminate redundant labels
     - High density ≠ cluttered
     - Iterate until nothing can be removed (Tufte's principle)

7. **Grammar of Graphics** (Slide 10)
   - Layered approach to visualization
   - Data, aesthetics, geometries, scales, coordinates

8. **Good vs Bad Visualization Examples** (Slide 11)
   - **Chart Type Selection:** 3D pie charts (bad) vs. 2D bar charts (good)
   - **Axis Management:** Dual y-axes mislead by arbitrary scaling (bad) vs. normalized or separate charts (good)
   - **Y-Axis Baseline:** Truncated y-axis exaggerates differences (bad) vs. start at zero or clearly indicate breaks (good)
   - **Color Application:** Too many colors causing confusion (bad) vs. intentional color use highlighting key insights (good)

#### Key Concepts:

- Visual encoding effectiveness follows a strict hierarchy with position being most accurate
- Human perception follows predictable patterns (Gestalt principles) that can be leveraged for better design
- Data-ink ratio maximization leads to clearer, more effective visualizations
- Accessibility considerations are fundamental, not optional
- Good visualizations avoid common pitfalls like misleading axes and inappropriate chart types

#### Learning Outcomes:

Students will be able to:
- Evaluate visualization effectiveness using encoding hierarchy principles
- Apply Gestalt principles to improve visual grouping and organization
- Calculate and maximize data-ink ratio in their visualizations
- Identify and avoid common visualization mistakes
- Design accessible visualizations for diverse audiences

---

### **Part 2: Mastering Basic Charts** (Slides 12-20)

#### Topics Covered:

9. **1D Data - Histogram, KDE** (Slide 13)
   - Distribution visualization for single variables
   - Histogram bin selection strategies
   - Kernel Density Estimation for smooth distributions
   - When to use each method

10. **2D Relationships - Scatter Plot, Bubble Chart** (Slide 14)
    - Visualizing correlations and relationships
    - Scatter plot for two continuous variables
    - Bubble charts adding a third dimension via size
    - Overplotting solutions (alpha transparency, jittering)

11. **Categorical Data - Bar Chart, Pie Chart** (Slide 15)
    - Bar charts for categorical comparisons
    - Horizontal vs. vertical orientation
    - Grouped and stacked bar charts
    - When pie charts are appropriate (rarely)
    - Avoiding 3D and exploded pie charts

12. **Distribution Comparison - Boxplot, Violin Plot** (Slide 16)
    - Boxplot components: quartiles, median, outliers, whiskers
    - Violin plots showing full distribution shape
    - Comparing multiple distributions simultaneously
    - Identifying skewness and outliers

13. **Time Series - Line Graph, Area Chart** (Slide 17)
    - Temporal trend visualization
    - Line graphs for continuous time series
    - Area charts for cumulative values
    - Multiple time series comparison
    - Handling seasonality and trends

14. **Correlation - Heatmap, Correlation Matrix** (Slide 18)
    - Visualizing correlation coefficients
    - Color schemes for correlation strength
    - Identifying multicollinearity
    - Clustering correlated features

15. **Multidimensional - Parallel Coordinates, Radar Chart** (Slide 19)
    - Parallel coordinates for high-dimensional data
    - Radar/spider charts for multivariate comparisons
    - Normalizing variables for fair comparison
    - Pattern identification across dimensions

16. **Geographic Data - Choropleth, Bubble Map** (Slide 20)
    - Choropleth maps for regional data
    - Color scales for geographic distributions
    - Bubble maps for location-based quantities
    - Projection considerations

#### Key Concepts:

- Different data structures require specific chart types for optimal representation
- Chart selection depends on data dimensionality, type (continuous/categorical), and analysis goals
- Distribution visualization is fundamental for understanding data characteristics
- Temporal and spatial data require specialized visualization approaches
- Correlation and multidimensional techniques reveal complex relationships

#### Learning Outcomes:

Students will be able to:
- Select appropriate chart types based on data structure and analysis objectives
- Implement histograms, KDE, and distribution plots for univariate analysis
- Create effective scatter plots and bubble charts for relationship exploration
- Visualize categorical data using bar charts with proper design principles
- Compare distributions using boxplots and violin plots
- Analyze time series patterns and trends
- Create correlation heatmaps for feature relationship analysis
- Visualize high-dimensional data using parallel coordinates
- Design geographic visualizations for spatial data

---

### **Part 3: Advanced Visualization for ML** (Slides 21-30)

#### Topics Covered:

17. **Visualization Strategy for EDA** (Slide 22)
    - **Systematic Workflow:**
      1. Data Loading: Import and understand data structure
      2. Univariate Analysis: Distributions, missing values, outliers
      3. Bivariate Relationships: Correlations, dependencies, patterns
      4. Multivariate Patterns: Interactions, clusters, dimensionality
      5. Insights & Hypotheses: Generate actionable insights
    - **Key Principles:**
      - Systematic pipeline approach
      - Automate repetitive plots
      - Use small multiples (facets)
      - Document insights iteratively
    - **Best Practices:**
      - Start simple, add complexity
      - Maintain visualization journal
      - Compare across categories
      - Iterate: hypothesis → visualization → insight
    - **Balance:** Breadth (many features) ⟷ Depth (detailed analysis)

18. **Feature Distribution and Outlier Detection** (Slide 23)
    - Distribution shape analysis (normal, skewed, bimodal)
    - Statistical outlier detection methods
    - Box plots for outlier identification
    - Impact of outliers on model performance
    - Handling strategies: removal, transformation, capping

19. **Visualizing Feature Interactions** (Slide 24)
    - Interaction plots for feature pairs
    - Conditional distributions
    - Faceted scatter plots
    - Feature importance in interactions
    - Non-linear relationship detection

20. **Dimensionality Reduction Visualization (PCA, t-SNE, UMAP)** (Slide 25)
    - **PCA (Principal Component Analysis):**
      - Linear dimensionality reduction
      - Variance preservation
      - Interpretable components
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
      - Non-linear manifold learning
      - Preserves local structure
      - Perplexity parameter tuning
    - **UMAP (Uniform Manifold Approximation and Projection):**
      - Faster than t-SNE
      - Preserves both local and global structure
      - Better scalability
    - Use cases: cluster visualization, data exploration, feature engineering validation

21. **Model Performance Visualization - Learning Curves** (Slide 26)
    - Training vs. validation performance over time
    - Identifying overfitting and underfitting
    - Convergence analysis
    - Sample size impact on performance
    - Early stopping decisions

22. **Classification Model Evaluation - Confusion Matrix, ROC, PR** (Slide 27)
    - **Confusion Matrix:**
      - True Positives, True Negatives, False Positives, False Negatives
      - Precision, Recall, F1-Score calculation
      - Multi-class confusion matrices
    - **ROC Curve (Receiver Operating Characteristic):**
      - TPR vs. FPR at different thresholds
      - AUC (Area Under Curve) interpretation
      - Model comparison
    - **PR Curve (Precision-Recall):**
      - Better for imbalanced datasets
      - Precision vs. Recall trade-off
      - Average Precision metric

23. **Regression Model Evaluation - Residuals, QQ Plot** (Slide 28)
    - **Residual Plots:**
      - Residuals vs. predicted values
      - Identifying heteroscedasticity
      - Non-linearity detection
      - Outlier and influential point identification
    - **QQ Plot (Quantile-Quantile):**
      - Normality assumption testing
      - Distribution comparison
      - Detecting heavy tails and skewness

24. **Model Interpretation - SHAP, LIME, Attention** (Slide 29)
    - **SHAP (SHapley Additive exPlanations):**
      - Feature importance across entire dataset
      - Individual prediction explanations
      - Summary plots, dependence plots, force plots
      - Game theory-based approach
    - **LIME (Local Interpretable Model-agnostic Explanations):**
      - Local surrogate models
      - Instance-level explanations
      - Model-agnostic approach
    - **Attention Mechanisms:**
      - Visualization for deep learning models
      - Attention weights heatmaps
      - Understanding model focus areas
      - Application in NLP and computer vision

25. **Dashboard Design and Storytelling** (Slide 30)
    - Dashboard layout principles
    - Information hierarchy
    - Interactive elements design
    - Narrative structure in data presentation
    - Audience-specific customization
    - Key metrics highlighting
    - Balancing detail and overview

#### Key Concepts:

- Systematic EDA workflows ensure comprehensive data understanding
- Dimensionality reduction techniques enable visualization of high-dimensional data
- Model performance visualization is critical for debugging and improvement
- Different evaluation metrics suit different problem types (classification vs. regression)
- Model interpretation techniques bridge the gap between complex models and human understanding
- Effective dashboards combine multiple visualizations into coherent narratives

#### Learning Outcomes:

Students will be able to:
- Implement systematic EDA workflows from data loading to insight generation
- Detect and visualize outliers and distribution anomalies
- Apply dimensionality reduction (PCA, t-SNE, UMAP) for data exploration
- Create and interpret learning curves for model diagnostics
- Evaluate classification models using confusion matrices, ROC, and PR curves
- Assess regression models using residual analysis and QQ plots
- Explain model predictions using SHAP and LIME
- Design comprehensive dashboards for ML project communication
- Tell compelling data stories that drive decision-making

---

## 🛠️ Prerequisites

### Knowledge Prerequisites:

- **Statistics:** Basic understanding of descriptive statistics, distributions, correlation, and probability
- **Programming:** Proficiency in Python (recommended) or R
- **Data Structures:** Familiarity with DataFrames, arrays, and basic data manipulation
- **Mathematics:** Basic linear algebra for dimensionality reduction techniques
- **Machine Learning (for Part 3):** Understanding of supervised learning, overfitting, train-test split concepts

### Software Requirements:

#### Python Stack (Recommended):
```
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- plotly >= 5.0.0 (for interactive visualizations)
- scikit-learn >= 0.24.0 (for ML visualizations)
- shap >= 0.40.0 (for model interpretation)
```

#### R Stack (Alternative):
```
- R >= 4.0.0
- ggplot2
- dplyr
- tidyr
- plotly
- caret (for ML visualizations)
```

#### Additional Tools:
- Jupyter Notebook or JupyterLab (for interactive development)
- Git (for version control)
- Any modern web browser (for viewing HTML slides)

### Hardware Requirements:
- **RAM:** Minimum 8GB (16GB recommended for large datasets)
- **Storage:** 5GB free space for software and datasets
- **Processor:** Modern multi-core CPU (for dimensionality reduction algorithms)

---

## 💻 Hands-on Components

### Project 1: Visualization Fundamentals (Part 1)
**Duration:** 1-2 hours  
**Objective:** Apply fundamental visualization principles

**Tasks:**
1. Create a dataset with 5 quantitative variables
2. Implement the same data using all 7 visual encoding methods (position, length, angle, area, color, shape, texture)
3. Compare effectiveness through user testing or self-evaluation
4. Redesign a "bad" visualization found in the wild following data-ink ratio principles
5. Create before/after comparison demonstrating:
   - Removal of chartjunk
   - Improved data-ink ratio (from ~30% to ~90%)
   - Enhanced accessibility (colorblind-safe palette)

**Deliverables:**
- Notebook with 7 encoding examples
- Before/after visualization pair with written analysis
- 500-word reflection on effectiveness differences

### Project 2: Chart Type Mastery (Part 2)
**Duration:** 2-3 hours  
**Objective:** Master selection and implementation of appropriate chart types

**Dataset:** Provided multi-dimensional dataset (or UCI ML Repository dataset)

**Tasks:**
1. **Univariate Analysis:**
   - Create histograms with 3 different bin sizes
   - Overlay KDE on histogram
   - Compare distribution shapes

2. **Bivariate Analysis:**
   - Generate scatter plot matrix for all numeric pairs
   - Create bubble chart with 3+ variables
   - Identify strongest correlations

3. **Categorical Analysis:**
   - Design grouped bar chart comparing categories
   - Create stacked bar chart for composition
   - Critique: when would a pie chart be appropriate?

4. **Temporal Analysis (if time-series data available):**
   - Plot trends with line graphs
   - Add area chart for cumulative metrics
   - Highlight anomalies or significant events

5. **Multidimensional Analysis:**
   - Implement parallel coordinates plot for 5+ variables
   - Create correlation heatmap with hierarchical clustering
   - Generate radar chart comparing instances

**Deliverables:**
- Comprehensive visualization notebook (15-20 plots)
- Chart type decision matrix (flowchart for selecting appropriate chart)
- 1000-word report analyzing patterns discovered

### Project 3: ML Visualization Pipeline (Part 3)
**Duration:** 4-5 hours  
**Objective:** Build complete visualization pipeline for ML workflow

**Dataset:** Classification or regression dataset (e.g., Titanic, Boston Housing, or custom)

**Phase 1: Exploratory Data Analysis (45 min)**
1. Follow systematic EDA workflow (Slide 22)
2. Univariate: distributions, missing values, outliers
3. Bivariate: correlation analysis, feature relationships
4. Multivariate: interaction effects, patterns
5. Document insights in visualization journal

**Phase 2: Feature Engineering Visualization (30 min)**
1. Visualize feature distributions before/after transformations
2. Outlier detection with box plots and statistical methods
3. Feature interaction plots for top 3 pairs
4. Justify engineering decisions with visualizations

**Phase 3: Dimensionality Reduction (45 min)**
1. Apply PCA and visualize explained variance
2. Create 2D scatter plot of first two components
3. Implement t-SNE with different perplexity values (5, 30, 50)
4. Compare UMAP results with t-SNE
5. Color points by target variable to assess separation

**Phase 4: Model Training and Evaluation (60 min)**
1. Train baseline model and visualize learning curves
2. Plot training vs. validation loss/accuracy over epochs/iterations
3. Identify overfitting/underfitting
4. **For Classification:**
   - Create confusion matrix (raw counts and normalized)
   - Plot ROC curve with AUC score
   - Generate PR curve (especially important for imbalanced data)
   - Compare multiple models on same ROC plot
5. **For Regression:**
   - Create residual plot (residuals vs. predicted)
   - Generate QQ plot for normality check
   - Plot actual vs. predicted values
   - Calculate and visualize error metrics (MAE, RMSE, R²)

**Phase 5: Model Interpretation (45 min)**
1. Calculate SHAP values for trained model
2. Create SHAP summary plot (feature importance)
3. Generate SHAP dependence plots for top 3 features
4. Create SHAP force plot for individual predictions
5. Apply LIME to explain 3 interesting predictions
6. Compare SHAP and LIME explanations

**Phase 6: Dashboard Creation (30 min)**
1. Design dashboard layout with 6-8 key visualizations
2. Include: data overview, EDA insights, model performance, interpretation
3. Add narrative text explaining insights
4. Ensure visual hierarchy guides viewer attention
5. Export as HTML or create interactive dashboard with Plotly Dash

**Deliverables:**
- Complete Jupyter notebook with all phases
- Interactive dashboard (HTML or Dash app)
- 1500-word technical report covering:
  - EDA findings
  - Feature engineering rationale
  - Model selection and performance
  - Key insights from interpretation
  - Recommendations for stakeholders
- 10-minute video presentation (optional)

**Evaluation Criteria:**
- Visualization quality and appropriateness (30%)
- Technical correctness (25%)
- Insight depth and clarity (25%)
- Code quality and documentation (10%)
- Dashboard design and storytelling (10%)

---

## 📖 Additional Resources

### Books:
1. **"The Visual Display of Quantitative Information"** by Edward Tufte
   - Classic text on information design and data-ink ratio
   - Essential reading for understanding visualization principles

2. **"Fundamentals of Data Visualization"** by Claus O. Wilke
   - Comprehensive guide to chart types and design principles
   - Available free online: https://clauswilke.com/dataviz/

3. **"Storytelling with Data"** by Cole Nussbaumer Knaflic
   - Practical guide to data storytelling and dashboard design
   - Focus on business communication

4. **"Python Data Science Handbook"** by Jake VanderPlas
   - Chapters on matplotlib, seaborn, and visualization best practices
   - Available free online: https://jakevdp.github.io/PythonDataScienceHandbook/

### Online Resources:

**Visualization Galleries:**
- D3.js Gallery: https://observablehq.com/@d3/gallery
- Python Graph Gallery: https://python-graph-gallery.com/
- Seaborn Example Gallery: https://seaborn.pydata.org/examples/index.html
- Plotly Chart Types: https://plotly.com/python/

**Interactive Tutorials:**
- DataCamp: Data Visualization courses
- Kaggle Learn: Data Visualization module
- Coursera: Data Visualization with Python (IBM)

**Color Tools:**
- ColorBrewer 2.0: https://colorbrewer2.org/ (colorblind-safe palettes)
- Coolors: https://coolors.co/ (color scheme generator)
- Adobe Color: https://color.adobe.com/ (color wheel and harmony rules)
- Viz Palette: https://projects.susielu.com/viz-palette (visualization color testing)

**Accessibility:**
- WebAIM Contrast Checker: https://webaim.org/resources/contrastchecker/
- Color Oracle: https://colororacle.org/ (colorblindness simulator)

**SHAP Documentation:**
- Official SHAP repository: https://github.com/slundberg/shap
- SHAP tutorials and examples: https://shap.readthedocs.io/

**Dimensionality Reduction:**
- Scikit-learn User Guide: https://scikit-learn.org/stable/modules/manifold.html
- UMAP documentation: https://umap-learn.readthedocs.io/

### Academic Papers:

1. **Visual Encoding:**
   - Cleveland & McGill (1984): "Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods"

2. **Dimensionality Reduction:**
   - Van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
   - McInnes et al. (2018): "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"

3. **Model Interpretation:**
   - Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions" (SHAP)
   - Ribeiro et al. (2016): "Why Should I Trust You?: Explaining the Predictions of Any Classifier" (LIME)

### Sample Datasets:

**For Practice:**
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
- Kaggle Datasets: https://www.kaggle.com/datasets
- Seaborn built-in datasets: `sns.load_dataset()`
- Scikit-learn toy datasets: `sklearn.datasets`

**Recommended Specific Datasets:**
- **Classification:** Titanic, Iris, Wine Quality, Credit Default
- **Regression:** Boston Housing, California Housing, Diabetes
- **Time Series:** Stock prices, Weather data, COVID-19 statistics
- **Geographic:** World Development Indicators, US Census data

---

## 🔗 How to Use These Materials

### For Self-Study:

1. **Week 1-2: Fundamentals (Part 1)**
   - Study slides 3-11 (1-2 hours)
   - Read Tufte's chapters on data-ink ratio
   - Complete Project 1: Visualization Fundamentals
   - Practice: Critique 5 visualizations from news websites

2. **Week 3-4: Chart Types (Part 2)**
   - Study slides 12-20 (2-3 hours)
   - Explore Python Graph Gallery for each chart type
   - Complete Project 2: Chart Type Mastery
   - Practice: Create visualization library template

3. **Week 5-7: ML Visualization (Part 3)**
   - Study slides 21-30 (3-4 hours)
   - Deep dive into SHAP and dimensionality reduction papers
   - Complete Project 3: ML Visualization Pipeline
   - Practice: Reproduce visualizations from ML papers

4. **Week 8: Integration and Review**
   - Review all materials
   - Create personal visualization style guide
   - Build portfolio of best visualizations
   - Prepare presentation of learnings

### For Classroom Instruction:

**Session 1 (2 hours): Fundamentals**
- Lecture: Slides 3-11 (60 min)
- Activity: Visual encoding effectiveness experiment (30 min)
- Discussion: Critique example visualizations (30 min)
- Homework: Project 1

**Session 2 (2 hours): Basic Charts I**
- Lecture: Slides 12-16 (45 min)
- Live coding: Create histogram, scatter, bar chart (45 min)
- Activity: Chart type selection exercise (30 min)

**Session 3 (2 hours): Basic Charts II**
- Lecture: Slides 17-20 (30 min)
- Live coding: Time series, heatmap, parallel coordinates (60 min)
- Activity: Dataset exploration with multiple chart types (30 min)
- Homework: Project 2

**Session 4 (2 hours): EDA and Feature Visualization**
- Lecture: Slides 21-24 (45 min)
- Live demo: Systematic EDA workflow (45 min)
- Activity: Outlier detection challenge (30 min)

**Session 5 (2 hours): Dimensionality Reduction**
- Lecture: Slide 25 (30 min)
- Live coding: PCA, t-SNE, UMAP comparison (60 min)
- Discussion: When to use each method (30 min)

**Session 6 (2 hours): Model Evaluation**
- Lecture: Slides 26-28 (45 min)
- Live demo: Learning curves, confusion matrix, ROC/PR curves (60 min)
- Activity: Model debugging with visualizations (15 min)

**Session 7 (2 hours): Interpretation and Storytelling**
- Lecture: Slides 29-30 (45 min)
- Live demo: SHAP values calculation and visualization (45 min)
- Activity: Dashboard design critique (30 min)
- Homework: Project 3

**Session 8 (2 hours): Project Presentations**
- Student presentations of Project 3 (90 min)
- Peer feedback session (30 min)

### Navigating the Slideshow:

The complete slideshow (`lecture02_slideshow.html`) provides an integrated viewing experience:

1. **Open the file** in any modern web browser
2. **Navigation controls:**
   - Use arrow keys (← →) to move between slides
   - Press Space bar to advance
   - Home key: jump to first slide
   - End key: jump to last slide
   - Click navigation buttons in bottom-right corner
3. **Features:**
   - Progress bar at top shows lecture completion
   - Slide counter displays current position (X / 31)
   - Slide title overlay shows current topic and part number
   - Keyboard hint appears briefly on load

4. **Viewing individual slides:**
   - All slides are HTML files that can be opened independently
   - Each slide is self-contained and styled consistently
   - Recommended for detailed study or printing specific topics

---

## 📊 Assessment Suggestions

### Formative Assessment (Ongoing):

1. **Weekly Quizzes (10%)**
   - 3 quizzes covering Parts 1, 2, and 3
   - 10 multiple choice + 2 short answer questions each
   - Topics: visual encoding hierarchy, chart selection, interpretation techniques
   - Auto-graded through LMS

2. **Visualization Critiques (15%)**
   - 3 submissions throughout course
   - Find and critique visualizations from news, research papers, or social media
   - Identify: violations of principles, missed opportunities, strengths
   - Suggest improvements with sketches or descriptions
   - 300-400 words each

3. **Code Reviews (10%)**
   - Peer review of project code
   - Evaluate: correctness, clarity, documentation, style
   - Provide constructive feedback
   - Reflection on feedback received

### Summative Assessment (Projects):

**Project 1: Visualization Fundamentals (15%)**
- Rubric:
  - Correct implementation of 7 encoding types (5%)
  - Before/after comparison quality (5%)
  - Written analysis depth (3%)
  - Visual design quality (2%)

**Project 2: Chart Type Mastery (20%)**
- Rubric:
  - Appropriate chart selection (6%)
  - Technical implementation (6%)
  - Insight discovery (4%)
  - Decision matrix completeness (4%)

**Project 3: ML Visualization Pipeline (30%)**
- Rubric breakdown:
  - EDA completeness and insights (6%)
  - Feature engineering visualization (4%)
  - Dimensionality reduction application (5%)
  - Model evaluation visualizations (6%)
  - Interpretation (SHAP/LIME) quality (4%)
  - Dashboard design and storytelling (5%)

### Final Assessment Options:

**Option A: Comprehensive Exam (10%)**
- 90-minute written exam
- Sections:
  1. Visual encoding principles (20 points)
  2. Chart type selection scenarios (25 points)
  3. Interpretation of ML visualizations (25 points)
  4. Design critique and improvement (30 points)

**Option B: Capstone Presentation (10%)**
- 15-minute presentation + 5-minute Q&A
- Present insights from Project 3
- Evaluated on:
  - Visualization quality
  - Storytelling effectiveness
  - Technical accuracy
  - Response to questions

### Grading Scale:
- A: 90-100% (Exceptional visualization skills, publication-ready work)
- B: 80-89% (Proficient visualization skills, professional quality)
- C: 70-79% (Competent visualization skills, some improvements needed)
- D: 60-69% (Basic visualization skills, significant improvements needed)
- F: <60% (Does not meet minimum competency)

### Extra Credit Opportunities (up to 5% bonus):

1. **Contribute to open-source visualization project** (3%)
   - Submit PR to matplotlib, seaborn, or plotly
   - Document contribution with screenshots and description

2. **Create visualization tutorial** (2%)
   - Blog post or video explaining advanced technique
   - Minimum 1000 words or 10-minute video
   - Share with class

3. **Attend visualization talk/webinar** (1%)
   - Write 500-word summary and reflection
   - Submit proof of attendance

---

## 📝 Notes for Implementation

### Technical Requirements:

**Browser Compatibility:**
- Slides tested on: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- JavaScript enabled required for slideshow navigation
- Recommended screen resolution: 1920x1080 or higher
- Mobile viewing supported but desktop experience recommended

**File Structure:**
```
lecture02-data-visualization/
│
├── README.md                          (this file)
├── lecture02_slideshow.html           (complete slideshow)
│
├── slides/
│   ├── Lecture02_01_Lecture 2 Data Visualization.html
│   ├── Lecture02_02_Lecture Contents.html
│   ├── Lecture02_03_Part 1_3 Visualization Fundamentals.html
│   ├── ... (all 31 individual slide files)
│   └── Lecture02_31_Thank you.html
│
├── projects/
│   ├── project1_fundamentals/
│   │   ├── instructions.md
│   │   ├── rubric.md
│   │   └── starter_notebook.ipynb
│   │
│   ├── project2_charts/
│   │   ├── instructions.md
│   │   ├── rubric.md
│   │   ├── starter_notebook.ipynb
│   │   └── sample_dataset.csv
│   │
│   └── project3_ml_pipeline/
│       ├── instructions.md
│       ├── rubric.md
│       ├── starter_notebook.ipynb
│       └── datasets/ (various ML datasets)
│
├── code_examples/
│   ├── part1_fundamentals/
│   │   ├── visual_encoding_comparison.ipynb
│   │   ├── gestalt_principles_demo.ipynb
│   │   └── data_ink_ratio_example.ipynb
│   │
│   ├── part2_charts/
│   │   ├── histogram_kde.ipynb
│   │   ├── scatter_bubble.ipynb
│   │   ├── categorical_viz.ipynb
│   │   ├── distribution_comparison.ipynb
│   │   ├── time_series.ipynb
│   │   ├── correlation_heatmap.ipynb
│   │   ├── multidimensional.ipynb
│   │   └── geographic.ipynb
│   │
│   └── part3_ml_viz/
│       ├── eda_workflow.ipynb
│       ├── feature_distribution.ipynb
│       ├── feature_interactions.ipynb
│       ├── dimensionality_reduction.ipynb
│       ├── learning_curves.ipynb
│       ├── classification_eval.ipynb
│       ├── regression_eval.ipynb
│       ├── shap_lime_demo.ipynb
│       └── dashboard_example.ipynb
│
├── datasets/
│   ├── sample_1d_data.csv
│   ├── sample_2d_data.csv
│   ├── titanic.csv
│   ├── boston_housing.csv
│   └── README_datasets.md
│
└── resources/
    ├── colorblind_palettes.py
    ├── chart_selection_flowchart.pdf
    ├── visualization_checklist.pdf
    └── recommended_reading.md
```

### Performance Considerations:

**Slideshow Performance:**
- Each slide loads independently via iframe
- Initial load time: ~2-3 seconds for all 31 slides
- Navigation is instant after initial load
- Memory usage: ~150-200MB for complete slideshow

**Jupyter Notebooks:**
- Large datasets (>100MB) may require increased memory
- Dimensionality reduction algorithms (t-SNE, UMAP) are computationally intensive
  - t-SNE: O(n²) time complexity, can be slow for n>10,000 samples
  - UMAP: O(n log n), much faster and scalable
  - PCA: O(n × min(n, p)²), fastest for linear reduction
- SHAP calculations can be slow for large models
  - TreeExplainer: Fast for tree-based models (~seconds)
  - KernelExplainer: Slower, model-agnostic (~minutes)
  - Recommend sampling for initial exploration

**Optimization Tips:**
```python
# For large datasets, sample for visualization
df_sample = df.sample(n=1000, random_state=42)

# Use appropriate backends for better performance
import matplotlib
matplotlib.use('Agg')  # For non-interactive, faster rendering

# Cache dimensionality reduction results
from joblib import Memory
memory = Memory(location='.cache', verbose=0)

@memory.cache
def compute_tsne(X, perplexity=30):
    return TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
```

### Customization Guide:

**Modifying Slides:**
- All slides use inline CSS for easy customization
- Primary color: `#1E64C8` (blue) - change for branding
- Font: Aptos, 'Segoe UI' - modify in `<style>` tag
- Animations can be disabled by removing `@keyframes` sections

**Adding New Content:**
1. Copy existing slide HTML file as template
2. Modify content in `<div class="container">` section
3. Update `lecture02_slideshow.html` slide array with new entry
4. Test navigation before/after new slide

**Language Localization:**
- HTML lang attribute set to "ko" (Korean)
- Change to "en" for English: `<html lang="en">`
- Update text content in each slide
- Consider cultural differences in color meanings

### Known Issues and Solutions:

**Issue 1: Slideshow not advancing**
- Solution: Ensure JavaScript is enabled in browser
- Alternative: Use individual HTML files

**Issue 2: Plots not rendering in Jupyter**
- Solution: Run `%matplotlib inline` at notebook start
- For interactive: Use `%matplotlib notebook` or `%matplotlib widget`

**Issue 3: SHAP installation fails**
- Solution: Install from conda-forge: `conda install -c conda-forge shap`
- Windows users may need Visual C++ build tools

**Issue 4: Memory errors with large datasets**
- Solution: Use sampling or chunking strategies
- Increase swap space or use cloud computing resources

**Issue 5: Font rendering differences across platforms**
- Solution: Fonts fall back to system fonts automatically
- For consistency, embed web fonts (e.g., Google Fonts)

### Accessibility Compliance:

**WCAG 2.1 AA Standards:**
- ✓ Color contrast ratios meet 4.5:1 minimum
- ✓ All interactive elements keyboard accessible
- ✓ Semantic HTML structure for screen readers
- ✓ Alt text should be added to any images (not present in these slides)
- ✓ Color is not the only means of conveying information

**Recommendations for Teaching:**
- Provide slide transcripts for hearing-impaired students
- Offer high-contrast version of slides for visually impaired
- Ensure code examples include descriptive comments
- Use large fonts (minimum 18pt) for presentations
- Test all visualizations with colorblind simulators

### Version Control and Updates:

**Current Version:** 1.0  
**Last Updated:** 2025  
**Change Log:**
- Initial release with 31 slides covering 3 major parts

**Planned Updates:**
- v1.1: Add interactive Plotly examples in slides
- v1.2: Include video lectures for each section
- v1.3: Expand geographic visualization with Folium examples
- v2.0: Add Part 4 on Deep Learning visualization techniques

**Contributing:**
If you find errors or have suggestions:
1. Document the issue with slide number and description
2. Propose solution or correction
3. Contact instructor: homin.park@ghent.ac.kr

---

## 🙏 Credits

**Course Development:**  
**Instructor:** Ho-min Park  
**Affiliation:** Ghent University Global Campus  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com

**Acknowledgments:**
- Edward Tufte for foundational principles of data visualization
- Hadley Wickham for Grammar of Graphics framework
- Jake VanderPlas for Python data science community contributions
- Scott Lundberg for SHAP library development
- Leland McInnes for UMAP algorithm development

**Design Inspiration:**
- Material Design (color palette and layouts)
- Apple Human Interface Guidelines (typography and spacing)
- Microsoft PowerPoint design templates (slide structure)

**Tools Used in Course Development:**
- Visualization: Python (matplotlib, seaborn, plotly)
- Slide Creation: HTML5, CSS3, JavaScript
- Diagrams: draw.io, Figma
- Documentation: Markdown, Jupyter Notebook

**License:**
These educational materials are provided for academic use. Please contact the instructor for permissions regarding commercial use or redistribution.

**Citation:**
```
Park, H. (2025). Lecture 2: Data Visualization. 
Ghent University Global Campus.
```

---

**Last Updated:** 2025  
**Version:** 1.0  
**Maintained by:** Ho-min Park (homin.park@ghent.ac.kr)

For questions, suggestions, or bug reports, please contact the instructor.
