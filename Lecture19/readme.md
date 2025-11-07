# Lecture 19: Model Explainability - XAI Fundamentals and Traditional Methods

## 📋 Overview

**Instructor:** Ho-min Park  
**Contact:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Total Slides:** 32  
**Lecture Duration:** 3-4 hours (including hands-on)  
**Difficulty Level:** Intermediate to Advanced  
**Prerequisites:** Machine Learning Fundamentals, Python Programming, scikit-learn  

This comprehensive lecture introduces Explainable AI (XAI) methodologies, covering both theoretical foundations and practical implementations. The course progresses from fundamental concepts through intrinsically interpretable models to advanced model-agnostic explanation techniques.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand XAI Fundamentals**: Articulate the importance of explainability in AI systems and distinguish between key concepts (interpretability, explainability, transparency)

2. **Apply Intrinsically Interpretable Models**: Implement and interpret linear models, decision trees, GAMs, rule-based models, and monotonic constraint models using scikit-learn

3. **Evaluate Feature Importance**: Calculate and visualize feature importance using permutation importance, drop-column importance, PDP, ICE, and ALE plots

4. **Implement Model-Agnostic Methods**: Apply surrogate models, LIME, and Anchor explanations to explain black-box model predictions

5. **Select Appropriate XAI Methods**: Choose suitable explanation techniques based on use case requirements (global vs. local, model-specific vs. agnostic)

---

## 📚 Lecture Structure

### Part 1/4: Introduction to XAI and Its Importance

**Topics Covered:**

1. **Course Introduction - The Era of Explainable AI**
   - Evolution timeline from simple models (1980s-1990s) to complex AI systems
   - Increasing need for XAI with model complexity growth
   - Historical context: Linear Regression → SVM/Random Forest → Deep Learning → Transformers/LLMs

2. **Model Complexity vs Interpretability Trade-off**
   - Visual representation of the inverse relationship between model complexity and interpretability
   - Spectrum from Linear Regression (high interpretability) to Deep Learning (high accuracy, low interpretability)
   - Understanding when to prioritize interpretability over performance

3. **Core Concepts and Terminology in XAI**
   - **Interpretability**: Understanding model mechanics and decision-making process
   - **Explainability**: Providing human-understandable reasons for specific predictions
   - **Transparency**: Visibility into model's internal workings
   - Global vs. Local explanations
   - Model-agnostic vs. Model-specific methods
   - Feature importance and internal mechanics

4. **Why Explainability is Necessary**
   - **Trust**: Users need to understand AI decisions before accepting them
   - **Accountability**: Identifying responsibility when AI makes errors
   - **Debugging**: Finding and fixing model biases or data issues
   - **Regulatory Compliance**: Legal requirements (GDPR, EU AI Act)
   - **Fairness**: Detecting and mitigating discrimination
   - **Scientific Discovery**: Learning new insights from model patterns
   - **Safety**: Ensuring AI behaves as intended in critical applications

5. **XAI Classification Framework**
   - **Timing**: Intrinsic (built-in), Post-hoc (after training), Ante-hoc (design phase)
   - **Scope**: Global (entire model), Local (single prediction), Example-based
   - **Applicability**: Model-specific vs. Model-agnostic
   - Transparent (white-box) vs. Black-box approaches

6. **XAI Applications by Industry**
   - Healthcare: Diagnosis predictions, treatment recommendations
   - Finance: Credit scoring, fraud detection reasoning
   - Autonomous Vehicles: Decision transparency for safety certification
   - Criminal Justice: Risk assessment for parole decisions
   - Human Resources: Fair hiring practices, bias detection
   - E-commerce: Recommendation system transparency
   - Manufacturing: Predictive maintenance, quality control

7. **XAI Evaluation Criteria**
   - **Fidelity**: How accurately explanations reflect actual model behavior
   - **Comprehensibility**: Human ability to understand explanations
   - **Consistency**: Similar instances receive similar explanations
   - **Contrastivity**: Explaining why this prediction vs. alternatives
   - **Actionability**: Ability to use explanations for decisions
   - **Stability**: Robustness to small input perturbations
   - **Efficiency**: Computational cost of generating explanations

**Key Concepts:**
- XAI has become critical as AI systems transition from interpretable models to complex black-box architectures
- Core distinction between interpretability (understanding mechanism) and explainability (understanding specific outputs)
- Multiple evaluation dimensions exist for assessing explanation quality
- Industry-specific requirements drive different XAI needs
- Classification frameworks help organize the diverse landscape of XAI methods

**Learning Outcomes:**
- Articulate the historical drivers for XAI development
- Differentiate between interpretability, explainability, and transparency
- Identify appropriate XAI applications for various industries
- Evaluate XAI methods using multiple criteria (fidelity, comprehensibility, etc.)

---

### Part 2/4: Intrinsically Interpretable Models

**Topics Covered:**

8. **Interpreting Linear Models**
   - Linear model equation: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
   - **Coefficient Magnitude**: Indicates feature importance (larger absolute values = stronger impact)
   - **Sign Direction**: Shows positive/negative relationship with target
   - **Regularization Techniques**: 
     - L1 (Lasso): Feature selection through sparsity
     - L2 (Ridge): Coefficient stability and multicollinearity handling
   - Visualization: Horizontal bar charts showing positive/negative coefficient impacts
   - Example coefficients: Age (+0.82), Income (+0.63), Distance (-0.55), Risk Score (-0.76)

9. **Transparency of Decision Trees**
   - **If-Then Rules**: Binary splits create human-readable decision paths
   - **Complete Path Visibility**: Full decision logic from root to leaf
   - **Feature Importance**: Determined by split frequency and information gain
   - **Natural Interaction Handling**: Hierarchical splits capture feature interactions automatically
   - **Trade-offs**: Deep trees lose interpretability; pruning required for balance
   - Visualization: Interactive tree structure showing decision nodes and leaf predictions
   - Example: Credit approval tree with Age, Income, Credit Score, Education splits

10. **Generalized Additive Models (GAM)**
    - Model structure: y = β₀ + f₁(x₁) + f₂(x₂) + f₃(x₃) + ...
    - Each feature contributes independently through smooth shape functions
    - **Key Properties**:
      - Independent: Features contribute separately (additive nature)
      - Non-linear: Capture complex patterns via smooth functions
      - Interpretable: Visualize individual feature effects through partial dependence plots
      - Separable: Effects remain isolated and analyzable
      - Balanced: Middle ground between linear and black-box models
    - Shape function examples: Linear increasing, U-shaped, S-curve, Bell curve, Step-like, Decay
    - Applications: Understanding non-linear relationships while maintaining interpretability

11. **Rule-Based Models**
    - **IF-THEN Structure**: Logical conditions leading to predictions
    - Rule format: IF (condition1 AND condition2) THEN decision
    - **Coverage**: Percentage of data instances satisfying rule conditions
    - **Confidence**: Prediction accuracy for instances matching the rule
    - **Interpretability Advantage**: Each prediction explained by triggering rules
    - **Trade-offs**: Balancing concise rule sets with high coverage and accuracy
    - Example rules:
      - Rule 1: IF Age > 30 AND Income > 50K THEN APPROVE (Coverage: 85%, Confidence: 92%)
      - Rule 2: IF Credit Score > 700 AND Debt Ratio < 0.3 THEN APPROVE (Coverage: 72%, Confidence: 88%)
      - Rule 3: IF Age < 25 AND Employment < 1 year THEN REJECT (Coverage: 65%, Confidence: 78%)

12. **Monotonic Constraint Models**
    - **Definition**: Enforcing logical relationships where feature increases always lead to prediction increases (or decreases)
    - **Problem with Non-Monotonic Models**: Counter-intuitive behaviors (e.g., higher credit score decreasing approval probability)
    - **Benefits of Monotonicity**:
      - Logical consistency with domain knowledge
      - Increased trust in model predictions
      - Regulatory compliance (especially in finance, healthcare)
      - Easier debugging and validation
    - **Implementation**: XGBoost, LightGBM with monotonic constraints, specialized neural networks
    - **Trade-offs**: Slight accuracy reduction for interpretability gains
    - **Critical Applications**: Regulated industries, financial decisions, healthcare, high-stakes predictions
    - Visual comparison: Erratic non-monotonic curve vs. smooth monotonic increasing curve

13. **Sparse Linear Models**
    - Using L1 regularization (Lasso) to drive coefficients to zero
    - **Feature Selection**: Automatic identification of most important features
    - **Interpretability Enhancement**: Fewer features = simpler explanations
    - **Regularization Strength**: α parameter controls sparsity level
    - **Applications**: High-dimensional data, when feature reduction is desired
    - Comparison with Ridge (L2): Lasso for selection, Ridge for stability

14. **Hands-on: Building Interpretable Models with scikit-learn**
    - **Implementation Workflow**:
      1. Import & Load Data: Load dataset and dependencies
      2. Preprocessing: Feature scaling, categorical encoding
      3. Train Models: Linear, Tree-based classifiers
      4. Extract Insights: Coefficients, feature importance
      5. Visualize & Compare: Accuracy vs interpretability trade-offs
    
    - **Code Examples**:
      ```python
      # Linear Model
      from sklearn.linear_model import LogisticRegression
      model = LogisticRegression()
      model.fit(X_train, y_train)
      coef = model.coef_
      
      # Decision Tree
      from sklearn.tree import DecisionTreeClassifier
      tree = DecisionTreeClassifier(max_depth=3)
      tree.fit(X_train, y_train)
      importance = tree.feature_importances_
      
      # Inspection Tools
      from sklearn.inspection import PartialDependenceDisplay
      PartialDependenceDisplay.from_estimator(model, X, features=[0, 1])
      ```
    
    - **Key sklearn Modules**: `linear_model`, `tree`, `inspection`, `preprocessing`

**Key Concepts:**
- Intrinsically interpretable models have built-in transparency mechanisms
- Linear models provide coefficient-based interpretability with clear directional effects
- Decision trees offer rule-based explanations through if-then logic
- GAMs balance non-linearity with interpretability through additive structure
- Monotonic constraints ensure logical consistency with domain knowledge
- Sparse models enhance interpretability through automatic feature selection

**Learning Outcomes:**
- Implement and interpret linear regression coefficients in context
- Visualize and explain decision tree structures
- Build and analyze GAMs for non-linear relationships
- Extract and evaluate rule-based explanations
- Apply monotonic constraints for domain-consistent predictions
- Use scikit-learn for building interpretable models in practice

---

### Part 3/4: Feature Importance Methodologies

**Topics Covered:**

15. **Permutation Importance**
    - **Methodology**: Randomly shuffle a feature and measure performance drop
    - **Algorithm**:
      1. Calculate baseline model performance on validation set
      2. For each feature: shuffle values, predict, measure performance
      3. Importance = baseline_performance - shuffled_performance
    - **Advantages**:
      - Model-agnostic (works with any black-box model)
      - Captures feature interactions
      - Reflects real prediction impact
    - **Limitations**:
      - Computationally expensive (requires multiple predictions)
      - Can be affected by correlated features
      - Variance across different shuffles
    - **Implementation**: `sklearn.inspection.permutation_importance()`
    - **Use Cases**: Understanding which features actually drive predictions

16. **Drop-Column Importance**
    - **Methodology**: Retrain model without each feature, measure performance change
    - **Algorithm**:
      1. Train baseline model with all features
      2. For each feature: remove from dataset, retrain model, evaluate
      3. Importance = baseline_performance - reduced_performance
    - **Advantages**:
      - More stable than permutation importance
      - Truly measures feature contribution
      - Better handles correlated features
    - **Limitations**:
      - Extremely computationally expensive (requires n retrainings)
      - Not suitable for models with long training times
      - May not be feasible for large feature sets
    - **Comparison with Permutation**: More accurate but much slower
    - **Best Practice**: Use for final model analysis with limited features

17. **Partial Dependence Plots (PDP)**
    - **Definition**: Shows marginal effect of a feature on predicted outcome
    - **Methodology**: Average predictions across all other feature values
    - **Formula**: PDP(xₛ) = 𝔼[f(xₛ, X_C)] where X_C are complementary features
    - **Interpretation**:
      - Y-axis: Predicted outcome (e.g., probability, regression value)
      - X-axis: Feature values
      - Curve shows average relationship between feature and prediction
    - **Types**:
      - 1D PDP: Single feature effect
      - 2D PDP: Interaction effects between two features (heatmap)
    - **Assumptions**: Features are independent (violated by correlation)
    - **Advantages**: Model-agnostic, intuitive visualization, shows non-linear relationships
    - **Limitations**: Assumes feature independence, can be misleading with strong interactions
    - **Implementation**: `sklearn.inspection.PartialDependenceDisplay`

18. **Individual Conditional Expectation (ICE)**
    - **Definition**: Shows how prediction changes for a single instance as feature varies
    - **Relationship to PDP**: PDP is the average of all ICE curves
    - **Visualization**: Multiple lines (one per instance) showing individual trajectories
    - **Advantages**:
      - Reveals heterogeneous effects across instances
      - Shows prediction variability
      - Identifies subgroups with different behaviors
      - More detailed than PDP
    - **Interpretation**:
      - Parallel lines: Consistent effect across all instances
      - Diverging lines: Heterogeneous effects, potential interactions
      - Crossing lines: Complex interactions present
    - **Use Cases**: 
      - Validating PDP assumptions
      - Finding outlier instances
      - Understanding prediction diversity
    - **Centered ICE (c-ICE)**: Anchors all curves at zero for easier comparison

19. **Accumulated Local Effects (ALE)**
    - **Motivation**: Overcome PDP's feature independence assumption
    - **Methodology**: Averages changes in predictions within local neighborhoods
    - **Algorithm**:
      1. Divide feature range into intervals
      2. Calculate prediction differences within each interval
      3. Accumulate differences to create global effect plot
    - **Advantages**:
      - Unbiased even with correlated features
      - Faster computation than PDP
      - More accurate for complex models
      - Handles feature interactions better
    - **Comparison with PDP**:
      - PDP: Marginal effects (assumes independence)
      - ALE: Conditional effects (handles correlations)
    - **Limitations**: More complex to understand conceptually
    - **Best Practice**: Use ALE when features are correlated, PDP for independent features
    - **Implementation**: `ALEPlot` from `alepython` library or custom implementation

20. **Feature Interaction Analysis**
    - **Definition**: Quantifying how features jointly influence predictions
    - **H-Statistic (Friedman's H)**: Measures interaction strength
      - H = 0: No interaction (purely additive)
      - H = 1: Full interaction (completely non-additive)
    - **2D Partial Dependence**: Heatmaps showing joint effects of two features
    - **Methodology**:
      1. Calculate predictions for all combinations of two features
      2. Compare joint effect vs. sum of individual effects
      3. Quantify deviation as interaction strength
    - **Visualization Techniques**:
      - 2D heatmaps for feature pairs
      - Interaction matrices showing all pairwise interactions
      - Network graphs for complex interaction patterns
    - **Applications**:
      - Feature engineering (creating interaction terms)
      - Model debugging (unexpected interactions)
      - Domain validation (confirming known interactions)
    - **Computational Cost**: O(n²) for all pairwise interactions

**Key Concepts:**
- Feature importance methods quantify which features most influence predictions
- Permutation and drop-column importance measure actual prediction impact
- PDP and ICE visualize feature-prediction relationships globally and locally
- ALE provides unbiased feature effects even with correlations
- Interaction analysis reveals how features jointly influence predictions
- Different methods have distinct computational costs and assumptions

**Learning Outcomes:**
- Calculate permutation and drop-column importance for any model
- Generate and interpret PDP and ICE plots
- Understand when to use ALE over PDP
- Quantify and visualize feature interactions
- Select appropriate importance methods based on computational budget and data characteristics

---

### Part 4/4: Model-Agnostic Methods

**Topics Covered:**

21. **Surrogate Models**
    - **Definition**: Approximating black-box model with interpretable model
    - **Process**:
      1. Train complex black-box model (neural network, ensemble, etc.)
      2. Generate predictions on data using black-box model
      3. Train interpretable surrogate model to mimic black-box predictions
      4. Interpret surrogate model to understand black-box behavior
    - **Types**:
      - **Global Surrogate**: Mimics entire model behavior across all data
      - **Local Surrogate**: Approximates model in specific region of interest
    - **Common Surrogate Models**:
      - Decision Trees (most popular due to rule extraction)
      - Linear Models (coefficient interpretation)
      - Rule Sets (logical rule extraction)
      - Generalized Additive Models (non-linear relationships)
    - **Fidelity**: Measure of how well surrogate matches original model
    - **Advantages**:
      - Leverage simple model interpretability for complex models
      - Model compression (smaller, faster models)
      - Debugging tool for black-box behavior
    - **Limitations**:
      - May not capture full complexity of original model
      - Fidelity-interpretability trade-off
      - Can be misleading if fidelity is low
    - **Use Cases**: Model explanation, compression, regulatory compliance

22. **Introduction to LIME (Local Interpretable Model-agnostic Explanations)**
    - **Core Idea**: Explain individual predictions by approximating model locally
    - **Methodology**:
      1. Select instance to explain
      2. Generate perturbed samples around instance
      3. Get black-box predictions for perturbed samples
      4. Weight samples by proximity to original instance
      5. Train interpretable model (e.g., linear regression) on weighted samples
      6. Interpret the local model as explanation
    - **Key Components**:
      - **Perturbed Dataset**: Created by sampling around original instance
      - **Proximity Weight**: Closer samples have higher weight (kernel function)
      - **Interpretable Representation**: Simplified feature space (e.g., binary presence/absence)
      - **Local Model**: Simple model (linear, tree) trained on perturbed data
    - **Explanation Output**:
      - Feature importance scores for the specific prediction
      - Positive/negative contribution visualization
      - Decision boundary in local region
    - **Advantages**:
      - Model-agnostic (works with any model)
      - Provides instance-specific explanations
      - Flexible perturbation strategies
      - Intuitive visual explanations
    - **Supported Data Types**:
      - Tabular data (feature perturbation)
      - Image data (superpixel perturbation)
      - Text data (word/token perturbation)
    - **Implementation**: `lime` Python library

23. **LIME Advanced Topics**
    - **Hyperparameter Tuning**:
      - **Kernel Width**: Controls locality of explanation (narrower = more local)
      - **Number of Samples**: More samples = more stable explanations (computational trade-off)
      - **Number of Features**: How many features to include in explanation
    - **Stability and Consistency**:
      - Running LIME multiple times may give different explanations
      - Solution: Use larger sample sizes, aggregate multiple runs
    - **Submodular Pick for Representative Explanations**:
      - Algorithm for selecting diverse, representative instances to explain
      - Provides coverage of model behavior across dataset
    - **Handling Different Data Types**:
      - **Tabular**: Gaussian perturbations or discretization
      - **Images**: Superpixel segmentation and masking
      - **Text**: Word removal or replacement
    - **Fidelity Metrics**: Measuring how well local model approximates black-box
    - **Limitations and Criticisms**:
      - Instability: Different runs produce different explanations
      - Locality: Explanation only valid in small neighborhood
      - Sampling: Quality depends on perturbation strategy
      - Interpretable representation may lose information
    - **Best Practices**:
      - Validate explanations with domain experts
      - Use consistent hyperparameters across comparisons
      - Check explanation stability by running multiple times
      - Combine with global methods for comprehensive understanding

24. **Anchor Explanations**
    - **Definition**: High-precision rules that "anchor" predictions
    - **Core Concept**: Find IF-THEN rules with high coverage and precision
    - **Anchor Properties**:
      - **Precision**: Probability that rule leads to same prediction (typically > 0.95)
      - **Coverage**: Proportion of data satisfying the rule
      - **Simplicity**: Minimal number of conditions
    - **Example Anchor**:
      - Rule: "IF Age > 30 AND Income > 50K THEN Approved (Precision: 97%)"
      - Means: 97% of instances meeting these conditions are approved
    - **Algorithm**:
      1. Start with empty rule
      2. Iteratively add conditions (features) that increase precision
      3. Stop when precision threshold reached
      4. Balance precision, coverage, and rule length
    - **Advantages over LIME**:
      - More interpretable (IF-THEN format)
      - Easier to validate (can check precision directly)
      - Provides decision boundaries
      - More stable explanations
    - **Limitations**:
      - May produce complex rules for difficult instances
      - Computational cost for finding optimal anchors
      - Coverage may be low for strict precision requirements
    - **Use Cases**:
      - Regulatory compliance (explainable decision rules)
      - Medical diagnosis (clear condition-outcome relationships)
      - Credit decisions (transparent approval criteria)
    - **Implementation**: `anchor` Python library

25. **Practical Guidelines and Best Practices**
    - **XAI Method Selection Decision Tree**:
      - **Step 1**: Need Global or Local explanation?
        - Global → Feature Importance, PDP, ALE, or Surrogate Models
        - Local → LIME, SHAP, or Anchor
      - **Step 2**: Model-specific method possible?
        - Yes → Use model's built-in interpretability (coefficients, feature importance)
        - No → Use model-agnostic methods
      - **Step 3**: Real-time explanation needed?
        - Yes → Use efficient methods (Anchor, pre-computed SHAP)
        - No → Use comprehensive methods (LIME, surrogate models)
    
    - **Best Practice Guidelines**:
      - ✓ Use multiple XAI methods (triangulation for robustness)
      - ✓ Validate with domain experts (check if explanations make sense)
      - ✓ Consider computational budget (real-time vs. batch processing)
      - ✓ Document limitations (assumptions, scope, potential biases)
      - ✓ Match stakeholder needs (technical vs. non-technical audience)
    
    - **Method Selection Criteria**:
      - **Audience**: Technical users can handle complex methods, non-technical need simple visuals
      - **Timeliness**: Real-time applications need fast methods
      - **Scope**: Global understanding vs. specific prediction explanation
      - **Model Access**: White-box vs. black-box constraints
      - **Regulation**: Legal requirements may mandate specific explanation types
    
    - **Validation Approaches**:
      - Match explanations with domain expertise
      - Check consistency across similar instances
      - Test explanation stability with perturbations
      - Verify fidelity of approximate methods
    
    - **Common Pitfalls**:
      - ⚠ Over-reliance on single XAI method
      - ⚠ Ignoring data distribution shifts
      - ⚠ Not validating explanation stability
      - ⚠ Assuming correlation implies causation
      - ⚠ Adversarial manipulation of explanations
    
    - **Documentation Requirements**:
      - Assumptions made by XAI method
      - Limitations and scope of explanations
      - Hyperparameters used
      - Validation procedures
      - Known failure modes
    
    - **Monitoring and Maintenance**:
      - Track explanation drift over time
      - Re-validate explanations when model updates
      - Monitor for adversarial attacks on explanations
      - Update documentation as understanding evolves

**Key Concepts:**
- Model-agnostic methods work with any black-box model architecture
- Surrogate models provide global approximations using interpretable models
- LIME creates local linear approximations for individual predictions
- Anchor explanations provide high-precision IF-THEN rules
- Method selection depends on scope (global/local), timeliness, and audience
- Multiple XAI methods should be combined for robust understanding

**Learning Outcomes:**
- Build and evaluate surrogate models for black-box approximation
- Implement LIME for local instance explanations
- Generate Anchor rules for high-precision decision explanations
- Select appropriate XAI methods based on use case requirements
- Validate and document XAI explanations properly
- Avoid common pitfalls in XAI implementation

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Machine Learning Fundamentals**: Supervised learning, model evaluation, overfitting/underfitting
- **Statistics**: Probability distributions, hypothesis testing, correlation vs. causation
- **Python Programming**: Data structures, functions, object-oriented basics
- **NumPy/Pandas**: Array manipulation, dataframe operations, data preprocessing
- **Matplotlib/Seaborn**: Basic plotting, customization

### Recommended Background:
- Experience with scikit-learn for model training and evaluation
- Understanding of regularization (L1/L2)
- Familiarity with tree-based models (Random Forest, Gradient Boosting)
- Basic knowledge of neural networks (optional but helpful)

### Required Software:
```
Python >= 3.8
scikit-learn >= 1.0
numpy >= 1.20
pandas >= 1.3
matplotlib >= 3.4
seaborn >= 0.11
lime >= 0.2
shap >= 0.40 (optional)
alepython >= 0.1 (for ALE plots)
```

### Installation:
```bash
pip install scikit-learn numpy pandas matplotlib seaborn lime shap alepython
```

### Hardware Requirements:
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB+ RAM, 4-core CPU (for faster permutation importance and LIME)
- **Storage**: 500MB for datasets and model files

---

## 💻 Hands-on Components

### Project 1: Building and Comparing Interpretable Models
**Objective**: Implement multiple interpretable models and compare their explanations

**Dataset**: Credit approval dataset (binary classification)
- Features: Age, Income, Credit Score, Employment Length, Education, Debt Ratio
- Target: Approval decision (Approve/Reject)
- Size: ~1000 samples

**Tasks**:
1. **Data Preparation**:
   - Load and explore dataset
   - Handle missing values and encode categorical features
   - Split into train/validation/test sets (60%/20%/20%)

2. **Model Training**:
   - Logistic Regression (baseline interpretable model)
   - Decision Tree (max_depth=5 for interpretability)
   - Random Forest (as comparison complex model)
   - GAM using splines or `interpret` library

3. **Interpretation**:
   - Extract and visualize linear model coefficients
   - Plot decision tree structure
   - Compare feature importance across models
   - Generate partial dependence plots for top 3 features

4. **Analysis**:
   - Accuracy vs. interpretability trade-off
   - Feature importance consistency across methods
   - Identify contradictory insights

**Expected Outputs**:
- Coefficient bar chart for logistic regression
- Decision tree visualization with leaf node statistics
- Comparative feature importance plot
- PDP/ICE plots showing non-linear relationships
- Written interpretation report (2-3 pages)

**Time Estimate**: 2-3 hours

---

### Project 2: Feature Importance Analysis Pipeline
**Objective**: Implement and compare multiple feature importance techniques

**Dataset**: House price prediction (regression)
- Features: Square footage, bedrooms, bathrooms, location, age, amenities
- Target: House price
- Size: ~5000 samples

**Tasks**:
1. **Train Baseline Models**:
   - Gradient Boosting Regressor
   - Random Forest Regressor
   - Neural Network (simple MLP)

2. **Compute Feature Importance**:
   - Built-in feature importance (tree-based models)
   - Permutation importance (all models)
   - Drop-column importance (Random Forest only, due to speed)

3. **Visualization**:
   - Generate PDP for top 5 features
   - Create ICE plots to show heterogeneity
   - Compute ALE plots and compare with PDP
   - Analyze 2-way interactions using H-statistic

4. **Comparative Analysis**:
   - Rank features by different importance metrics
   - Identify disagreements between methods
   - Investigate why certain features rank differently
   - Validate with domain knowledge (e.g., square footage should be important)

**Expected Outputs**:
- Feature importance comparison table
- Side-by-side PDP vs. ALE plots
- ICE plot showing instance-level heterogeneity
- Interaction heatmap for top features
- Analysis report explaining findings

**Time Estimate**: 3-4 hours

---

### Project 3: LIME and Anchor Explanations for Black-Box Models
**Objective**: Explain individual predictions from a complex black-box model

**Dataset**: Medical diagnosis dataset (binary classification)
- Features: Patient age, symptoms, lab values, medical history
- Target: Disease presence (Yes/No)
- Size: ~2000 samples
- Note: Use synthetic or public dataset due to sensitivity

**Tasks**:
1. **Train Black-Box Model**:
   - Gradient Boosting Classifier or Neural Network
   - Optimize for high accuracy (>85%)
   - Intentionally create a complex, non-interpretable model

2. **LIME Implementation**:
   - Install and configure `lime` library
   - Explain 5 diverse predictions:
     - 2 high-confidence correct predictions
     - 2 low-confidence borderline predictions
     - 1 misclassified instance
   - Tune hyperparameters:
     - Number of samples: [1000, 5000, 10000]
     - Kernel width: [0.25, 0.75, 1.0]
     - Number of features: [5, 10, all]
   - Analyze stability by running LIME 10 times per instance

3. **Anchor Explanations**:
   - Generate Anchor rules for same 5 instances
   - Set precision threshold = 0.95
   - Compare Anchor rules with LIME explanations

4. **Validation**:
   - Check if explanations align with medical knowledge
   - Test explanation stability
   - Measure fidelity of LIME approximations
   - Evaluate Anchor coverage

**Expected Outputs**:
- LIME explanation visualizations (bar charts showing feature contributions)
- Anchor rules with precision/coverage statistics
- Stability analysis (variance across LIME runs)
- Comparison table: LIME vs. Anchor explanations
- Validation report with domain expert feedback (simulated or real)

**Time Estimate**: 4-5 hours

---

### Project 4: Surrogate Model Construction
**Objective**: Build global and local surrogate models to approximate a complex model

**Dataset**: Customer churn prediction (binary classification)
- Features: Account age, usage metrics, support interactions, billing history
- Target: Churn (Yes/No)
- Size: ~10000 samples

**Tasks**:
1. **Train Complex Black-Box Model**:
   - Ensemble model (e.g., stacking classifier with XGBoost + LightGBM + Neural Network)
   - Achieve high performance as baseline

2. **Global Surrogate**:
   - Use black-box to predict entire dataset
   - Train decision tree surrogate (various max_depths: 3, 5, 7, 10)
   - Measure fidelity: agreement between surrogate and original
   - Extract and interpret decision rules

3. **Local Surrogate**:
   - Select region of interest (e.g., high-risk customers)
   - Train local surrogate only on this subset
   - Compare local vs. global surrogate explanations

4. **Evaluation**:
   - Fidelity metrics: Agreement %, AUC comparison
   - Interpretability: Rule complexity, tree depth
   - Trade-off analysis: Fidelity vs. interpretability

**Expected Outputs**:
- Fidelity vs. tree depth plot
- Extracted decision rules from surrogate tree
- Confusion matrix: surrogate vs. original predictions
- Local vs. global surrogate comparison
- Recommendations for deployment

**Time Estimate**: 3-4 hours

---

## 📖 Additional Resources

### Academic Papers:
1. **LIME**: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." ACM SIGKDD.
2. **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
3. **Anchors**: Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). "Anchors: High-Precision Model-Agnostic Explanations." AAAI.
4. **ALE**: Apley, D. W., & Zhu, J. (2020). "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models." JRSS-B.
5. **Interpretable ML Book**: Molnar, C. (2022). "Interpretable Machine Learning: A Guide for Making Black Box Models Explainable." https://christophm.github.io/interpretable-ml-book/

### Python Libraries:
- **scikit-learn**: Core ML and inspection tools
  - Documentation: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection
- **LIME**: Local explanations
  - GitHub: https://github.com/marcotcr/lime
- **SHAP**: Unified explanation framework (optional, not covered in depth)
  - GitHub: https://github.com/slundberg/shap
- **interpret**: Microsoft's interpretability library
  - GitHub: https://github.com/interpretml/interpret
- **alepython**: ALE plots
  - GitHub: https://github.com/blent-ai/ALEPython
- **alibi**: Comprehensive XAI library
  - Documentation: https://docs.seldon.io/projects/alibi/en/latest/

### Textbooks:
- Molnar, C. (2022). *Interpretable Machine Learning*. (Free online)
- Biecek, P., & Burzykowski, T. (2021). *Explanatory Model Analysis*. Chapman & Hall/CRC.
- Linardatos, P., Papastefanopoulos, V., & Kotsiantis, S. (2021). *Explainable AI: A Review of Machine Learning Interpretability Methods*. Entropy.

### Online Courses:
- Coursera: "Explainable AI (XAI)" Specialization
- fast.ai: "Practical Deep Learning" (includes interpretability module)
- Kaggle Learn: "Machine Learning Explainability" (free micro-course)

### Video Lectures:
- Christoph Molnar's YouTube series on Interpretable ML
- LIME/SHAP tutorials by Data Science Dojo
- Google's People + AI Research (PAIR) talks on XAI

### Regulatory Guidelines:
- **GDPR Article 22**: Right to explanation for automated decisions
- **EU AI Act**: Transparency requirements for high-risk AI systems
- **NIST AI Risk Management Framework**: Includes explainability considerations

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Sequential Learning**: Follow slides in order (Part 1 → Part 2 → Part 3 → Part 4)
2. **Hands-on Practice**: Complete each project after finishing relevant lecture part
3. **Interactive Slideshow**: Open `lecture19_slideshow.html` in web browser for navigation
4. **Code-Along**: Replicate examples from Slide 18 in Jupyter notebook
5. **Deep Dive**: Read referenced papers for theoretical foundations

### For Classroom Instruction:
1. **Lecture Format** (3-4 hours total):
   - Part 1: 45 minutes (introduction, motivation, framework)
   - Part 2: 60 minutes (intrinsically interpretable models + hands-on demo)
   - Break: 15 minutes
   - Part 3: 45 minutes (feature importance methods)
   - Part 4: 45 minutes (model-agnostic methods + practical guidelines)

2. **Workshop Format** (Full day):
   - Morning: Parts 1-2 with extensive hands-on exercises
   - Lunch: 60 minutes
   - Afternoon: Parts 3-4 with project work
   - Final hour: Project presentations and Q&A

3. **Flipped Classroom**:
   - Pre-class: Students review slides and read Molnar chapters
   - In-class: Focus on hands-on projects and troubleshooting
   - Post-class: Complete remaining projects and peer review

### Navigation:
- **Slideshow Controls**: Use arrow keys (← →), space bar, or on-screen buttons
- **Direct Access**: Open individual HTML files for specific topics
- **Progress Tracking**: Progress bar shows lecture completion percentage
- **Slide Titles**: Displayed in top-left corner for orientation

---

## 📊 Assessment Suggestions

### Formative Assessments:

1. **Concept Check Quizzes** (After each Part):
   - Part 1: XAI terminology, evaluation criteria (10 questions, 15 minutes)
   - Part 2: Interpreting coefficients, tree structures (10 questions, 15 minutes)
   - Part 3: Feature importance methods comparison (10 questions, 15 minutes)
   - Part 4: LIME vs. Anchor trade-offs (10 questions, 15 minutes)

2. **Hands-on Coding Checkpoints**:
   - During Project 1: Verify students can extract and plot coefficients
   - During Project 2: Check PDP/ALE implementation correctness
   - During Project 3: Validate LIME configuration and output interpretation
   - During Project 4: Review surrogate model fidelity calculations

3. **Peer Review**:
   - Students exchange explanation reports
   - Evaluate clarity, correctness, and completeness
   - Provide constructive feedback using rubric

### Summative Assessments:

1. **Take-Home Project** (40% of grade):
   - **Task**: Apply XAI methods to novel dataset (provided or student-selected)
   - **Requirements**:
     - Train at least one interpretable and one black-box model
     - Apply minimum 3 different XAI techniques
     - Compare and validate explanations
     - Write 5-7 page report with visualizations
   - **Rubric**:
     - Technical correctness (40%): Proper implementation, correct interpretation
     - Analysis depth (30%): Comparative insights, validation
     - Presentation (20%): Clarity, visualization quality, organization
     - Creativity (10%): Novel applications, thoughtful method selection
   - **Due**: 2 weeks after lecture

2. **Written Exam** (30% of grade):
   - **Format**: 2-hour closed-book exam
   - **Content**:
     - Short answer: XAI concepts, method characteristics (30%)
     - Interpretation: Given visualizations (PDP, LIME, etc.), explain insights (40%)
     - Problem-solving: Select appropriate XAI method for scenarios (30%)
   - **Sample Questions**:
     - "Explain when ALE is preferred over PDP and why."
     - "Given this LIME explanation, what are the top 3 factors for the prediction?"
     - "A bank wants real-time explanations for loan decisions. Recommend an XAI method and justify."

3. **Presentation** (20% of grade):
   - **Task**: Present take-home project findings to class
   - **Duration**: 10 minutes + 5 minutes Q&A
   - **Evaluation Criteria**:
     - Content clarity (40%): Clear explanation of methods and results
     - Visual quality (30%): Effective use of figures and tables
     - Time management (15%): Within allocated time
     - Q&A handling (15%): Thoughtful responses to questions

4. **Class Participation** (10% of grade):
   - Active engagement during discussions
   - Thoughtful questions during lecture
   - Helping peers during hands-on sessions
   - Contribution to collaborative learning

### Grading Rubric Example (for Take-Home Project):

| Criterion | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (<70%) |
|-----------|---------------------|---------------|------------------------|--------------------------|
| **Implementation** | Flawless code, all methods correctly implemented, excellent documentation | Minor bugs, mostly correct, good documentation | Some errors, functional but incomplete, basic documentation | Major errors, non-functional code, poor documentation |
| **Interpretation** | Deep insights, correct understanding, validates explanations | Mostly correct interpretation, some validation | Surface-level interpretation, limited validation | Incorrect or missing interpretation |
| **Comparison** | Thorough comparison of methods, identifies trade-offs, actionable insights | Good comparison, identifies main differences | Basic comparison, lists differences | Weak or missing comparison |
| **Presentation** | Professional quality, clear visualizations, well-organized | Good quality, effective visuals, organized | Adequate quality, some unclear visuals, loosely organized | Poor quality, confusing visuals, disorganized |

---

## 📝 Notes for Implementation

### Technical Requirements:

1. **Computing Environment**:
   - **Jupyter Notebook/Lab**: Recommended for interactive development
   - **IDE**: VS Code, PyCharm, or Spyder for script development
   - **Version Control**: Git for tracking code changes
   - **Environment Management**: `conda` or `venv` for dependency isolation

2. **Performance Considerations**:
   - **Permutation Importance**: O(n × m) where n=features, m=samples
     - Optimization: Use multiprocessing with `n_jobs=-1`
     - Time estimate: ~10 seconds for 10 features, 1000 samples, simple model
   - **LIME**: 
     - Bottleneck: Number of perturbation samples (default 5000)
     - Optimization: Reduce samples to 1000 for faster explanations
     - Time estimate: ~2-5 seconds per explanation with 1000 samples
   - **PDP/ALE**: 
     - Computation scales with grid resolution
     - Optimization: Use `n_jobs=-1` for parallel computation
     - Time estimate: ~5-10 seconds for single feature with 100 grid points

3. **Memory Requirements**:
   - **Large Datasets** (>100K samples):
     - Use sampling for LIME (select representative subset)
     - Compute PDP/ALE on validation set rather than full training set
     - Consider using Dask or distributed computing for permutation importance
   - **Expected Memory Usage**:
     - Small dataset (1K samples): <500 MB
     - Medium dataset (10K samples): ~2 GB
     - Large dataset (100K samples): ~10 GB (use sampling strategies)

4. **Reproducibility**:
   - Set random seeds: `np.random.seed(42)`, `random.seed(42)`
   - Save model checkpoints before explanation generation
   - Document library versions in `requirements.txt`
   - Store hyperparameters in configuration files (JSON/YAML)

### Common Implementation Pitfalls:

1. **Feature Scaling Issues**:
   - **Problem**: Coefficients not comparable when features have different scales
   - **Solution**: Standardize features before training linear models
   - **Example**: Income (0-100K) vs. Age (0-100) need scaling for fair coefficient comparison

2. **LIME Instability**:
   - **Problem**: Different runs produce different explanations
   - **Solution**: Increase number of samples (>5000), fix random seed, or average multiple runs
   - **Best Practice**: Report confidence intervals for LIME explanations

3. **Correlated Features**:
   - **Problem**: Permutation importance misleading, PDP assumes independence
   - **Solution**: Use ALE instead of PDP, compute VIF before interpretation
   - **Example**: Square footage and number of rooms are highly correlated

4. **Overfitting in Surrogate Models**:
   - **Problem**: Surrogate overfits to black-box predictions, not true patterns
   - **Solution**: Prune decision trees, use validation set for surrogate training
   - **Validation**: Compare surrogate predictions on holdout test set

5. **Misinterpreting Causation**:
   - **Problem**: Feature importance ≠ causal effect
   - **Reminder**: XAI shows correlation and model behavior, not causal relationships
   - **Example**: "Time of day" important for predictions doesn't mean it causes the outcome

### Debugging Checklist:

- [ ] Data loaded correctly (check shapes, missing values)
- [ ] Train/test split performed before any preprocessing
- [ ] Features scaled appropriately for linear models
- [ ] Random seeds set for reproducibility
- [ ] Model converged (check training logs, warnings)
- [ ] Predictions reasonable (sanity check on sample instances)
- [ ] Explanations make domain sense (validate with expert if possible)
- [ ] Visualizations labeled and titled correctly
- [ ] Code documented with comments
- [ ] Results saved to files (models, figures, reports)

### Extension Ideas:

1. **Advanced Topics** (for motivated students):
   - SHAP values: TreeExplainer, DeepExplainer
   - Counterfactual explanations: "What-if" scenarios
   - Contrastive explanations: Why A instead of B?
   - Adversarial robustness of explanations

2. **Domain-Specific Applications**:
   - Healthcare: Medical diagnosis with LIME
   - Finance: Credit risk with monotonic constraints
   - Computer Vision: Image classification with attention maps
   - NLP: Text classification with word importance

3. **Research Projects**:
   - Compare explanation stability across XAI methods
   - Develop new evaluation metrics for explanation quality
   - Human-subject study on explanation effectiveness
   - Adversarial attacks on LIME/SHAP explanations

---

## 🙏 Credits

**Instructor**: Ho-min Park, Ph.D.  
**Affiliation**: Ghent University Global Campus  
**Email**: homin.park@ghent.ac.kr | powersimmani@gmail.com

**Lecture Materials**:
- All slide designs and content created by Ho-min Park
- Visualizations and examples developed specifically for this course
- HTML/CSS slide implementation for interactive learning

**Acknowledgments**:
- Inspired by Christoph Molnar's "Interpretable Machine Learning" book
- LIME and Anchor methods by Marco Tulio Ribeiro et al.
- scikit-learn documentation and community
- Students and colleagues who provided feedback on materials

**License**: These materials are provided for educational purposes. Please cite appropriately if using in academic work or courses.

**Version**: 1.0 (November 2025)

---

**Last Updated**: November 7, 2025  
**Course Code**: [Your course code]  
**Semester**: [Your semester]