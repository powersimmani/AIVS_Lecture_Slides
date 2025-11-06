# Lecture 6: Supervised Learning Evaluation

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Course Level:** Intermediate  
**Total Slides:** 32  
**Estimated Duration:** 120-150 minutes  
**Prerequisites:** Understanding of supervised learning algorithms, basic statistics  

This comprehensive lecture provides an in-depth exploration of evaluation methodologies for supervised machine learning models. The course covers both theoretical foundations and practical implementation strategies for assessing regression and classification models. Students will learn to select appropriate evaluation metrics, implement robust validation techniques, and avoid common pitfalls such as data leakage and overfitting.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Design robust evaluation frameworks** that properly split data and prevent information leakage between training, validation, and test sets
2. **Select and interpret appropriate evaluation metrics** for both regression (MSE, RMSE, MAE, R², MAPE) and classification (Precision, Recall, F1, AUC-ROC) tasks based on problem requirements
3. **Implement advanced validation techniques** including K-fold cross-validation, stratified sampling, and bootstrapping to obtain reliable performance estimates
4. **Diagnose model issues** through residual analysis, confusion matrices, and learning curves to identify overfitting, underfitting, and bias-variance tradeoffs
5. **Optimize hyperparameters systematically** using grid search, random search, and Bayesian optimization while maintaining proper validation protocols

---

## 📚 Lecture Structure

### Part 1/4: Importance and Fundamentals of Evaluation (Slides 3-11)

#### Topics Covered:

**1. Why is Evaluation Important?**
- Validates model performance before deployment in real-world scenarios
- Prevents costly errors from poorly performing models
- Guides model selection and hyperparameter tuning decisions
- Identifies overfitting and underfitting issues early
- Builds trust and confidence in ML systems
- Ensures models generalize well to unseen data
- Enables objective comparison between different algorithms

**2. Train vs Validation vs Test**
- Training Set (60-70%): Fit model parameters and learn patterns
- Validation Set (15-20%): Tune hyperparameters and act as proxy for real-world performance
- Test Set (15-20%): Final unbiased evaluation, completely unseen until final assessment
- Best Practice: Never use test data during model development

**3. Overfitting and Underfitting**
- Underfitting: Too simple model, high training error, high validation error
- Good Fit: Balance between bias and variance, low training error, low validation error
- Overfitting: Memorizes training data, low training error, high validation error
- Key Insight: Model complexity should match problem complexity

**4. Bias-Variance Tradeoff**
- Bias: Error from oversimplified assumptions (leads to underfitting)
- Variance: Error from sensitivity to training data fluctuations (leads to overfitting)
- Total Error = Bias² + Variance + Irreducible Error
- Goal: Minimize both bias and variance simultaneously

**5. Data Splitting Strategies**
- Random Split: Simple approach for large, i.i.d. datasets
- Stratified Split: Maintains class distribution in each subset
- Time-based Split: Essential for temporal or sequential data
- Group-based Split: Prevents data leakage from related samples

**6. Stratified Sampling**
- Maintains class distribution across all splits
- Critical for imbalanced datasets
- Prevents biased evaluation
- Reduces variance in performance estimates

**7. Time Series Data Splitting**
- Never shuffle temporal data
- Use forward chaining (train on past, validate on future)
- Two approaches: Sliding window (fixed size) and Expanding window (growing size)
- Respect temporal dependencies and model drift

**8. Preventing Data Leakage**
- Definition: Information from test set influences training
- Common sources: Using test statistics, features with future info, duplicate samples, improper CV folds
- Correct workflow: Split FIRST, then normalize using only train statistics
- Consequence: Overly optimistic performance leading to production failures

#### Key Concepts:
1. **Proper data partitioning** is the foundation of honest model evaluation
2. **The bias-variance tradeoff** explains the fundamental tension in model complexity
3. **Data leakage prevention** requires careful attention to the order of operations in ML pipelines
4. **Different data types** (tabular, time series, grouped) require different splitting strategies
5. **Stratification** is essential for maintaining representative samples in imbalanced datasets

#### Learning Outcomes:
- Students can design a complete evaluation framework from data splitting to final testing
- Students understand when and why each splitting strategy should be applied
- Students can identify and prevent common data leakage scenarios
- Students can diagnose overfitting vs underfitting from learning curves

---

### Part 2/4: Regression Evaluation Metrics (Slides 12-16)

#### Topics Covered:

**9. MSE, RMSE, MAE**
- Mean Squared Error (MSE): Σ(yᵢ - ŷᵢ)² / n
  - Squares differences, heavily penalizes large errors
  - Units are squared, less interpretable
  - High outlier sensitivity
- Root Mean Squared Error (RMSE): √(MSE)
  - Same units as target variable
  - More interpretable than MSE
  - Still sensitive to outliers
- Mean Absolute Error (MAE): Σ|yᵢ - ŷᵢ| / n
  - Treats all errors equally
  - Robust to outliers
  - Same units as target

**10. R², Adjusted R²**
- R² (Coefficient of Determination): R² = 1 - (SS_res / SS_tot)
  - Measures proportion of variance explained
  - Range: -∞ to 1.0 (higher is better)
  - Interpretation: <0 (terrible), 0.25 (weak), 0.50 (moderate), 0.75 (strong), 1.0 (perfect)
  - Can artificially increase by adding features
- Adjusted R²: R²_adj = 1 - [(1-R²)(n-1)/(n-p-1)]
  - Penalizes adding features that don't help
  - Better for model comparison
  - Use for high-dimensional data

**11. MAPE, SMAPE**
- Mean Absolute Percentage Error (MAPE): (100/n) × Σ|yᵢ - ŷᵢ| / |yᵢ|
  - Scale-independent metric
  - Easy to interpret and communicate
  - Undefined when actual = 0
  - Asymmetric (penalizes over-prediction more)
- Symmetric MAPE (SMAPE): (100/n) × Σ|yᵢ - ŷᵢ| / [(|yᵢ| + |ŷᵢ|)/2]
  - Bounded: 0% to 200%
  - More stable than MAPE
  - Symmetric treatment of errors
  - Handles zero values better

**12. Residual Analysis and Diagnostics**
- Residuals: y - ŷ (differences between observed and predicted)
- Residual Plot: Check for random scatter around zero
- Q-Q Plot: Check normal distribution of residuals
- Scale-Location Plot: Detect heteroscedasticity
- What to check: Non-linearity, heteroscedasticity, outliers, normal distribution

#### Key Concepts:
1. **MSE/RMSE vs MAE**: Choose based on whether large errors should be penalized more heavily
2. **R² limitations**: Can be inflated by adding features; use Adjusted R² for model comparison
3. **Percentage metrics** (MAPE/SMAPE) enable comparison across different scales
4. **Residual patterns** reveal violations of model assumptions
5. **No single metric** tells the complete story; use multiple complementary metrics

#### Learning Outcomes:
- Students can select appropriate regression metrics based on problem characteristics
- Students understand the mathematical relationships between different error metrics
- Students can interpret R² values in the context of their domain
- Students can perform residual analysis to diagnose model problems

---

### Part 3/4: Classification Evaluation Metrics (Slides 17-24)

#### Topics Covered:

**13. Confusion Matrix**
- True Positive (TP): Correctly predicted positive cases
- False Positive (FP): Incorrectly predicted as positive (Type I Error)
- False Negative (FN): Incorrectly predicted as negative (Type II Error)
- True Negative (TN): Correctly predicted negative cases
- Foundation: All classification metrics derive from these four components

**14. Accuracy and Its Limitations**
- Formula: Accuracy = (TP + TN) / Total
- Advantages: Simple to calculate, easy to interpret, intuitive (0-100%)
- The Accuracy Paradox: 99% accuracy on 1% disease prevalence (by always predicting "healthy")
- Limitations: Misleading with imbalanced data, ignores different costs of FP vs FN, provides no error context
- Critical: Always consider class distribution before relying on accuracy alone

**15. Precision and Recall**
- Precision = TP / (TP + FP)
  - "When model says positive, how often is it right?"
  - Focuses on predicted positives
  - High precision needed when false positives are costly (e.g., spam filter)
- Recall = TP / (TP + FN)
  - "Of all actual positives, how many did we find?"
  - Focuses on actual positives
  - High recall needed when false negatives are costly (e.g., cancer screening)
- The Precision-Recall Tradeoff: Improving one often decreases the other

**16. F1 Score and Fβ**
- F1 Score: 2 × P × R / (P + R)
  - Harmonic mean of precision and recall
  - Balances both metrics equally
  - Single metric (0 to 1)
  - Penalizes extreme values
- Fβ Score: (1+β²) × P × R / (β²P + R)
  - β controls weight: β > 1 favors recall, β < 1 favors precision
  - F0.5: Emphasizes precision (minimize false alarms)
  - F2: Emphasizes recall (minimize misses)

**17. ROC Curve and AUC**
- ROC Curve: True Positive Rate vs False Positive Rate at various thresholds
- AUC (Area Under Curve): 0.5 (random), 0.7 (acceptable), 0.8 (good), 0.9+ (excellent)
- Advantages: Threshold-independent, good for comparing models
- Use when: Classes relatively balanced, care about ranking quality
- Provides comprehensive view of classifier performance across all thresholds

**18. Precision-Recall Curve**
- Plots Precision vs Recall at various thresholds
- Better than ROC for imbalanced datasets
- Area under PR curve (AP) summarizes performance
- Use when: Positive class is rare, focus on minority class
- More informative for highly skewed distributions

**19. Multi-class Metrics**
- Macro-average: Average metric per class (treats all classes equally)
- Weighted-average: Weighted by class support (accounts for imbalance)
- Micro-average: Aggregate contributions of all classes
- One-vs-Rest (OvR): Binary classification for each class
- Confusion matrix extends to n×n for n classes

#### Key Concepts:
1. **Confusion matrix components** form the foundation for all classification metrics
2. **Accuracy fails on imbalanced data**; always examine class distribution first
3. **Precision-Recall tradeoff** requires careful consideration of business costs
4. **ROC-AUC** works well for balanced classes; **PR curves** excel with imbalance
5. **Multi-class evaluation** requires aggregation strategies that match problem goals

#### Learning Outcomes:
- Students can construct and interpret confusion matrices for binary and multi-class problems
- Students understand when accuracy is misleading and can select better alternatives
- Students can explain precision-recall tradeoffs in domain-specific contexts
- Students can choose between ROC-AUC and PR curves based on class distribution

---

### Part 4/4: Model Validation Techniques (Slides 25-31)

#### Topics Covered:

**20. K-fold Cross Validation**
- Process: Split data into K folds, train K times using K-1 folds for training and 1 for validation
- Common: K=5 or K=10
- Final score: Mean of K validation scores
- Standard deviation indicates model stability
- Benefits: Every sample used for both training and validation, more reliable performance estimate, reduces variance

**21. Stratified K-fold**
- Maintains class distribution in each fold
- Essential for imbalanced datasets
- Regular K-fold may create folds with skewed distributions
- Reduces variance in performance estimates
- Always prefer for classification problems

**22. Leave-One-Out CV (LOOCV)**
- Special case: K = N (dataset size)
- Each sample used as test set exactly once
- Advantages: Maximum data use (N-1 for training), nearly unbiased, deterministic
- Disadvantages: Computationally expensive (N trainings), high variance, not practical for large datasets
- Use when: Small datasets where data is precious

**23. Bootstrapping**
- Sampling with replacement from original dataset
- Create B bootstrap samples of same size as original
- Train model on each sample
- Aggregate results for mean and confidence intervals
- Use for: Estimating model uncertainty, building ensembles (Bagging), small dataset validation
- Key feature: Same data point can appear multiple times in a bootstrap sample

**24. Hyperparameter Tuning**
- Grid Search: Exhaustive search through parameter combinations
  - Comprehensive, guaranteed to find best in grid
  - Can be computationally expensive
- Random Search: Random sampling from parameter space
  - More efficient, explores wider range
  - Often finds good solutions faster
- Bayesian Optimization: Uses probabilistic model to guide search
  - Smart search toward promising regions
  - Fewer iterations needed
- Workflow: Define parameter space → Choose strategy → Use CV for each config → Select best → Evaluate on test
- Best Practices: Use validation set (never test), start broad then narrow, log-scale for learning rates

**25. Model Selection Strategy**
- Complete workflow from data splitting through final evaluation
- Proper data partitioning: 60% train, 20% validation, 20% test
- Use cross-validation during model development
- Hyperparameter tuning on validation set only
- Final evaluation on completely held-out test set
- Never leak information between splits
- Document all decisions and performance metrics

#### Key Concepts:
1. **Cross-validation** provides more reliable estimates than single train-test splits
2. **Stratification** is critical for classification with imbalanced classes
3. **LOOCV** maximizes data use but at computational cost
4. **Bootstrapping** quantifies uncertainty and enables ensemble methods
5. **Systematic hyperparameter tuning** prevents overfitting to validation data

#### Learning Outcomes:
- Students can implement K-fold cross-validation with proper stratification
- Students understand computational tradeoffs of different validation schemes
- Students can design hyperparameter tuning experiments using appropriate search strategies
- Students can construct complete model selection pipelines that prevent data leakage

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Machine Learning Fundamentals:**
  - Understanding of supervised learning (regression and classification)
  - Familiarity with common algorithms (linear regression, logistic regression, decision trees)
  - Concept of training and testing
  - Basic understanding of model parameters vs hyperparameters

- **Statistics:**
  - Descriptive statistics (mean, median, standard deviation, variance)
  - Probability distributions (normal distribution, sampling)
  - Hypothesis testing concepts
  - Understanding of correlation and causation

- **Mathematics:**
  - Linear algebra basics (vectors, matrices)
  - Calculus fundamentals (derivatives, optimization concepts)
  - Understanding of squared errors and absolute values

- **Programming (Recommended):**
  - Python basics for hands-on implementation
  - Familiarity with numpy for numerical operations
  - Basic data manipulation concepts

### Software Requirements:
- **Python 3.8+** for practical implementation
- **Core Libraries:**
  - `scikit-learn>=1.0`: For model evaluation and validation
  - `numpy>=1.21`: For numerical computations
  - `pandas>=1.3`: For data manipulation
  - `matplotlib>=3.4` and `seaborn>=0.11`: For visualization

- **Recommended Tools:**
  - Jupyter Notebook or JupyterLab for interactive exploration
  - Git for version control of experiments

### Recommended Pre-readings:
- Chapter on Model Evaluation from "Introduction to Statistical Learning" (James et al.)
- Scikit-learn documentation on model evaluation metrics
- Understanding of Type I and Type II errors in hypothesis testing

---

## 💻 Hands-on Components

### Lab Exercise 1: Regression Model Evaluation
**Objective:** Implement comprehensive regression evaluation pipeline

**Tasks:**
1. Load California Housing dataset
2. Split data using proper train-validation-test protocol
3. Train multiple regression models (Linear Regression, Ridge, Random Forest)
4. Calculate all regression metrics: MSE, RMSE, MAE, R², Adjusted R², MAPE
5. Perform residual analysis with diagnostic plots
6. Compare models and justify selection

**Expected Output:**
- Comparison table of all metrics across models
- Residual diagnostic plots (residual plot, Q-Q plot, scale-location plot)
- Written justification (~200 words) for model selection
- Code notebook with clear documentation

**Evaluation Criteria:**
- Proper data splitting (no leakage)
- Correct metric calculations
- Quality of residual analysis
- Clarity of model comparison

---

### Lab Exercise 2: Classification with Imbalanced Data
**Objective:** Handle evaluation challenges in imbalanced classification

**Tasks:**
1. Use credit card fraud detection dataset (highly imbalanced)
2. Implement stratified train-test split
3. Train baseline classifier and evaluate with multiple metrics
4. Generate confusion matrix, calculate precision, recall, F1, F2
5. Plot ROC curve and Precision-Recall curve
6. Compare AUC-ROC vs Average Precision
7. Discuss why accuracy is misleading for this dataset

**Expected Output:**
- Confusion matrix visualization
- ROC and PR curves on same plot
- Metric comparison table showing why accuracy fails
- Recommendations for metric selection (500 words)

**Evaluation Criteria:**
- Proper handling of imbalanced data
- Correct calculation of all metrics
- Quality of visualizations
- Understanding demonstrated in write-up

---

### Lab Exercise 3: Cross-Validation and Hyperparameter Tuning
**Objective:** Implement robust validation and optimization

**Tasks:**
1. Use Iris dataset or similar multi-class classification problem
2. Implement 5-fold cross-validation manually (without sklearn)
3. Compare results with stratified vs non-stratified CV
4. Implement grid search for SVM hyperparameters (C, gamma)
5. Implement random search with same search space
6. Compare computational time and best scores
7. Evaluate final model on held-out test set

**Expected Output:**
- CV implementation code with clear documentation
- Comparison of stratified vs non-stratified results
- Grid search vs random search comparison (time, performance)
- Learning curves showing overfitting/underfitting
- Final test set evaluation

**Evaluation Criteria:**
- Correct CV implementation
- Proper isolation of test set
- Comprehensive comparison of tuning strategies
- Understanding of bias-variance tradeoff

---

### Lab Exercise 4: Time Series Validation
**Objective:** Apply proper validation to temporal data

**Tasks:**
1. Use stock price or energy consumption time series dataset
2. Implement time-based train-validation-test split (never shuffle!)
3. Implement sliding window cross-validation
4. Implement expanding window cross-validation
5. Compare results and discuss temporal considerations
6. Demonstrate data leakage with incorrect (shuffled) split

**Expected Output:**
- Visualization of different splitting strategies
- Performance comparison across validation methods
- Demonstration of performance degradation with data leakage
- Discussion of temporal considerations (500 words)

**Evaluation Criteria:**
- Respect for temporal ordering
- Correct implementation of time-based CV
- Clear demonstration of leakage effects
- Quality of temporal analysis

---

### Project: Complete ML Pipeline with Evaluation
**Objective:** Integrate all lecture concepts into production-ready pipeline

**Requirements:**
1. Select dataset from provided list or propose your own
2. Perform exploratory data analysis
3. Design appropriate data splitting strategy with justification
4. Train at least 3 different model types
5. Implement comprehensive evaluation framework:
   - All relevant metrics for your problem type
   - Cross-validation with appropriate strategy
   - Hyperparameter tuning with systematic approach
   - Residual or error analysis
6. Document all decisions and tradeoffs
7. Provide final model recommendation with evidence

**Deliverables:**
- Jupyter notebook with complete pipeline
- PDF report (8-10 pages) including:
  - Problem statement and dataset description
  - EDA insights
  - Methodology justification
  - Results with visualizations
  - Model comparison and selection
  - Limitations and future work
- Presentation slides (10 minutes)

**Grading Rubric:**
- Data splitting and leakage prevention (20%)
- Metric selection and justification (20%)
- Validation methodology (20%)
- Model comparison quality (20%)
- Documentation and reproducibility (10%)
- Presentation clarity (10%)

---

## 📖 Additional Resources

### Textbooks:
1. **"An Introduction to Statistical Learning"** by James, Witten, Hastie, Tibshirani
   - Chapter 5: Resampling Methods (CV, Bootstrap)
   - Chapter 6: Linear Model Selection and Regularization
   - Excellent for understanding bias-variance tradeoff

2. **"Pattern Recognition and Machine Learning"** by Christopher Bishop
   - Chapter 1.5: Decision Theory
   - Chapter 3: Linear Models for Regression

3. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron
   - Chapter 2: End-to-End Machine Learning Project
   - Chapter 3: Classification

### Online Resources:

**Documentation:**
- [Scikit-learn Model Evaluation Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Yellowbrick: Visual Diagnostic Tools](https://www.scikit-yb.org/)

**Interactive Tutorials:**
- [Distill.pub: Understanding ROC and PR Curves](https://distill.pub/)
- [Google's Machine Learning Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification)
- [Kaggle Learn: Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning)

**Video Lectures:**
- StatQuest with Josh Starmer: ROC and AUC Explained
- Andrew Ng's Machine Learning Course (Coursera): Week 6 on Model Evaluation
- Fast.ai Practical Deep Learning: Validation Sets and Cross-Validation

**Research Papers:**
1. "A Survey on Evaluation Metrics for Machine Learning Classification" (2020)
2. "The Relationship Between Precision-Recall and ROC Curves" (ICML 2006)
3. "An Introduction to Variable and Feature Selection" (JMLR 2003)

### Datasets for Practice:
- **Regression:** California Housing, Boston Housing, Ames Housing
- **Classification (Balanced):** Iris, Wine Quality, MNIST
- **Classification (Imbalanced):** Credit Card Fraud, Anomaly Detection, Medical Diagnosis
- **Time Series:** Stock Prices, Energy Consumption, Traffic Volume

### Code Examples Repository:
GitHub repository (to be provided) contains:
- Complete implementations of all evaluation metrics
- Visualization functions for confusion matrices, ROC, PR curves
- Cross-validation framework templates
- Hyperparameter tuning examples
- Real-world case studies

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Review slides sequentially** - Each part builds on previous concepts
2. **Pause after each part** to complete relevant lab exercises
3. **Work through code examples** in Jupyter notebooks before attempting labs
4. **Join study group or forum** to discuss tricky concepts (bias-variance tradeoff, metric selection)
5. **Estimated timeline:** 
   - Week 1: Part 1 + Lab Exercise 1
   - Week 2: Part 2 + Lab Exercise 2
   - Week 3: Part 3 + Lab Exercise 3
   - Week 4: Part 4 + Lab Exercise 4
   - Week 5-6: Final Project

### As Course Material:
1. **Lecture format** (3-hour class):
   - Part 1: 45 minutes + 10-minute break
   - Part 2: 30 minutes + 10-minute break
   - Part 3: 45 minutes + 10-minute break
   - Part 4: 30 minutes + Q&A

2. **Flipped classroom approach:**
   - Students review slides before class
   - Class time: Work through labs with instructor guidance
   - Discussions on real-world applications and edge cases

3. **Online/hybrid format:**
   - Release one part per week with corresponding lab
   - Weekly live Q&A sessions
   - Discussion forum for metric selection debates

### Viewing the Slideshow:
The complete slideshow is available in `lecture06_slideshow.html`:
- Open in any modern web browser
- Navigate with arrow keys (← →) or space bar
- Shows progress bar and slide counter
- Each slide is a standalone HTML file for modular teaching

---

## 📊 Assessment Suggestions

### Formative Assessment:

**1. Concept Checks (After Each Part):**
- Part 1: "Design a data splitting strategy for patient medical records. Explain your reasoning."
- Part 2: "Given R²=0.95 but high RMSE, what might be wrong?"
- Part 3: "For cancer screening, which is worse: FP or FN? Which metric matters most?"
- Part 4: "Why might 10-fold CV give different results than LOOCV on small datasets?"

**2. Quick Quizzes:**
- 5-question multiple choice after each part
- Focus on metric selection, common pitfalls, interpretation
- Example: "When is accuracy misleading? (A) Always (B) Never (C) With imbalanced classes (D) With small datasets"

**3. Code Reviews:**
- Peer review of lab exercise implementations
- Checklist: Data leakage checks, metric calculations, visualization quality
- Encourages learning from others' approaches

### Summative Assessment:

**1. Lab Exercises (40% of grade):**
- 4 labs weighted equally (10% each)
- Graded on correctness, code quality, documentation, insights

**2. Final Project (40% of grade):**
- Complete ML pipeline with evaluation
- Graded using provided rubric
- Includes code, report, and presentation

**3. Written Exam (20% of grade):**
- Part A: Metric selection justification (3 scenarios)
- Part B: Identify and fix evaluation mistakes (code debugging)
- Part C: Interpret confusion matrix and recommend actions
- Duration: 90 minutes, open notes

### Example Exam Questions:

**Question 1 (Metric Selection):**
"You're building a model to predict equipment failure in a manufacturing plant. False negatives cost $100,000 in downtime, while false positives cost $500 for unnecessary inspection. Your model achieves: Accuracy=92%, Precision=85%, Recall=78%, F1=81%.
(a) Which metric should you optimize? Explain why.
(b) Is this performance acceptable? Justify with cost analysis.
(c) What threshold adjustment might improve your target metric?"

**Question 2 (Data Leakage Debugging):**
```python
# Find the error in this code
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```
Explain the problem and provide corrected code.

**Question 3 (Interpretation):**
Given confusion matrix:
```
              Predicted
            Pos    Neg
Actual Pos  850    150
       Neg   50   2950
```
(a) Calculate accuracy, precision, recall, F1
(b) Is this an imbalanced dataset?
(c) Which metric is most informative? Why?
(d) If this is medical diagnosis, what's your recommendation?

---

## 📝 Notes for Implementation

### Technical Requirements:

**Minimum System Requirements:**
- Processor: Dual-core 2.0 GHz or better
- RAM: 8GB minimum (16GB recommended for larger datasets)
- Storage: 5GB free space for datasets and outputs
- Operating System: Windows 10+, macOS 10.14+, or Linux (Ubuntu 20.04+)

**Software Environment Setup:**

```bash
# Create conda environment
conda create -n ml_eval python=3.9
conda activate ml_eval

# Install required packages
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install scikit-learn==1.2.2
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install jupyter==1.0.0

# Optional: Advanced visualization
pip install yellowbrick==1.5
pip install shap==0.41.0

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

**Jupyter Notebook Extensions (Optional):**
```bash
pip install jupyterlab
pip install jupyterlab-git
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Dataset Preparation:

**Provided Datasets Location:**
All datasets should be organized in a `data/` directory:
```
data/
├── regression/
│   ├── california_housing.csv
│   ├── ames_housing.csv
│   └── energy_consumption.csv
├── classification/
│   ├── credit_fraud.csv
│   ├── medical_diagnosis.csv
│   └── customer_churn.csv
└── time_series/
    ├── stock_prices.csv
    └── traffic_volume.csv
```

**Loading Data Template:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load with proper handling
def load_dataset(path, has_header=True, index_col=None):
    """Load dataset with error handling"""
    try:
        df = pd.read_csv(path, 
                        header=0 if has_header else None,
                        index_col=index_col)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print(f"Error: {path} not found. Check your data directory.")
        return None
```

### Common Implementation Pitfalls:

**1. Data Leakage Prevention Checklist:**
```python
# ✓ CORRECT: Split before preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform

# ✗ WRONG: Preprocessing before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leaks test statistics
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

**2. Stratification for Imbalanced Data:**
```python
# Always use stratify for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintains class distribution
)

# Check distribution
print(f"Train distribution: {np.bincount(y_train)}")
print(f"Test distribution: {np.bincount(y_test)}")
```

**3. Cross-Validation Proper Usage:**
```python
from sklearn.model_selection import cross_val_score

# ✓ CORRECT: CV on training data only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# ✗ WRONG: CV on entire dataset
cv_scores = cross_val_score(model, X, y, cv=5)
# Then trying to use separate test set - information leakage!
```

**4. Metric Calculation Verification:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Always verify with multiple metrics
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")

# Check for class imbalance
print(f"Class distribution: {np.bincount(y_test)}")
```

### Performance Optimization Tips:

**1. Efficient Cross-Validation:**
```python
# Use n_jobs for parallel processing
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X_train, y_train,
    cv=5,
    scoring=['accuracy', 'precision', 'recall'],
    n_jobs=-1,  # Use all CPU cores
    return_train_score=True
)
```

**2. Hyperparameter Tuning Acceleration:**
```python
from sklearn.model_selection import RandomizedSearchCV

# Random search is faster than grid search
param_dist = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

random_search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist,
    n_iter=20,  # Try 20 random combinations
    cv=5,
    n_jobs=-1,
    random_state=42
)
```

### Debugging Checklist:

Before submitting any evaluation code, verify:
- [ ] Data split performed before any preprocessing
- [ ] Test set never used during development
- [ ] Cross-validation only on training data
- [ ] Stratification used for classification
- [ ] Random seeds set for reproducibility
- [ ] All metrics calculated on test predictions
- [ ] No data leakage in feature engineering
- [ ] Proper handling of time series (no shuffling)
- [ ] Confusion matrix labels match class names
- [ ] Visualizations have proper titles and labels

---

## 🙏 Credits

**Instructor:** Ho-min Park  
**Affiliation:** Ghent University Global Campus  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com

### Acknowledgments:
This lecture material draws upon concepts and best practices from:
- Scikit-learn documentation and tutorials
- Andrew Ng's Machine Learning course
- Fast.ai Practical Deep Learning course
- Research literature on model evaluation methodologies

### Citation:
If using these materials in your course or research, please cite as:
```
Park, H. (2024). Lecture 6: Supervised Learning Evaluation. 
Ghent University Global Campus.
```

### License:
These materials are provided for educational purposes. Please contact the instructor for permissions regarding commercial use or redistribution.

---

**Last Updated:** 2024  
**Version:** 1.0  
**Slide Count:** 32  
**Estimated Teaching Time:** 120-150 minutes