# Lecture 3: From Set Theory to Linear Regression

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Total Slides:** 31  
**Difficulty Level:** Intermediate to Advanced  
**Format:** HTML-based interactive slides with visual demonstrations

This comprehensive lecture bridges fundamental mathematical concepts with practical machine learning applications, specifically linear regression. The course systematically builds from abstract mathematical foundations through probability theory to culminate in a complete understanding of linear regression models.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Apply mathematical foundations** including set theory, linear algebra, and calculus to formulate and solve machine learning problems
2. **Understand probability and statistics** concepts essential for statistical inference and parameter estimation in regression models
3. **Derive and implement** linear regression solutions using both analytical (normal equation) and geometric interpretations
4. **Diagnose and validate** regression models through assumption checking, residual analysis, and appropriate diagnostic techniques
5. **Implement regression models** in Python using NumPy, scikit-learn, and statsmodels with proper validation methodology

---

## 📚 Lecture Structure

### Part 1/3: Mathematical Foundations (12 slides)

This section establishes the mathematical framework necessary for understanding machine learning algorithms, particularly regression models.

#### Topics Covered:

**1. Course Overview - From Sets to Regression**
- Journey from abstract mathematics to practical ML applications
- How sets define data spaces and domains
- Functions model relationships between variables
- Linear algebra provides computational framework
- Calculus enables model optimization
- Statistics connects theory to real-world data

**2. Set Theory Basics and Notation**
- Sets as collections of distinct objects: X = {x₁, x₂, ..., xₙ}
- Common sets: ℝ (real numbers), ℝⁿ (n-dimensional space)
- Membership notation: x ∈ X
- Set operations: Union (∪), Intersection (∩), Complement (')
- Cartesian product: X × Y = {(x,y) | x ∈ X, y ∈ Y}
- ML application: Data points live in feature space ℝⁿ (e.g., 28×28 image = ℝ⁷⁸⁴)

**3. Functions and Mapping Concepts**
- Function definition: f: X → Y (each input maps to exactly one output)
- Domain X: set of all possible inputs
- Codomain Y: set of all possible outputs
- Range: actual outputs achieved by f
- Function types: One-to-one (injective), Onto (surjective), Bijection
- Concrete examples with mathematical properties:
  - Natural number division: f: ℕ × ℕ\{0} → ℚ
  - Square function: f: ℝ → ℝ≥0 (surjective, not injective)
  - Absolute value: f: ℝ → ℝ≥0 (surjective, not injective)

**4. Vector Spaces and Basis**
- Vector space V: closed under addition and scalar multiplication
- Examples: ℝⁿ, polynomial space, function space
- Linear combination: v = c₁v₁ + c₂v₂ + ... + cₙvₙ
- Span: all possible linear combinations of vectors
- Linear independence: no vector is combination of others
- Basis: minimal spanning set (linearly independent)
- Standard basis in ℝⁿ: e₁ = (1,0,...,0), e₂ = (0,1,...,0)
- Any vector expressed as: v = c₁e₁ + c₂e₂ + ... + cₙeₙ

**5. Inner Product and Orthogonality**
- Inner product (dot product): ⟨x, y⟩ = x₁y₁ + x₂y₂ + ... + xₙyₙ
- Geometric interpretation: ⟨x, y⟩ = ‖x‖‖y‖cos(θ)
- Measures alignment between vectors
- Norm (length): ‖x‖ = √(⟨x, x⟩) = √(x₁² + x₂² + ... + xₙ²)
- Orthogonality: x ⊥ y when ⟨x, y⟩ = 0 (perpendicular, θ = 90°)
- Projection: proj_y(x) = (⟨x,y⟩/⟨y,y⟩) · y
- ML application: Projecting data onto subspaces, residuals orthogonal to fitted values

**6. Matrix Operations and Properties**
- Matrix A ∈ ℝᵐˣⁿ: rectangular array of numbers
- Matrix-vector multiplication: Ax represents linear transformation
- Matrix multiplication: (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ
- Dimension rule: (m×k) × (k×n) = (m×n)
- Transpose: Aᵀ swaps rows and columns
- Symmetric matrix: A = Aᵀ (important in regression)
- Identity matrix: AI = IA = A
- Properties: (AB)ᵀ = BᵀAᵀ, (AB)C = A(BC)

**7. Inverse Matrices and Determinants**
- Inverse matrix: AA⁻¹ = A⁻¹A = I
- Existence condition: det(A) ≠ 0
- Determinant properties and 2×2 formula
- Cramer's rule for solving systems
- Ill-conditioned matrices and numerical stability
- ML application: Normal equation requires (XᵀX)⁻¹

**8. Eigenvalues and Eigenvectors**
- Eigenvector v: direction unchanged by matrix A
- Eigenvalue equation: Av = λv
- λ: scaling factor for the eigenvector
- Finding eigenvalues: det(A - λI) = 0 (characteristic equation)
- n×n matrix has n eigenvalues (counting multiplicity)
- Spectral theorem: symmetric matrices have orthogonal eigenvectors
- Eigendecomposition: A = QΛQᵀ (for symmetric A)
- Example calculation for 2×2 matrix with step-by-step solution
- ML applications: PCA, understanding data variance structure, regression diagnostics

**9. Differentiation and Partial Derivatives**
- Derivative as rate of change: df/dx
- Partial derivatives: ∂f/∂xᵢ (holding other variables constant)
- Gradient vector: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
- Critical points: ∇f = 0
- Hessian matrix: second-order partial derivatives
- Chain rule for composite functions
- ML application: Finding minimum of loss function in regression

#### Key Concepts:
- **Mathematical Rigor**: Precise definitions and notation establishing foundation for statistical learning
- **Vector Space Theory**: Understanding data as points in high-dimensional spaces
- **Linear Transformations**: Matrices as operators that transform data
- **Optimization Foundations**: Calculus tools for finding optimal parameters
- **Geometric Intuition**: Visual understanding of mathematical operations

#### Learning Outcomes:
- Express machine learning problems using precise mathematical notation
- Manipulate matrices and vectors efficiently for computational tasks
- Understand the geometric interpretation of linear algebra operations
- Apply differentiation techniques to optimize model parameters
- Recognize how abstract mathematics connects to practical ML applications

---

### Part 2/3: Probability and Statistics Fundamentals (9 slides)

This section covers probability theory and statistical inference methods that form the theoretical foundation for regression analysis.

#### Topics Covered:

**10. Probability Spaces and Random Variables**
- Sample space Ω: set of all possible outcomes
- Event A: subset of sample space (A ⊆ Ω)
- Probability measure P: P(A) ∈ [0,1], P(Ω) = 1
- Random variable X: function mapping outcomes to real numbers (X: Ω → ℝ)
- Types of random variables:
  - Discrete RV: countable values (coin flips, dice rolls)
  - Continuous RV: uncountable values (heights, temperatures)
- Cumulative distribution function: F(x) = P(X ≤ x)
- Visual representation of sample space and events

**11. Probability Distributions - Discrete and Continuous**
- Discrete distributions:
  - Probability mass function (PMF): P(X = x)
  - Examples: Bernoulli, Binomial, Poisson
  - Binomial: n trials, probability p, X ~ Bin(n,p)
- Continuous distributions:
  - Probability density function (PDF): f(x)
  - P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx
  - Normal distribution: X ~ N(μ, σ²)
  - Standard normal: Z ~ N(0, 1)
- Properties and visualization of distribution shapes
- ML application: Error terms in regression assumed normal

**12. Expectation, Variance, and Covariance**
- Expectation E[X]: average value
  - Discrete: E[X] = Σ xP(X=x)
  - Continuous: E[X] = ∫ xf(x)dx
- Properties: E[aX + b] = aE[X] + b, E[X + Y] = E[X] + E[Y]
- Variance Var(X): spread around mean
  - Var(X) = E[(X - μ)²]
- Standard deviation: σ = √Var(X)
- Covariance: Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]
  - Positive: X and Y increase together
  - Negative: X increases as Y decreases
- Visual interpretation with scatter plots showing positive/negative covariance
- Covariance matrix Σ: [Cov(Xᵢ, Xⱼ)] for multiple variables
- ML application: Understanding variable relationships in regression

**13. Conditional Probability and Bayes' Theorem**
- Conditional probability: P(A|B) = P(A ∩ B) / P(B)
- Probability of A given B has occurred
- Multiplication rule: P(A ∩ B) = P(A|B)P(B)
- Independence: X ⊥ Y when P(X|Y) = P(X)
- Bayes' Theorem: P(A|B) = P(B|A)P(A) / P(B)
- Components: Prior P(A) → Likelihood P(B|A) → Posterior P(A|B)
- Applications: updating beliefs, medical diagnosis, spam filtering
- Visual representation with Venn diagrams
- ML foundation: Bayesian regression and inference

**14. Central Limit Theorem and Law of Large Numbers**
- Law of Large Numbers: sample mean converges to population mean as n → ∞
- Central Limit Theorem (CLT): sample means follow normal distribution
  - X̄ ~ N(μ, σ²/n) for large n
- Applies regardless of original distribution shape
- Visual demonstration of CLT convergence
- Standard error: SE = σ/√n
- Confidence intervals depend on CLT
- ML application: Justifies normality assumptions in regression inference

**15. Parameter Estimation - MLE and MAP**
- Maximum Likelihood Estimation (MLE):
  - Objective: argmax L(θ|data)
  - Likelihood function: L(θ) = P(data|θ) = ∏ᵢ P(xᵢ|θ)
  - Log-likelihood: ℓ(θ) = log L(θ) (easier to optimize)
  - Example: MLE for normal distribution gives μ̂ = X̄, σ̂² = (1/n)Σ(xᵢ - X̄)²
- Maximum A Posteriori (MAP):
  - Objective: argmax P(θ|data)
  - Using Bayes' theorem: P(θ|data) = P(data|θ)P(θ) / P(data)
  - MAP = MLE + Prior (incorporates prior knowledge)
- Comparison table: prior usage, focus areas
- ML application: Estimating regression coefficients

**16. Hypothesis Testing and Confidence Intervals**
- Hypothesis testing framework:
  - Null hypothesis H₀ vs Alternative hypothesis H₁
  - Test statistic calculation
  - p-value interpretation: probability under H₀
  - Significance level α (commonly 0.05)
- Type I and Type II errors:
  - Type I (α): rejecting true H₀ (false positive)
  - Type II (β): failing to reject false H₀ (false negative)
  - Power = 1 - β
- Confidence intervals: range likely containing true parameter
  - 95% CI: [θ̂ - 1.96·SE, θ̂ + 1.96·SE]
- Common tests: t-test, z-test, F-test
- ML application: Testing significance of regression coefficients

**17. Correlation vs Causation**
- Correlation: statistical association between variables
  - Pearson correlation: ρ = Cov(X,Y) / (σₓσᵧ)
  - Range: ρ ∈ [-1, 1]
  - ρ = -1: perfect negative, ρ = 0: no linear, ρ = 1: perfect positive
- Causation: X directly influences Y
  - Requires: temporal precedence, mechanism, control of confounders
- **Critical warning**: Correlation does NOT imply causation
- Confounding variables: Z affects both X and Y, creating spurious correlation
- Classic example: Ice cream sales and drownings (both caused by temperature)
- Visual diagrams showing causal vs correlational relationships
- ML limitation: Regression shows association, not necessarily causation

#### Key Concepts:
- **Probabilistic Thinking**: Understanding uncertainty and randomness in data
- **Statistical Inference**: Drawing conclusions from data samples
- **Distribution Theory**: Mathematical models for random phenomena
- **Bayesian Reasoning**: Updating beliefs with evidence
- **Critical Statistical Awareness**: Distinguishing correlation from causation

#### Learning Outcomes:
- Calculate and interpret probabilities, expectations, and variances
- Apply Bayes' theorem to update probabilistic beliefs
- Understand the theoretical basis for confidence intervals and hypothesis tests
- Estimate parameters using MLE and MAP approaches
- Critically evaluate claims of causation vs mere correlation
- Recognize distributional assumptions underlying regression models

---

### Part 3/3: Linear Regression Model (9 slides)

This section synthesizes all previous concepts to develop a complete understanding of linear regression from problem formulation through implementation.

#### Topics Covered:

**18. Linear Regression Problem Definition**
- Supervised learning task: predicting continuous output
- Goal: model relationship Y = f(X) + ε
- Components:
  - Response variable Y (dependent, target)
  - Predictor variables X (independent, features)
  - Model parameters to estimate
  - Random error term ε
- Assumptions about relationship (linearity, additivity)
- Applications: house price prediction, sales forecasting, risk assessment
- Why linear models: interpretability, computational efficiency, strong baseline
- Connects to all previous mathematical concepts

**19. Simple Linear Regression Model**
- Model equation: Y = β₀ + β₁X + ε
- Parameters:
  - β₀: Intercept (value when X = 0)
  - β₁: Slope (change in Y per unit change in X)
  - ε: Error term, ε ~ N(0, σ²)
- Fitted line: Ŷ = β̂₀ + β̂₁x (estimated parameters)
- Residual: eᵢ = yᵢ - ŷᵢ (prediction error)
- Visual representation with scatter plot, regression line, and residuals
- Geometric interpretation: intercept and slope
- Real-world example: Salary = β₀ + β₁ × (Years of Experience) + ε
- Interactive simulators:
  - 2D simulator: https://www.mladdict.com/linear-regression-simulator
  - 3D simulator: https://miabellaai.net/regression.html
- Objective: minimize total prediction error

**20. Deriving the Least Squares Method**
- Loss function (Sum of Squared Errors): L(β₀, β₁) = Σᵢ(yᵢ - β₀ - β₁xᵢ)²
- Why squares:
  - Penalizes large errors more heavily
  - Mathematically convenient (differentiable)
  - No cancellation of positive/negative errors
- Optimization goal: Find β̂₀, β̂₁ that minimize L
- Derivation steps:
  1. Define loss function
  2. Take partial derivatives: ∂L/∂β₀ = 0, ∂L/∂β₁ = 0
  3. Solve system of equations (normal equations)
  4. Obtain optimal parameters
- Unique solution exists when X has full rank
- Statistical connection: Least Squares = Maximum Likelihood under normal errors

**21. Normal Equation Solution**
- Matrix formulation: Y = Xβ + ε
- Design matrix X includes intercept column
- Normal equation: β̂ = (XᵀX)⁻¹XᵀY
- Derivation from calculus:
  - Loss function: L(β) = ‖Y - Xβ‖²
  - Gradient: ∇L(β) = -2Xᵀ(Y - Xβ)
  - Set to zero: XᵀXβ̂ = XᵀY
- Analytical solution (closed-form)
- Computational considerations:
  - Matrix inversion: O(n³) complexity
  - Numerical stability issues when XᵀX near singular
  - Alternative: use gradient descent for large datasets
- Properties: β̂ is unbiased estimator under assumptions
- Example calculation with numerical values

**22. Geometric Interpretation**
- Column space of X: all possible linear combinations of predictors
- Prediction Ŷ: projection of Y onto column space of X
- Residual e: orthogonal to column space (perpendicular)
- Visual representation:
  - Y vector in n-dimensional space
  - Column space as plane/subspace
  - Ŷ as closest point in column space to Y
  - e perpendicular to column space
- Projection matrix: P = X(XᵀX)⁻¹Xᵀ
  - Ŷ = PY
- Residual maker: M = I - P
  - e = MY
- Orthogonality condition: ⟨e, Xⱼ⟩ = 0 for all predictors
- Unifies linear algebra and statistics perspectives
- Key insight: Ŷ is the closest point in column space to Y

**23. Multiple Linear Regression Extension**
- Model with p predictors: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
- Matrix form: Y = Xβ + ε
- Design matrix X: n × (p+1) including intercept column
- Parameter vector: β = [β₀, β₁, ..., βₚ]ᵀ
- Same normal equation solution: β̂ = (XᵀX)⁻¹XᵀY
- Interpretation of coefficients:
  - βⱼ: change in Y for unit change in Xⱼ, holding others constant
  - Partial effects, controlling for other variables
- Model evaluation metrics:
  - R²: proportion of variance explained (0 to 1)
  - Adjusted R²: penalizes model complexity
  - RMSE: root mean squared error
  - F-statistic: overall model significance
- Example: House price prediction with multiple features (size, bedrooms, location)
- Multicollinearity concerns when predictors highly correlated

**24. Model Assumptions and Diagnostics**
- Five key assumptions:
  1. **Linearity**: True relationship is linear (or approximately)
  2. **Independence**: Observations are independent
  3. **Homoscedasticity**: Constant error variance across X
  4. **Normality**: Errors ~ N(0, σ²)
  5. **No Multicollinearity**: Predictors not highly correlated
- Diagnostic tools:
  - Residual plots: check for patterns (good = random scatter)
  - Q-Q plots: assess normality (points should follow diagonal)
  - Leverage plots: identify influential observations
  - VIF scores: detect multicollinearity (VIF > 10 problematic)
- Visual diagnostics examples:
  - Good residual plot: random scatter around zero
  - Q-Q plot: points on diagonal line indicates normal errors
- Violations and remedies:
  - Transformations (log, sqrt) for non-linearity or heteroscedasticity
  - Robust regression methods for outliers
  - Regularization (Ridge, Lasso) for multicollinearity
- Model validation approaches:
  - Train-test split
  - Cross-validation (k-fold)
  - Bootstrap methods

**25. Python Implementation and Practice**
- Key Python libraries:
  - **NumPy**: Matrix operations and numerical computing
  - **Scikit-learn**: ML convenience functions and pipelines
  - **Pandas**: Data manipulation and preprocessing
  - **Statsmodels**: Detailed statistical output and diagnostics
  - **Matplotlib/Seaborn**: Visualization and plotting
- Implementation approaches:
  - NumPy (from scratch): β = np.linalg.inv(X.T @ X) @ X.T @ y
  - Scikit-learn (easy): LinearRegression().fit(X, y)
  - Statsmodels (detailed stats): sm.OLS(y, X).fit()
- Standard workflow:
  1. Load Data (read CSV, handle missing values)
  2. Explore & Visualize (scatter plots, correlations)
  3. Fit Model (choose implementation method)
  4. Validate (diagnostics, residual analysis)
  5. Predict (apply to new data)
- Best practices:
  - Feature scaling (standardization, normalization)
  - Handle missing data (imputation, removal)
  - Train-test split (typically 80-20 or 70-30)
  - Cross-validation for robust evaluation
- Practice datasets:
  - Boston Housing: predict house prices
  - California Housing: larger-scale regression
  - Diabetes: medical data prediction
- Code examples with syntax highlighting for each approach

#### Key Concepts:
- **Model Formulation**: Expressing relationships mathematically
- **Optimization**: Finding best-fit parameters through least squares
- **Matrix Solution**: Analytical closed-form solution via linear algebra
- **Geometric Understanding**: Projection interpretation of regression
- **Practical Implementation**: From theory to working Python code

#### Learning Outcomes:
- Formulate regression problems mathematically
- Derive the least squares solution from first principles
- Implement linear regression using multiple Python libraries
- Diagnose model violations and apply appropriate remedies
- Validate models using proper statistical methodology
- Interpret regression coefficients in context
- Understand the geometric meaning of regression as projection
- Critically evaluate when linear regression is appropriate

---

## 🛠️ Prerequisites

### Mathematical Background
- **Linear Algebra**: Matrix operations, vector spaces, eigenvalues (review if needed)
- **Calculus**: Partial derivatives, gradients, optimization basics
- **Probability**: Basic probability theory, distributions, expectation
- **Statistics**: Descriptive statistics, hypothesis testing fundamentals

### Programming Skills
- **Python 3.7+**: Basic to intermediate proficiency
- **NumPy**: Array manipulation, linear algebra operations
- **Familiarity with Jupyter**: Notebooks for interactive development

### Software Requirements
```bash
# Core libraries
numpy >= 1.19.0
pandas >= 1.2.0
matplotlib >= 3.3.0
seaborn >= 0.11.0

# Machine learning
scikit-learn >= 0.24.0
statsmodels >= 0.12.0

# Development environment
jupyter >= 1.0.0
```

### Hardware Recommendations
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 500MB for datasets and libraries
- **Processor**: Any modern CPU (no GPU required for linear regression)

---

## 💻 Hands-on Components

### Interactive Simulators

**2D Linear Regression Simulator**
- URL: https://www.mladdict.com/linear-regression-simulator
- Features:
  - Add/remove data points by clicking
  - Real-time regression line updates
  - Visualize residuals dynamically
  - Adjust parameters and see effects
- Learning objectives:
  - Understand how data points influence the fit
  - Observe least squares optimization visually
  - Experiment with outliers and their impact

**3D Multiple Regression Simulator**
- URL: https://miabellaai.net/regression.html
- Features:
  - 3D visualization of regression plane
  - Multiple predictor variables
  - Interactive rotation and zoom
  - Real-time coefficient updates
- Learning objectives:
  - Visualize higher-dimensional regression
  - Understand partial effects of predictors
  - See geometric interpretation in 3D space

### Practice Projects

**Project 1: Boston Housing Price Prediction**
- Dataset: 506 samples, 13 features
- Task: Predict median house value
- Skills practiced:
  - Data exploration and visualization
  - Feature engineering
  - Model fitting and interpretation
  - Diagnostic checking
- Expected time: 2-3 hours

**Project 2: California Housing Analysis**
- Dataset: 20,640 samples, 8 features
- Task: Large-scale regression with geographic data
- Skills practiced:
  - Handling larger datasets
  - Computational efficiency considerations
  - Cross-validation implementation
  - Model comparison
- Expected time: 3-4 hours

**Project 3: Medical Data - Diabetes Progression**
- Dataset: 442 samples, 10 physiological features
- Task: Predict disease progression
- Skills practiced:
  - Domain-specific interpretation
  - Handling medical/scientific data
  - Feature importance analysis
  - Ethical considerations in prediction
- Expected time: 2-3 hours

### Coding Exercises

**Exercise 1: Normal Equation from Scratch**
- Implement `β̂ = (XᵀX)⁻¹XᵀY` using only NumPy
- Add intercept column to design matrix
- Compare results with scikit-learn
- Handle edge cases (singular matrices)

**Exercise 2: Diagnostic Plots**
- Create residual vs fitted plot
- Implement Q-Q plot for normality checking
- Generate scale-location plot for homoscedasticity
- Calculate and visualize Cook's distance

**Exercise 3: Cross-Validation**
- Implement k-fold cross-validation manually
- Compare with sklearn's cross_val_score
- Plot learning curves (training vs validation error)
- Analyze bias-variance tradeoff

---

## 📖 Additional Resources

### Textbooks
- **"An Introduction to Statistical Learning"** by James, Witten, Hastie, Tibshirani
  - Chapter 3: Linear Regression (detailed coverage)
  - Free PDF available online
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, Friedman
  - Section 3.2: Linear Regression Models (advanced treatment)
  - Comprehensive mathematical derivations
- **"Pattern Recognition and Machine Learning"** by Bishop
  - Chapter 3: Linear Models for Regression
  - Bayesian perspective

### Online Courses
- **Stanford CS229**: Machine Learning (Andrew Ng)
  - Lecture 2: Linear Regression and Gradient Descent
- **MIT 18.065**: Matrix Methods in Data Analysis
  - Focus on linear algebra for ML
- **Khan Academy**: Linear Algebra and Statistics modules
  - Review fundamentals as needed

### Video Lectures
- 3Blue1Brown: "Essence of Linear Algebra" series
  - Visual geometric intuition
- StatQuest: "Linear Regression" explanations
  - Clear statistical concepts

### Python Documentation
- NumPy Documentation: https://numpy.org/doc/
- Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
  - Section 1.1: Linear Models
- Statsmodels Documentation: https://www.statsmodels.org/
  - OLS regression examples

### Mathematical Resources
- Matrix Cookbook: Common matrix identities and derivatives
- Probability and Statistics Cookbook: Quick reference formulas

---

## 🔗 How to Use These Materials

### For Self-Study

1. **Sequential Learning Path**:
   - Work through slides in order (Parts 1 → 2 → 3)
   - Do not skip mathematical foundations
   - Take notes on key equations and concepts

2. **Active Learning**:
   - Pause at each mathematical derivation
   - Work through derivations on paper independently
   - Verify understanding before proceeding

3. **Interactive Exploration**:
   - Use both simulators extensively
   - Add data points, observe effects
   - Test edge cases and extreme values

4. **Coding Practice**:
   - Complete all three practice projects
   - Implement exercises from scratch before using libraries
   - Compare your results with library implementations

5. **Review and Reinforce**:
   - Revisit difficult concepts after completing full lecture
   - Create summary sheets for each part
   - Test yourself with practice problems

### For Instructors

**Lecture Planning**:
- **Part 1** (Mathematical Foundations): 90-120 minutes
  - Focus on geometric interpretations
  - Use visual demonstrations from slides
  - Emphasize connections between concepts
- **Part 2** (Probability/Statistics): 90 minutes
  - Interactive probability demonstrations
  - Real-world examples for each concept
  - Connect to regression assumptions
- **Part 3** (Linear Regression): 120 minutes
  - Live coding demonstration
  - Work through derivation on board
  - Show diagnostic workflow

**Classroom Activities**:
- Small group work on mathematical derivations
- Pair programming for Python implementation
- Class discussion on correlation vs causation examples

**Assessment Integration**:
- Use slides as lecture material
- Assign practice projects as homework
- Create quizzes based on key concepts

### Slide Navigation

**Using the Slideshow**:
- Open `lecture03_slideshow.html` in web browser
- Keyboard controls:
  - `Arrow Left/Right`: Navigate slides
  - `Space`: Next slide
  - `Home/End`: First/last slide
- Click navigation buttons for mouse control
- Progress bar shows current position

**Individual Slides**:
- Each HTML file is standalone
- Can be used for focused review
- All styling is self-contained

---

## 📊 Assessment Suggestions

### Formative Assessment

**Concept Checks** (During Lecture):
- Quick polls on mathematical properties
- Ask students to predict regression line behavior
- Identify assumptions from diagnostic plots

**Weekly Quizzes** (10-15 minutes):
1. **Week 1**: Set theory, functions, vector spaces
2. **Week 2**: Probability distributions, expectation, covariance
3. **Week 3**: Regression model, normal equation, diagnostics

### Summative Assessment

**Problem Sets** (Weight: 30%):
- **PS1**: Mathematical foundations
  - Matrix operations
  - Eigenvalue calculations
  - Gradient computations
- **PS2**: Probability and statistics
  - Probability calculations
  - Bayes' theorem applications
  - Distribution properties
- **PS3**: Regression derivations
  - Least squares derivation
  - Geometric interpretation questions
  - Assumption verification

**Programming Projects** (Weight: 40%):
- **Project 1**: Implement regression from scratch
  - Normal equation implementation
  - Comparison with libraries
  - Documentation and code quality
- **Project 2**: Real-world data analysis
  - Full workflow from EDA to validation
  - Diagnostic checking
  - Written interpretation report
- **Project 3**: Model comparison study
  - Multiple models on same data
  - Performance metrics
  - Critical evaluation

**Final Exam** (Weight: 30%):
- **Part A** (40 points): Conceptual questions
  - Multiple choice on key concepts
  - Short answer interpretations
  - True/false with justification
- **Part B** (30 points): Mathematical derivations
  - Derive normal equation
  - Prove properties of least squares
  - Show geometric relationships
- **Part C** (30 points): Applied problems
  - Interpret regression output
  - Diagnose model issues
  - Recommend remedies

### Rubric Example (Programming Project)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) |
|-----------|---------------|----------|------------------|----------------|
| **Implementation** | Correct, efficient, well-documented | Correct with minor issues | Partially correct | Significant errors |
| **Analysis** | Thorough diagnostics, clear insights | Good analysis, some missing checks | Basic analysis | Minimal analysis |
| **Visualization** | Clear, informative plots with labels | Good plots, minor improvements needed | Basic plots | Poor or missing plots |
| **Interpretation** | Insightful, contextual, precise | Good interpretation, mostly accurate | Basic interpretation | Incorrect or missing |
| **Code Quality** | Clean, modular, documented | Readable with some improvements | Works but messy | Poorly organized |

---

## 📝 Notes for Implementation

### Technical Specifications

**Browser Compatibility**:
- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+
- JavaScript must be enabled

**File Structure**:
```
lecture03/
├── Lecture03_01_*.html         # Title slide
├── Lecture03_02_*.html         # Contents
├── Lecture03_03-12_*.html      # Part 1: Math foundations
├── Lecture03_13-21_*.html      # Part 2: Probability/stats
├── Lecture03_22-30_*.html      # Part 3: Linear regression
├── Lecture03_31_*.html         # Thank you slide
└── lecture03_slideshow.html    # Master slideshow file
```

**Viewing Options**:
1. **Slideshow Mode**: Open `lecture03_slideshow.html`
   - Full presentation with navigation
   - Progress tracking
   - Keyboard shortcuts
2. **Individual Slides**: Open specific HTML files
   - Focused review
   - Printing individual slides
   - Embedding in other materials

### Performance Considerations

**Loading Time**:
- Each slide: < 100KB
- Total package: ~400KB
- Loads quickly even on slow connections

**Rendering**:
- Pure HTML/CSS (no external dependencies)
- Fast rendering on all devices
- Works offline once downloaded

**Memory Usage**:
- Browser tab: ~50-100MB
- Suitable for older computers
- No GPU acceleration required

### Customization Options

**For Instructors**:
- Edit HTML files directly for content changes
- Modify CSS `<style>` blocks for appearance
- Add institutional branding
- Insert additional examples

**Color Scheme**:
- Primary: #1E64C8 (blue)
- Accent: #2874d8 (lighter blue)
- Background: #ffffff (white)
- Text: #333 (dark gray)
- Easily customizable in style blocks

**Typography**:
- Primary font: Aptos, 'Segoe UI'
- Monospace: 'Courier New' (for code/math)
- Responsive sizing based on screen size

### Accessibility

**Features**:
- High contrast color scheme
- Clear typography hierarchy
- Keyboard navigation support
- Semantic HTML structure

**Recommendations**:
- Use screen reader for text content
- Keyboard-only navigation available
- Visual diagrams supplemented with text

---

## 🙏 Credits

**Instructor**: Ho-min Park

**Contact**:
- Email: homin.park@ghent.ac.kr
- Email: powersimmani@gmail.com

**Institution**: Ghent University (assumed from email domain)

**Course Materials**:
- All slides designed and created by Ho-min Park
- Interactive visualizations and diagrams
- Comprehensive coverage from foundations to implementation

**External Resources Referenced**:
- Interactive simulators (mladdict.com, miabellaai.net)
- Standard ML textbooks and literature
- Python scientific computing ecosystem

---

**License**: Educational use permitted with attribution  
**Last Updated**: November 2025  
**Version**: 1.0

---

## 📌 Quick Reference

**Key Equations**:
- Normal Equation: `β̂ = (XᵀX)⁻¹XᵀY`
- Linear Model: `Y = Xβ + ε`
- Loss Function: `L(β) = ‖Y - Xβ‖²`
- Projection Matrix: `P = X(XᵀX)⁻¹Xᵀ`

**Python Quick Start**:
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X_test)
```

**Diagnostic Checklist**:
- [ ] Residual plot shows random scatter
- [ ] Q-Q plot approximately linear
- [ ] No obvious patterns in residuals
- [ ] VIF < 10 for all predictors
- [ ] R² and adjusted R² reasonable
