# Lecture 20: Model Explainability - SHAP and Deep Learning XAI

## 📋 Overview

**Instructor:** Ho-min Park, PhD  
**Contact:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Institution:** Ghent University  
**Total Slides:** 32 slides  
**Duration:** ~3-4 hours (including hands-on sessions)  
**Difficulty Level:** Advanced (Graduate level or senior undergraduate)  
**Prerequisites Required:** Yes (see Prerequisites section)

This comprehensive lecture provides an in-depth exploration of Explainable AI (XAI) with a focus on SHAP (SHapley Additive exPlanations) and advanced deep learning explainability techniques. The course bridges game-theoretic foundations with practical implementation, covering both classical and cutting-edge methods in model interpretability.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand the theoretical foundations** of SHAP through cooperative game theory and Shapley values, including their mathematical properties and guarantees
2. **Implement and compare** multiple SHAP explainers (KernelSHAP, TreeSHAP, DeepSHAP, GradientSHAP) for different model architectures
3. **Create and interpret** various SHAP visualizations (waterfall plots, force plots, summary plots, dependence plots) for both tabular and complex data types
4. **Apply advanced deep learning XAI techniques** including attention mechanisms, gradient-based methods (Integrated Gradients, SmoothGrad), and CAM-family methods
5. **Critically evaluate** XAI methods based on faithfulness, computational efficiency, and applicability to different domains (medical imaging, NLP, time series)

---

## 📚 Lecture Structure

### **Part 1/4: SHAP Theory and Fundamentals** (8 topics)

#### Topics Covered:

**1. Course Introduction - Game Theory-Based Explainability**
- XAI taxonomy and positioning of SHAP in the explainability landscape
- Model-agnostic vs. model-specific approaches
- Local vs. global explainability methods
- Comparison with other XAI methods (LIME, CAM, Attention Maps)

**2. Cooperative Game Theory Basics**
- Introduction to cooperative game theory concepts
- Coalition formation and value functions
- Coalition tree structure for feature subsets
- Mathematical framework: v(S) for coalition S ⊆ N
- Example: House price prediction with 3 features (2³ = 8 coalitions)

**3. Shapley Values Mathematical Definition**
- Complete mathematical formulation of Shapley values
- Permutation-based interpretation
- Marginal contribution calculation
- Fair credit allocation principles
- Computational complexity: O(2ⁿ) for n features

**4. Core Ideas of SHAP**
- Additive feature attribution: f(x) = φ₀ + Σφᵢ
- Decomposition of predictions into feature contributions
- Base value (φ₀) as average prediction
- Individual SHAP values (φᵢ) as deviations
- Visual example: House price $350K = $250K base + contributions

**5. SHAP vs LIME Comparison**
- Side-by-side feature comparison
- **Approach:** Game-theoretic (SHAP) vs. Local linear approximation (LIME)
- **Consistency:** Fairness axioms (SHAP) vs. No guarantees (LIME)
- **Speed:** More intensive (SHAP) vs. Faster (LIME)
- **Use cases:** Critical decisions vs. Quick prototyping

**6. Interpreting SHAP Values**
- Credit approval decision example (Applicant #247)
- Base value: 0.42 → Prediction: 0.78 (Approved)
- Positive impacts: Income (+0.18), Credit History (+0.15)
- Negative impacts: Debt Ratio (-0.05)
- Magnitude interpretation and sum property

**7. Mathematical Properties of SHAP**
- **Local Accuracy:** f(x) = φ₀ + Σφᵢ(x)
- **Missingness:** xᵢ = 0 ⇒ φᵢ = 0
- **Consistency:** Δfᵢ ↑ ⇒ φᵢ ↑
- **Efficiency:** Σφᵢ = f(x) - E[f(x)]
- **Symmetry:** i ≡ j ⇒ φᵢ = φⱼ
- **Linearity:** φ(f+g) = φ(f) + φ(g)
- Visual proof with numerical example

**8. Hands-on: First SHAP Analysis**
- Complete workflow: Setup → Load Data → Train Model → Explain
- Installation: `pip install shap`
- Code example with RandomForestRegressor on Boston housing
- TreeExplainer initialization and SHAP value computation
- Waterfall plot visualization
- Interpretation tips: red bars (increase), blue bars (decrease)

#### Key Concepts:
- Shapley values as unique fair allocation scheme from game theory
- SHAP values satisfy desirable mathematical properties (efficiency, symmetry, linearity)
- Additive decomposition enables intuitive interpretation
- Model-agnostic framework applicable to any ML model
- Computational complexity requires efficient approximation methods

#### Learning Outcomes:
Students will be able to explain the game-theoretic foundations of SHAP, calculate simple Shapley values manually, understand why SHAP provides theoretically sound explanations, and implement basic SHAP analysis on tabular data.

---

### **Part 2/4: SHAP Implementation Methods** (7 topics)

#### Topics Covered:

**9. KernelSHAP**
- Model-agnostic approximation algorithm
- Five-step workflow:
  1. Sample coalitions systematically
  2. Create perturbed instances with background data
  3. Get model predictions for each coalition
  4. Apply SHAP kernel weights
  5. Weighted linear regression to solve for φᵢ
- Trade-off: computational cost vs. accuracy
- Works with any black-box model (neural networks, ensembles, custom models)
- Implementation: `shap.KernelExplainer(model.predict, X_train)`

**10. TreeSHAP**
- Exact computation for tree-based models
- Polynomial time algorithm: O(TLD²) where T=trees, L=leaves, D=depth
- Optimized for Random Forest, XGBoost, LightGBM, CatBoost
- Path-dependent calculation using tree structure
- Conditional expectation: E[f(x)|xₛ] computed exactly
- Speed comparison: 100-1000x faster than KernelSHAP
- No sampling required, deterministic results
- Implementation: `shap.TreeExplainer(model)`

**11. DeepSHAP (DeepLIFT + SHAP)**
- Specialized for deep neural networks
- Combines DeepLIFT's efficient backpropagation with SHAP framework
- Layer-by-layer contribution propagation
- Reference baseline selection strategies
- Handles non-linear activations (ReLU, tanh, sigmoid)
- Supports CNN and RNN architectures
- Significantly faster than KernelSHAP for deep learning
- Implementation: `shap.DeepExplainer(model, background_data)`

**12. GradientSHAP**
- Gradient-based approximation for differentiable models
- Combines Integrated Gradients with SHAP
- Multiple reference points using baseline distribution
- Mathematical formulation: φᵢ = ∫₀¹ (∂f/∂xᵢ)(x' + α(x - x'))dα
- Handles gradient saturation better than vanilla gradients
- Applicable to any differentiable model
- Trade-off between DeepSHAP speed and KernelSHAP generality

**13. SHAP Approximation Techniques**
- Sampling strategies for large feature sets
- **Auto sampling:** Adaptive sample size based on convergence
- **Independence assumption:** Marginal vs. conditional distributions
- **Paired sampling:** Efficient coalition generation
- **Stratified sampling:** Ensuring representative coverage
- Background data selection: training set sample (50-100 instances)
- Convergence criteria and error bounds
- Practical guidelines: start with 100 samples, increase if unstable

**14. SHAP Interaction Values**
- Capturing feature interactions: φᵢⱼ measures joint effect
- Interaction matrix visualization
- Example: Age × Income interaction in credit scoring
- Matrix properties:
  - Symmetric: φᵢⱼ = φⱼᵢ
  - Diagonal: main effects (φᵢᵢ)
  - Off-diagonal: pairwise interactions
- Decomposition: φ_total = φᵢ + Σⱼ φᵢⱼ
- TreeSHAP enables exact interaction computation
- Use cases: identifying synergistic and antagonistic effects

**15. Hands-on: Comparing Different SHAP Explainers**
- Workflow: Load Dataset → Train Models → Apply Explainers → Compare Results
- Performance comparison table:
  - **TreeSHAP:** Fast (1x baseline), Exact accuracy, ~0.5s for 1000 samples
  - **DeepSHAP:** Moderate (5x), High accuracy, ~2.5s for 1000 samples
  - **GradientSHAP:** Moderate (4x), High accuracy, ~2s for 1000 samples
  - **KernelSHAP:** Slow (20x), Approximate, ~10s for 1000 samples
- Code examples for each explainer
- Selection criteria based on model type and requirements

#### Key Concepts:
- Different SHAP implementations optimize for specific model architectures
- TreeSHAP provides exact solutions in polynomial time for tree ensembles
- Deep learning requires specialized methods (DeepSHAP, GradientSHAP) for efficiency
- KernelSHAP serves as universal fallback for any model
- SHAP interaction values reveal feature synergies and dependencies

#### Learning Outcomes:
Students will be able to select appropriate SHAP explainers for different model types, understand the computational trade-offs, implement multiple SHAP variants, and analyze feature interactions using SHAP interaction values.

---

### **Part 3/4: SHAP Visualization and Analysis** (5 topics)

#### Topics Covered:

**16. Waterfall Plot**
- Feature contribution breakdown for single predictions
- Visual flow: Base Value → Feature Contributions → Final Prediction
- Example: House price $350K = $250K + $80K (size) + $50K (location) + $20K (bedrooms) - $30K (age) - $20K (condition)
- Color coding: Red (positive impact), Blue (negative impact)
- Bar length indicates magnitude of contribution
- Reading guide: start at base, each bar adds/subtracts, final value at top
- Use cases: explaining individual predictions, debugging model decisions, stakeholder communication
- Code: `shap.plots.waterfall(shap_values[0])`

**17. Force Plot and Decision Plot**
- **Force Plot:**
  - Horizontal push/pull visualization
  - Interactive HTML output with hover details
  - Base value at center, features push left/right
  - Gradient coloring by contribution strength
  - Example: Credit approval with base 0.42 → prediction 0.62
  - Code: `shap.plots.force(base_value, shap_values[i])`
- **Decision Plot:**
  - Cumulative SHAP value progression
  - Y-axis: prediction value, X-axis: features ordered by importance
  - Comparing multiple predictions simultaneously
  - Identifying critical decision thresholds
  - Path visualization showing prediction evolution
  - Code: `shap.plots.decision(base_value, shap_values)`

**18. Summary Plot and Dependence Plot**
- **Summary Plot (Beeswarm):**
  - Global feature importance ranking
  - Dot density shows value distribution
  - Color gradient: red (high feature value), blue (low feature value)
  - Horizontal spread shows SHAP value distribution
  - Reveals patterns: monotonic, threshold, or complex relationships
- **Dependence Plot:**
  - Scatter plot: feature value vs. SHAP value
  - Reveals non-linear relationships and interactions
  - Color coding by interaction feature
  - Identifies thresholds and saturation points
  - Example: Income effect on loan approval varies by age group
- Combined insights for comprehensive model understanding

**19. SHAP for Time Series**
- Temporal feature importance visualization
- Handling sequential dependencies
- Recurrent pattern identification
- Example applications:
  - Stock price prediction: time-varying feature importance
  - Anomaly detection: unusual pattern explanation
  - Forecasting: horizon-specific driver analysis
- Sliding window approaches
- Lagged feature importance
- Seasonal decomposition integration
- Visualization adaptations: heatmaps over time, temporal dependence plots

**20. SHAP for Text and Images**
- **Text/NLP Applications:**
  - Token-level SHAP values
  - Transformer model explanations
  - Word importance highlighting
  - Context-dependent attribution
  - Example: Sentiment analysis word contributions
  - Integration with attention mechanisms
- **Image/Computer Vision:**
  - Pixel-level or superpixel-level attribution
  - Heatmap overlays on original images
  - Partition-based explanations
  - Example: Medical image pathology localization
  - Integration with CAM methods
  - Multi-scale explanations
- Challenges: high dimensionality, perceptual alignment

#### Key Concepts:
- Multiple visualization types serve different analytical purposes
- Waterfall/Force plots for individual prediction explanation
- Summary plots for global model behavior
- Dependence plots reveal feature relationships and interactions
- Domain-specific adaptations for time series, text, and images

#### Learning Outcomes:
Students will be able to create and interpret all major SHAP visualization types, select appropriate plots for specific analysis goals, apply SHAP to time series and unstructured data, and communicate model behavior to technical and non-technical audiences.

---

### **Part 4/4: Advanced Deep Learning XAI Techniques** (5 topics)

#### Topics Covered:

**21. Attention Mechanisms as Explanations**
- Self-attention weights as interpretability signals
- Multi-head attention visualization
- Limitations: attention ≠ explanation debate
- Attention rollout and flow techniques
- Transformer explainability strategies
- Example: BERT attention patterns for NLP tasks
- ViT (Vision Transformer) attention maps
- Practical considerations:
  - Which attention layer to visualize?
  - How to aggregate multi-head attention?
  - Correlation with other XAI methods
- Code integration with Hugging Face transformers

**22. Gradient-based Methods**
- **Vanilla Gradients/Saliency Maps:**
  - Basic approach: ∂f/∂x
  - Limitations: noisy, gradient saturation
- **Integrated Gradients:**
  - Path integral from baseline to input
  - Mathematical rigor and axiom satisfaction
  - Formula: IGᵢ(x) = (xᵢ - x'ᵢ) × ∫₀¹ (∂f/∂xᵢ)(x' + α(x - x')))dα
  - Baseline selection strategies
- **SmoothGrad:**
  - Adding Gaussian noise for stability
  - Averaging over multiple noisy versions
  - Reduces gradient noise artifacts
- **GradCAM (covered in next section)**
- **Guided Backpropagation:**
  - Modified ReLU gradients
  - Sharper visualizations
- Comparison table: faithfulness, computational cost, visual quality

**23. CAM-family Methods**
- Evolution timeline: CAM → Grad-CAM → Grad-CAM++ → Score-CAM → Layer-CAM
- **CAM (2016):**
  - Original class activation mapping
  - Requires global average pooling
  - Formula: L_CAM = Σ wₖ · Aₖ
  - Architecture limitation: only GAP-based networks
- **Grad-CAM (2017):**
  - Any CNN architecture
  - Gradient-weighted activation: L = ReLU(Σ αₖ · Aₖ)
  - αₖ = (1/Z) Σᵢ Σⱼ ∂y^c/∂Aₖⁱʲ
  - Most widely used variant
- **Grad-CAM++ (2018):**
  - Better localization for multiple instances
  - Weighted combination of gradients
  - Improved spatial resolution
- **Score-CAM (2020):**
  - Gradient-free approach
  - Ablation-based scoring
  - More stable, no gradient saturation issues
- **Layer-CAM (2021):**
  - Layer-wise activation maps
  - Finer-grained spatial resolution
- Medical imaging example: X-ray pneumonia detection with heatmap overlay
- Applications: medical diagnosis, object detection, quality control, defect detection

**24. Concept-based Explanations**
- High-level concept understanding vs. pixel/feature-level
- **TCAV (Testing with Concept Activation Vectors):**
  - User-defined concept testing
  - Linear separability in activation space
  - Directional derivative: concept sensitivity score
  - Example: "striped" concept for zebra classification
- **ACE (Automated Concept Extraction):**
  - Unsupervised concept discovery
  - Clustering in activation space
  - No manual concept definition required
- **Concept Bottleneck Models:**
  - Explicit concept predictions as intermediate layer
  - Architecture: input → concepts → prediction
  - Interpretable by design
  - Intervention capabilities
- Human-aligned explanations
- Domain expert validation
- Use cases: medical diagnosis, legal compliance, scientific discovery

**25. Future of XAI and Challenges**
- **Key Challenges:**
  - Standardization: evaluation metrics for explanation quality
  - Faithfulness vs. Plausibility: do explanations reflect true model reasoning?
  - Counterfactual Explanations: minimal changes for different outcomes
  - Causal Inference Integration: combining XAI with causal reasoning
  - Explanation Stability: robustness to input perturbations
- **Regulatory Landscape:**
  - EU AI Act: transparency requirements for high-risk AI
  - FDA Guidelines: medical AI explainability standards
  - GDPR: right to explanation
- **Research Roadmap:**
  - 2016-2018: Foundation Era (LIME, SHAP, CAM, Attention)
  - 2019-2021: Refinement Period (Integrated Gradients, faithfulness studies)
  - 2022-2024: Integration Phase (concept-based, causal XAI, regulatory compliance) ← Current
  - 2025-2027: Multimodal XAI (cross-modal explanations, unified frameworks)
  - 2028+: Autonomous XAI (self-explaining models, AGI integration)
- Emerging research directions
- Industry adoption challenges
- Ethical considerations

#### Key Concepts:
- Multiple complementary XAI techniques for deep learning models
- Gradient-based methods leverage model differentiability
- CAM family provides spatial visual explanations for CNNs
- Concept-based explanations align with human reasoning
- XAI field rapidly evolving with regulatory and ethical pressures

#### Learning Outcomes:
Students will be able to implement gradient-based explanation methods, apply CAM variants to computer vision tasks, understand concept-based explanation frameworks, critically evaluate XAI methods for faithfulness and stability, and anticipate future directions in explainable AI research.

---

## 🛠️ Prerequisites

### Required Knowledge:
1. **Machine Learning Fundamentals:**
   - Supervised learning (regression, classification)
   - Model evaluation metrics
   - Overfitting, underfitting, bias-variance tradeoff

2. **Deep Learning:**
   - Neural network architectures (MLP, CNN, RNN)
   - Backpropagation and gradient descent
   - Activation functions, loss functions
   - Experience with PyTorch or TensorFlow

3. **Mathematics:**
   - Linear algebra: vectors, matrices, matrix operations
   - Calculus: derivatives, gradients, chain rule
   - Probability: expectations, conditional probability
   - Basic game theory concepts (helpful but not required)

4. **Programming:**
   - Python proficiency (intermediate to advanced)
   - NumPy, pandas for data manipulation
   - Matplotlib/seaborn for visualization
   - Experience with scikit-learn

### Software Requirements:
```bash
# Core libraries
pip install shap>=0.41.0
pip install scikit-learn>=1.0.0
pip install xgboost>=1.5.0
pip install lightgbm>=3.3.0

# Deep learning (choose one or both)
pip install torch>=1.10.0 torchvision
pip install tensorflow>=2.8.0

# Visualization
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install plotly>=5.0.0

# Optional but recommended
pip install jupyter notebook
pip install pandas>=1.3.0
pip install numpy>=1.21.0
```

### Recommended Hardware:
- CPU: 4+ cores for efficient tree ensemble training
- RAM: 8GB minimum, 16GB recommended for large datasets
- GPU: Optional but beneficial for deep learning sections (CUDA-capable with 4GB+ VRAM)
- Storage: 5GB for datasets and model checkpoints

---

## 💻 Hands-on Components

### **Lab 1: First SHAP Analysis (Part 1 - Topic 8)**
**Duration:** 30 minutes  
**Objective:** Implement basic SHAP workflow on tabular data

**Tasks:**
1. Load Boston housing dataset (or California housing as alternative)
2. Train RandomForestRegressor with `n_estimators=100`
3. Initialize `shap.TreeExplainer` with trained model
4. Compute SHAP values for test set
5. Create waterfall plot for single prediction
6. Interpret feature contributions

**Expected Outputs:**
- Working Jupyter notebook with complete pipeline
- Waterfall plot visualization
- Written interpretation (2-3 sentences) of one prediction

**Key Code Snippet:**
```python
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# Load and train
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X.iloc[0],
    feature_names=X.columns
))
```

---

### **Lab 2: Comparing SHAP Explainers (Part 2 - Topic 15)**
**Duration:** 45 minutes  
**Objective:** Understand trade-offs between different SHAP implementations

**Tasks:**
1. Train three models on same dataset:
   - XGBoost (for TreeSHAP)
   - Neural network (for DeepSHAP)
   - Logistic regression (for KernelSHAP)
2. Apply appropriate SHAP explainer to each
3. Measure computation time for 100 samples
4. Compare SHAP values between methods
5. Visualize results with summary plots

**Expected Outputs:**
- Performance comparison table (computation time, memory usage)
- Side-by-side summary plots
- Analysis document discussing when to use each method

**Evaluation Criteria:**
- Correct implementation of all three explainers
- Accurate timing measurements
- Insightful comparison of results
- Clear documentation

---

### **Lab 3: SHAP Visualization Gallery (Part 3)**
**Duration:** 45 minutes  
**Objective:** Master all major SHAP visualization types

**Tasks:**
1. Using a classification dataset (e.g., credit approval, medical diagnosis):
   - Create waterfall plot for 3 different instances
   - Generate force plot with interactive HTML output
   - Produce summary plot (beeswarm)
   - Create dependence plots for top 3 features
   - Generate decision plot comparing 5 instances
2. Customize color schemes and labels
3. Export publication-ready figures

**Expected Outputs:**
- 5+ high-quality visualizations
- README explaining each plot's purpose
- Presentation slide deck (5 slides) using the visualizations

**Bonus Challenge:**
- Apply SHAP to time series data with custom temporal visualization
- Create animated force plot showing prediction evolution

---

### **Lab 4: Deep Learning XAI (Part 4)**
**Duration:** 60 minutes  
**Objective:** Apply advanced XAI techniques to deep neural networks

**Tasks:**
1. **Gradient-based explanations:**
   - Implement Integrated Gradients on image classifier
   - Apply SmoothGrad with different noise levels
   - Compare with vanilla gradients
2. **CAM methods:**
   - Use Grad-CAM on pre-trained ResNet for ImageNet
   - Visualize heatmaps on medical images (chest X-rays)
   - Try Score-CAM and compare results
3. **Analysis:**
   - Identify failure cases where explanations are misleading
   - Test stability under input perturbations

**Expected Outputs:**
- Image classification model with explanations
- Heatmap overlay visualizations (5+ examples)
- Report on method comparison (faithfulness, stability, interpretability)

**Advanced Extension:**
- Implement attention visualization for Transformer model
- Create concept activation vectors for user-defined concepts
- Quantify explanation quality using deletion/insertion metrics

---

## 📖 Additional Resources

### Research Papers (Foundational):
1. **Lundberg & Lee (2017):** "A Unified Approach to Interpreting Model Predictions" - Original SHAP paper
2. **Shapley (1953):** "A Value for n-person Games" - Game theory foundations
3. **Ribeiro et al. (2016):** "Why Should I Trust You? Explaining the Predictions of Any Classifier" - LIME
4. **Selvaraju et al. (2017):** "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
5. **Sundararajan et al. (2017):** "Axiomatic Attribution for Deep Networks" - Integrated Gradients

### Research Papers (Advanced):
6. **Kim et al. (2018):** "Interpretability Beyond Feature Attribution: Testing with Concept Activation Vectors (TCAV)"
7. **Lundberg et al. (2020):** "From local explanations to global understanding with explainable AI for trees"
8. **Ghorbani et al. (2019):** "Towards Automatic Concept-based Explanations" - ACE
9. **Chattopadhay et al. (2018):** "Grad-CAM++: Generalized Gradient-based Visual Explanations"
10. **Wang et al. (2020):** "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"

### Books:
- **Molnar (2022):** "Interpretable Machine Learning" - Free online book, comprehensive coverage
- **Samek et al. (2019):** "Explainable AI: Interpreting, Explaining and Visualizing Deep Learning"

### Online Resources:
- **SHAP GitHub:** https://github.com/slundberg/shap - Official repository with examples
- **SHAP Documentation:** https://shap.readthedocs.io/ - API reference and tutorials
- **Distill.pub:** https://distill.pub/ - High-quality interactive XAI visualizations
- **Captum (PyTorch):** https://captum.ai/ - Model interpretability library

### Datasets for Practice:
1. **Tabular:**
   - UCI Adult Income (classification)
   - California Housing (regression)
   - Credit Card Default (classification)
2. **Images:**
   - ImageNet subset
   - CIFAR-10/100
   - Medical: ChestX-ray14, MIMIC-CXR
3. **Text:**
   - IMDB sentiment
   - 20 Newsgroups
   - SQuAD (question answering)
4. **Time Series:**
   - Stock market data
   - ECG signals (MIT-BIH)
   - Weather forecasting

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Sequential Learning Path:**
   - Follow Part 1 → Part 2 → Part 3 → Part 4 order
   - Complete hands-on labs after each theory section
   - Spend ~1-1.5 hours per part

2. **Slide Navigation:**
   - Open `lecture20_slideshow.html` in web browser
   - Use arrow keys (← →) or Space for navigation
   - Press Home/End for first/last slide

3. **Practical Implementation:**
   - Set up Python environment before starting
   - Work through code examples in Jupyter notebooks
   - Modify parameters and observe effects

### For Instructors:
1. **Course Integration:**
   - Can be delivered as 2-session workshop (2 hours each)
   - Or integrated into 4-week module (1 part per week)
   - Adapt depth based on student background

2. **Lecture Delivery:**
   - Part 1-2: Focus on theory and foundations (1.5-2 hours)
   - Part 3-4: Emphasize applications and practice (1.5-2 hours)
   - Allocate 40% theory, 60% hands-on time

3. **Assessment Options:**
   - Weekly quizzes on theoretical concepts
   - Hands-on lab submissions (4 labs × 20% = 80%)
   - Final project (20%): Apply SHAP to novel dataset

### For Practitioners:
1. **Quick Start:**
   - Review Part 1 topics 4-6 for SHAP overview (30 min)
   - Jump to Part 2 topic 10/11 for your model type (TreeSHAP or DeepSHAP)
   - Use Part 3 visualizations as reference guide

2. **Production Deployment:**
   - Follow performance guidelines from Lab 2
   - Use TreeSHAP for tree models (fastest)
   - Cache explanations when possible
   - Consider approximate methods for large-scale applications

---

## 📊 Assessment Suggestions

### Theoretical Understanding (30%):

**Quiz 1: SHAP Foundations (after Part 1)**
- Multiple choice: XAI taxonomy, SHAP vs other methods (5 questions)
- Short answer: Explain Shapley value fair allocation (1 question)
- Problem: Calculate simple Shapley value manually (2 features)

**Quiz 2: Implementation Methods (after Part 2)**
- Matching: SHAP explainer to appropriate model type (5 pairs)
- Short answer: Explain computational complexity trade-offs (2 questions)
- True/False: Properties of different SHAP variants (10 statements)

### Practical Skills (50%):

**Lab Assignments (10% each):**
1. First SHAP Analysis - Correctness, interpretation quality
2. Explainer Comparison - Comprehensive testing, clear documentation
3. Visualization Gallery - Aesthetic quality, appropriate usage
4. Deep Learning XAI - Technical implementation, insightful analysis

**Grading Rubric per Lab:**
- Code functionality (40%): Runs without errors, correct implementation
- Documentation (20%): Clear comments, README with explanations
- Visualizations (20%): Proper labels, publication-quality
- Insights (20%): Meaningful interpretation, identifies patterns

### Final Project (20%):

**Project Requirements:**
- Apply SHAP (and optionally other XAI methods) to a dataset of your choice
- Minimum 1000 samples, 10+ features
- Must include:
  - Problem statement and motivation
  - Model training and evaluation
  - SHAP analysis with 3+ visualization types
  - Interpretation and insights
  - Discussion of limitations

**Deliverables:**
- Jupyter notebook with complete analysis
- 10-minute presentation (recorded or live)
- 5-page report (IEEE format)

**Evaluation Criteria:**
- Technical correctness (30%)
- Depth of analysis (25%)
- Quality of visualizations (20%)
- Clarity of communication (15%)
- Originality and creativity (10%)

---

## 📝 Notes for Implementation

### Technical Considerations:

**Memory Management:**
- SHAP values stored as arrays of shape `(n_samples, n_features)`
- TreeSHAP with 1000 samples × 100 features ≈ 1MB
- Background data for KernelSHAP: 50-100 samples recommended
- Interaction values: `(n_samples, n_features, n_features)` - memory intensive

**Performance Optimization:**
- **TreeSHAP:** Use `feature_perturbation="tree_path_dependent"` for speed
- **KernelSHAP:** Start with `nsamples=100`, increase only if unstable
- **DeepSHAP:** Process in batches for large datasets
- Parallel processing: `n_jobs=-1` where supported

**Visualization Best Practices:**
- **Waterfall plots:** Best for single predictions, limit to top 10-15 features
- **Summary plots:** Show all features or use `max_display=20`
- **Force plots:** Use `matplotlib=True` for static export, HTML for interactivity
- **Color consistency:** Stick to default red/blue for positive/negative

### Common Pitfalls:

1. **Background Data Selection:**
   - ❌ Using test set as background (data leakage)
   - ✅ Use representative sample from training set (50-100 instances)

2. **Interpretation Errors:**
   - ❌ Confusing SHAP value magnitude with feature importance
   - ✅ SHAP value = contribution to *this prediction*, not global importance

3. **Computational Issues:**
   - ❌ Running KernelSHAP on 10,000 samples with 100 features (too slow)
   - ✅ Use TreeSHAP/DeepSHAP when possible, sample strategically

4. **Visualization Mistakes:**
   - ❌ Showing all 100 features in waterfall plot (cluttered)
   - ✅ Focus on top contributors, aggregate remaining as "Other"

### Troubleshooting:

**Issue: SHAP values don't sum to prediction difference**
- Solution: Check that you're using correct base value (explainer.expected_value)
- Verify model output type (probabilities vs. logits for classifiers)

**Issue: Slow computation with KernelSHAP**
- Solution: Reduce `nsamples`, use fewer background instances
- Consider switching to model-specific explainer if possible

**Issue: Unstable SHAP values between runs**
- Solution: Set random seed in KernelSHAP
- Increase number of samples for better convergence

**Issue: Out of memory with interaction values**
- Solution: Compute interactions only for subset of features
- Use hierarchical analysis: global → top features → interactions

### Extensions and Customization:

**Custom Models:**
```python
# Wrap any prediction function
def custom_predict(X):
    # Your custom preprocessing and prediction
    return predictions

explainer = shap.KernelExplainer(custom_predict, background_data)
```

**Custom Visualizations:**
```python
# Access raw SHAP values for custom plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_names, shap_values[0])
plt.xlabel('SHAP Value')
plt.title('Custom SHAP Visualization')
```

**Integration with MLOps:**
- Save explainer: `pickle.dump(explainer, open('explainer.pkl', 'wb'))`
- Compute explanations in inference pipeline
- Store SHAP values with predictions for audit trail
- Use SHAP for model monitoring and drift detection

---

## 🙏 Credits

**Instructor:** Ho-min Park, PhD  
**Affiliation:** Ghent University  
**Contact:** homin.park@ghent.ac.kr | powersimmani@gmail.com  

**Acknowledgments:**
- SHAP library developed by Scott Lundberg and team at Microsoft Research
- Game theory foundations from Lloyd Shapley's seminal work
- Visualization design inspired by interpretable ML community best practices
- Lecture materials developed for graduate-level AI course at Ghent University

**License:**
These educational materials are provided for academic and educational purposes. Code examples are provided under MIT License. Please cite appropriately if using in research or publications.

**Suggested Citation:**
```
Park, H. (2024). Lecture 20: Model Explainability - SHAP and Deep Learning XAI. 
Ghent University. Retrieved from [URL]
```

---

**Last Updated:** 2024  
**Version:** 1.0  
**Slide Count:** 32 slides across 4 parts  
**Estimated Learning Time:** 8-12 hours (including hands-on practice)

For questions, corrections, or suggestions, please contact: homin.park@ghent.ac.kr