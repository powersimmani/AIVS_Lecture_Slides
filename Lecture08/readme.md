# Lecture 8: Loss, Optimization and Scheduling

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Total Slides:** 31  
**Difficulty Level:** Intermediate to Advanced  
**Estimated Duration:** 3-4 hours  
**Course Type:** Deep Learning / Machine Learning

This comprehensive lecture covers three fundamental pillars of neural network training: loss function design, optimization algorithms, and learning rate scheduling. Students will gain both theoretical understanding and practical insights into how these components work together to enable effective model training.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Design and select appropriate loss functions** for different machine learning tasks (regression, classification, metric learning) and understand their mathematical properties and use cases
2. **Implement and compare modern optimization algorithms** including SGD, Momentum, Adam, and AdamW, understanding their convergence characteristics and computational trade-offs
3. **Apply advanced learning rate scheduling strategies** such as step decay, exponential decay, cosine annealing, and warm-up schedules to improve model training dynamics
4. **Diagnose training problems** by analyzing loss curves and selecting appropriate combinations of loss functions, optimizers, and schedules for specific scenarios
5. **Understand the mathematical foundations** behind regularization, adaptive learning rates, and second-order optimization methods

---

## 📚 Lecture Structure

### **Part 1/3: Loss Function Design** (Slides 3-12)

#### Topics Covered:

**1. Role and Importance of Loss Functions** (Slide 4)
- Quantifies difference between predictions and actual values
- Provides gradient direction for optimization
- Task-specific design requirements (regression vs classification)
- Direct impact on model performance and behavior
- Acts as objective function for minimization
- Affects convergence speed and final accuracy

**2. Regression Losses - MSE, MAE, Huber** (Slide 5)
- **MSE (Mean Squared Error - L2):**
  - Strong penalty for large errors through squaring
  - Sensitive to outliers
  - Use when outliers are important signals
- **MAE (Mean Absolute Error - L1):**
  - Robust to outliers
  - Uniform penalty regardless of error size
  - Use when robustness is required
- **Huber Loss (Hybrid):**
  - Combines MSE and MAE advantages
  - Quadratic for small errors, linear for large errors
  - Adjustable with delta parameter (typically δ=1)
  - Provides balanced approach between sensitivity and robustness
- **Practical Example:** With 5 data points including one outlier (error=-20):
  - MSE = 83.6 (heavily penalized)
  - Huber = 19.3 (balanced)
  - MAE = 5.6 (moderate)

**3. Classification Loss - Cross-Entropy** (Slide 6)
- Measures dissimilarity between predicted and true probability distributions
- Uses Softmax activation for probability output
- **Binary Cross-Entropy:** For 2-class problems (0 or 1)
  - Formula: `-[y·log(p) + (1-y)·log(1-p)]`
  - Example: True=1, Pred=0.95 → Loss=0.051 (good)
  - Example: True=1, Pred=0.20 → Loss=1.609 (poor)
- **Categorical Cross-Entropy:** For multi-class problems (N classes)
  - Formula: `-Σ y·log(p)` (sum over correct class only)
  - Example: True=[0,1,0], Pred=[0.1,0.8,0.1] → Loss=0.223
- Encourages high confidence for correct classes
- Larger penalty for confident wrong predictions

**4. Hinge Loss and SVM** (Slide 7)
- Primary loss function for Support Vector Machines
- Focuses on maximizing margin between classes
- **Linear Hinge Loss:** `max(0, 1 - y·f(x))`
  - Margin-based linear penalty
- **Squared Hinge Loss:** `max(0, 1 - y·f(x))²`
  - Smoother, differentiable variant
- Only penalizes predictions within margin or misclassified
- Less sensitive to outliers compared to cross-entropy
- Use cases: binary classification, large-margin classifiers

**5. Focal Loss - Class Imbalance** (Slide 8)
- Designed for severe class imbalance problems
- Formula: `FL = -α(1-p)^γ log(p)`
- **Key Parameters:**
  - **γ (Focusing Parameter):** Typical value = 2
    - Controls down-weighting rate
    - Reduces contribution of well-classified examples
  - **α (Balancing Parameter):** Addresses class imbalance
    - Adjusts per-class weights
- Down-weights loss for well-classified examples
- Focuses learning on hard, misclassified examples
- Main applications: Object Detection (RetinaNet), imbalanced datasets

**6. Contrastive Loss** (Slide 9)
- Learns meaningful representations in embedding space
- Training on sample pairs (similar/dissimilar)
- **Similar Pairs:** Minimize distance (place close together)
- **Dissimilar Pairs:** Maintain margin (keep apart)
- Can learn without explicit labels
- Foundation for Siamese Networks
- Applications: Face Recognition, Signature Verification, Image Retrieval

**7. Triplet Loss** (Slide 10)
- Extension of Contrastive Loss with triplet structure
- **Components:**
  - **Anchor:** Reference sample
  - **Positive:** Sample similar to anchor
  - **Negative:** Sample different from anchor
- Formula: `max(d(a,p) - d(a,n) + margin, 0)`
- More efficient than Contrastive Loss (one optimization step per triplet)
- **Triplet Mining Strategies:**
  - Hard Negatives: Most difficult samples
  - Semi-Hard Negatives: Medium difficulty
  - Online vs Offline mining approaches
- Applications: Face Recognition (FaceNet), Person Re-identification

**8. Regularization Terms (L1, L2, L1+L2)** (Slide 11)
- **L2 Regularization (Ridge):** `λ Σ w²`
  - Penalizes sum of squared weights
  - Encourages small weights uniformly
  - Smooth solutions, no sparsity
- **L1 Regularization (Lasso):** `λ Σ |w|`
  - Penalizes sum of absolute weights
  - Promotes sparsity (sets weights to zero)
  - Enables automatic feature selection
- **Elastic Net (L1+L2):** `λ₁ Σ |w| + λ₂ Σ w²`
  - Combines advantages of both
  - Balances sparsity and grouping effect
  - Handles correlated features effectively
- **λ (Lambda):** Controls regularization strength (larger = stronger)

**9. Custom Loss Function Design** (Slide 12)
- Aligning loss with business metrics or task requirements
- Combining multiple loss components
- Domain-specific loss engineering

#### Key Concepts:
- Loss functions quantify model error and guide optimization
- Different tasks require different loss functions (regression vs classification vs metric learning)
- Robustness vs sensitivity trade-offs (MAE vs MSE vs Huber)
- Class imbalance handling through specialized losses (Focal Loss)
- Regularization prevents overfitting by constraining model complexity

#### Learning Outcomes:
- Select appropriate loss functions for regression and classification tasks
- Understand mathematical properties and use cases of common losses
- Design custom loss functions for specific requirements
- Apply regularization techniques to prevent overfitting
- Implement metric learning losses for similarity tasks

---

### **Part 2/3: Optimization Algorithms** (Slides 13-22)

#### Topics Covered:

**10. Gradient Descent Review** (Slide 14)
- **Core Concept:** Iterative optimization to minimize loss function
- **Update Rule:** `θ = θ - η∇L(θ)` where η is learning rate
- **Mechanism:**
  1. Calculate loss gradient for all parameters
  2. Move parameters opposite to gradient direction
  3. Repeat until convergence
- Foundation of all modern neural network training
- **Challenges:**
  - Can get trapped in local minima
  - Saddle point problems
  - Trade-off: Convergence speed ↔ Stability

**11. Batch vs Mini-Batch vs Stochastic** (Slide 15)
- **Batch Gradient Descent:**
  - Uses entire dataset per update
  - Stable but slow and memory-intensive
- **Stochastic Gradient Descent (SGD):**
  - One sample per update
  - Fast but noisy
- **Mini-Batch Gradient Descent:**
  - Batch size typically 32-512
  - Balance between speed and stability
  - Enables GPU parallelization

**12. Momentum Method** (Slide 16)
- **Core Concept:** Accelerates gradient descent by accumulating velocity
- **Update Rules:**
  1. Velocity Update: `v = βv + ∇L(θ)`
  2. Parameter Update: `θ = θ - ηv`
- **Momentum Coefficient β:** Typically 0.9 (retains 90% of previous velocity)
- **Benefits:**
  - Helps overcome local minima
  - Navigates ravines in loss landscape
  - Reduces oscillations in high curvature directions
  - Accelerates progress in consistent gradient directions
- **Physical Analogy:** Like a ball rolling down a hill gaining momentum
- **Effect:** Smooth progress vs zigzag movement without momentum

**13. Nesterov Accelerated Gradient (NAG)** (Slide 17)
- **Key Innovation:** "Look-ahead" mechanism
- Calculates gradient at anticipated future position
- **Update Rules:**
  1. Velocity Update (Look-ahead): `v = βv + ∇L(θ - βv)`
  2. Parameter Update: `θ = θ - ηv`
- **vs Standard Momentum:**
  - Momentum: Gradient at current position
  - Nesterov: Gradient at future position
- **Advantages:**
  - Better convergence than standard momentum
  - Reduces overshooting through early trajectory correction
  - Particularly effective for convex optimization
- **Trade-off:** Slightly higher computational cost
- Widely used in PyTorch and TensorFlow

**14. AdaGrad (Adaptive Gradient Algorithm)** (Slide 18)
- **Core Concept:** Adaptively adjusts learning rate per parameter
- **Update Rule:** `θ = θ - η/√(G + ε) ⊙ ∇L(θ)`
  - G: Cumulative sum of squared gradients
  - ⊙: Element-wise multiplication
  - ε: Prevents division by zero (typically 1e-8)
- **Adaptation Mechanism:**
  - Infrequent parameters: Larger updates
  - Frequent parameters: Smaller updates
- **Benefits:**
  - No manual learning rate tuning needed
  - Advantageous for sparse data
- **Main Drawbacks:**
  - Learning rate monotonically decreases over time
  - Aggressive decay can cause premature training stop
- **Best Use Cases:** Sparse data processing, NLP

**15. RMSprop (Root Mean Square Propagation)** (Slide 19)
- **Key Improvement:** Solves AdaGrad's monotonically decreasing LR problem
- **Innovation:** Uses exponential moving average of squared gradients
- **Update Rules:**
  1. EMA of Squared Gradients: `E[g²] = ρE[g²] + (1-ρ)g²`
  2. Parameter Update: `θ = θ - η/√(E[g²] + ε) ⊙ g`
- **Decay Rate ρ:** Typically 0.9 (maintains moving average)
- **vs AdaGrad:**
  - AdaGrad: Monotonically decreasing LR
  - RMSprop: LR can dynamically increase/decrease
- **Advantages:**
  - Effective for non-stationary objectives
  - Well-suited for Recurrent Neural Networks
  - Good default choice for various problems
- **Origin:** Introduced in Geoff Hinton's Coursera lectures (unpublished)

**16. Adam and AdamW** (Slide 20)
- **Adam (Adaptive Moment Estimation):**
  - Most popular optimization algorithm
  - Combines Momentum and RMSprop
  - Estimates both first moment (mean) and second moment (variance)
  - **Default Parameters:**
    - β₁ = 0.9 (Momentum)
    - β₂ = 0.999 (RMSprop)
    - ε = 1e-8 (numerical stability)
  - Includes bias correction for early iterations
- **AdamW (Adam with Weight Decay):**
  - Improved Adam using decoupled weight decay
  - Fixes weight decay implementation bug in original Adam
  - Decouples L2 regularization from gradient-based optimization
  - Generally better generalization than Adam
- **Key Difference:**
  - Adam: Weight decay in gradient
  - AdamW: Weight decay decoupled
- **Recommendation:** AdamW is preferred in modern deep learning

**17. Second-Order Optimization - L-BFGS** (Slide 21)
- **Core Concept:** Uses second-order information (Hessian)
- Efficient inverse Hessian approximation
- **First-Order vs Second-Order:**
  - First-Order: Uses gradient only, more iterations, lower cost per iteration
  - Second-Order: Uses Hessian, fewer iterations, higher cost per iteration (O(n²))
- **Characteristics:**
  - ✓ Fewer iterations than first-order methods
  - ✓ Deterministic method
  - ✗ High computational cost per iteration
  - ✗ Requires full batch evaluation
- **Limitations:**
  - Unsuitable for large-scale deep learning (memory constraints)
  - Not suitable for mini-batch optimization
- **Suitable Applications:**
  - Small-scale model optimization
  - Scientific computing
  - Traditional ML (Logistic Regression)
  - When full batch evaluation is feasible

**18. Comparison of Optimization Algorithms** (Slide 22)
- **Performance Metrics Comparison:**

| Metric | SGD | Momentum | RMSprop | Adam/AdamW |
|--------|-----|----------|---------|------------|
| Training Speed | Slow | Moderate | Fast | Fastest |
| Generalization | Good | Best | Moderate | Good |
| Memory Usage | Lowest | Low | Moderate | High |
| HP Sensitivity | High | Moderate | Low | Lowest |

- **Characteristics Summary:**
  - **SGD + Momentum:** Simple, reliable, good generalization, requires tuning
  - **Adam/AdamW:** Adaptive, fast convergence, out-of-the-box
  - **RMSprop:** Good for RNNs, non-stationary problems
  - **Speed Ranking:** Adam > RMSprop > Momentum > SGD
- **Recommendation:** Start with Adam/AdamW, then fine-tune with SGD+Momentum
- **Practice Resource:** Gradient Descent Visualization (GitHub: lilipads/gradient_descent_viz)

#### Key Concepts:
- Gradient descent forms the foundation of neural network optimization
- Momentum methods accelerate convergence by accumulating velocity
- Adaptive learning rate methods (AdaGrad, RMSprop, Adam) eliminate manual tuning
- Second-order methods are powerful but computationally expensive
- Trade-offs exist between speed, memory, and generalization

#### Learning Outcomes:
- Implement various optimization algorithms from scratch
- Select appropriate optimizers for different training scenarios
- Understand convergence properties and computational requirements
- Diagnose optimization problems through loss curve analysis
- Apply best practices for optimizer selection and hyperparameter tuning

---

### **Part 3/3: Learning Rate Scheduling** (Slides 23-30)

#### Topics Covered:

**19. Importance of Learning Rate** (Slide 24)
- **Most critical hyperparameter** affecting training dynamics
- **Learning Rate Impact:**
  - **Too Large:** Training divergence, loss oscillation, loss explosion
  - **Too Small:** Slow convergence, stuck in poor local minima
  - **Optimal (Dynamic):** Changes throughout training
- **Training Phase Strategy:**
  - **Early Training:** Large LR for fast initial progress
  - **Late Training:** Small LR for fine-tuning and convergence
- **Key Insights:**
  - Different layers may benefit from different learning rates
  - LR scheduling improves final model performance

**20. Fixed vs Adaptive Learning Rate** (Slide 25)
- **Fixed Learning Rate:**
  - ✓ Simple implementation
  - ✗ Suboptimal performance
  - ✗ Requires careful manual tuning
  - ✗ Difficult to find good value
- **Adaptive Learning Rate:**
  - ✓ Improved convergence
  - ✓ Better final accuracy
  - ✓ Fast early training
  - ✓ Careful late refinement
- **Adaptive LR Types:**
  - **Schedule-based:** Predefined decay (Step, Exponential)
  - **Performance-based:** Reduce on validation plateau
- **Key Benefits:** Combines fast early training with careful late refinement
- **Modern Practice:** Always use some form of LR Scheduling

**21. Step Decay** (Slide 26)
- **Core Concept:** Reduce LR by fixed factor at specific epochs
- **Formula:** `LR = initial_lr × 0.1^(epoch // step_size)`
- **Common Schedule Example:** Reduce by 0.1× every 30 epochs
  - Epoch 0-29: LR = 1.0 × initial_lr
  - Epoch 30-59: LR = 0.1 × initial_lr
  - Epoch 60-89: LR = 0.01 × initial_lr
- **Characteristics:**
  - ✓ Simple to implement and understand
  - ✓ Effective when training duration is known
  - ✓ Can align drops with learning regime changes
  - ⚠ May cause sudden jumps in loss when LR drops
- **Widely Used In:** Computer Vision (ResNet Training Schedule)

**22. Exponential Decay** (Slide 27)
- **Core Concept:** Gradual continuous reduction following exponential function
- **Formula:** `LR = initial_lr × decay_rate^(epoch / decay_steps)`
- **Decay Rate:** Typically 0.94-0.99 per epoch or step
- **vs Step Decay:**
  - Step Decay: Sudden changes, staircase reduction
  - Exponential: Smooth transition, continuous reduction
- **Characteristics:**
  - ✓ Smoother than step decay
  - ✓ No sudden changes
  - ✓ Effective for long training runs
  - ⚠ Can be too aggressive if decay rate poorly tuned
- **Commonly Used In:** Reinforcement Learning, NLP

**23. Cosine Annealing** (Slide 28)
- **Core Concept:** LR follows cosine function from max to min
- **Formula:** `LR = min_lr + (max_lr - min_lr) × (1 + cos(πt/T)) / 2`
  - t: current epoch, T: total epochs
- **Decay Behavior:**
  - Early Phase: Fast reduction
  - Late Phase: Slow reduction (fine-tuning)
- **Key Advantages:**
  - ✓ Smooth and gradual decay
  - ✓ No hyperparameters to tune (only max/min LR)
  - ✓ Improves generalization (empirically proven)
- **Warm Restarts (SGDR):** Can include restarts to escape local minima
- **Popular In:** Computer Vision, Transformer Training

**24. Warm-up and Linear Schedule** (Slide 29)
- **Phase 1 - Warm-up:**
  - Gradual increase from low value
  - Prevents large gradient updates from random initialization
  - Typically first 5-10% of training
- **Phase 2 - Linear Decay:**
  - Linear decrease from peak to minimum
  - Remaining 90-95% of training
- **Combined Schedule:** Warm-up → Linear or Cosine Decay
- **Key Benefits:**
  - ✓ Stabilizes training in early phase
  - ✓ Especially effective with large batch sizes
  - ✓ Essential for large-scale model training
- **Essential For:** BERT, GPT, Vision Transformers

**25. Cyclical Learning Rates** (Slide 30)
- **Core Concept:** LR varies cyclically between min and max values
- **Range Policies:**
  - **Triangular:** Linear increase/decrease
  - **Triangular2:** Range halves each cycle
  - **Exponential:** Exponential variation
- **Parameters:**
  - Cycle Length: Typically 2-10 epochs per cycle
- **1cycle Policy:** Single cycle with warm-up and cool-down phases
- **Key Benefits:**
  - ✓ Escape saddle points & explore loss landscape
  - ✓ Better generalization performance
  - ✓ Faster convergence
  - ✓ Regularization effect
  - ✓ Find better local minima through exploration
- **Successfully Used In:** fast.ai training methodology
- **Additional Resource:** PyTorch Learning Rate Schedulers guide

#### Key Concepts:
- Learning rate is the most important hyperparameter
- Fixed learning rates are suboptimal; adaptive schedules improve performance
- Different schedules suit different scenarios (step vs exponential vs cosine)
- Warm-up prevents instability in early training
- Cyclical rates enable exploration and better generalization

#### Learning Outcomes:
- Implement various LR scheduling strategies
- Select appropriate schedules for different model architectures
- Understand the mathematical basis of each schedule type
- Combine warm-up with decay schedules effectively
- Diagnose training issues related to learning rate

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Mathematics:**
  - Calculus: Derivatives, gradients, chain rule
  - Linear Algebra: Matrix operations, vector norms
  - Probability: Basic distributions, expectation
  - Optimization: Convex functions, local/global minima
- **Machine Learning:**
  - Supervised learning fundamentals
  - Neural network architecture basics
  - Backpropagation algorithm
  - Training/validation/test splits
- **Programming:**
  - Python proficiency (intermediate level)
  - NumPy for numerical operations
  - Basic understanding of PyTorch or TensorFlow

### Software Requirements:
- Python 3.8+
- PyTorch 2.0+ or TensorFlow 2.10+
- NumPy, Matplotlib, Pandas
- Jupyter Notebook or JupyterLab
- GPU recommended but not required (CPU training acceptable for exercises)

### Recommended Prior Courses:
- Introduction to Machine Learning
- Neural Networks and Deep Learning Fundamentals
- Linear Algebra and Calculus for ML

---

## 💻 Hands-on Components

### **Exercise 1: Loss Function Comparison**
**Objective:** Implement and compare MSE, MAE, and Huber loss on regression tasks

**Dataset:** Boston Housing or California Housing dataset

**Tasks:**
1. Implement all three loss functions from scratch
2. Train models with each loss function
3. Analyze performance with and without outliers
4. Visualize loss landscapes
5. Compare convergence speed and final performance

**Expected Outcomes:**
- Understanding of robustness vs sensitivity trade-offs
- Practical experience with outlier handling
- Ability to select appropriate loss for regression tasks

**Estimated Time:** 2-3 hours

---

### **Exercise 2: Optimizer Shootout**
**Objective:** Compare SGD, Momentum, RMSprop, Adam, and AdamW on image classification

**Dataset:** CIFAR-10 or Fashion-MNIST

**Architecture:** ResNet-18 or similar CNN

**Tasks:**
1. Train identical models with each optimizer
2. Track training/validation loss and accuracy
3. Measure training time and memory usage
4. Analyze convergence curves
5. Test generalization performance
6. Experiment with different learning rates per optimizer

**Metrics to Compare:**
- Training speed (epochs to convergence)
- Final validation accuracy
- Generalization gap (train-val difference)
- Memory footprint
- Sensitivity to learning rate

**Expected Outcomes:**
- Hands-on experience with all major optimizers
- Understanding of speed vs generalization trade-offs
- Ability to select and tune optimizers for specific tasks

**Estimated Time:** 4-5 hours

---

### **Exercise 3: Learning Rate Schedule Design**
**Objective:** Implement and evaluate different LR schedules on Transformer training

**Dataset:** Text classification (e.g., IMDB sentiment) or small-scale translation

**Model:** BERT-tiny or 2-layer Transformer

**Tasks:**
1. Implement step decay, exponential decay, cosine annealing
2. Add warm-up phase to each schedule
3. Experiment with cyclical learning rates (1cycle policy)
4. Visualize LR progression throughout training
5. Compare final model performance
6. Analyze training stability

**Deliverables:**
- Code implementing all schedules
- Comparative plots of LR curves
- Training loss curves for each schedule
- Performance comparison table
- Written analysis of results (1-2 pages)

**Expected Outcomes:**
- Practical experience with LR scheduling
- Understanding of schedule selection criteria
- Ability to debug training instability issues

**Estimated Time:** 3-4 hours

---

### **Exercise 4: Custom Loss Function Design**
**Objective:** Design a custom loss function for imbalanced classification

**Scenario:** Medical diagnosis with 95% negative, 5% positive class distribution

**Dataset:** Chest X-ray pathology detection or similar

**Tasks:**
1. Analyze baseline performance with standard cross-entropy
2. Implement Focal Loss with different γ values
3. Design custom class-weighted loss
4. Combine classification loss with auxiliary losses
5. Evaluate using precision, recall, F1-score, AUROC
6. Optimize loss hyperparameters

**Expected Outcomes:**
- Ability to design task-specific loss functions
- Experience with imbalanced data challenges
- Understanding of loss function impact on metrics

**Estimated Time:** 3-4 hours

---

## 📖 Additional Resources

### Academic Papers:
1. **Adam Optimizer:** Kingma & Ba (2015) - "Adam: A Method for Stochastic Optimization"
2. **AdamW:** Loshchilov & Hutter (2019) - "Decoupled Weight Decay Regularization"
3. **Focal Loss:** Lin et al. (2017) - "Focal Loss for Dense Object Detection"
4. **Cyclical Learning Rates:** Smith (2017) - "Cyclical Learning Rates for Training Neural Networks"
5. **Cosine Annealing:** Loshchilov & Hutter (2017) - "SGDR: Stochastic Gradient Descent with Warm Restarts"
6. **Huber Loss:** Huber (1964) - "Robust Estimation of a Location Parameter"

### Online Resources:
- **PyTorch Optimizer Documentation:** https://pytorch.org/docs/stable/optim.html
- **PyTorch LR Scheduler Guide:** https://www.leoniemonigatti.com/blog/pytorch-learning-rate-schedulers.html
- **Gradient Descent Visualization:** https://github.com/lilipads/gradient_descent_viz
- **fast.ai Course:** Practical Deep Learning (covers 1cycle policy)
- **distill.pub:** Visual explanations of optimization algorithms

### Books:
- **Deep Learning** by Goodfellow, Bengio & Courville (Chapter 8: Optimization)
- **Hands-On Machine Learning** by Aurélien Géron (Chapter 11: Training Deep Neural Networks)
- **Deep Learning with PyTorch** by Stevens, Antiga & Viehmann

### Video Lectures:
- Stanford CS231n (Lecture 7: Training Neural Networks II)
- fast.ai Practical Deep Learning (Lesson 5: Other Computer Vision Problems)
- Coursera: Deep Learning Specialization (Course 2, Week 2)

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Week 1:** Study Part 1 (Loss Functions)
   - Read slides 3-12
   - Complete Exercise 1 (Loss Function Comparison)
   - Read Huber (1964) and Focal Loss paper
   
2. **Week 2:** Study Part 2 (Optimization)
   - Read slides 13-22
   - Complete Exercise 2 (Optimizer Shootout)
   - Experiment with gradient descent visualization tool
   
3. **Week 3:** Study Part 3 (LR Scheduling)
   - Read slides 23-30
   - Complete Exercise 3 (LR Schedule Design)
   - Read Smith (2017) on cyclical LR
   
4. **Week 4:** Integration Project
   - Complete Exercise 4 (Custom Loss Design)
   - Apply all concepts to personal project
   - Review and consolidate learning

### For Classroom Use:
**Session 1 (90 min):** Loss Functions
- Lecture: Slides 1-12 (45 min)
- Interactive demo: Loss landscape visualization (15 min)
- Group discussion: When to use which loss? (15 min)
- Q&A (15 min)

**Session 2 (90 min):** Optimization Algorithms
- Lecture: Slides 13-22 (50 min)
- Live coding: Implementing momentum (20 min)
- Comparison demo: SGD vs Adam (10 min)
- Q&A (10 min)

**Session 3 (90 min):** Learning Rate Scheduling
- Lecture: Slides 23-30 (40 min)
- Live demo: LR finder technique (20 min)
- Case study: Transformer training with warm-up (20 min)
- Q&A (10 min)

**Session 4 (120 min):** Lab Session
- Hands-on: Exercise 2 or 3 (90 min)
- Group presentations (20 min)
- Discussion and wrap-up (10 min)

### Slideshow Navigation:
- The complete slideshow is available in `lecture08_slideshow.html`
- Use arrow keys (← →) or spacebar to navigate
- Progress bar shows current position
- Each slide has its own file for focused study

---

## 📊 Assessment Suggestions

### Formative Assessment (During Learning):
1. **Concept Checks (Multiple Choice):**
   - Which loss function is most robust to outliers?
   - What is the typical value of momentum coefficient β?
   - When should you use warm-up in LR scheduling?
   
2. **Short Answer Questions:**
   - Explain why AdaGrad's learning rate decreases monotonically
   - Compare Focal Loss and standard Cross-Entropy
   - Describe the look-ahead mechanism in Nesterov momentum

3. **Code Review:**
   - Identify errors in optimizer implementations
   - Suggest improvements to LR schedule design
   - Debug common training issues

### Summative Assessment (End of Topic):
1. **Programming Assignment (40%):**
   - Implement all four exercises
   - Grading rubric:
     - Code quality and documentation (10%)
     - Correct implementation (15%)
     - Experimental design (10%)
     - Analysis and insights (5%)

2. **Written Report (30%):**
   - 5-7 pages analyzing results from exercises
   - Must include:
     - Methodology description
     - Quantitative results (tables, graphs)
     - Comparative analysis
     - Insights and recommendations
     - Discussion of limitations

3. **Final Project (30%):**
   - Apply concepts to novel problem
   - Requirements:
     - Custom loss function design
     - Optimizer comparison
     - LR schedule optimization
     - Comprehensive evaluation
   - Deliverables:
     - Code repository
     - Technical report (8-10 pages)
     - 10-minute presentation

### Grading Criteria:
- **Technical Correctness (40%):** Proper implementation, valid experiments
- **Analysis Depth (30%):** Insightful interpretation, thorough comparison
- **Presentation Quality (20%):** Clear writing, effective visualizations
- **Innovation (10%):** Creative solutions, going beyond requirements

---

## 📝 Notes for Implementation

### Technical Requirements:
- **Minimum Hardware:**
  - CPU: 4+ cores recommended
  - RAM: 16GB minimum, 32GB recommended
  - Storage: 20GB free space for datasets and checkpoints
  
- **Recommended Hardware:**
  - GPU: NVIDIA RTX 3060 or better (12GB+ VRAM)
  - RAM: 32GB+
  - Storage: SSD with 50GB+ free space

### Software Environment Setup:
```bash
# Create conda environment
conda create -n dl_opt python=3.10
conda activate dl_opt

# Install PyTorch (with CUDA if available)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional packages
pip install numpy pandas matplotlib seaborn jupyter
pip install scikit-learn tensorboard
```

### Common Issues and Solutions:
1. **Loss exploding (NaN values):**
   - Reduce learning rate
   - Add gradient clipping
   - Check for numerical instability in loss function

2. **Slow convergence:**
   - Increase learning rate (if stable)
   - Try adaptive optimizer (Adam/AdamW)
   - Add momentum

3. **Overfitting:**
   - Increase regularization (L1/L2)
   - Add dropout
   - Use early stopping

4. **GPU out of memory:**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

### Best Practices:
- Always use tensorboard or wandb for tracking
- Save checkpoints regularly
- Log all hyperparameters
- Use random seeds for reproducibility
- Validate on separate set before testing
- Document all experiments thoroughly

### Expected Training Times:
- Exercise 1: ~30 min on CPU, ~5 min on GPU
- Exercise 2: ~2 hours on GPU (complete comparison)
- Exercise 3: ~1 hour on GPU (all schedules)
- Exercise 4: ~1.5 hours on GPU (full evaluation)

---

## 🙏 Credits

**Lecturer:** Ho-min Park  
**Institution:** Ghent University  
**Contact:**  
- Email: homin.park@ghent.ac.kr  
- Email: powersimmani@gmail.com

**Course Materials:**
- All slide materials designed and created by Ho-min Park
- Visualization tools and code examples adapted from open-source resources
- Practice materials reference publicly available datasets and papers

**Acknowledgments:**
- PyTorch and TensorFlow documentation teams
- fast.ai community for educational resources
- Research paper authors whose work is referenced throughout

**License:** Educational use only. Please contact the instructor for permission to use materials outside of educational contexts.

---

**Last Updated:** Based on lecture materials provided  
**Version:** 1.0  
**Total Slides:** 31 slides covering comprehensive deep learning optimization curriculum