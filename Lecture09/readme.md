# Lecture 9: Initialization and Normalization

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr, powersimmani@gmail.com  
**Total Slides:** 31  
**Difficulty Level:** Intermediate to Advanced  
**Estimated Duration:** 3-4 hours

This lecture provides a comprehensive exploration of initialization strategies and normalization techniques in deep learning. Students will learn how proper weight initialization prevents gradient problems, how normalization techniques stabilize training, and how various regularization methods improve model generalization. The lecture bridges theoretical foundations with practical implementation strategies used in state-of-the-art deep learning models.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand and apply** various weight initialization strategies (Xavier, He, LSUV) and explain why zero initialization fails due to symmetry problems
2. **Implement and compare** normalization techniques (Batch Norm, Layer Norm, Group Norm, Instance Norm) and select appropriate methods based on architecture and batch size constraints
3. **Analyze** the Internal Covariate Shift problem and explain how normalization techniques address training instability
4. **Apply** regularization techniques including Dropout variants, Stochastic Depth, and data augmentation strategies to prevent overfitting
5. **Evaluate** trade-offs between different initialization and normalization approaches for specific deep learning tasks (CNNs, RNNs, Transformers, GANs)

---

## 📚 Lecture Structure

### Part 1: Initialization Strategies (9 topics)

**Topics Covered:**

1. **Why is Initialization Important?**
   - Sets starting point for gradient descent optimization
   - Affects convergence speed and final model performance
   - Prevents vanishing/exploding gradient problems
   - Determines learning effectiveness and training stability
   - Critical importance increases with network depth

2. **Problems with Zero Initialization**
   - Symmetry problem: all neurons compute identical outputs
   - Gradients are identical for all neurons in the same layer
   - Network fails to break symmetry during backpropagation
   - Results in redundant neurons learning same features
   - Effectively reduces network to single neuron per layer
   - No learning occurs - weights remain constant

3. **Random Initialization and Breaking Symmetry**
   - Small random values break symmetry between neurons
   - Enables different neurons to learn different features
   - Foundation for all modern initialization strategies
   - Must be carefully scaled to avoid gradient problems

4. **Gradient Vanishing/Exploding**
   - Vanishing gradients: extremely small gradients in early layers preventing effective learning
   - Exploding gradients: exponentially large gradients causing training instability
   - Caused by repeated multiplication of small/large values during backpropagation
   - Sigmoid/tanh activations particularly vulnerable
   - Proper initialization maintains gradient magnitude throughout network

5. **Xavier/Glorot Initialization**
   - Designed for tanh/sigmoid activation functions
   - Maintains variance of activations across layers
   - Weights sampled from distribution with variance = 2/(n_in + n_out)
   - Prevents gradients from vanishing or exploding
   - Standard choice for symmetric activation functions

6. **He Initialization (for ReLU)**
   - Specifically designed for ReLU and variants (LeakyReLU, PReLU)
   - Variance = 2/n_in to account for half of neurons being zero
   - Maintains proper signal propagation through ReLU networks
   - Industry standard for modern CNN architectures
   - Critical for training very deep networks (50+ layers)

7. **LSUV Initialization**
   - Layer-Sequential Unit-Variance initialization
   - Two-step process: orthogonal initialization + variance normalization
   - Uses small batch of actual data for variance estimation
   - More sophisticated than Xavier/He methods
   - Particularly effective for very deep networks (100+ layers)
   - Additional resource: [DeepLearning.AI Weight Initialization Guide](https://www.deeplearning.ai/ai-notes/initialization/index.html)

8. **Leveraging Pre-trained Weights**
   - Transfer learning from large-scale datasets (ImageNet, COCO)
   - Lower layers: freeze general features (edges, textures, patterns)
   - Higher layers: fine-tune for task-specific features
   - Dramatically reduces training time and data requirements
   - Standard practice in Computer Vision and NLP
   - Three-step process: pre-training → transfer & fine-tune → deploy

9. **Comparison of Initialization Strategies**
   - Zero Init: ❌ Never use - causes symmetry problem
   - Random Small Values: ⚠️ Basic - often insufficient for deep networks
   - Xavier/Glorot: ✓ Recommended for tanh/sigmoid activations
   - He Initialization: ✓ Recommended for ReLU and variants
   - LSUV: ✓✓ Advanced - best for extremely deep networks (100+ layers)
   - Pre-trained Weights: ✓✓ Preferred when transfer learning is applicable

**Key Concepts:**
- Symmetry breaking through random initialization
- Gradient flow preservation through proper variance scaling
- Activation function-specific initialization strategies
- Transfer learning as superior initialization method
- Trade-offs between computational cost and initialization quality

**Learning Outcomes:**
- Students can diagnose gradient problems and select appropriate initialization
- Students can implement Xavier and He initialization from scratch
- Students understand when to use transfer learning vs. training from scratch
- Students can explain the mathematical foundations of variance-preserving initialization

---

### Part 2: Normalization Techniques (9 topics)

**Topics Covered:**

10. **Internal Covariate Shift**
    - Definition: change in distribution of layer inputs during training
    - Causes training instability and requires lower learning rates
    - Slows down convergence significantly
    - Addressed by normalization techniques
    - Critical problem in deep networks (10+ layers)

11. **Batch Normalization (Batch Norm)**
    - Normalizes activations across batch dimension
    - Formula: (x - μ_batch) / σ_batch
    - Includes learnable scale (γ) and shift (β) parameters
    - Maintains running statistics (mean, variance) for inference
    - Enables higher learning rates (often 10-100x higher)
    - Acts as regularizer, reducing need for Dropout
    - Best for: CNNs with large batch sizes (≥32)
    - Limitation: performance degrades with small batches

12. **Layer Normalization (Layer Norm)**
    - Normalizes across feature dimension per sample
    - Independent of batch size - critical for small batches
    - Standard in Transformers and RNNs
    - Computes statistics per sample, not across batch
    - Best for: sequence models, small batch training
    - Used in: BERT, GPT, T5, and most modern NLP models

13. **Instance Normalization**
    - Normalizes per instance, per channel independently
    - Removes instance-specific contrast information
    - Normalizes style while preserving content
    - Each spatial map normalized separately
    - Primary applications: Style Transfer and GANs
    - Better than Batch Norm for artistic/generative tasks

14. **Group Normalization**
    - Divides channels into groups (typically 32 channels per group)
    - Each group normalized independently
    - Compromise between Layer Norm and Instance Norm
    - Independent of batch size like Layer Norm
    - Better than Layer Norm for visual tasks (CNNs)
    - Ideal for: small batch training in object detection and segmentation
    - Stable performance across different batch sizes

15. **Weight Normalization**
    - Reparameterizes weights: w = g · (v / ||v||)
    - Decouples magnitude (g) and direction (v) of weights
    - Normalizes weights rather than activations
    - Fewer parameters compared to Batch Norm
    - Faster training than Batch Norm in some cases
    - Works well with: RNNs and Reinforcement Learning
    - Less commonly used in modern architectures

16. **Spectral Normalization**
    - Constrains spectral norm of weight matrices: W_SN = W / σ(W)
    - Divides weights by largest singular value (σ)
    - Ensures Lipschitz continuity of network layers
    - Prevents gradient explosion in discriminator
    - Critical for stable GAN training
    - Standard in modern GANs: BigGAN, StyleGAN
    - Improves training stability without batch statistics

17. **Comparison of Normalization Techniques**
    - Batch Norm: CNNs with large batches - normalizes across batch
    - Layer Norm: RNNs, Transformers, small batches - batch-independent
    - Instance Norm: Style Transfer, GANs - per instance, per channel
    - Group Norm: Small batch vision tasks, detection, segmentation - groups channels
    - Weight Norm: RNNs, Reinforcement Learning - normalizes weights not activations
    - Spectral Norm: GAN discriminators - ensures Lipschitz continuity

18. **When to Use Which Normalization?**
    - 🖼️ Large Batch CNNs → Batch Normalization
    - 🔤 Transformers / RNNs → Layer Normalization
    - 🎨 Style Transfer → Instance Normalization
    - 🔍 Small Batch Detection → Group Normalization
    - 🎭 GANs (Discriminator) → Spectral Normalization
    - 📊 Online Learning / Streaming Data → Layer / Group Norm
    - Key considerations: batch size constraints, computational cost, architecture type, task requirements

**Key Concepts:**
- Internal Covariate Shift as fundamental training problem
- Batch dependency vs. batch-independent normalization
- Normalization target: activations vs. weights
- Task-specific normalization requirements (vision vs. NLP vs. generation)
- Trade-offs between normalization methods

**Learning Outcomes:**
- Students can implement Batch Norm and Layer Norm from scratch
- Students can select appropriate normalization based on architecture and constraints
- Students understand the mathematical foundations of each normalization technique
- Students can diagnose normalization-related training issues
- Students can explain why different tasks require different normalization approaches

---

### Part 3: Normalization and Generalization (7 topics)

**Topics Covered:**

19. **Principles of Dropout**
    - Randomly drops neurons during training (output set to 0)
    - Typical dropout rate: 0.2 - 0.5 (20% - 50%)
    - Prevents co-adaptation of neurons
    - Trains ensemble of sub-networks simultaneously
    - At inference: use all neurons with activation × (1 - dropout_rate)
    - Powerful regularization for large networks
    - Reduces overfitting significantly

20. **Dropout Variants (DropConnect, DropBlock)**
    - **DropConnect**: Drops weights instead of activations
    - **DropBlock**: Drops contiguous regions in feature maps
      - Better for CNNs than standard Dropout
      - Prevents information leaking through spatial correlation
    - **Spatial Dropout**: Drops entire feature channels
    - **DropPath**: Drops entire paths in multi-branch architectures
    - Each variant addresses specific architectural challenges

21. **Stochastic Depth**
    - Randomly drops entire layers during training
    - Creates shorter networks during training
    - Survival probability decreases with layer depth
    - Early layers: high survival probability (0.9)
    - Deep layers: lower survival probability (0.5)
    - Improves gradient flow to early layers
    - Reduces training time significantly
    - Enables training of extremely deep networks (1000+ layers)

22. **Data Augmentation Strategies**
    - **Geometric**: Flips, rotations, random crops
    - **Color Jittering**: Brightness, contrast, saturation adjustment
    - **Cutout**: Randomly mask image patches
    - **Random Erasing**: Erase random rectangular regions
    - Increases training data diversity without collecting new data
    - Improves model generalization and robustness
    - No additional storage requirements

23. **Mixup and CutMix**
    - **Mixup**: Linear interpolation of entire images
      - Formula: x_mix = λ·x_i + (1-λ)·x_j, y_mix = λ·y_i + (1-λ)·y_j
      - Smooth blending of images and labels
    - **CutMix**: Cut and paste regions between images
      - Retains local features better than Mixup
      - More natural appearance
      - Mixing ratio λ ~ Beta(α, α)
    - Shared benefits: smoother decision boundaries, improved robustness, better calibration

24. **Early Stopping**
    - Monitors validation performance during training
    - Stops when validation loss stops improving
    - Uses patience parameter: wait N epochs before stopping
    - Prevents overfitting to training data
    - Saves best model based on validation metric
    - Simple and effective regularization technique
    - Often combined with learning rate scheduling

25. **Ensemble Methods**
    - **Averaging**: Mean predictions from multiple models
    - **Voting**: Majority vote for classification tasks
    - **Stacking**: Meta-model trained on base model predictions
    - **Typical Size**: 3-10 models in ensemble
    - Benefits:
      - Reduces variance across predictions
      - Improves robustness of the model
    - Trade-off: computational cost vs. higher accuracy

**Key Concepts:**
- Dropout as implicit ensemble training
- Layer-wise vs. neuron-wise regularization strategies
- Data augmentation as regularization without additional data
- Label smoothing through mixing strategies
- Ensemble diversity for robust predictions

**Learning Outcomes:**
- Students can implement Dropout and its variants in neural networks
- Students understand the connection between Dropout and ensemble learning
- Students can design effective data augmentation pipelines for specific tasks
- Students can implement early stopping with appropriate patience values
- Students can build and evaluate ensemble models
- Students understand trade-offs between regularization techniques

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Deep Learning Fundamentals**
  - Forward and backward propagation
  - Gradient descent optimization
  - Loss functions and evaluation metrics
  - Understanding of overfitting and underfitting

- **Mathematics**
  - Linear algebra: matrix operations, eigenvalues, singular values
  - Calculus: partial derivatives, chain rule
  - Probability: distributions (Normal, Uniform), variance, expectation
  - Statistics: mean, variance, normalization, standardization

- **Neural Network Architectures**
  - Fully connected networks
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs) - basic understanding
  - Familiarity with activation functions (ReLU, sigmoid, tanh)

### Software Requirements:
- **Python 3.8+** with the following libraries:
  - PyTorch 1.12+ or TensorFlow 2.8+
  - NumPy 1.21+
  - Matplotlib for visualization
  - scikit-learn for utilities

- **Development Environment**
  - Jupyter Notebook or Google Colab
  - GPU access recommended for hands-on exercises (2GB+ VRAM)
  - Code editor (VS Code, PyCharm)

- **Optional Tools**
  - TensorBoard for training visualization
  - Weights & Biases for experiment tracking

---

## 💻 Hands-on Components

### Lab 1: Initialization Experiments (45-60 minutes)
**Objective:** Compare different initialization strategies on a deep network

**Tasks:**
1. Implement a 10-layer fully connected network
2. Train with different initialization methods:
   - Zero initialization (observe failure)
   - Random small values (0.01 * randn)
   - Xavier/Glorot initialization
   - He initialization with ReLU
3. Monitor and plot:
   - Training loss curves
   - Gradient magnitudes per layer
   - Activation distributions
   - Convergence speed (epochs to reach target accuracy)
4. Compare final test accuracy across methods

**Expected Results:**
- Zero init: No learning (loss stays constant)
- Small random: Slow convergence, potential vanishing gradients
- Xavier: Good performance with tanh/sigmoid
- He: Best performance with ReLU (2-3x faster convergence)

**Dataset:** MNIST or Fashion-MNIST (28×28 grayscale images)

---

### Lab 2: Normalization Techniques (60-75 minutes)
**Objective:** Implement and compare Batch Norm, Layer Norm, and Group Norm

**Tasks:**
1. Build a CNN for image classification (5-7 conv layers)
2. Create three variants:
   - Baseline (no normalization)
   - With Batch Normalization
   - With Layer Normalization
   - With Group Normalization (32 channels per group)
3. Train with different batch sizes: [2, 8, 32, 128]
4. Measure for each configuration:
   - Training stability (loss variance)
   - Convergence speed
   - Final test accuracy
   - Training time per epoch
5. Analyze how normalization affects training with small batches

**Expected Insights:**
- Batch Norm: Best with large batches (≥32), degrades with small batches
- Layer Norm: Consistent across all batch sizes
- Group Norm: Good compromise for small batch training
- All normalization methods: ~2-5x faster convergence than baseline

**Dataset:** CIFAR-10 (32×32 RGB images, 10 classes)

---

### Lab 3: Dropout and Regularization (45-60 minutes)
**Objective:** Apply dropout variants and evaluate regularization effects

**Tasks:**
1. Train an overfitting-prone model (large capacity, small dataset)
2. Implement and compare:
   - No dropout (baseline overfitting)
   - Standard Dropout (p=0.5)
   - Spatial Dropout for CNN layers
   - DropBlock with block_size=7
3. Evaluate on training vs. validation accuracy gap
4. Implement data augmentation pipeline:
   - Random flips, rotations (±15°)
   - Color jittering
   - Random crops
5. Compare training and validation curves with/without augmentation

**Analysis Questions:**
- How does dropout rate affect overfitting? Test p ∈ {0.2, 0.3, 0.5}
- Which dropout variant works best for your CNN architecture?
- What's the combined effect of dropout + data augmentation?

**Dataset:** Smaller subset of ImageNet or custom small dataset (1000-5000 images)

---

### Lab 4: Transfer Learning Project (90-120 minutes)
**Objective:** Apply pre-trained weights and fine-tuning strategies

**Tasks:**
1. Load a pre-trained ResNet-50 or EfficientNet from PyTorch/TensorFlow
2. Freeze initial layers (up to layer 3)
3. Replace final classification layer for your target task
4. Train with two strategies:
   - **Strategy A**: Freeze all layers, train only final layer
   - **Strategy B**: Freeze early layers, fine-tune layers 3-4 + final layer
5. Compare with training from random initialization (baseline)
6. Measure:
   - Training time to convergence
   - Final accuracy on test set
   - Number of trainable parameters
   - Data efficiency (accuracy vs. training set size)

**Expected Results:**
- Transfer learning: 10-50x faster than training from scratch
- Strategy B typically best: 5-10% higher accuracy than Strategy A
- Requires 10-100x less data than random initialization

**Dataset:** Custom classification task (e.g., flower species, food categories, medical images)

---

### Lab 5: Advanced Regularization (Optional Challenge, 90+ minutes)
**Objective:** Implement Mixup/CutMix and ensemble methods

**Tasks:**
1. Implement Mixup data augmentation:
   - Sample λ from Beta(α=1.0, β=1.0)
   - Mix two random images and their labels
2. Implement CutMix:
   - Cut random rectangular region from one image
   - Paste into another image
   - Adjust labels proportionally
3. Train models with:
   - Baseline augmentation
   - + Mixup
   - + CutMix
4. Build a 5-model ensemble using different:
   - Random seeds
   - Architectures (ResNet, DenseNet, EfficientNet)
   - Training strategies
5. Evaluate ensemble using averaging and majority voting

**Challenge Metrics:**
- Individual model accuracy vs. ensemble accuracy
- Calibration: plot confidence vs. accuracy
- Robustness: test on out-of-distribution samples

**Dataset:** CIFAR-100 or ImageNet subset

---

## 📖 Additional Resources

### Academic Papers (Foundational):
1. **Batch Normalization**
   - Ioffe & Szegedy (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML 2015.
   - [Paper Link](https://arxiv.org/abs/1502.03167)

2. **Initialization Strategies**
   - Glorot & Bengio (2010). "Understanding the difficulty of training deep feedforward neural networks." AISTATS 2010.
   - He et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." ICCV 2015.

3. **Layer Normalization**
   - Ba et al. (2016). "Layer Normalization." arXiv:1607.06450
   - Critical for Transformers and modern NLP

4. **Dropout**
   - Srivastava et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR 2014.

5. **Group Normalization**
   - Wu & He (2018). "Group Normalization." ECCV 2018.
   - [Paper Link](https://arxiv.org/abs/1803.08494)

### Online Resources:
- **DeepLearning.AI - Weight Initialization Guide**
  - Comprehensive visual guide to initialization strategies
  - https://www.deeplearning.ai/ai-notes/initialization/

- **Distill.pub - Understanding Batch Normalization**
  - Interactive visualizations of normalization effects
  - https://distill.pub (search for normalization articles)

- **PyTorch Documentation**
  - Official docs for nn.BatchNorm2d, nn.LayerNorm, nn.Dropout
  - https://pytorch.org/docs/stable/nn.html

- **Papers with Code**
  - Implementation comparisons and benchmarks
  - https://paperswithcode.com/methods/category/normalization

### Video Lectures:
- **Stanford CS231n** (Lectures 6-7)
  - Initialization, Batch Normalization, and Dropout
  - http://cs231n.stanford.edu/

- **deeplearning.ai Specialization** (Course 2, Week 3)
  - Hyperparameter tuning and normalization
  - https://www.coursera.org/specializations/deep-learning

### Books:
- **Deep Learning** by Goodfellow, Bengio, Courville
  - Chapter 8: Optimization for Training Deep Models
  - Chapter 7.8: Normalization
- **Dive into Deep Learning** (d2l.ai)
  - Chapter 7.5: Batch Normalization
  - Free online: https://d2l.ai/

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Sequential Learning Path:**
   - Review slides in order: Part 1 → Part 2 → Part 3
   - Complete prerequisite knowledge gaps before starting
   - Estimated time: 6-8 hours including labs

2. **Active Learning Approach:**
   - After each theory slide, pause and implement the concept
   - Complete corresponding lab exercises immediately
   - Test understanding by explaining concepts to peers

3. **Practice Schedule (Recommended):**
   - **Session 1 (2-3 hours):** Part 1 slides + Lab 1
   - **Session 2 (2-3 hours):** Part 2 slides + Lab 2
   - **Session 3 (2-3 hours):** Part 3 slides + Lab 3
   - **Session 4 (3-4 hours):** Labs 4-5 (project work)

### For Instructors:

**Lecture Delivery Options:**

**Option A - Traditional Lecture (3 hours):**
- Hour 1: Part 1 (Initialization) with live coding demos
- Hour 2: Part 2 (Normalization) with visualization examples
- Hour 3: Part 3 (Regularization) with case studies

**Option B - Flipped Classroom (2 class sessions):**
- Pre-class: Students review slides and complete Labs 1-2
- Session 1: Discussion, Q&A, advanced topics, start Lab 3
- Session 2: Student presentations of Lab 4-5 results

**Option C - Workshop Format (Full day):**
- Morning: Theory overview (all slides, 2-3 hours)
- Afternoon: Intensive lab work (Labs 1-5, 4-5 hours)
- Evening: Group presentations and discussion (1 hour)

**Teaching Tips:**
- Use visualization tools (TensorBoard) to show training dynamics
- Have students predict outcomes before running experiments
- Encourage comparison of results across different hyperparameters
- Facilitate peer code review for lab implementations

**Discussion Questions:**
1. "Why does zero initialization fail? Explain using gradient flow."
2. "When would you choose Layer Norm over Batch Norm? Give 3 scenarios."
3. "How does Dropout relate to ensemble methods mathematically?"
4. "Design a training strategy for a model with only 100 training samples."

---

## 📊 Assessment Suggestions

### Formative Assessment (During Learning):

**1. Concept Check Questions (After Each Part):**

*Part 1 Questions:*
- Q: "A 20-layer ReLU network with Xavier initialization shows vanishing gradients. What's wrong?"
  - A: Xavier is designed for tanh/sigmoid; use He initialization for ReLU
- Q: "You initialize all weights to 0.01. After 10 epochs, your network hasn't learned. Why?"
  - A: Still too small, creates symmetry problem; neurons compute identical outputs

*Part 2 Questions:*
- Q: "Your batch size is 2 due to GPU memory limits. Batch Norm performs poorly. What should you try?"
  - A: Layer Norm or Group Norm - both are batch-independent
- Q: "When would you use Instance Norm over Batch Norm?"
  - A: Style transfer, GANs, or when you want to normalize style while preserving content

*Part 3 Questions:*
- Q: "Your model overfits heavily. List 5 techniques you could try, in order of ease."
  - A: (1) Data augmentation, (2) Dropout, (3) Early stopping, (4) Reduce model size, (5) Collect more data
- Q: "Why does Mixup improve calibration?"
  - A: Soft labels create smoother decision boundaries, reducing overconfident predictions

**2. Code Review Exercises:**
- Review peer lab implementations
- Identify bugs in provided buggy code snippets
- Optimize given code for better performance

**3. Lab Checkpoints:**
- Require students to show intermediate results
- Instructor reviews training curves and provides feedback
- Peer discussions about unexpected results

---

### Summative Assessment (End of Module):

**Option 1: Lab Report (40% of grade)**
*Format:* Written report (5-7 pages) covering Labs 1-4

*Required Sections:*
1. **Introduction:** Problem statement and objectives
2. **Methods:** Implementation details for each lab
3. **Results:** 
   - Training curves with multiple methods
   - Comparison tables (accuracy, training time, convergence speed)
   - Visualization of gradient flow, activation distributions
4. **Analysis:**
   - Why do certain methods work better?
   - How do batch size and architecture affect normalization choice?
   - What's the relationship between regularization strength and overfitting?
5. **Conclusion:** Best practices and recommendations

*Grading Rubric:*
- Correctness of implementations (15%)
- Quality of experimental comparisons (10%)
- Depth of analysis and insights (10%)
- Code quality and documentation (5%)

---

**Option 2: Technical Exam (30% of grade)**
*Duration:* 60 minutes | *Format:* Mixed (multiple choice + short answer + coding)

*Sample Questions:*

**Multiple Choice (20 points):**
1. Which initialization is most appropriate for a 50-layer network with ReLU activations?
   - a) Zero initialization
   - b) Xavier/Glorot
   - c) He initialization ✓
   - d) Uniform(-0.01, 0.01)

2. Batch Normalization performance degrades when:
   - a) Using large batch sizes
   - b) Training on GPUs
   - c) Batch size < 8 ✓
   - d) Using momentum optimizer

**Short Answer (30 points):**
1. (10 pts) Explain why Batch Norm acts as a regularizer. Mention at least two mechanisms.
2. (10 pts) Compare Dropout vs. Stochastic Depth. When would you use each?
3. (10 pts) A GAN's discriminator training is unstable. Recommend a normalization technique and explain why.

**Coding Problem (50 points):**
```python
# Complete this PyTorch module with proper initialization
class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO: Implement 5-layer network with:
        # - Proper He initialization for ReLU
        # - Batch Normalization after each hidden layer
        # - Dropout (p=0.3) before final layer
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Students must implement initialization in __init__
# and proper train/eval mode handling in forward
```

*Expected Answer Key:*
- Correct use of `nn.init.kaiming_normal_` or `nn.init.kaiming_uniform_`
- Proper placement of BatchNorm layers
- Correct Dropout implementation with `self.training` check
- Clean code structure

---

**Option 3: Final Project (30% of grade)**
*Objective:* Apply all learned techniques to a real-world problem

*Project Requirements:*
1. **Problem Selection:**
   - Choose a dataset (image classification, NLP, time series)
   - Define success metrics (accuracy, F1, etc.)

2. **Implementation Requirements:**
   - Implement at least 3 different initialization strategies
   - Use at least 2 normalization techniques
   - Apply 3+ regularization methods
   - Compare all variants systematically

3. **Deliverables:**
   - GitHub repository with clean, documented code
   - 10-minute presentation with slides
   - Technical report (8-10 pages)
   - Trained models with reproducible results

*Grading Criteria:*
- Technical correctness (35%)
- Experimental rigor (25%)
- Code quality and reproducibility (20%)
- Analysis depth and insights (15%)
- Presentation quality (5%)

*Timeline:*
- Week 1: Proposal submission (dataset, baseline model)
- Week 2: Checkpoint 1 (initialization experiments)
- Week 3: Checkpoint 2 (normalization comparisons)
- Week 4: Final submission and presentation

---

### Assessment Alignment with Learning Objectives:

| Learning Objective | Lab Report | Exam | Final Project |
|-------------------|-----------|------|---------------|
| LO1: Initialization strategies | ✓✓ | ✓ | ✓✓ |
| LO2: Normalization techniques | ✓✓ | ✓✓ | ✓✓ |
| LO3: Internal Covariate Shift | ✓ | ✓ | ✓ |
| LO4: Regularization methods | ✓✓ | ✓ | ✓✓ |
| LO5: Trade-off evaluation | ✓ | ✓✓ | ✓✓ |

✓ = Assessed | ✓✓ = Heavily assessed

---

## 📝 Notes for Implementation

### Technical Requirements:

**Hardware:**
- **Minimum:** CPU with 8GB RAM (Labs 1-3 feasible)
- **Recommended:** GPU with 4GB+ VRAM (GTX 1050 Ti or better)
- **Optimal:** GPU with 8GB+ VRAM (RTX 2060 or better) for Labs 4-5
- **Cloud Options:** Google Colab (free tier sufficient), AWS SageMaker, Paperspace

**Software Environment Setup:**

```bash
# Create conda environment
conda create -n dl_lecture9 python=3.9
conda activate dl_lecture9

# Install PyTorch (GPU version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# OR Install TensorFlow (GPU version)
pip install tensorflow[and-cuda]

# Install additional packages
pip install numpy matplotlib scikit-learn jupyter tensorboard
pip install timm  # For pre-trained models
pip install torchinfo  # For model summaries
```

**Dataset Preparation:**
```python
# MNIST/Fashion-MNIST (Lab 1)
from torchvision import datasets, transforms
train_data = datasets.MNIST(root='./data', train=True, download=True)

# CIFAR-10 (Lab 2-3)
train_data = datasets.CIFAR10(root='./data', train=True, download=True)

# Pre-trained models (Lab 4)
import timm
model = timm.create_model('resnet50', pretrained=True, num_classes=10)
```

---

### Common Implementation Pitfalls:

**1. Batch Normalization Issues:**
```python
# ❌ WRONG: Forgetting to set train/eval mode
model.eval()  # Must call this during inference
output = model(test_data)

# ❌ WRONG: Using Batch Norm with batch_size=1
# Solution: Use Layer Norm or Group Norm instead

# ✓ CORRECT: Proper Batch Norm usage
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),  # After linear layer
    nn.ReLU(),
    nn.Dropout(0.3)
)
```

**2. Initialization Mistakes:**
```python
# ❌ WRONG: Initializing after moving to GPU
model = Model().to(device)
model.apply(init_weights)  # Too late!

# ✓ CORRECT: Initialize before moving to GPU
model = Model()
model.apply(init_weights)
model = model.to(device)

# ✓ CORRECT: He initialization for ReLU
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
```

**3. Dropout During Inference:**
```python
# ❌ WRONG: Dropout active during testing
model.train()  # Keeps dropout active!
test_acc = evaluate(model, test_loader)

# ✓ CORRECT: Disable dropout for inference
model.eval()  # Disables dropout
with torch.no_grad():
    test_acc = evaluate(model, test_loader)
```

**4. Transfer Learning Pitfalls:**
```python
# ❌ WRONG: Unfreezing all layers immediately
model = timm.create_model('resnet50', pretrained=True)
# All layers will be updated (wastes pre-training)

# ✓ CORRECT: Freeze early layers
for param in model.parameters():
    param.requires_grad = False
# Only train final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

---

### Performance Benchmarks (Expected Results):

**Lab 1 - Initialization (MNIST):**
| Method | Epochs to 95% | Final Accuracy | Gradient Flow |
|--------|---------------|----------------|---------------|
| Zero Init | Never | ~10% | Blocked |
| Small Random | 50-100 | 96-97% | Weak |
| Xavier | 20-30 | 97-98% | Good |
| He (ReLU) | 10-15 | 98-99% | Excellent |

**Lab 2 - Normalization (CIFAR-10):**
| Method | Batch Size | Epochs to 70% | Final Accuracy |
|--------|------------|---------------|----------------|
| Baseline | 32 | 80-100 | 72-75% |
| Batch Norm | 32 | 30-40 | 82-85% |
| Batch Norm | 2 | 60-80 | 75-78% |
| Layer Norm | 2 | 40-50 | 80-82% |
| Group Norm | 2 | 35-45 | 81-84% |

**Lab 3 - Regularization:**
| Method | Train Acc | Val Acc | Overfitting Gap |
|--------|-----------|---------|-----------------|
| No regularization | 95% | 70% | 25% ⚠️ |
| Dropout (0.5) | 88% | 78% | 10% ✓ |
| Data Augmentation | 85% | 80% | 5% ✓✓ |
| Both | 82% | 82% | 0% ✓✓✓ |

**Lab 4 - Transfer Learning (Custom Dataset, 1000 images):**
| Strategy | Training Time | Final Accuracy | Data Efficiency |
|----------|---------------|----------------|-----------------|
| Random Init | 2-3 hours | 65-70% | Requires 10k+ samples |
| Frozen | 10-15 min | 78-82% | Works with 500+ samples |
| Fine-tuned | 30-45 min | 85-90% | Works with 200+ samples |

---

### Debugging Tips:

**Issue 1: Loss is NaN**
- Check learning rate (try 1e-3 → 1e-4)
- Verify initialization (no inf/nan values)
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

**Issue 2: Model not learning (loss stuck)**
- Check if zero initialization was used
- Verify data normalization (mean=0, std=1)
- Ensure labels are correct format
- Try simpler model first

**Issue 3: Batch Norm performs poorly**
- Check batch size (increase to ≥16)
- Verify train/eval mode switching
- Try Layer Norm or Group Norm instead
- Check if momentum is too high (try 0.9 → 0.99)

**Issue 4: Overfitting despite regularization**
- Increase dropout rate (0.3 → 0.5)
- Add more data augmentation
- Reduce model capacity
- Use early stopping (patience=10)
- Try L2 regularization (weight_decay=1e-4)

---

### Code Templates:

**Template 1: Initialization Helper**
```python
def initialize_weights(model, init_type='he'):
    """
    Initialize model weights based on activation function.
    
    Args:
        model: PyTorch model
        init_type: 'xavier' for tanh/sigmoid, 'he' for ReLU
    """
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', 
                                       nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```

**Template 2: Training Loop with Normalization**
```python
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()  # Enable BatchNorm/Dropout
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()  # Disable BatchNorm/Dropout
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    accuracy = 100. * correct / len(loader.dataset)
    return total_loss / len(loader), accuracy
```

**Template 3: Data Augmentation Pipeline**
```python
from torchvision import transforms

# Training augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                          saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation/test (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**Template 4: Mixup Implementation**
```python
def mixup_data(x, y, alpha=1.0):
    """
    Applies Mixup augmentation.
    
    Returns:
        mixed_x: Mixed inputs
        y_a, y_b: Original labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Usage in training loop
mixed_input, target_a, target_b, lam = mixup_data(inputs, targets, alpha=1.0)
output = model(mixed_input)
loss = mixup_criterion(criterion, output, target_a, target_b, lam)
```

---

## 🙏 Credits

**Instructor:** Ho-min Park  
**Affiliation:** Ghent University  
**Contact:** 
- homin.park@ghent.ac.kr
- powersimmani@gmail.com

**Lecture Development:**
This lecture material synthesizes foundational research in deep learning initialization, normalization, and regularization techniques. Key contributions from:
- Sergey Ioffe and Christian Szegedy (Batch Normalization)
- Xavier Glorot and Yoshua Bengio (Xavier/Glorot Initialization)
- Kaiming He et al. (He Initialization, ResNet)
- Jimmy Lei Ba et al. (Layer Normalization)
- Yuxin Wu and Kaiming He (Group Normalization)
- Geoffrey Hinton et al. (Dropout)

**Course Context:**
This is Lecture 9 in a comprehensive Deep Learning course series. For complete course materials, please contact the instructor.

**License:**
These materials are provided for educational purposes. Please cite appropriately when using for teaching or research.

**Acknowledgments:**
- Students and TAs who provided feedback on earlier versions
- Open-source community for PyTorch and TensorFlow frameworks
- DeepLearning.AI for publicly available educational resources

---

**Last Updated:** 2025  
**Version:** 1.0

For questions, clarifications, or reporting issues with these materials, please contact the instructor directly.