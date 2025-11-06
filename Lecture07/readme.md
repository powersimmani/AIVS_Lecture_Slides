# Lecture 7: From Logistic Regression to Multi-layer Perceptrons

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Institution:** Ghent University  
**Total Slides:** 31  
**Difficulty Level:** Intermediate to Advanced  
**Estimated Duration:** 3-4 hours (including hands-on activities)  
**Format:** Interactive lecture with visual demonstrations and practical examples

This lecture provides a comprehensive exploration of deep neural network architectures, covering the theoretical foundations, modern activation functions, and advanced architectural patterns that have revolutionized deep learning. The course bridges the gap between basic logistic regression and sophisticated multi-layer perceptrons used in state-of-the-art applications.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand Deep Architecture Principles**: Explain why depth is crucial in neural networks and articulate the limitations of shallow networks versus the advantages of hierarchical representations
2. **Master Modern Activation Functions**: Compare and select appropriate activation functions (ReLU, Leaky ReLU, ELU, GELU, Swish) based on specific use cases and understand their impact on gradient flow
3. **Analyze Gradient Problems**: Identify and diagnose vanishing and exploding gradient problems, and apply appropriate solutions including activation function selection and architectural modifications
4. **Implement Advanced Architectures**: Design and construct networks using skip connections (ResNet), dense connections (DenseNet), bottleneck architectures, and inception modules
5. **Optimize for Efficiency**: Apply model compression techniques and design efficient architectures suitable for deployment in resource-constrained environments

---

## 📚 Lecture Structure

### **Part 1: The Need for Deep Neural Networks** (Slides 3-11)

#### Topics Covered:

**Slide 4: Limitations of Shallow Networks**
- Universal Approximation Theorem and its practical limitations
- Why exponentially many hidden units may be required
- Four key limitations:
  - No hierarchical pattern capture (images, text)
  - Compositional struggle (learning combinations from scratch)
  - Poor generalization on high-dimensional data
  - Limited abstraction requiring manual feature engineering
- Practical example: Face recognition challenges in shallow networks

**Slide 5: The Power of Depth - Hierarchical Representations**
- Three-layer hierarchical learning:
  - Layer 1: Low-level features (edges, textures, simple patterns)
  - Layer 2: Mid-level features (parts, components, object structures)
  - Layer 3+: High-level features (complete objects, semantic concepts)
- Human visual cortex analogy: V1 → V2 → V4 → IT pathway
- Transformation to linearly separable representation spaces
- Transfer learning capabilities of lower layers

**Slide 6: Feature Reuse and Composition**
- Compositional efficiency formula: k features → k² patterns
- Architecture visualization showing feature combination across layers
- Three key benefits:
  - Reusable components across different contexts
  - Exponential expressiveness with depth
  - Modular learning approach
- Example: "Eye" feature reused for faces, animals, reducing redundancy

**Slide 7: Parameter Efficiency**
- Comparison: Shallow (1 layer, 1000 units) vs Deep (3 layers, 100 units each)
- Parameter count: 1M+ (shallow) vs 30K (deep)
- Computational complexity: O(2ⁿ) vs O(n²)
- Benefits: Better generalization, faster training/inference, reduced redundancy

**Slide 8: Challenges of Deep Networks**
- Four categories of challenges:
  1. **Training Complexity**: Non-convex optimization, longer training time, high computational cost
  2. **Gradient Problems**: Vanishing gradients in early layers, exploding gradients, unstable updates
  3. **Generalization Issues**: Overfitting risk, memorization, poor test performance
  4. **Configuration Sensitivity**: Critical initialization, hyperparameter sensitivity, architecture design
- Historical barrier: Pre-2006, deep networks often underperformed shallow ones

**Slide 9: Vanishing Gradient Problem**
- Mathematical explanation: ∂L/∂w₁ ∝ ∏ σ'(zᵢ)
- Sigmoid derivative maximum: 0.25
- Effect across 10 layers: 0.25¹⁰ ≈ 9.5 × 10⁻⁷
- Consequences: Early layers learn very slowly, network acts effectively shallow
- Visual representation of gradient flow weakening through layers

**Slide 10: Exploding Gradient Problem**
- Causes: Large weight initialization, deep networks with improper scaling, ReLU without normalization
- Warning signs: NaN/Inf values, loss oscillation/divergence, extremely large weight updates
- Solutions: Gradient clipping (threshold at max norm), careful weight initialization, batch normalization, lower learning rates

**Slide 11: Overview of Solutions**
- Six solution categories:
  1. **Better Activation Functions**: ReLU, Leaky ReLU, ELU, GELU (→ Vanishing Gradient)
  2. **Smart Initialization**: Xavier, He Initialization (→ Gradient Issues)
  3. **Normalization Techniques**: Batch Norm, Layer Norm (→ Training Stability)
  4. **Skip Connections**: ResNet, Highway Networks (→ Vanishing Gradient)
  5. **Careful Optimization**: Adam, RMSprop, AdaGrad (→ Training Efficiency)
  6. **Regularization & Clipping**: Dropout, Weight Decay, Gradient Clipping (→ Overfitting & Explosion)
- Key insight: Robust training requires combining multiple techniques

#### Key Concepts:
- **Hierarchical Learning**: Deep networks learn progressively abstract representations
- **Compositional Efficiency**: Reusing features leads to exponential expressiveness
- **Gradient Flow**: Understanding how gradients propagate backward is crucial for training
- **Universal Approximation vs Practical Efficiency**: Theory vs practice in network design

#### Learning Outcomes:
- Articulate why depth matters more than width in most applications
- Diagnose gradient-related training problems from loss curves and weight statistics
- Justify architecture choices based on computational efficiency and parameter count
- Connect biological vision systems to artificial hierarchical feature learning

---

### **Part 2: Modern Activation Functions** (Slides 12-20)

#### Topics Covered:

**Slide 13: The ReLU Revolution**
- Definition: ReLU(x) = max(0, x)
- Historical impact: Enabled AlexNet (2012) to win ImageNet, sparked deep learning revolution
- Four advantages:
  - Prevents vanishing gradients (derivative = 1 for x > 0)
  - Computational efficiency (no exponential calculations)
  - Sparse activation (~50% neurons output zero)
  - Unchanged gradient flow through active neurons
- Limitations: Dead ReLU problem, non-zero centered, unbounded output

**Slide 14: Leaky ReLU & PReLU**
- **Leaky ReLU**: f(x) = max(αx, x) with α = 0.01 (fixed)
  - Prevents dead neurons with small negative slope
  - Minimal computational cost
- **PReLU**: Same formula but α is learned during training
  - Adaptive per channel/layer
  - Better performance in deep CNNs
- Performance comparison table: flexibility, complexity, use cases

**Slide 15: ELU & SELU**
- **ELU (Exponential Linear Unit)**:
  - f(x) = x if x > 0, f(x) = α(eˣ - 1) if x ≤ 0
  - Smooth exponential curve, negative values push toward zero mean
  - Reduces bias shift, more robust to noise
- **SELU (Scaled Exponential Linear Unit)**:
  - f(x) = λ × ELU(x), where λ ≈ 1.05, α ≈ 1.67
  - Special property: Self-normalizing without Batch Normalization
  - Maintains mean and variance automatically
- Usage comparison: ELU for general improvements, SELU for fully-connected networks

**Slide 16: Swish & GELU**
- **Swish (Self-Gating Activation)**:
  - f(x) = x × σ(βx), where β = 1 for Swish-1
  - Self-gating mechanism, smooth and non-monotonic
  - Discovered through neural architecture search
- **GELU (Gaussian Error Linear Unit)**:
  - f(x) = x × Φ(x), where Φ is Gaussian CDF
  - Probabilistic interpretation
  - Default activation in BERT and GPT models
- Key properties: Both smooth everywhere, outperform ReLU in Transformers, higher computational cost

**Slide 17: Activation Function Selection Guide**
- **Default choice**: Start with ReLU (fast, works well for most cases)
- **Scenario-based recommendations**:
  - Deep CNNs → Leaky ReLU / PReLU (prevents dead neurons)
  - Faster convergence → ELU (slightly slower but better accuracy)
  - Fully-connected deep networks → SELU (eliminates need for Batch Normalization)
  - Transformers / NLP → GELU (standard in BERT, GPT)
  - Limited compute → ReLU / Leaky ReLU (fastest speed)
  - Experimental / Research → Swish / GELU (potential accuracy gains)
- General rule: Modern smooth activations (GELU, Swish) often worth the extra cost

**Slide 18: Dead ReLU Problem**
- Definition: Neurons permanently output zero for all inputs
- Cause → Consequence → Impact chain:
  - Large negative gradients push weights negative
  - Neuron stops learning, effectively removed
  - Reduced capacity, wasted parameters
- Symptoms: >40% neurons inactive, monitor always-zero activations
- Solutions: Leaky ReLU/PReLU, lower learning rate, better initialization, gradient clipping

**Slide 19: Gradient Flow Analysis**
- Goal: Understand gradient propagation through different activations
- Comparison of four activation types:
  1. **ReLU**: Binary gradient (0 or 1) - ⚠️ Risky, can block gradients completely
  2. **Sigmoid/Tanh**: Max derivative 0.25 - ✗ Poor, exponential shrinkage
  3. **Leaky ReLU**: Always allows gradient - ✓ Good, small negative slope
  4. **Smooth (ELU, GELU, Swish)**: Continuous gradients - ★ Best for optimization
- Ideal property: Gradients neither vanish nor explode across depth
- Modern approach: Good activations + normalization + skip connections

**Slide 20: Layer-wise Activation Patterns**
- Key observation: Different layers may benefit from different activations
- Three-stage architecture:
  - **Early Layers**: ReLU (low-level features, input → features)
  - **Middle Layers**: ELU / GELU (complex patterns, feature processing)
  - **Output Layer**: Task-specific (Sigmoid/Softmax for predictions)
- Empirical findings:
  - Uniform activation often works well, mixed can be better
  - Neural Architecture Search explores per-layer choices
  - Practical tip: If experimenting, start from output and work backward
- Transformer pattern example: GELU in feedforward layers, linear (no activation) in attention

#### Key Concepts:
- **Activation Function Trade-offs**: Speed vs accuracy vs gradient properties
- **Dead Neuron Problem**: Understanding and preventing neuron death
- **Self-Normalization**: SELU's unique property of maintaining statistics
- **Task-Specific Selection**: Matching activation functions to architecture and domain

#### Learning Outcomes:
- Select appropriate activation functions based on network depth, architecture type, and computational constraints
- Diagnose dead ReLU problems through activation statistics monitoring
- Implement layer-wise activation strategies for improved performance
- Understand the mathematical foundations of gradient flow through different activations

---

### **Part 3: Advanced Architecture Patterns** (Slides 21-30)

#### Topics Covered:

**Slide 22: Skip Connection (ResNet)**
- Core formula: y = F(x) + x
- Architecture: Input → Layers → Addition with identity shortcut → Output
- Four key benefits:
  - Extreme depth: Enabled 152+ layer networks
  - Gradient highway: Direct path for gradients to early layers
  - Residual learning: Learns F(x) instead of full H(x)
  - Math benefit: Gradient always has component of 1
- Impact: ResNet-50/101/152 ImageNet champions, universal standard adoption

**Slide 23: Dense Connection (DenseNet)**
- Core idea: Each layer connects to ALL subsequent layers
- Formula: xₗ = Hₗ([x₀, x₁, ..., xₗ₋₁])
- Connectivity pattern visualization showing L₀→L₁→L₂→L₃→L₄ with all connections
- Four aspects:
  - **Advantages**: Maximum information flow, feature reuse, parameter efficiency
  - **Efficiency**: Fewer parameters than ResNet for same accuracy
  - **Memory consideration**: Requires storing all intermediate features
  - **Use cases**: Image classification, feature extraction, limited parameters
- Growth rate: Each layer adds k feature maps (typically k=12 or k=32)

**Slide 24: Bottleneck Architecture**
- Purpose: Reduce computational cost while maintaining expressiveness
- Structure: Input (256) → 1×1 reduce (64) → 3×3 process (64) → 1×1 expand (256)
- Dimensional flow example: 256 → 64 → 64 → 256 channels
- Three benefits:
  - Computational savings: ~70% fewer FLOPs
  - Trade-off: More layers, fewer operations
  - Usage: ResNet, Inception, and most modern architectures
- Modern standard: Almost all efficient architectures use bottlenecks

**Slide 25: The Role of 1×1 Convolution**
- Three primary functions:
  1. **Channel Reduction**: Projects high-dimensional features to lower dimensions
  2. **Channel Expansion**: Increases feature map channels without spatial processing
  3. **Cross-Channel Mixing**: Learns relationships between different channels
- Example: 256 input channels → 1×1 Conv → 64 output channels
- Benefits: Computational efficiency vs larger kernels, Network-in-Network adds depth
- Universal use: Inception, ResNet, MobileNet, Transformers (feedforward layers)

**Slide 26: Inception Module**
- Core idea: Multiple parallel conv paths with different receptive fields
- Structure: Input → Four parallel paths:
  1. 1×1 reduce → 1×1 conv
  2. 1×1 reduce → 3×3 conv
  3. 1×1 reduce → 5×5 conv
  4. Max pool → 1×1 conv
  → Concatenation (filter concat)
- Three features:
  - Multi-scale: Captures features at different scales
  - Efficiency: 1×1 convolutions reduce dimensions first
  - Diversity: Rich feature diversity
- Evolution: GoogLeNet (v1, ImageNet 2014) → Inception v2/v3 (factorized convs) → Inception-ResNet → Modern influence (EfficientNet, NAS)

**Slide 27: Depthwise Separable Convolution**
- Key insight: Spatial and channel-wise processing can be separated
- **Standard Convolution**: Single operation, all channels together (Params: D² × M × N)
- **Depthwise Separable**: Two-step process:
  1. Depthwise Conv: Each channel processed separately
  2. Pointwise (1×1): Combine channels
  - Parameters: D² × M + M × N
- Efficiency gains: 8-9× fewer operations, massive parameter reduction
- Applications: MobileNet, EfficientNet, Xception
- Trade-off: Slightly lower accuracy, massive efficiency

**Slide 28: Neural Architecture Search (NAS)**
- Goal: Automatically discover optimal network architectures using ML
- Three-step process:
  1. **Search Space**: Operations, connections, layer counts, kernel sizes, channel counts
  2. **Search Strategy**: Reinforcement learning, evolutionary algorithms, gradient-based
  3. **Optimal Architecture**: Result
- Four components:
  - Search space: Define possible operations and connections
  - Search strategies: RL, evolutionary, gradient-based methods
  - Success stories: NASNet, EfficientNet, AmoebaNet (state-of-the-art results)
  - Computational cost: Originally thousands of GPU-hours, now more efficient
- Future direction: Architecture search becoming standard practice for deployment optimization

**Slide 29: Model Compression Techniques**
- Five techniques:
  1. **Pruning**: Remove unimportant weights/neurons (structured or unstructured)
  2. **Quantization**: Reduce precision FP32 → INT8 (~4× memory/speed improvement)
  3. **Knowledge Distillation**: Train small student to mimic large teacher
  4. **Low-Rank Factorization**: Decompose weight matrices into smaller components
  5. **Weight Sharing**: Group weights to reduce unique parameters
- Typical results: 5-10× compression ratio, <1% accuracy loss
- Deployment contexts: Mobile (essential for phones, IoT devices), Tools (TensorFlow Lite, PyTorch Mobile, ONNX Runtime)

**Slide 30: Practical Design Guidelines**
- Starting point: Begin with proven architectures (ResNet, EfficientNet), modify as needed
- Four guideline categories:
  1. **Architecture Choices**: Depth vs width (deeper generally better), residual connections for >10 layers
  2. **Key Components**: Batch norm after conv, before activation; ReLU/GELU (avoid sigmoid/tanh)
  3. **Regularization Stack**: Data augmentation, dropout, weight decay
  4. **Architecture Principles**: Gradually reduce spatial size, increase channels through network
- Validation: Consider inference speed and memory, not just accuracy

#### Key Concepts:
- **Skip Connections**: Creating gradient highways through identity mappings
- **Dense Connectivity**: Maximizing information flow between all layers
- **Efficient Convolutions**: Separating spatial and channel-wise operations
- **Automated Design**: Using ML to discover optimal architectures

#### Learning Outcomes:
- Implement ResNet-style skip connections to enable training of very deep networks
- Design bottleneck architectures for computational efficiency
- Apply depthwise separable convolutions for mobile and edge deployment
- Evaluate trade-offs between model complexity, accuracy, and inference speed
- Select appropriate compression techniques for deployment scenarios

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Mathematics**:
  - Linear algebra: Matrix operations, vector spaces, eigenvalues
  - Calculus: Partial derivatives, chain rule, gradient descent
  - Probability: Basic probability theory, distributions, expectation
- **Machine Learning Fundamentals**:
  - Supervised learning concepts
  - Loss functions and optimization
  - Overfitting, underfitting, and regularization
  - Basic neural networks (perceptrons, feedforward networks)
- **Programming**:
  - Python proficiency (intermediate level)
  - NumPy for numerical computations
  - Familiarity with at least one deep learning framework (PyTorch or TensorFlow)

### Software Requirements:
```
Python >= 3.8
PyTorch >= 2.0 or TensorFlow >= 2.10
NumPy >= 1.21
Matplotlib >= 3.5
Jupyter Notebook or JupyterLab
```

### Optional but Recommended:
- GPU access (CUDA-enabled) for hands-on experiments
- Understanding of computer vision basics
- Familiarity with backpropagation algorithm
- Experience with model training and evaluation

---

## 💻 Hands-on Components

### Project 1: Activation Function Comparison
**Objective**: Empirically compare different activation functions on a classification task

**Tasks**:
1. Implement a deep feedforward network (10+ layers) with different activation functions:
   - Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, SELU, GELU
2. Train on MNIST or CIFAR-10 dataset
3. Monitor and visualize:
   - Training/validation loss curves
   - Gradient magnitudes across layers
   - Activation statistics (mean, std, sparsity)
   - Dead neuron percentages for ReLU variants
4. Compare convergence speed, final accuracy, and training stability

**Expected Deliverables**:
- Jupyter notebook with implementation
- Comparative plots and analysis report
- Recommendations for when to use each activation

**Estimated Time**: 3-4 hours

---

### Project 2: Implementing ResNet from Scratch
**Objective**: Build a ResNet-style architecture and understand skip connections

**Tasks**:
1. Implement basic residual block: y = F(x) + x
2. Build ResNet-18 architecture (or similar depth)
3. Compare with plain network of same depth (without skip connections)
4. Visualize gradient flow through the network
5. Experiment with different skip connection patterns:
   - Identity shortcuts
   - Projection shortcuts (1×1 convolutions)
   - Dense connections (DenseNet style)

**Expected Deliverables**:
- Clean, modular PyTorch/TensorFlow code
- Training comparison plots (plain vs ResNet)
- Gradient flow analysis
- Written explanation of why skip connections help

**Estimated Time**: 5-6 hours

---

### Project 3: Efficient Architecture Design
**Objective**: Design and deploy an efficient model for mobile/edge devices

**Tasks**:
1. Start with a baseline CNN on CIFAR-10
2. Apply compression techniques:
   - Replace standard convolutions with depthwise separable convolutions
   - Add bottleneck blocks
   - Implement 1×1 convolutions for dimension reduction
3. Measure:
   - Parameter count (aim for <500K parameters)
   - FLOPs (floating-point operations)
   - Inference time on CPU
   - Accuracy (target: <2% drop from baseline)
4. Optionally: Apply quantization (FP32 → INT8)

**Expected Deliverables**:
- Efficient model implementation
- Comparison table (baseline vs optimized):
  - Parameters, FLOPs, inference time, accuracy
- Analysis of accuracy-efficiency trade-offs
- Deployment-ready model

**Estimated Time**: 6-8 hours

---

### Project 4 (Advanced): Neural Architecture Search Exploration
**Objective**: Implement a simple NAS algorithm to discover architectures

**Tasks**:
1. Define a search space:
   - Layer types: Conv, Depthwise Sep Conv, Residual Block
   - Number of layers: 5-15
   - Channel sizes: 32, 64, 128, 256
2. Implement a simple search strategy:
   - Random search (baseline)
   - Evolutionary algorithm or reinforcement learning
3. Evaluate discovered architectures on a small dataset
4. Analyze:
   - Pareto frontier (accuracy vs efficiency)
   - Common patterns in successful architectures
   - Computational cost of search process

**Expected Deliverables**:
- NAS implementation code
- Top 5 discovered architectures
- Analysis report with visualizations
- Comparison with hand-designed baselines

**Estimated Time**: 10-12 hours

---

## 📖 Additional Resources

### Research Papers (Foundational):
1. **Deep Learning Book** (Goodfellow et al., 2016) - Chapters 6-9
2. **ImageNet Classification with Deep CNNs** (Krizhevsky et al., 2012) - AlexNet paper
3. **Deep Residual Learning for Image Recognition** (He et al., 2016) - ResNet paper
4. **Densely Connected Convolutional Networks** (Huang et al., 2017) - DenseNet paper
5. **Going Deeper with Convolutions** (Szegedy et al., 2015) - Inception paper
6. **MobileNets: Efficient CNNs for Mobile Vision** (Howard et al., 2017)
7. **Neural Architecture Search with Reinforcement Learning** (Zoph & Le, 2017)

### Activation Functions:
- **Rectifier Nonlinearities Improve Neural Network Acoustic Models** (Maas et al., 2013) - Leaky ReLU
- **Fast and Accurate Deep Network Learning by ELU** (Clevert et al., 2016)
- **Self-Normalizing Neural Networks** (Klambauer et al., 2017) - SELU paper
- **Gaussian Error Linear Units (GELUs)** (Hendrycks & Gimpel, 2016)
- **Searching for Activation Functions** (Ramachandran et al., 2017) - Swish paper

### Online Resources:
- **CS231n: CNNs for Visual Recognition** (Stanford) - Lectures 6-9
  - Video lectures and notes on network architectures
- **Deep Learning Specialization** (Coursera) - Course 4: CNNs
- **Distill.pub** - Interactive visualizations of network architectures
- **Papers With Code** - Implementation benchmarks and leaderboards
- **PyTorch Tutorials** - Official tutorials on building custom architectures
- **TensorFlow Model Garden** - Reference implementations of major architectures

### Tools and Frameworks:
- **TorchVision Models** - Pre-trained ResNet, DenseNet, Inception, MobileNet
- **Keras Applications** - High-level API for standard architectures
- **ONNX** - Model exchange format for deployment
- **TensorBoard / Weights & Biases** - Experiment tracking and visualization
- **NetScope** - CNN architecture visualization tool

### Books:
- **Hands-On Machine Learning** (Géron, 2019) - Chapters 10-11
- **Deep Learning with Python** (Chollet, 2021) - Chapters 5-9
- **Dive into Deep Learning** (Zhang et al., 2021) - Free online book with code

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Sequential Learning Path**:
   - Start with Part 1 slides (3-11) to understand foundational concepts
   - Move to Part 2 (12-20) for activation functions - implement each one
   - Complete Part 3 (21-30) for advanced patterns - build incrementally

2. **Hands-on First Approach**:
   - Read slides 1-11 for theory
   - Immediately start Project 1 (Activation Functions)
   - Return to slides as questions arise
   - Progress through projects in order

3. **Interactive Slideshow**:
   - Open `lecture07_slideshow.html` in a web browser
   - Navigate using arrow keys or space bar
   - Practice explaining each concept out loud
   - Take notes on areas needing deeper study

### For Instructors:

#### Lecture Format (3-hour session):
- **Hour 1** (0:00-1:00):
  - Slides 1-11: Deep network motivation and challenges
  - Include: Live coding demo of vanishing gradient
  - Interactive: Poll students on gradient problem experiences
  
- **Hour 2** (1:00-2:00):
  - Slides 12-20: Activation functions and gradient flow
  - Include: Activation function visualization demo
  - Activity: Students sketch activation functions and derivatives
  
- **Hour 3** (2:00-3:00):
  - Slides 21-30: Advanced architectures
  - Include: ResNet implementation walkthrough
  - Discussion: Trade-offs in architecture design

#### Workshop Format (Full day):
- **Morning** (9:00-12:00):
  - Theory: All three parts with extended Q&A
  - Live demos of gradient problems and solutions
  
- **Lunch Break** (12:00-1:00)

- **Afternoon** (1:00-5:00):
  - Hands-on: Projects 1 and 2 in pairs
  - Instructor support and code reviews
  - Group presentations of findings

#### Assignment Suggestions:
- **Week 1**: Theory quiz + Project 1 (Activation Functions)
- **Week 2**: Project 2 (ResNet Implementation)
- **Week 3**: Project 3 (Efficient Architecture)
- **Optional**: Project 4 for advanced students or final project

### Individual Slide Usage:
Each HTML slide can be opened independently in a browser for focused study:
- Use individual slides for review of specific topics
- Share specific slides with students for pre-class preparation
- Reference during office hours for targeted explanations

---

## 📊 Assessment Suggestions

### Quiz Questions (Theory):

**Basic Level:**
1. What is the Universal Approximation Theorem and why doesn't it guarantee practical shallow networks?
2. Explain the vanishing gradient problem with mathematical notation
3. What is the derivative of ReLU? Why does this help with gradient flow?
4. Draw and explain a residual block in ResNet

**Intermediate Level:**
1. Compare ReLU, Leaky ReLU, and ELU: when would you use each?
2. Why do skip connections help train very deep networks? Provide mathematical intuition
3. Explain how depthwise separable convolution reduces computational cost
4. What is the "Dead ReLU" problem and how can it be prevented?

**Advanced Level:**
1. Analyze the trade-offs between ResNet and DenseNet architectures
2. Design an efficient architecture for a resource-constrained device (< 100K parameters)
3. Explain why SELU has self-normalizing properties
4. Discuss the search space and strategy in Neural Architecture Search

### Practical Assessment Rubric:

**Code Quality (30%)**:
- Clean, modular, well-documented code
- Proper use of PyTorch/TensorFlow APIs
- Reproducible experiments with random seeds
- Efficient implementation (no obvious bottlenecks)

**Experimental Rigor (30%)**:
- Multiple runs with error bars
- Proper train/val/test splits
- Ablation studies showing component contributions
- Fair comparisons (same training conditions)

**Analysis and Insights (25%)**:
- Clear visualizations with proper labels
- Insightful interpretation of results
- Connection to theoretical concepts
- Discussion of limitations and future work

**Documentation (15%)**:
- Clear README with setup instructions
- Well-commented code
- Comprehensive results report
- Proper citations of papers and resources

### Project Grading Examples:

**Project 1 (Activation Functions):**
- A Grade: All 7 activations implemented, comprehensive comparison, insightful analysis of gradient flow, clear recommendations
- B Grade: 5+ activations, basic comparison metrics, correct implementation, adequate visualizations
- C Grade: 3+ activations, working code, minimal analysis

**Project 2 (ResNet):**
- A Grade: Working ResNet with clear improvement over plain network, gradient flow visualization, experimentation with variations
- B Grade: Correct implementation, basic comparison, trains successfully
- C Grade: Implemented but may have issues, shows understanding of core concept

---

## 📝 Notes for Implementation

### Technical Requirements:

**Hardware Recommendations:**
- **Minimum**: CPU with 8GB RAM (Projects 1-2)
- **Recommended**: NVIDIA GPU with 6GB+ VRAM (Projects 3-4)
- **Cloud Alternatives**: Google Colab (free), Kaggle Kernels, AWS/Azure credits

**Dataset Sizes:**
- MNIST: ~50MB (simple experiments, fast iteration)
- CIFAR-10: ~170MB (standard benchmark)
- CIFAR-100: ~170MB (more challenging classification)
- ImageNet subset: ~5GB (for advanced projects)

### Common Implementation Issues:

1. **Vanishing Gradients**:
   ```python
   # Monitor gradient magnitudes during training
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: {param.grad.abs().mean()}")
   ```

2. **Dead ReLU Detection**:
   ```python
   # Count neurons that always output zero
   def count_dead_neurons(activations):
       dead = (activations == 0).all(dim=0).sum()
       return dead / activations.shape[1]
   ```

3. **Memory Issues with DenseNet**:
   - Use gradient checkpointing for memory efficiency
   - Reduce batch size if OOM errors occur
   - Consider memory-efficient implementation

4. **Skip Connection Dimension Mismatch**:
   ```python
   # Use 1x1 projection when dimensions don't match
   if x.shape[1] != out_channels:
       x = self.projection(x)
   return F.relu(self.conv(x) + x)
   ```

### Debugging Tips:
- **Gradient checking**: Compare numerical and analytical gradients
- **Overfitting test**: Can model memorize small training set?
- **Activation visualization**: Plot activation distributions per layer
- **Weight initialization**: Check initial weight statistics
- **Learning rate**: Use learning rate finder before full training

### Performance Optimization:
- Use DataLoader with `num_workers > 0` for faster data loading
- Enable `torch.backends.cudnn.benchmark = True` for consistent input sizes
- Use mixed precision training (FP16) for faster training on modern GPUs
- Profile code with `torch.profiler` or `cProfile` to find bottlenecks

---

## 🙏 Credits

**Instructor**: Ho-min Park, PhD  
**Affiliation**: Ghent University  
**Contact**: homin.park@ghent.ac.kr | powersimmani@gmail.com

### Acknowledgments:
This lecture synthesizes knowledge from seminal papers in deep learning, particularly:
- The ResNet paper (He et al., 2016) for skip connections
- The DenseNet paper (Huang et al., 2017) for dense connectivity
- The Inception papers (Szegedy et al., 2015-2017) for multi-scale architecture
- The MobileNet papers (Howard et al., 2017-2019) for efficient design
- Numerous activation function papers (Maas, Clevert, Klambauer, Hendrycks, et al.)

The visual design and pedagogical approach draw inspiration from CS231n (Stanford), Fast.ai courses, and the broader deep learning education community.

### License:
These materials are intended for educational purposes. Please contact the instructor for permission to reuse or redistribute.

---

**Last Updated**: November 2024  
**Version**: 1.0  
**Lecture Series**: Deep Learning Fundamentals