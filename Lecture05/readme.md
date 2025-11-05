# Lecture 5: From Logistic Regression to Multi-layer Perceptrons

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr, powersimmani@gmail.com  
**Total Slides:** 31  
**Lecture Duration:** Approximately 2.5-3 hours  
**Difficulty Level:** Intermediate  
**Course Type:** Machine Learning / Deep Learning Fundamentals

This lecture provides a comprehensive introduction to Multi-layer Perceptrons (MLPs), bridging the gap between simple logistic regression and modern deep neural networks. Students will understand the theoretical foundations, architectural design principles, and the backpropagation algorithm that enables training of deep networks.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Analyze the limitations** of linear models and understand why multi-layer architectures are necessary for solving non-linearly separable problems like XOR
2. **Design and implement** multi-layer perceptron architectures with appropriate activation functions, layer sizes, and connectivity patterns
3. **Derive and apply** the backpropagation algorithm using the chain rule to compute gradients efficiently through multiple layers
4. **Implement** forward propagation and backpropagation algorithms using modern deep learning frameworks (PyTorch/TensorFlow)
5. **Evaluate** trade-offs between network depth, width, and computational complexity in neural network design

---

## 📚 Lecture Structure

### Part 1/3: Neural Network Motivation (Slides 3-11)

#### Topics Covered:

**Slide 4: Limitations of Logistic Regression**
- Linear decision boundaries restrict model expressiveness
- Cannot handle complex feature interactions without manual engineering
- Single-layer architecture prevents hierarchical representation learning
- Low model capacity insufficient for high-dimensional non-linear data
- Solution: Multi-layer neural networks with non-linear activations

**Slide 5: XOR Problem - Linear Inseparability**
- Classical example demonstrating linear model failure
- Truth table: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
- Visual proof: no single line can separate the classes
- Diagonal separator fails to correctly classify all points
- Motivates need for non-linear decision boundaries

**Slide 6: Need for Feature Space Transformation**
- Original feature space: data is entangled and not linearly separable
- Transformation function φ(x) maps inputs to new representation
- Transformed space: classes become linearly separable
- Neural networks learn φ(x) automatically through hidden layers
- Eliminates need for manual feature engineering

**Slide 7: The Idea of Multi-layer Structure**
- Input layer → Hidden layer 1 → Hidden layer 2 → Output layer
- Each layer performs feature transformation: h[l] = σ(W[l]h[l-1] + b[l])
- Layer hierarchy enables increasingly abstract representations
- Non-linear activations critical for learning complex patterns
- Depth provides exponential expressiveness with polynomial parameters

**Slide 8: Biological Neuron vs Artificial Neuron**
- Biological: dendrites (inputs), soma (cell body), axon (output), synapses (weights)
- Artificial: weighted sum z = Σwixi + b, activation function a = σ(z)
- Inspiration from neuroscience but simplified mathematical model
- Activation threshold mimics neuron firing
- Multiple neurons form layers in artificial networks

**Slide 9: Role of Activation Functions**
- Without activation: multiple layers collapse to single linear transformation
- With activation: enables non-linear transformations and complex patterns
- Common functions: Sigmoid σ(z) = 1/(1+e^-z), Tanh, ReLU
- Introduces non-linearity essential for learning
- Enables gradient flow during backpropagation

**Slide 10: Universal Approximation Theorem**
- Single hidden layer with finite neurons can approximate any continuous function
- Theoretical guarantee of neural network expressiveness
- Approximation accuracy improves as number of neurons increases
- Practical limitation: may require exponentially many neurons
- Deep networks more efficient than wide shallow networks

**Slide 11: Expressiveness and Depth**
- Shallow & wide: single hidden layer with many neurons (O(2^n) parameters)
- Deep & narrow: multiple hidden layers with fewer neurons per layer (O(n²) parameters)
- Deep networks achieve hierarchical feature learning
- Compositional depth provides exponential expressiveness
- Better generalization and stronger representation power

#### Key Concepts:
- **Linear Inseparability**: Fundamental limitation of single-layer models
- **Feature Space Transformation**: Hidden layers learn φ(x) automatically
- **Universal Approximation**: Theoretical foundation for neural network power
- **Depth vs Width**: Trade-off between network architecture choices
- **Non-linear Activations**: Essential component for learning complex functions

#### Learning Outcomes:
Students will understand why simple linear models fail on complex problems, how neural networks overcome these limitations through multi-layer architectures, and the theoretical guarantees provided by the universal approximation theorem.

---

### Part 2/3: MLP Structure and Forward Propagation (Slides 12-20)

#### Topics Covered:

**Slide 13: Single-layer Perceptron Review**
- Architecture: inputs (x₁, x₂, x₃) → weighted sum → activation → output ŷ
- Mathematical form: z = Σwixi + b, ŷ = σ(z)
- Components: weights (feature importance), bias (threshold), activation (non-linearity)
- Limitation: only learns linear decision boundaries
- Cannot solve XOR problem

**Slide 14: Multi-layer Perceptron Architecture**
- 5-layer network: Input (n features) → Hidden1 (m₁ units) → Hidden2 (m₂ units) → Hidden3 (m₃ units) → Output (k outputs)
- Fully connected: every neuron in layer l connects to all neurons in layer l+1
- Layer-wise computation: h[l] = σ(W[l]h[l-1] + b[l])
- Parameters: weights W[l] and biases b[l] for each layer
- Total parameters scale with layer sizes

**Slide 15: Weights and Biases**
- **Weights (W)**: Connection strength between neurons
  - Matrix form: W[l] ∈ ℝ^(m×n)
  - Larger |w| indicates stronger feature influence
  - Learned during training via gradient descent
- **Biases (b)**: Shifts activation to control threshold
  - Vector form: b[l] ∈ ℝ^m
  - One bias per neuron in the layer
  - Provides model flexibility for better data fitting
- Complete formula: z[l] = W[l]h[l-1] + b[l] → h[l] = σ(z[l])

**Slide 16: Forward Propagation Algorithm**
- Step-by-step computation from input to output
- For each layer l = 1, 2, ..., L:
  1. Compute pre-activation: z[l] = W[l]h[l-1] + b[l]
  2. Apply activation: h[l] = σ(z[l])
- Input layer: h[0] = x (raw features)
- Output layer: ŷ = h[L] (predictions)
- Sequential computation: information flows forward only

**Slide 17: Activation Functions - Sigmoid, Tanh**
- **Sigmoid**: σ(z) = 1/(1+e^-z)
  - Range: (0, 1)
  - Not zero-centered
  - Max gradient: 0.25
  - Use case: output layer for binary classification
- **Tanh**: tanh(z) = (e^z - e^-z)/(e^z + e^-z)
  - Range: (-1, 1)
  - Zero-centered (better for optimization)
  - Max gradient: 1.0
  - Use case: hidden layers
- Common issue: vanishing gradient for large |z|

**Slide 18: ReLU and Its Variants**
- **ReLU**: f(z) = max(0, z)
  - Simple, fast computation
  - No vanishing gradient for z > 0
  - Sparse activation
  - Problem: dead neurons (z < 0)
- **Leaky ReLU**: f(z) = max(αz, z) where α=0.01
  - Fixes dying ReLU problem
  - Small negative slope maintains gradient
  - All neurons remain active
- **ELU**: f(z) = z if z>0, else α(e^z-1)
  - Smooth everywhere
  - Pushes mean activation toward zero
  - Slower due to exponential computation

**Slide 19: Output Layer Design (Regression vs Classification)**
- **Regression**: Linear activation or identity function
  - Output: ŷ ∈ ℝ (unbounded continuous values)
  - Loss: Mean Squared Error (MSE)
  - Example: predicting house prices, temperature
- **Binary Classification**: Sigmoid activation
  - Output: ŷ ∈ (0,1) (probability)
  - Loss: Binary Cross-Entropy
  - Example: spam detection, disease diagnosis
- **Multi-class Classification**: Softmax activation
  - Output: probability distribution over K classes
  - Loss: Categorical Cross-Entropy
  - Example: image classification, language modeling

**Slide 20: Network Capacity and Complexity**
- **Capacity**: Model's ability to fit a wide variety of functions
- Controlled by: number of layers, neurons per layer, total parameters
- **Width**: More neurons per layer increases representational power
- **Depth**: More layers enables hierarchical abstractions
- Trade-off: higher capacity risks overfitting, requires more data
- Regularization needed to prevent overfitting (dropout, weight decay)

#### Key Concepts:
- **Forward Propagation**: Sequential layer-by-layer computation from input to output
- **Activation Functions**: Non-linear transformations enabling complex pattern learning
- **Weight Matrices & Bias Vectors**: Learnable parameters defining network behavior
- **Network Architecture**: Design choices affecting capacity and performance
- **Output Layer Design**: Task-specific activation and loss function selection

#### Learning Outcomes:
Students will be able to design MLP architectures appropriate for different tasks, implement forward propagation algorithms, select appropriate activation functions for hidden and output layers, and understand the impact of network depth and width on model capacity.

---

### Part 3/3: Backpropagation and Learning (Slides 21-30)

#### Topics Covered:

**Slide 22: Loss Function Definition**
- **Purpose**: Quantifies prediction error to guide learning
- **Regression Loss (MSE)**: L = (1/2n)Σ(yi - ŷi)²
  - Penalizes large errors quadratically
  - Differentiable everywhere
  - Sensitive to outliers
- **Classification Loss (Cross-Entropy)**: L = -Σyi log(ŷi)
  - Measures probability distribution difference
  - Stronger gradient signal than MSE
  - Numerically stable with softmax
- **Optimization Goal**: Find W, b that minimize L(W, b)

**Slide 23: Chain Rule**
- **Mathematical Foundation**: d/dx f(g(x)) = f'(g(x)) · g'(x)
- Decomposes complex derivatives into simpler parts
- Example: d/dx sin(x²) = cos(x²) · 2x
- **Multi-layer Composition**: y = f₃(f₂(f₁(x)))
  - ∂y/∂x = ∂f₃/∂f₂ · ∂f₂/∂f₁ · ∂f₁/∂x
- **Backpropagation Application**: ∂L/∂w[l] = ∂L/∂z[l] · ∂z[l]/∂w[l]
- Enables efficient gradient computation through network

**Slide 24: Backpropagation Algorithm Derivation**
- **Forward Pass**: Compute and store all activations h[l] and pre-activations z[l]
- **Output Layer Gradient**: δ[L] = ∂L/∂z[L] = (ŷ - y) ⊙ σ'(z[L])
- **Hidden Layer Gradients** (backward recursion): δ[l] = (W[l+1]ᵀδ[l+1]) ⊙ σ'(z[l])
- **Parameter Gradients**:
  - ∂L/∂W[l] = δ[l] · h[l-1]ᵀ
  - ∂L/∂b[l] = δ[l]
- **Update Rule**: W[l] ← W[l] - η∂L/∂W[l], b[l] ← b[l] - η∂L/∂b[l]

**Slide 25: Computational Graph**
- Visual representation of computation flow
- Nodes: operations (matrix multiply, addition, activation)
- Edges: data flow (forward) and gradient flow (backward)
- Forward pass: compute outputs following graph edges
- Backward pass: compute gradients in reverse order
- Facilitates automatic differentiation implementation

**Slide 26: Gradient Calculation Example**
- Concrete numerical example with 2-layer network
- Forward pass: compute layer outputs step-by-step
- Loss calculation: compare prediction with target
- Backward pass: propagate gradients layer-by-layer
- Parameter updates: apply gradient descent with learning rate
- Demonstrates gradient flow through entire network

**Slide 27: Automatic Differentiation (Autograd)**
- **Dynamic Computation Graphs**: Build graph during forward pass
- **Operator Overloading**: Track operations on tensors automatically
- **Reverse Mode AD**: Efficiently computes all gradients in one backward pass
- **PyTorch Example**: `loss.backward()` computes all gradients automatically
- **TensorFlow/JAX**: Similar autograd capabilities
- Eliminates manual gradient derivation errors

**Slide 28: Mini-batch Gradient Descent**
- **Batch Gradient Descent**: Use all training samples (slow, stable)
- **Stochastic Gradient Descent (SGD)**: Use 1 sample (fast, noisy)
- **Mini-batch**: Use B samples (balanced trade-off)
- Common batch sizes: 32, 64, 128, 256
- **Advantages**:
  - Computational efficiency with vectorization
  - Faster convergence than full batch
  - More stable than pure SGD
  - Better GPU utilization
- Update rule: W ← W - η(1/B)Σ∇L(W; xᵢ, yᵢ)

**Slide 29: Implementation Tips and Debugging**
- **Weight Initialization**: Use Xavier/He initialization, avoid zeros
- **Learning Rate**: Start with 0.001-0.01, use learning rate schedules
- **Gradient Checking**: Numerical approximation to verify backprop
- **Monitoring**: Track train/validation loss and accuracy
- **Common Issues**:
  - Exploding gradients: use gradient clipping
  - Vanishing gradients: use ReLU, batch normalization
  - Overfitting: apply dropout, L2 regularization
- **Debugging Tools**: Print shapes, visualize activations, check gradient norms

**Slide 30: PyTorch/TensorFlow Hands-on**
- **Define Model**: Create nn.Module subclass with layers
- **Forward Pass**: Implement forward() method
- **Loss Function**: Use built-in losses (nn.CrossEntropyLoss, nn.MSELoss)
- **Optimizer**: Choose Adam, SGD, or RMSprop
- **Training Loop**:
  1. Forward pass: predictions = model(inputs)
  2. Compute loss: loss = criterion(predictions, targets)
  3. Backward pass: loss.backward()
  4. Update weights: optimizer.step()
  5. Zero gradients: optimizer.zero_grad()
- Practical implementation on real datasets

#### Key Concepts:
- **Backpropagation**: Efficient algorithm for computing gradients using chain rule
- **Loss Functions**: Task-specific objectives for optimization
- **Automatic Differentiation**: Modern frameworks handle gradient computation automatically
- **Mini-batch Training**: Balanced approach for efficient learning
- **Implementation Best Practices**: Initialization, hyperparameter tuning, debugging strategies

#### Learning Outcomes:
Students will master the backpropagation algorithm, implement training loops in PyTorch/TensorFlow, debug common training issues, and apply best practices for neural network implementation and optimization.

---

## 🛠️ Prerequisites

### Mathematical Background:
- **Linear Algebra**: Matrix multiplication, vector operations, transposes
- **Calculus**: Partial derivatives, chain rule, gradient computation
- **Probability**: Basic probability theory, probability distributions
- **Optimization**: Gradient descent, convex vs non-convex optimization

### Programming Skills:
- **Python**: Intermediate level (functions, classes, NumPy)
- **NumPy**: Array operations, broadcasting, vectorization
- **Matplotlib**: Basic plotting for visualization
- **Familiarity** with Jupyter notebooks

### Machine Learning Basics:
- Supervised learning concepts (training, validation, testing)
- Logistic regression and classification
- Loss functions and optimization
- Overfitting and regularization concepts

### Software Requirements:
- Python 3.8+
- PyTorch 2.0+ or TensorFlow 2.x
- NumPy 1.20+
- Matplotlib 3.5+
- Jupyter Lab or Google Colab
- GPU access recommended but not required (CPU sufficient for exercises)

---

## 💻 Hands-on Components

### Exercise 1: XOR Problem Implementation
**Objective**: Implement a 2-layer MLP to solve the XOR problem

**Requirements**:
- Input layer: 2 neurons (x₁, x₂)
- Hidden layer: 4 neurons with sigmoid/tanh activation
- Output layer: 1 neuron with sigmoid activation
- Train using mini-batch gradient descent

**Expected Outcome**: 
- Network correctly classifies all 4 XOR cases
- Decision boundary visualization shows non-linear separation
- Training loss decreases below 0.01 within 1000 epochs

**Skills Practiced**: Forward propagation, backpropagation, activation functions, loss computation

---

### Exercise 2: MNIST Digit Classification
**Objective**: Build and train an MLP for handwritten digit recognition

**Dataset**: MNIST (60,000 training images, 10,000 test images, 28×28 pixels)

**Architecture Specifications**:
- Input: 784 neurons (28×28 flattened)
- Hidden layer 1: 256 neurons, ReLU activation
- Hidden layer 2: 128 neurons, ReLU activation
- Output: 10 neurons (digits 0-9), Softmax activation

**Training Parameters**:
- Batch size: 64
- Learning rate: 0.001 (with Adam optimizer)
- Epochs: 10-20
- Loss: Cross-entropy

**Expected Performance**:
- Test accuracy: >95% (target: 97-98%)
- Training time: ~2-3 minutes on GPU, ~15-20 minutes on CPU
- Convergence: smooth loss curve without overfitting

**Skills Practiced**: Multi-class classification, larger networks, hyperparameter tuning, performance evaluation

---

### Exercise 3: Activation Function Comparison
**Objective**: Empirically compare different activation functions

**Task**: Train identical network architectures with different activations:
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- ELU

**Metrics to Compare**:
- Training speed (epochs to convergence)
- Final accuracy
- Gradient magnitudes (vanishing gradient analysis)
- Computation time per epoch

**Visualization Requirements**:
- Plot training/validation curves for all variants
- Compare activation distributions across layers
- Gradient flow analysis through network depth

**Expected Insights**:
- ReLU trains fastest and achieves best accuracy
- Sigmoid/Tanh suffer from vanishing gradients in deep networks
- Leaky ReLU prevents dead neurons compared to standard ReLU

**Skills Practiced**: Experimental design, comparative analysis, visualization, scientific reporting

---

### Exercise 4: Gradient Descent Optimizer Comparison
**Objective**: Compare SGD, SGD with momentum, Adam, and RMSprop

**Setup**:
- Fixed network architecture (same as MNIST exercise)
- Train each optimizer for 20 epochs
- Learning rates: 0.01 (SGD), 0.001 (Adam/RMSprop)

**Analysis**:
- Convergence speed
- Final performance
- Sensitivity to learning rate
- Memory requirements

**Deliverable**: Report with plots and recommendations for optimizer selection

---

### Exercise 5: Debugging Neural Networks
**Objective**: Practice identifying and fixing common training problems

**Scenarios Provided**:
1. **Scenario A**: Network with all zeros initialization (no learning)
2. **Scenario B**: Exploding gradients (loss → NaN)
3. **Scenario C**: Severe overfitting (train acc 99%, val acc 60%)
4. **Scenario D**: Dead ReLU neurons (50% neurons output zero)

**Task**: Diagnose each problem and implement solutions

**Skills Practiced**: Debugging, initialization strategies, regularization, gradient clipping

---

## 📖 Additional Resources

### Textbooks:
1. **Deep Learning** by Goodfellow, Bengio, and Courville (Chapter 6: Deep Feedforward Networks)
   - Comprehensive theoretical foundation
   - Available free online: www.deeplearningbook.org

2. **Neural Networks and Deep Learning** by Michael Nielsen
   - Intuitive explanations with interactive visualizations
   - Free online book: neuralnetworksanddeeplearning.com

3. **Pattern Recognition and Machine Learning** by Christopher Bishop (Chapter 5: Neural Networks)
   - Mathematical rigor and probabilistic perspective

### Online Courses:
- **Stanford CS231n**: Convolutional Neural Networks (Lecture 4: Backpropagation)
- **deeplearning.ai**: Neural Networks and Deep Learning (Week 3-4)
- **Fast.ai**: Practical Deep Learning (Lesson 3-4)

### Papers:
1. **"Learning representations by back-propagating errors"** - Rumelhart, Hinton, Williams (1986)
   - Original backpropagation paper

2. **"Understanding the difficulty of training deep feedforward neural networks"** - Glorot & Bengio (2010)
   - Xavier initialization

3. **"Delving Deep into Rectifiers"** - He et al. (2015)
   - He initialization for ReLU networks

### Interactive Tools:
- **TensorFlow Playground**: playground.tensorflow.org
  - Interactive visualization of neural networks
  - Experiment with architectures and hyperparameters in browser

- **Neural Network Playground**: cs.stanford.edu/people/karpathy/convnetjs
  - ConvNetJS demos by Andrej Karpathy

- **Distill.pub Articles**: distill.pub
  - Visual, interactive explanations of ML concepts

### Documentation:
- **PyTorch Tutorials**: pytorch.org/tutorials
- **TensorFlow Guides**: tensorflow.org/guide
- **NumPy Documentation**: numpy.org/doc

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Review slides sequentially** (Part 1 → Part 2 → Part 3)
2. **Work through mathematical derivations** with pen and paper
3. **Complete hands-on exercises** after each part
4. **Implement algorithms from scratch** before using frameworks
5. **Visualize concepts** using provided code examples

### For Classroom Teaching:
1. **Week 1**: Parts 1-2 (Motivation + Architecture) + Exercise 1
2. **Week 2**: Part 3 (Backpropagation) + Exercises 2-3
3. **Week 3**: Advanced topics + Exercises 4-5 + Final project

### Recommended Pace:
- **Slides 1-11**: 45-60 minutes (with discussion)
- **Slides 12-20**: 45-60 minutes (with coding demo)
- **Slides 21-30**: 60-75 minutes (with derivation walkthrough)
- **Hands-on exercises**: 2-4 hours outside class time

### Lab Session Structure:
1. **Setup (15 min)**: Environment configuration, import libraries
2. **Guided coding (45 min)**: Instructor demonstrates implementation
3. **Independent work (60 min)**: Students complete exercises
4. **Discussion (30 min)**: Review solutions, common mistakes, Q&A

---

## 📊 Assessment Suggestions

### Formative Assessment (During Learning):
- **Concept checks**: Short quiz after each part (5-10 questions)
- **Coding checkpoints**: Verify forward propagation implementation
- **Peer review**: Students explain concepts to each other
- **Office hours**: Individual gradient descent debugging sessions

### Summative Assessment (End of Module):

#### Assignment 1: Theory (30 points)
- Derive backpropagation for 3-layer network (10 pts)
- Compare activation functions theoretically (10 pts)
- Analyze gradient flow in deep networks (10 pts)

#### Assignment 2: Implementation (40 points)
- Implement MLP from scratch in NumPy (20 pts)
- Train on custom dataset and report results (15 pts)
- Code quality and documentation (5 pts)

#### Assignment 3: Experimentation (30 points)
- Design experiments comparing architectures (10 pts)
- Analyze results with proper visualizations (10 pts)
- Write technical report with conclusions (10 pts)

### Grading Rubric for Coding Assignments:

**Correctness (50%)**:
- Forward propagation works correctly
- Backpropagation computes accurate gradients
- Training loop converges successfully
- Final model achieves target performance

**Code Quality (25%)**:
- Proper variable naming and commenting
- Modular, reusable functions
- Efficient vectorized operations
- Error handling

**Analysis (25%)**:
- Clear visualizations (loss curves, accuracy plots)
- Thoughtful interpretation of results
- Comparison with baselines
- Discussion of limitations

---

## 📝 Notes for Implementation

### Technical Requirements:

#### Computational Resources:
- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB+ RAM, quad-core CPU, GPU with 4GB+ VRAM
- **Cloud Options**: Google Colab (free GPU), Kaggle Kernels, AWS SageMaker

#### Memory Considerations:
- **MNIST MLP**: ~2MB model size, ~500MB training memory
- **Batch size impact**: Memory usage ≈ batch_size × model_size × 10
- **Gradient checkpointing**: Can reduce memory by 30-50% at cost of 20% slower training

### Common Implementation Pitfalls:

1. **Broadcasting Errors**: 
   - Always check tensor shapes before operations
   - Use `.reshape()` explicitly rather than relying on automatic broadcasting

2. **In-place Operations**:
   - Avoid `x += y` in backward pass computation
   - Use `x = x + y` to maintain computation graph

3. **Learning Rate**:
   - Too high: Loss oscillates or diverges
   - Too low: Training too slow, may get stuck
   - Use learning rate finder or start with 1e-3

4. **Weight Initialization**:
   - Never use all zeros (neurons learn identical features)
   - Use Xavier for sigmoid/tanh: `std = sqrt(2/(fan_in + fan_out))`
   - Use He for ReLU: `std = sqrt(2/fan_in)`

5. **Numerical Stability**:
   - Use log-softmax instead of softmax for stability
   - Clip gradients: `torch.nn.utils.clip_grad_norm_()`
   - Monitor for NaN/Inf values in tensors

### Performance Optimization Tips:

- **Vectorization**: Replace loops with matrix operations (10-100× speedup)
- **GPU Utilization**: Keep batch size large enough to saturate GPU
- **Data Loading**: Use multi-threaded data loaders (num_workers=4)
- **Mixed Precision**: Use float16 for 2× speedup on modern GPUs
- **Profiling**: Use PyTorch profiler to identify bottlenecks

### Reproducibility:

```python
# Set random seeds for reproducibility
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 🙏 Credits

**Lecture Developed By:**  
Ho-min Park, Ph.D.  
Ghent University Global Campus  

**Contact:**
- Email: homin.park@ghent.ac.kr
- Email: powersimmani@gmail.com

**Acknowledgments:**
This lecture draws upon foundational work in neural networks and deep learning by researchers including Geoffrey Hinton, Yann LeCun, Yoshua Bengio, and many others in the machine learning community.

**License:**
These materials are provided for educational purposes. Please cite appropriately if using in your courses or research.

**Last Updated:** 2025

---

## 📌 Additional Notes

### Slide Navigation:
The complete slideshow (`lecture05_slideshow.html`) provides an integrated viewing experience with:
- Keyboard navigation (Arrow keys, Space)
- Progress tracking bar
- Slide counter (current/total)
- Part indicators for each section

### File Structure:
```
Lecture05/
├── Lecture05_01_Title.html
├── Lecture05_02_Contents.html
├── Part1/ (Slides 03-11)
├── Part2/ (Slides 12-20)
├── Part3/ (Slides 21-30)
├── Lecture05_31_Thank_you.html
└── lecture05_slideshow.html (integrated viewer)
```

### Customization:
Instructors can:
- Reorder slides for different teaching approaches
- Add institution-specific examples
- Extend hands-on exercises with domain-specific datasets
- Integrate with existing course management systems

---

**Ready to dive into Multi-layer Perceptrons? Start with Part 1 to understand the motivation, then progress through architecture design and backpropagation!** 🚀
