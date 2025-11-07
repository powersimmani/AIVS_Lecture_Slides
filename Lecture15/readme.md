# Lecture 15: Generative Models - GAN

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Institution:** Ghent University  
**Total Slides:** 33  
**Estimated Duration:** 3-4 hours  
**Difficulty Level:** Advanced Undergraduate / Graduate  
**Course:** Deep Learning

This lecture provides a comprehensive introduction to Generative Adversarial Networks (GANs), covering foundational theory, mathematical formulations, training algorithms, challenges, and modern improvements. The material balances theoretical rigor with practical implementation guidance.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand the theoretical foundations** of GANs including the minimax objective, value function analysis, and Nash equilibrium concepts
2. **Derive and explain** the optimal discriminator formula and global optimum conditions from first principles
3. **Implement** GAN training algorithms with proper architectural choices (DCGAN guidelines) and optimization strategies
4. **Identify and address** common failure modes including mode collapse, vanishing gradients, and training instability
5. **Apply** advanced GAN variants (cGAN, WGAN, DCGAN) to appropriate problem domains with understanding of their trade-offs

---

## 📚 Lecture Structure

### Part 1: Introduction & Motivation (Slides 3-6)

#### Topics Covered:
- **Review of Generative Models** (Slide 4)
  - Comparison of VAE, GAN, and autoregressive models
  - Advantages and disadvantages of each approach
  - When to use which generative model

- **Why GANs?** (Slide 5)
  - Historical context: Birth of GANs (NIPS 2014)
  - Ian Goodfellow et al.'s original paper
  - Impact: 50,000+ citations, revolutionary approach
  - Timeline: VAE Era (2013) → GAN Born (2014) → Rapid Growth (2015+)

- **Intuitive Understanding: Counterfeiter Analogy** (Slide 6)
  - Generator as counterfeiter creating fake money
  - Discriminator as police detecting fakes
  - Zero-sum game dynamics
  - Iterative improvement process leading to equilibrium

#### Key Concepts:
- GANs as game-theoretic framework for implicit density modeling
- Adversarial training as alternative to explicit likelihood maximization
- Competitive dynamics driving both networks toward excellence

#### Learning Outcomes:
- Articulate the motivation for GANs over previous generative approaches
- Explain the adversarial training process using intuitive analogies
- Understand GANs' historical significance in the AI landscape

---

### Part 2: Mathematical Foundations (Slides 7-12)

#### Topics Covered:
- **Probability Distribution Perspective** (Slide 8)
  - Real data distribution: p_data(x)
  - Generator distribution: p_g(x)
  - Objective: Minimize distance D(p_data || p_g)
  - Sampling process: z ~ p_z(z) → G(z) → x ~ p_g(x)
  - Visual convergence from initial to trained distribution

- **Mathematical Definition of GAN** (Slide 9)
  - Minimax objective: min_G max_D V(D, G)
  - Two-player game architecture
  - Value function: V(D,G) = 𝔼_x~p_data[log D(x)] + 𝔼_z~p_z[log(1 - D(G(z)))]
  - Real data term vs. Fake data term interpretation
  - Neural network formulations: G(z; θ_g) and D(x; θ_d)

- **Value Function Analysis** (Slide 10)
  - Discriminator maximizes: max_D V(D,G)
  - Generator minimizes: min_G V(D,G)
  - Theoretical insights: Cross-entropy loss interpretation
  - Jensen-Shannon divergence connection
  - Global optimality conditions: p_g = p_data, D*(x) = 1/2, V(D*, G*) = -log(4)

- **Deriving Optimal Discriminator** (Slide 11)
  - Step 1: Fix generator G, optimize D
  - Step 2: Apply calculus of variations (∂V/∂D(x) = 0)
  - Step 3: Solve for optimal D
  - Result: D*(x) = p_data(x) / (p_data(x) + p_g(x))
  - Decision boundary visualization at equilibrium

- **Global Optimum** (Slide 12)
  - Theorem: Global minimum achieved if and only if p_g = p_data
  - Convergence trajectory visualization
  - JS divergence connection: JS(p_data || p_g) = 0 ⟺ p_g = p_data
  - Practical challenges: mode collapse, training instability, non-convex optimization

#### Key Concepts:
- GANs minimize Jensen-Shannon divergence between distributions
- Optimal discriminator depends on density ratio of real vs. generated data
- Theoretical guarantees exist under ideal conditions (infinite capacity, optimal training)
- Cross-entropy loss naturally emerges from the value function formulation

#### Learning Outcomes:
- Derive the optimal discriminator formula from the value function
- Prove global optimum conditions using divergence theory
- Connect game-theoretic formulation to practical optimization objectives
- Understand the mathematical relationship between discriminator confidence and distribution distance

---

### Part 3: Training Algorithm (Slides 13-18)

#### Topics Covered:
- **Training Process Overview** (Slide 14)
  - Alternating optimization loop
  - Step 1: Update Discriminator (k steps) - train D to distinguish real from fake
  - Step 2: Update Generator (1 step) - train G to fool discriminator
  - Typical practice: k=1
  - Strategy: Mini-batch SGD with separate optimizers for G and D
  - Monitoring: Track loss curves, gradient norms, output quality

- **Detailed Algorithm** (Slide 15)
  - Sample minibatch of m noise samples {z^(1)...z^(m)}
  - Sample minibatch of m real samples {x^(1)...x^(m)}
  - Update D by ascending gradient: ∇_θd [1/m Σ (log D(x^(i)) + log(1-D(G(z^(i)))))]
  - Sample new minibatch of noise
  - Update G by descending gradient: ∇_θg [1/m Σ log(1-D(G(z^(i))))]
  - Repeat until convergence

- **Gradient Flow** (Slide 16)
  - Backpropagation through discriminator to generator
  - Gradient flow visualization
  - Importance of discriminator not being too strong
  - Balance between G and D capacity

- **Non-Saturating Loss** (Slide 17)
  - Problem: Original loss saturates when D is confident
  - Solution: Maximize log D(G(z)) instead of minimizing log(1-D(G(z)))
  - Provides stronger gradients early in training
  - Same fixed point, better optimization dynamics

- **Practical Tips** (Slide 18)
  - **Data & Activation:**
    - Normalize inputs to [-1, 1]
    - Use tanh activation for generator output
    - LeakyReLU in D, ReLU in G
  - **Normalization & Regularization:**
    - Batch normalization (except last G layer, first D layer)
    - Label smoothing: real=0.9, fake=0.1
    - Add noise to discriminator inputs
  - **Optimization:**
    - Use Adam optimizer with β₁=0.5, β₂=0.999
    - Monitor D loss - shouldn't go to 0
    - Learning rate scheduling
  - **Architecture Guide:**
    - Generator: Dense/Conv → ReLU → BatchNorm → ... → No BN (last) → tanh
    - Discriminator: No BN (first) → LeakyReLU → BatchNorm → ... → Sigmoid
  - **Interactive Practice:** GAN Lab (https://poloclub.github.io/ganlab/)

#### Key Concepts:
- Alternating optimization prevents one network from dominating
- Non-saturating loss provides better gradients for generator
- Architectural choices (BatchNorm, LeakyReLU) critical for stability
- Adam with modified momentum (β₁=0.5) works better than standard settings

#### Learning Outcomes:
- Implement the complete GAN training loop with proper gradient updates
- Apply non-saturating loss to prevent vanishing gradients
- Configure neural network architectures following DCGAN guidelines
- Debug training issues using loss curves and monitoring strategies

---

### Part 4: Key Challenges (Slides 19-24)

#### Topics Covered:
- **Mode Collapse** (Slide 20)
  - Definition: Generator produces limited variety of samples
  - Visual comparison:
    - Healthy: Diverse outputs (A, B, C, D, E, F)
    - Partial collapse: Limited modes (A, A, B, B, C, C)
    - Complete collapse: Single output (A, A, A, A, A, A)
  - Types:
    - Partial collapse: Missing some data modes
    - Complete collapse: Single output regardless of input
  - Solutions: Minibatch discrimination, unrolled GANs, feature matching, diversity penalties

- **Training Instability** (Slide 21)
  - Characteristics: Oscillating losses, no clear convergence
  - Loss curves show non-convergent patterns
  - Key issues:
    - No convergence criterion
    - G/D imbalance
    - Loss ≠ Quality correlation
    - No stopping criterion
  - Core problems:
    - Generator/Discriminator imbalance
    - Loss doesn't correlate with generation quality
  - Solutions: Spectral normalization, self-attention, progressive growing, LR scheduling

- **Evaluation Difficulties** (Slide 22)
  - Challenge: No single metric captures quality completely
  - Metrics comparison:
    - **Inception Score (IS):** High quality & diversity
    - **Fréchet Inception Distance (FID):** High quality & diversity (most popular)
    - **Precision:** High quality, low diversity
    - **Recall:** Low quality, high diversity
    - **Human Evaluation:** High quality & diversity (gold standard)
  - Challenges: Mode coverage assessment, computational cost of metrics

- **Vanishing Gradient Details** (Slide 23)
  - Problem cascade:
    1. Discriminator becomes too successful
    2. D correctly classifies fakes with high confidence: D(G(z)) → 0
    3. Original loss saturates: log(1 - D(G(z))) → log(1) = 0
    4. Gradients vanish: ∇_G → 0
  - When it occurs: D is too successful, no learning signal for G
  - Solutions: Non-saturating loss, WGAN, gradient penalty, balance G and D capacity

- **Common Failure Patterns** (Slide 24)
  - **Visual artifacts:**
    - Checkerboard artifacts from deconvolution
    - Unrealistic color distributions
    - Inconsistent texture transitions
  - **Anatomical errors:** Incorrect features in faces, hands, body proportions
  - **Training collapse:** Performance degrades after initial success
  - **Video issues:** Temporal inconsistency between frames
  - **Resolution limits:** Quality degrades without progressive techniques

#### Key Concepts:
- Mode collapse results from generator exploiting discriminator weaknesses
- Training instability stems from minimax game dynamics and lack of convergence guarantees
- Evaluation requires multiple metrics as no single metric is sufficient
- Vanishing gradients occur when discriminator becomes too strong

#### Learning Outcomes:
- Diagnose mode collapse from generated sample distributions
- Identify training instability from loss curves and implement stabilization techniques
- Select appropriate evaluation metrics for different GAN applications
- Explain the mathematical cause of vanishing gradients and apply corrective measures

---

### Part 5: Improvement Techniques (Slides 25-29)

#### Topics Covered:
- **DCGAN (2015)** (Slide 26)
  - Deep Convolutional GAN architecture breakthrough
  - **Key features:**
    - Replace fully connected layers with convolutions
    - No pooling layers - use strided convolutions
    - Batch normalization in both G and D
  - **Generator architecture:**
    - z (Noise) → Project & Reshape → Fractional-Strided Conv + BN + ReLU (×3) → Conv + Tanh → Image
  - **Discriminator architecture:**
    - Image → Strided Conv + LeakyReLU → Strided Conv + BN + LeakyReLU (×2) → Strided Conv + LeakyReLU → Sigmoid
  - **Benefits:** Stable training, higher resolutions, better image quality, foundation for future GANs
  - Original paper: Radford et al., 2015 (arXiv:1511.06434)

- **Conditional GAN (cGAN)** (Slide 27)
  - Core concept: Condition generation on additional information y
  - Architecture: Both G(z, y) and D(x, y) receive conditioning
  - Conditioning types: Class labels, text descriptions, images, structured information
  - **Applications:**
    - Class-conditional generation
    - Image-to-image translation (Pix2Pix)
    - Text-to-image synthesis
    - Super-resolution
    - Style transfer
  - Key benefit: Direct control over output characteristics
  - Original paper: Mirza & Osindero, 2014 (arXiv:1411.1784)

- **Wasserstein GAN (WGAN)** (Slide 28)
  - **Standard GAN vs WGAN comparison:**
    - Distance metric: JS divergence → Wasserstein distance (Earth Mover)
    - Output layer: Sigmoid (Discriminator) → Linear (Critic, no sigmoid)
    - Constraint: No explicit constraint → Lipschitz constraint (weight clipping)
    - Training: Unstable, oscillating → Stable, meaningful loss
  - **Improvements:**
    - Stable training with consistent convergence
    - Loss correlates with generation quality
    - Better mode coverage
  - **WGAN-GP enhancement:** Gradient penalty replaces weight clipping
  - Original papers: Arjovsky et al., 2017 (WGAN); Gulrajani et al., 2017 (WGAN-GP)

- **Other Improvements** (Slide 29)
  - **ProGAN (2017):** Progressive growing for high-resolution images
  - **SAGAN (2018):** Self-attention mechanisms for long-range dependencies
  - **CycleGAN (2017):** Unpaired image-to-image translation
  - **StyleGAN (2018):** Style-based architecture with unprecedented control
  - **BigGAN (2018):** Large-scale training with architectural tricks
  - **GauGAN (2019):** NVIDIA's semantic image synthesis
  - Note: Diffusion models emerged as powerful alternative approach

#### Key Concepts:
- DCGAN architectural guidelines became standard for convolutional GANs
- Conditional GANs enable controllable generation through auxiliary information
- WGAN addresses training instability through Wasserstein distance
- Progressive methods (ProGAN, StyleGAN) enable high-resolution generation

#### Learning Outcomes:
- Implement DCGAN architecture with proper convolutional layers and normalization
- Design conditional GANs for task-specific controllable generation
- Understand Wasserstein distance advantages over JS divergence
- Select appropriate GAN variant based on application requirements (resolution, controllability, stability)

---

### Part 6: Hands-on & Applications (Slide 30)

#### Topics Covered:
- Hands-on implementation guide
- Real-world applications across domains
- Best practices for deployment

#### Key Concepts:
- Practical implementation considerations
- Application-specific architectural choices
- Production deployment strategies

#### Learning Outcomes:
- Apply GAN knowledge to real-world projects
- Make informed decisions about architecture and training strategies

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Deep Learning Fundamentals:**
  - Neural network architectures (CNNs, fully connected networks)
  - Backpropagation and gradient descent
  - Optimization algorithms (SGD, Adam)
  - Regularization techniques (dropout, batch normalization)
  
- **Mathematics:**
  - Multivariable calculus (gradients, chain rule)
  - Probability theory (distributions, expectations, KL/JS divergence)
  - Linear algebra (matrix operations, vector spaces)
  - Basic game theory concepts (minimax, Nash equilibrium)

- **Machine Learning:**
  - Supervised learning fundamentals
  - Loss functions and their properties
  - Overfitting and generalization
  - Cross-validation techniques

### Recommended Background:
- Experience with PyTorch or TensorFlow
- Understanding of convolutional neural networks
- Familiarity with image processing concepts
- Prior exposure to autoencoders or VAEs (helpful but not required)

### Software Requirements:
- **Python 3.8+** with following libraries:
  - PyTorch >= 1.10 or TensorFlow >= 2.5
  - NumPy >= 1.19
  - Matplotlib for visualization
  - torchvision or tensorflow-datasets
  
- **Hardware:**
  - GPU strongly recommended (NVIDIA with CUDA support)
  - Minimum 8GB RAM, 16GB+ preferred
  - 50GB+ free disk space for datasets

- **Development Tools:**
  - Jupyter Notebook or JupyterLab
  - TensorBoard or Weights & Biases for experiment tracking
  - Git for version control

---

## 💻 Hands-on Components

### Interactive Lab Exercise: GAN Lab
- **Platform:** https://poloclub.github.io/ganlab/
- **Duration:** 30-45 minutes
- **Objectives:**
  - Visualize GAN training dynamics in real-time
  - Experiment with hyperparameters (learning rate, network architecture)
  - Observe mode collapse and training instability firsthand
  - Understand the relationship between discriminator strength and generator learning

### Implementation Project 1: MNIST GAN
**Difficulty:** Beginner  
**Duration:** 2-3 hours  
**Objectives:**
- Implement vanilla GAN from scratch on MNIST dataset
- Architecture:
  - Generator: 100D noise → 256 → 512 → 1024 → 784 (28×28 image)
  - Discriminator: 784 → 1024 → 512 → 256 → 1 (real/fake)
- Training considerations:
  - Batch size: 128
  - Learning rate: 0.0002
  - Optimizer: Adam (β₁=0.5, β₂=0.999)
  - Training epochs: 50-100
- Deliverables:
  - Training loss curves (G and D)
  - Generated digit samples at epochs 10, 25, 50, 100
  - Analysis of mode collapse or other failure modes

### Implementation Project 2: DCGAN on CIFAR-10
**Difficulty:** Intermediate  
**Duration:** 4-6 hours  
**Objectives:**
- Implement DCGAN following architectural guidelines from Slide 26
- Architecture specifications:
  - **Generator:**
    - Input: 100D noise vector
    - Project to 4×4×1024
    - Fractional-strided conv: 8×8×512 (stride 2)
    - Fractional-strided conv: 16×16×256 (stride 2)
    - Fractional-strided conv: 32×32×3 (stride 2)
    - All layers except output: BatchNorm + ReLU
    - Output: Tanh activation
  - **Discriminator:**
    - Input: 32×32×3 image
    - Strided conv: 16×16×64 (stride 2, no BN)
    - Strided conv: 8×8×128 (stride 2)
    - Strided conv: 4×4×256 (stride 2)
    - Flatten → Dense → Sigmoid
    - All conv layers except first: BatchNorm + LeakyReLU (α=0.2)
- Training hyperparameters:
  - Batch size: 64
  - Learning rate: 0.0002 for both G and D
  - β₁=0.5, β₂=0.999
  - Training epochs: 200
  - Label smoothing: real=0.9, fake=0.0
- Evaluation metrics:
  - Inception Score (IS)
  - Fréchet Inception Distance (FID)
  - Visual quality assessment
- Deliverables:
  - Trained model checkpoints
  - Generated image grid (8×8)
  - Training curves with IS/FID over epochs
  - Written report analyzing results and failure modes

### Implementation Project 3: Conditional GAN for Class-Specific Generation
**Difficulty:** Intermediate-Advanced  
**Duration:** 6-8 hours  
**Objectives:**
- Extend DCGAN to conditional setting using class labels
- Dataset: CIFAR-10 or Fashion-MNIST
- Modifications:
  - Embedding layer for class labels (dimension 50)
  - Concatenate embeddings with noise (Generator) and image features (Discriminator)
  - Train with paired (image, label) data
- Experiments:
  - Generate images conditioned on specific classes
  - Interpolate between different class conditions
  - Analyze class separation in latent space
- Deliverables:
  - Class-conditional image generation (10 classes × 10 samples each)
  - Interpolation visualizations
  - Comparison with unconditional DCGAN performance

### Implementation Project 4: WGAN with Gradient Penalty
**Difficulty:** Advanced  
**Duration:** 8-10 hours  
**Objectives:**
- Implement WGAN-GP as described in Slide 28
- Key implementation details:
  - Remove sigmoid from discriminator (now called "critic")
  - Implement Wasserstein loss: E[D(x_real)] - E[D(x_fake)]
  - Add gradient penalty term: λ * E[(||∇D(x_interpolated)||₂ - 1)²]
  - Use λ=10 for gradient penalty coefficient
  - Critic updates: 5 per generator update
  - No batch normalization in critic
- Comparative study:
  - Compare WGAN-GP vs standard GAN on same dataset
  - Analyze loss curves (should be more stable)
  - Compare mode coverage and sample diversity
  - Measure FID scores for both approaches
- Deliverables:
  - Side-by-side comparison of loss curves
  - Generated samples from both methods
  - Quantitative comparison (FID, IS)
  - Written analysis of stability improvements

### Optional Advanced Project: StyleGAN Implementation
**Difficulty:** Expert  
**Duration:** 15-20 hours  
**Objectives:**
- Implement simplified StyleGAN architecture
- Study style-based generator design
- Experiment with style mixing and truncation trick
- Requires strong PyTorch/TensorFlow skills and computational resources

---

## 📖 Additional Resources

### Original Papers:
1. **Goodfellow et al. (2014)** - "Generative Adversarial Networks"  
   arXiv:1406.2661 | NIPS 2014  
   The foundational GAN paper - essential reading

2. **Radford et al. (2015)** - "Unsupervised Representation Learning with Deep Convolutional GANs"  
   arXiv:1511.06434  
   DCGAN architecture guidelines that became standard practice

3. **Mirza & Osindero (2014)** - "Conditional Generative Adversarial Nets"  
   arXiv:1411.1784  
   Introduces conditioning for controllable generation

4. **Arjovsky et al. (2017)** - "Wasserstein GAN"  
   arXiv:1701.07875 | ICML 2017  
   Addresses training stability through Wasserstein distance

5. **Gulrajani et al. (2017)** - "Improved Training of Wasserstein GANs"  
   arXiv:1704.00028 | NIPS 2017  
   Gradient penalty for enforcing Lipschitz constraint

### Textbooks and Tutorials:
- **Deep Learning** by Goodfellow, Bengio, Courville (2016)  
  Chapter 20: Deep Generative Models
  
- **GANs in Action** by Jakub Langr & Vladimir Bok (2019)  
  Practical implementation guide with code examples

- **CS231n: Convolutional Neural Networks for Visual Recognition**  
  Stanford University - Lecture 13 on Generative Models  
  http://cs231n.stanford.edu/

- **Spinning Up in Deep RL** by OpenAI  
  Has excellent GAN tutorials and implementations

### Code Repositories:
- **PyTorch GAN Zoo:** https://github.com/facebookresearch/pytorch_GAN_zoo  
  High-quality implementations of major GAN variants

- **TensorFlow GAN (TF-GAN):** https://github.com/tensorflow/gan  
  Official TensorFlow library for GAN research

- **Keras-GAN:** https://github.com/eriklindernoren/Keras-GAN  
  Collection of 30+ GAN implementations in Keras

### Interactive Tools:
- **GAN Lab:** https://poloclub.github.io/ganlab/  
  Interactive visualization of GAN training (mentioned in lecture)

- **TensorFlow Playground:** https://playground.tensorflow.org/  
  While not GAN-specific, helps build intuition for neural networks

### Survey Papers:
- **Goodfellow (2016)** - "NIPS 2016 Tutorial: Generative Adversarial Networks"  
  arXiv:1701.00160  
  Comprehensive tutorial by the inventor of GANs

- **Creswell et al. (2018)** - "Generative Adversarial Networks: An Overview"  
  arXiv:1710.07035  
  Broad survey of GAN variants and applications

### Video Lectures:
- **Ian Goodfellow's NIPS 2016 Tutorial** (2.5 hours)  
  Available on YouTube - definitive introduction

- **Stanford CS236: Deep Generative Models**  
  https://deepgenerativemodels.github.io/  
  Full course on generative models including GANs

- **Two Minute Papers** - GAN series on YouTube  
  Visual explanations of latest GAN research

### Evaluation Metrics:
- **Inception Score:** Salimans et al. (2016) - "Improved Techniques for Training GANs"
- **Fréchet Inception Distance:** Heusel et al. (2017) - "GANs Trained by a Two Time-Scale Update Rule"
- **Precision and Recall for GANs:** Sajjadi et al. (2018)

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Week 1: Theory Foundation (Parts 1-2)**
   - Review slides 1-12 carefully
   - Work through mathematical derivations on paper
   - Prove the optimal discriminator formula yourself
   - Study time: 4-6 hours

2. **Week 2: Training & Implementation (Part 3)**
   - Study training algorithm details (slides 13-18)
   - Complete GAN Lab interactive exercise
   - Begin MNIST GAN implementation project
   - Study time: 8-10 hours (including coding)

3. **Week 3: Challenges & Solutions (Parts 4-5)**
   - Understand failure modes (slides 19-24)
   - Study improvement techniques (slides 25-29)
   - Implement DCGAN on CIFAR-10
   - Study time: 10-12 hours (including coding)

4. **Week 4: Advanced Topics & Projects**
   - Implement cGAN or WGAN-GP
   - Compare different variants empirically
   - Write technical report on findings
   - Study time: 12-15 hours

### For Classroom Instruction:

**Session 1 (90 minutes): Introduction & Theory**
- Cover Parts 1-2 (slides 1-12)
- Focus on intuition before mathematics
- Interactive discussion: Compare GANs with other generative models
- Assign reading: Original GAN paper (Goodfellow et al., 2014)

**Session 2 (90 minutes): Training Algorithm**
- Cover Part 3 (slides 13-18)
- Live coding demo: Simple GAN on MNIST
- In-class activity: GAN Lab exploration (30 minutes)
- Assign: MNIST GAN implementation project (due in 1 week)

**Session 3 (90 minutes): Challenges**
- Cover Part 4 (slides 19-24)
- Case study: Analyzing real GAN failures
- Group discussion: Solutions to mode collapse
- Lab session: Debug provided broken GAN code

**Session 4 (90 minutes): Modern GANs**
- Cover Part 5 (slides 25-29)
- Guest lecture (if possible): Industry application of GANs
- Project presentations: Students share MNIST GAN results
- Assign: DCGAN or WGAN implementation (final project)

### Suggested Lab Schedule:
- **Lab 1 (Week 2):** GAN Lab + Basic implementation setup
- **Lab 2 (Week 3):** MNIST GAN debugging and improvement
- **Lab 3 (Week 4):** DCGAN implementation and training
- **Lab 4 (Week 5):** Advanced project work and presentations

### Recommended Study Path by Background:

**For ML/DL Beginners:**
- Focus heavily on Part 1-2 for conceptual understanding
- Use GAN Lab extensively before coding
- Start with 2D toy datasets before images
- Implement vanilla GAN thoroughly before variants

**For Intermediate Students:**
- Skim Part 1, focus on mathematical derivations in Part 2
- Jump quickly to DCGAN implementation
- Experiment with hyperparameters systematically
- Compare multiple GAN variants quantitatively

**For Advanced Researchers:**
- Focus on Parts 4-5 (challenges and solutions)
- Implement WGAN-GP or StyleGAN
- Reproduce results from recent papers
- Consider research project on GAN improvements

---

## 📊 Assessment Suggestions

### Formative Assessment (During Learning):

**Concept Check Questions (After Each Part):**

*Part 1-2: Theory*
1. Derive the optimal discriminator D*(x) from the value function V(D,G)
2. Prove that the global optimum occurs when p_g = p_data
3. Explain why JS divergence is minimized in the minimax game
4. What is the equilibrium value of the discriminator at optimum?

*Part 3: Training*
5. Why is alternating optimization used instead of joint optimization?
6. Explain the vanishing gradient problem in the original GAN formulation
7. What is the purpose of the non-saturating loss?
8. Why is batch normalization excluded from certain layers in DCGAN?

*Part 4: Challenges*
9. Describe three different failure modes and their causes
10. How can you detect mode collapse during training?
11. Why is FID preferred over Inception Score for evaluation?

*Part 5: Improvements*
12. Compare the loss functions of vanilla GAN vs WGAN
13. How does conditioning enable controllable generation in cGAN?
14. What problem does progressive growing solve?

### Summative Assessment Options:

**Option 1: Implementation Project (40%)**
- Complete one of the three main implementation projects
- Submit code, trained models, and generated samples
- Grading rubric:
  - Code quality and documentation: 25%
  - Training convergence and stability: 25%
  - Generated sample quality: 25%
  - Analysis and discussion: 25%

**Option 2: Written Exam (30%)**
- Theoretical questions on minimax formulation
- Derivations (optimal discriminator, gradient calculations)
- Conceptual questions on training dynamics
- Analysis of provided loss curves and generated samples

**Option 3: Comparative Study Report (30%)**
- Implement two GAN variants (e.g., DCGAN vs WGAN-GP)
- Compare on same dataset with consistent evaluation
- Report structure:
  - Introduction and motivation (10%)
  - Methodology and implementation details (20%)
  - Experimental results with visualizations (30%)
  - Quantitative analysis (FID, IS) (20%)
  - Discussion and conclusions (20%)
- Length: 6-8 pages, conference paper format

### Grading Rubric for Implementation Projects:

**Code Quality (25 points):**
- [ ] Proper project structure and organization (5 pts)
- [ ] Clear variable naming and code comments (5 pts)
- [ ] Implementation follows best practices (5 pts)
- [ ] Reproducibility (random seeds, saved configs) (5 pts)
- [ ] Git version control with meaningful commits (5 pts)

**Technical Implementation (35 points):**
- [ ] Generator architecture correctly implemented (10 pts)
- [ ] Discriminator architecture correctly implemented (10 pts)
- [ ] Training loop with proper gradient updates (10 pts)
- [ ] Loss functions and optimization correctly configured (5 pts)

**Experimental Results (25 points):**
- [ ] Model successfully trains without crashing (5 pts)
- [ ] Generated samples show learning progress (10 pts)
- [ ] Quantitative metrics reported (FID/IS) (5 pts)
- [ ] Training curves visualized and analyzed (5 pts)

**Analysis and Documentation (15 points):**
- [ ] README with clear setup instructions (3 pts)
- [ ] Discussion of hyperparameter choices (4 pts)
- [ ] Analysis of failure modes encountered (4 pts)
- [ ] Comparison with baseline or literature results (4 pts)

### Project Presentation (Optional, 10% bonus):
- 10-minute presentation + 5-minute Q&A
- Demonstrate trained model generating samples
- Discuss technical challenges and solutions
- Compare results with published benchmarks

---

## 📝 Notes for Implementation

### Technical Requirements:

**Minimum Hardware:**
- **CPU:** 4+ cores (8+ recommended)
- **RAM:** 16GB (32GB for larger models)
- **GPU:** NVIDIA GPU with 6GB+ VRAM
  - Recommended: RTX 3060 (12GB), RTX 3070 (8GB), or better
  - Budget option: Google Colab Pro (~$10/month) with T4/V100 GPUs
  - Alternative: Kaggle Kernels (free GPU access)
- **Storage:** 50GB free space (for datasets and checkpoints)

**Software Environment Setup:**

```bash
# Create conda environment
conda create -n gan-lecture python=3.9
conda activate gan-lecture

# Install PyTorch (CUDA 11.8 version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional packages
pip install numpy matplotlib tensorboard jupyter
pip install scipy scikit-learn pillow
pip install pytorch-fid  # For FID calculation
pip install lpips  # For perceptual loss

# Install development tools
pip install black flake8 pytest
```

**Dataset Downloads:**
- MNIST: Automatically downloaded via torchvision
- CIFAR-10: ~170MB, automatically downloaded
- CelebA: ~1.4GB, requires manual download from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- ImageNet (optional): ~150GB, requires significant storage

### Common Implementation Pitfalls:

**1. Discriminator Too Strong:**
- **Symptom:** Generator loss increases, discriminator loss near 0
- **Solution:** 
  - Reduce discriminator learning rate
  - Increase generator updates per discriminator update
  - Add noise to discriminator inputs
  - Use label smoothing

**2. Mode Collapse:**
- **Symptom:** All generated samples look similar
- **Solution:**
  - Implement minibatch discrimination
  - Try different loss formulations (WGAN)
  - Increase model capacity
  - Modify training dynamics (unrolled GANs)

**3. Non-Convergence:**
- **Symptom:** Losses oscillate without improvement
- **Solution:**
  - Reduce learning rates for both networks
  - Ensure proper gradient flow (check architecture)
  - Try different optimizers (RMSprop, SGD with momentum)
  - Verify batch normalization usage

**4. Poor Sample Quality:**
- **Symptom:** Blurry or unrealistic outputs
- **Solution:**
  - Increase model capacity (deeper/wider networks)
  - Train longer (more epochs)
  - Improve discriminator architecture
  - Use perceptual losses

**5. Training Instability:**
- **Symptom:** Sudden collapse after initial success
- **Solution:**
  - Save checkpoints frequently
  - Implement learning rate scheduling
  - Use spectral normalization
  - Monitor gradient norms

### Debugging Checklist:

- [ ] Verify data normalization ([-1, 1] for tanh output)
- [ ] Check gradient flow (print gradient norms)
- [ ] Visualize generated samples every N iterations
- [ ] Plot loss curves in real-time (TensorBoard)
- [ ] Verify discriminator not at 0 or 1 constantly
- [ ] Test on toy dataset first (2D Gaussians)
- [ ] Compare against reference implementation
- [ ] Check for NaN in losses or gradients
- [ ] Verify batch sizes are consistent
- [ ] Ensure proper train/eval mode switching

### Performance Optimization Tips:

**Training Speed:**
- Use mixed precision training (torch.cuda.amp) for 2-3x speedup
- Increase batch size up to GPU memory limit
- Use DataLoader with num_workers=4-8
- Pin memory in DataLoader (pin_memory=True)
- Use torch.compile() in PyTorch 2.0+

**Memory Optimization:**
- Use gradient checkpointing for large models
- Clear cache periodically: torch.cuda.empty_cache()
- Reduce image resolution during initial experiments
- Use smaller batch sizes with gradient accumulation

**Experiment Tracking:**
```python
# Example using TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/gan_experiment1')

# Log scalars
writer.add_scalar('Loss/Generator', g_loss, epoch)
writer.add_scalar('Loss/Discriminator', d_loss, epoch)

# Log images
writer.add_images('Generated', generated_images, epoch)

# Log model graph
writer.add_graph(generator, sample_noise)

writer.close()
```

### Reproducibility Best Practices:

```python
# Set random seeds for reproducibility
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Recommended Hyperparameters (Starting Points):

**DCGAN on CIFAR-10:**
```python
# Generator
latent_dim = 100
g_channels = [1024, 512, 256, 128]
g_lr = 0.0002

# Discriminator  
d_channels = [128, 256, 512, 1024]
d_lr = 0.0002

# Training
batch_size = 64
num_epochs = 200
beta1 = 0.5  # Adam parameter
beta2 = 0.999

# Regularization
label_smoothing_real = 0.9
label_smoothing_fake = 0.0
```

**WGAN-GP on CelebA:**
```python
# Critic (Discriminator)
c_lr = 0.0001
c_updates = 5  # per generator update

# Generator
g_lr = 0.0001

# Gradient Penalty
lambda_gp = 10

# Training
batch_size = 64
num_epochs = 100

# No batch normalization in critic
# RMSprop optimizer instead of Adam
```

---

## 🙏 Credits

**Lecturer:** Ho-min Park  
**Affiliation:** Ghent University  
**Contact:** 
- Email: homin.park@ghent.ac.kr
- Email: powersimmani@gmail.com

**Course:** Deep Learning  
**Lecture Number:** 15  
**Topic:** Generative Models - Generative Adversarial Networks (GANs)

**Acknowledgments:**
- Original GAN paper: Ian Goodfellow et al. (2014)
- DCGAN architecture: Alec Radford et al. (2015)
- WGAN formulation: Martin Arjovsky et al. (2017)
- Course materials inspired by Stanford CS231n and CS236

**License:** These educational materials are provided for academic use. Please cite appropriately when using for research or teaching purposes.

**Slide Count:** 33 slides covering 6 major parts

**Last Updated:** 2025

---

## 📌 Quick Reference

### Key Equations:

**Minimax Objective:**
```
min_G max_D V(D, G)
```

**Value Function:**
```
V(D,G) = 𝔼_x~p_data[log D(x)] + 𝔼_z~p_z[log(1 - D(G(z)))]
```

**Optimal Discriminator:**
```
D*(x) = p_data(x) / (p_data(x) + p_g(x))
```

**Global Optimum:**
```
p_g = p_data  ⟺  D*(x) = 1/2  ⟺  V(D*, G*) = -log(4)
```

**Non-Saturating Loss (Generator):**
```
max_G 𝔼_z~p_z[log D(G(z))]
```

### Recommended Reading Order:
1. Slides 1-6: Build intuition
2. Slides 7-12: Understand mathematics
3. Original paper: Goodfellow et al. (2014)
4. Slides 13-18: Learn training procedures
5. Implement MNIST GAN
6. Slides 19-24: Study failure modes
7. Slides 25-29: Modern improvements
8. DCGAN paper: Radford et al. (2015)
9. Implement DCGAN on CIFAR-10
10. Explore advanced variants (WGAN, StyleGAN)

### Essential Resources:
- 🔗 GAN Lab: https://poloclub.github.io/ganlab/
- 📄 Original Paper: https://arxiv.org/abs/1406.2661
- 💻 PyTorch Examples: https://github.com/pytorch/examples/tree/master/dcgan
- 📚 GAN Zoo: https://github.com/hindupuravinash/the-gan-zoo (500+ GAN variants)

---

**Happy Learning! 🚀**

For questions or clarifications, please contact the instructor at homin.park@ghent.ac.kr