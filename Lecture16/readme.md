# Lecture 16: Generative Models - Diffusion

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr, powersimmani@gmail.com  
**Total Slides:** 34  
**Difficulty Level:** Advanced (Graduate-level)  
**Estimated Duration:** 3-4 hours  
**Format:** Lecture slides with theoretical foundations and practical applications

This comprehensive lecture provides an in-depth exploration of diffusion models, covering everything from fundamental concepts to state-of-the-art applications. The material is designed for graduate students and researchers in computer vision, machine learning, and artificial intelligence.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand the mathematical foundations** of diffusion models, including the forward and reverse processes, Markov chains, and score-based formulations
2. **Implement core components** of diffusion models including noise scheduling, U-Net architectures, and attention mechanisms
3. **Apply advanced techniques** such as classifier-free guidance, DDIM sampling, and latent diffusion for efficient generation
4. **Compare and contrast** diffusion models with other generative approaches (GANs, VAEs, Flow-based models)
5. **Deploy diffusion models** for real-world applications including text-to-image generation, image editing, and conditional generation

---

## 📚 Lecture Structure

### Part 1/7: Introduction & Motivation

**Slides:** 3-6 (4 slides)

#### Topics Covered:

**Slide 4: Evolution of Generative Models**
- Timeline of generative model development (2013-2023)
- VAEs (2013): Probabilistic encoding with KL divergence
- GANs (2014): Adversarial training approach
- Diffusion Models (2015-2020): Gradual noise addition & removal
- Flow-based Models (2018): Invertible transformations
- Score-based Models (2020+): Score matching connection
- Recent successes: DALL-E 2, Stable Diffusion, Midjourney (2022-2023)
- Key advantages: High quality, stable training, mode coverage

**Slide 5: Intuitive Understanding**
- Analogy: Sculptor gradually refining marble from rough block to statue
- Forward Process: Original data → +Gaussian Noise → Intermediate States → Pure Noise
- Reverse Process: Pure Noise → Neural Network → Iterative Denoising → Generated Data
- Markov Chain structure connecting both processes
- Key insights: Easier to learn small denoising steps, each step depends only on previous, more controllable than direct generation

**Slide 6: Comparison with GANs**
- **Training Stability:** Diffusion (✓) more stable, no adversarial collapse vs GANs (unstable, mode collapse risk)
- **Mode Coverage:** Diffusion (✓) captures all modes effectively vs GANs (may miss modes)
- **Sample Quality:** Diffusion (✓) comparable, often more detailed vs GANs (high quality, less detail)
- **Generation Speed:** GANs (✓) faster (1 step) vs Diffusion (slower, 100-1000 steps)
- **Training Complexity:** Diffusion (✓) simpler loss, no discriminator vs GANs (complex discriminator tuning)
- **Controllability:** Diffusion (✓) easier to guide and condition vs GANs (challenging)
- **Likelihood Evaluation:** Diffusion (✓) tractable likelihood vs GANs (no direct estimation)

#### Key Concepts:
- Evolution from VAEs/GANs to modern diffusion models
- Intuitive understanding through sculptor analogy
- Markov chain formulation of diffusion processes
- Systematic advantages of diffusion over GANs
- State-of-the-art applications (DALL-E 2, Stable Diffusion, Midjourney)

#### Learning Outcomes:
- Students can explain the historical context and evolution of generative models
- Students can articulate the intuitive process of diffusion-based generation
- Students can compare diffusion models with GANs across multiple dimensions
- Students understand why diffusion models have become dominant in recent years

---

### Part 2/7: Forward Process

**Slides:** 7-12 (6 slides)

#### Topics Covered:

**Slide 8: Forward Process Definition**
- Goal: Gradually transform data into pure Gaussian noise
- Markov Chain formulation: q(x_t | x_{t-1})
- Process flow: x₀ (Original Data) → [t = 1,2,...,T] → x_T ~ N(0,I)
- Gaussian transition (distribution form): q(x_t|x_{t-1}) = N(x_t; √α_t·x_{t-1}, (1-α_t)I) where α_t = 1-β_t
- Sampling form (reparameterization): x_t = √α_t·x_{t-1} + √(1-α_t)·ε, where ε ~ N(0,I)
- Noise schedule β_t controls amount of noise at step t
- Fixed schedule: β_t increases from 0.0001 to 0.02
- Time steps: Usually T = 1000 steps
- End result: x_T ~ N(0,I) is pure Gaussian noise

**Slide 9: Cumulative Effect**
- Problem: Iterative approach (x₀ → x₁ → x₂ → ... → x_t) is slow and requires all intermediate steps
- Solution: Direct sampling (x₀ ⟹ x_t) with one-step computation
- Alpha definitions: α_t = 1-β_t
- Cumulative product: ᾱ_t = ∏ᵢ₌₁ᵗ αᵢ
- Direct sampling formula: x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε
- Benefits: Efficient training, signal-to-noise control, reparameterization trick

**Slide 10: Mathematical Properties**
- **Mean Preservation:** E[x_t] = √ᾱ_t·x₀ (expected value scaled by √ᾱ_t)
- **Variance Growth:** Var[x_t] = (1-ᾱ_t)·I (variance increases as noise accumulates)
- **Gaussian Distribution:** q(x_t|x₀) = N(x_t; √ᾱ_t·x₀, (1-ᾱ_t)·I) - complete distributional characterization
- **Posterior Distribution:** q(x_{t-1}|x_t,x₀) also Gaussian with closed-form solution
- **Tractable Likelihood:** Can compute exact log-likelihood for model evaluation
- **Reversibility:** Theoretical foundation enables the reverse generative process
- **Noise Independence:** Noise at different timesteps is independent, simplifying analysis
- Key insight: All properties maintain Gaussian structure, enabling tractable inference and generation

**Slide 11: Forward Process Visualization**
- Visual timeline from t=0 to t=1000 showing gradual corruption
- **t=0:** Original clear image (🖼️)
- **t=250:** Slight noise, details visible
- **t=500:** Moderate noise, structure recognizable (🔳)
- **t=750:** Heavy noise, vague outline (▢)
- **t=1000:** Pure noise, no information (❄️)
- Demonstrates information loss: Gradual and controlled degradation over timesteps
- Validates noise schedule effectiveness through visual inspection

**Slide 12: Why is this Process Necessary?**
- **Tractable Posterior:** Known distribution q(x_{t-1}|x_t,x₀) enables efficient training
- **Curriculum Learning:** Start easy (small noise), gradually increase difficulty
- **Stable Gradients:** Small steps ensure smooth gradient flow during training
- **Mode Coverage:** Noise prevents collapse to single mode
- **Theoretical Foundation:** Enables variational bound derivation for optimization
- **Flexible Design:** Can adjust β_t schedule for different data types
- **Reversibility:** Ensures reverse generative process is well-defined
- Key insight: The forward process creates a tractable learning framework that makes the reverse process trainable

#### Key Concepts:
- Forward diffusion as a Markov chain with Gaussian transitions
- Reparameterization trick enabling direct sampling at any timestep
- Closed-form solutions for cumulative noise effects (ᾱ_t formulation)
- Mathematical properties maintaining Gaussian structure throughout
- Noise schedule β_t controlling the rate of corruption

#### Learning Outcomes:
- Students can derive the forward process equations from first principles
- Students can implement direct sampling using the cumulative α formulation
- Students understand the mathematical properties that enable tractable training
- Students can visualize and interpret the gradual corruption process
- Students can explain why the forward process is necessary for tractable learning

---

### Part 3/7: Reverse Process

**Slides:** 13-18 (6 slides)

#### Topics Covered:

**Slide 14: Reverse Process Goal**
- Objective: Learn to reverse the forward diffusion process
- Markov chain formulation: p_θ(x_{t-1}|x_t)
- Start point: x_T ~ N(0,I) (pure noise)
- Process: Iterative denoising through neural network θ
- End point: x₀ ~ p_data(x) (data sample)
- Parameterized by neural network
- Models conditional Gaussian transitions
- Direction: Remove added noise step-by-step
- Result: Iteratively denoise from noise to create new high-quality samples

**Slide 15: Neural Network Parameterization**
- **Input 1:** Noisy image x_t
- **Input 2:** Timestep t
- **Network Core:** U-Net θ with attention layers and time encoding
- **Output:** ε_θ(x_t,t) or predicted x₀
- Mean prediction formula: μ_θ(x_t,t) = 1/√α_t · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t,t))
- Variance: σ_t² (fixed or learned)
- Reverse transition distribution: p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²I)

**Slide 16: Deep Dive into Score Function**
- Score definition: ∇_x log p(x) indicates direction of density increase
- **Connection:** Denoising ≈ Score matching
- **Tweedie's Formula:** Score relates to optimal denoising estimation
- **Score Estimation:** ε_θ(x_t,t) ≈ -√(1-ᾱ_t) · ∇_{x_t} log p(x_t)
- **Langevin Dynamics:** Reverse process as Langevin sampling from score function
- **Continuous Limit:** Score-based SDEs when T → ∞
- **Unified View:** Diffusion and score-based models are equivalent

**Slide 17: Denoising Objective Function**
- **Simplified Loss (Preferred):** L_simple = E_{t,x₀,ε}[||ε - ε_θ(x_t,t)||²] - works better in practice
- **Variational Lower Bound:** L_vlb = E[log p(x₀) - log p_θ(x₀)] - theoretical foundation
- Practical training process:
  1. Sample timestep: t ~ Uniform{1,...,T}
  2. Sample noise: ε ~ N(0,I)
  3. Create noisy sample: x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε
  4. Gradient descent: Update θ to minimize loss
- Optimization objective: minimize ||ε - ε_θ(x_t,t)||²

**Slide 18: Training Algorithm**
- Complete algorithm implementation with pseudocode structure
- Iterative training over dataset with random timestep sampling
- Loss computation using predicted vs actual noise
- Backpropagation and parameter updates
- Typically trained for hundreds of thousands to millions of iterations

#### Key Concepts:
- Reverse process as learned Markov chain p_θ(x_{t-1}|x_t)
- U-Net architecture with time embedding for noise prediction
- Score function interpretation connecting denoising and score matching
- Simplified denoising objective (noise prediction) as practical training goal
- Equivalence between diffusion models and score-based generative models

#### Learning Outcomes:
- Students can implement the reverse process neural network architecture
- Students understand the connection between denoising and score estimation
- Students can derive and implement the training objective
- Students can explain the equivalence between diffusion and score-based approaches
- Students can write training loops for diffusion models from scratch

---

### Part 4/7: Sampling

**Slides:** 19-23 (5 slides)

#### Topics Covered:

**Slide 20: DDPM Sampling**
- **Step 1:** Initialize from pure noise: x_T ~ N(0,I)
- **Step 2:** Iterative denoising for t = T, T-1, ..., 1:
  - ① Predict noise: ε̂ = ε_θ(x_t,t)
  - ② Compute mean: μ_θ(x_t,t)
  - ③ Sample noise: z ~ N(0,I) if t>1, else z=0
  - ④ Update state: x_{t-1} = μ_θ(x_t,t) + σ_t·z
- **Step 3:** Final output: x₀ is the generated sample
- Time complexity: O(T) forward passes, typically T = 1000

**Slide 21: DDIM - Fast Sampling**
- **Problem:** DDPM requires many steps (1000) for good quality
- **Solution:** DDIM uses deterministic, non-Markovian process
- DDPM characteristics: All timesteps (1000), stochastic process, random noise added, slower generation
- DDIM characteristics: Subset only (e.g., 50 steps), deterministic (σ_t=0), skip connections τ=[1,20,40,...,1000], comparable quality
- **20x speedup** compared to DDPM
- DDIM update formula: x_{t-1} = √ᾱ_{t-1} · x̂₀ + √(1-ᾱ_{t-1}) · ε̂_t
- Trade-off: Can interpolate between DDPM and DDIM for speed-quality balance

**Slide 22: Conditional Generation**
- Goal: Generate samples conditioned on class, text, or image
- **Classifier Guidance Method:**
  - Base score (s) + Classifier gradient (w·∇_x log p(y|x_t)) = Modified score (s̃)
  - Modified score formula: s̃ = s + w·∇_x log p(y|x_t)
  - Guidance scale w controls conditioning strength
- Applications: Class-conditional ImageNet generation
- Limitations: Requires trained classifier, less flexible, need separate classifier for each type
- Key insight: Use classifier gradients to guide generation toward desired conditions

**Slide 23: Classifier-Free Guidance**
- **Innovation:** No separate classifier needed
- Training strategy: Randomly drop condition c with probability p (e.g., 10%)
- Guided prediction formula: ε̃ = ε_θ(x_t,t,∅) + w·(ε_θ(x_t,t,c) - ε_θ(x_t,t,∅))
- Components: Unconditional prediction + Weight × Difference
- **Guidance scale:** w > 1 increases strength
- **Trade-off:** Higher w = better alignment with condition, less diversity
- **Flexibility:** Single model handles multiple condition types
- State-of-the-art: Used in DALL-E 2, Stable Diffusion, Imagen

#### Key Concepts:
- DDPM sampling as full iterative denoising (1000 steps)
- DDIM acceleration through deterministic sampling and timestep skipping
- Classifier guidance using external classifier gradients
- Classifier-free guidance eliminating need for separate classifiers
- Trade-offs between speed, quality, and conditioning strength

#### Learning Outcomes:
- Students can implement both DDPM and DDIM sampling algorithms
- Students understand the mathematical differences enabling DDIM speedup
- Students can apply conditional generation techniques for controlled synthesis
- Students can implement classifier-free guidance for flexible conditioning
- Students can balance generation speed and quality through hyperparameter tuning

---

### Part 5/7: Architecture

**Slides:** 24-27 (4 slides)

#### Topics Covered:

**Slide 25: U-Net Structure**
- **Encoder Path (Downsampling):**
  - Level 1: 256×256
  - Level 2: 128×128
  - Level 3: 64×64
  - Level 4: 32×32
- **Bottleneck:** 16×16 with highest number of channels
- **Decoder Path (Upsampling):**
  - Level 4: 32×32
  - Level 3: 64×64
  - Level 2: 128×128
  - Level 1: 256×256
- **Skip Connections:** Concatenate encoder features to decoder at each level
- **ResNet Blocks:** Two convolutions + residual connections
- **Normalization:** Group normalization
- **Time Embedding:** Adaptive normalization with timestep information
- Architecture designed for preserving both local and global information

**Slide 26: Attention Mechanism**
- **Self-Attention:**
  - Models long-range dependencies in images
  - Q, K, V all from same source
  - Applied at 16×16, 32×32 resolutions
  - Captures spatial relationships within the image
- **Cross-Attention:**
  - For text conditioning
  - Text embeddings provide Q, image features provide K, V
  - Text→Image attention enables controllable generation
  - Essential for text-to-image models
- **Key Properties:**
  - Multi-head attention for capturing multiple patterns
  - Spatial attention across different locations
  - Complexity: O(n²) in sequence length
  - Standard transformer mechanism
  - Applied at low resolutions to manage computational cost
  - Crucial for long-range dependencies
- Impact: Crucial for coherent, high-quality generation

**Slide 27: Condition Injection Methods**
- **Time Embedding:** Sinusoidal positional encoding + MLP (primary method)
- **Class Embedding:** Learned embedding for each class
- **Text Conditioning:** CLIP or T5 text encoder → cross-attention (highlighted)
- **Adaptive Group Norm:** Scale and shift from condition embedding
- **FiLM Layers:** Feature-wise Linear Modulation
- **Concatenation:** Directly concat condition to input channels
- **Cross-Attention:** Text tokens attend to image features (highlighted)
- **Hybrid Approaches:** Combine multiple conditioning methods for optimal performance
- Categorization by type:
  - Temporal: Time embedding
  - Categorical: Class embedding
  - Textual: Text + cross-attention
- Note: Modern models often combine multiple injection methods for best results

#### Key Concepts:
- U-Net architecture with encoder-decoder structure and skip connections
- Self-attention for spatial coherence and cross-attention for conditioning
- Multiple methods for injecting conditional information
- Time embedding as critical component for diffusion models
- Multi-scale processing through hierarchical U-Net structure

#### Learning Outcomes:
- Students can implement a complete U-Net architecture for diffusion models
- Students understand the role of attention mechanisms in generation quality
- Students can integrate various conditioning methods (text, class, time)
- Students can design hybrid architectures combining multiple techniques
- Students understand computational trade-offs in attention layer placement

---

### Part 6/7: Advanced Techniques

**Slides:** 28-31 (4 slides)

#### Topics Covered:

**Slide 29: Latent Diffusion (Stable Diffusion)**
- **Problem:** High-resolution image generation is computationally expensive
- **Solution:** Operate in latent space instead of pixel space
- **Architecture:**
  - Encoder: Compress image to latent representation (e.g., 512×512 → 64×64)
  - Diffusion Process: Apply diffusion in compressed latent space
  - Decoder: Reconstruct image from denoised latent
- **VAE Component:** Pre-trained autoencoder for compression/reconstruction
- **Benefits:**
  - Significantly reduced computational cost (8× memory reduction)
  - Faster training and sampling
  - Maintains image quality
  - Enables larger model capacity in latent space
- **Stable Diffusion:** Most prominent implementation
  - Open-source availability
  - Text-to-image generation
  - Fine-tunable for specific domains
  - Community-driven improvements

**Slide 30: Noise Schedule Improvements**
- **Evolution of Noise Schedules:**
  - **Linear Schedule (Original):** β_t: 1e-4 → 0.02, original DDPM approach
  - **Cosine Schedule (Improved):** Smoother transition, better for high-resolution images
  - **Learned Schedule (Advanced):** Network learns optimal β_t during training
- **Key Concepts & Innovations:**
  - **Signal-to-Noise Ratio:** SNR(t) = ᾱ_t / (1-ᾱ_t)
  - **V-parameterization:** Predict velocity v_t instead of noise
  - **EDM Framework:** Exponential noise for better scaling
  - **Continuous Time:** Formulate as continuous SDE
- Impact: Better sample quality and training stability

**Slide 31: Other Improvement Techniques**
- **Quality Improvements:**
  - **Cascaded Diffusion:** Low-res generation → super-resolution
  - **Upsampler:** Specialized diffusion model for upscaling
  - **Adversarial Diffusion:** Add adversarial loss for sharper samples
- **Speed Improvements:**
  - **Progressive Distillation:** Student mimics T-step teacher in T/2 steps
  - **Consistency Models:** One-step generation via consistency training
- **Alternative Approaches:**
  - **Flow Matching:** Alternative to diffusion with optimal transport
  - **Rectified Flow:** Straightens probability paths for efficiency
- **Control Techniques:**
  - **Truncation Tricks:** Early stopping or partial diffusion for control
- Categorization: Quality (3 techniques), Speed (2 techniques), Alternative (2 techniques), Control (1 technique)

#### Key Concepts:
- Latent space diffusion for computational efficiency
- Evolution from linear to learned noise schedules
- Alternative formulations (flow matching, rectified flow)
- Distillation techniques for faster sampling
- Quality-speed trade-offs in model design

#### Learning Outcomes:
- Students can implement latent diffusion models with VAE compression
- Students understand various noise schedule designs and their impacts
- Students can apply distillation techniques for model acceleration
- Students can evaluate trade-offs between different advanced techniques
- Students are aware of state-of-the-art improvements beyond basic DDPM

---

### Part 7/7: Applications and Extensions

**Slides:** 32-34 (3 slides)

#### Topics Covered:

**Slide 33: Applications and Conclusion**
- **Creative Applications:**
  - **Text-to-Image (Popular):** DALL-E 2, Stable Diffusion, Midjourney, Imagen
  - **Image Editing:** Inpainting, outpainting, style transfer
  - **Video Generation:** Temporal diffusion models
  - **3D Generation:** NeRF + diffusion (DreamFusion)
- **Audio & Media:**
  - **Audio Synthesis:** Diffusion for music and speech generation
- **Scientific Applications:**
  - **Molecular Design:** Drug discovery and protein design
  - **Medical Imaging:** Denoising, reconstruction, synthesis
- **Future Directions:**
  - Real-time generation (⚡)
  - Better control mechanisms (🎛️)
  - Improved efficiency (📈)

**Slide 34: Thank You**
- Contact information for further questions and collaboration
- Instructor: Ho-min Park
- Emails: homin.park@ghent.ac.kr, powersimmani@gmail.com

#### Key Concepts:
- Text-to-image as killer application of diffusion models
- Diverse applications across creative and scientific domains
- Extension to video, 3D, and audio generation
- Medical and molecular applications leveraging controlled generation
- Future research directions in efficiency and control

#### Learning Outcomes:
- Students can identify appropriate applications for diffusion models
- Students understand the breadth of diffusion model applications
- Students are aware of cutting-edge research directions
- Students can propose new applications based on diffusion principles
- Students understand the societal impact and future potential

---

## 🛠️ Prerequisites

### Mathematical Background:
- **Probability Theory:** Understanding of probability distributions, expectation, variance, Gaussian distributions
- **Linear Algebra:** Matrix operations, eigenvalues, vector spaces
- **Calculus:** Multivariable calculus, gradients, chain rule
- **Statistics:** Markov chains, conditional probability, Bayes' theorem

### Machine Learning Knowledge:
- **Deep Learning Fundamentals:** Neural networks, backpropagation, optimization (SGD, Adam)
- **Convolutional Neural Networks:** Understanding of convolutions, pooling, residual connections
- **Attention Mechanisms:** Transformer architecture, self-attention, cross-attention
- **Generative Models:** Basic familiarity with GANs or VAEs is helpful but not required

### Programming Skills:
- **Python:** Proficiency in Python 3.7+
- **Deep Learning Frameworks:** Experience with PyTorch or TensorFlow
- **NumPy/SciPy:** For numerical operations and scientific computing
- **Git:** For version control and accessing implementations

### Software Requirements:
- **Python 3.8+** with pip or conda
- **PyTorch 2.0+** or TensorFlow 2.10+
- **CUDA 11.8+** (for GPU acceleration)
- **8GB+ GPU memory** recommended (16GB+ for high-resolution training)
- **50GB+ disk space** for datasets and model checkpoints

### Recommended Libraries:
```bash
pip install torch torchvision torchaudio
pip install transformers diffusers
pip install numpy scipy matplotlib
pip install pillow opencv-python
pip install wandb tensorboard  # for experiment tracking
```

---

## 💻 Hands-on Components

### Project 1: Implementing Forward Process (Part 2)
**Objective:** Build the forward diffusion process from scratch

**Tasks:**
1. Implement noise scheduling (linear, cosine)
2. Code the reparameterization trick for direct sampling
3. Visualize the corruption process at different timesteps
4. Compute and verify mathematical properties (mean, variance)

**Expected Output:**
- Function `forward_diffusion(x0, t, alpha_bar)` returning noisy samples
- Visualization showing x₀ → x_T progression
- Verification that x_T ≈ N(0,I)

**Estimated Time:** 4-6 hours

---

### Project 2: Training a Simple Diffusion Model (Part 3)
**Objective:** Train a diffusion model on a simple dataset (MNIST or CIFAR-10)

**Tasks:**
1. Implement a basic U-Net architecture with time embedding
2. Code the denoising objective L_simple
3. Create training loop with proper noise sampling
4. Monitor training with loss curves and sample quality

**Expected Output:**
- Working training script achieving convergence
- Generated samples showing learned data distribution
- Training curves demonstrating loss reduction
- Model checkpoint saved for inference

**Estimated Time:** 8-12 hours

**Resources:**
- Starter code template with U-Net skeleton
- Pre-processed dataset loaders
- Hyperparameter configuration file

---

### Project 3: Implementing DDIM Sampling (Part 4)
**Objective:** Accelerate sampling using DDIM

**Tasks:**
1. Implement DDPM sampling (1000 steps baseline)
2. Code DDIM deterministic sampling
3. Experiment with different timestep schedules (50, 100, 250 steps)
4. Compare speed vs quality trade-offs

**Expected Output:**
- Side-by-side comparison: DDPM vs DDIM samples
- Timing measurements for different step counts
- Quality metrics (FID score, visual inspection)
- Report on optimal speed-quality balance

**Estimated Time:** 6-8 hours

---

### Project 4: Text-to-Image Generation (Part 5)
**Objective:** Implement conditional generation with text prompts

**Tasks:**
1. Integrate CLIP text encoder for embeddings
2. Add cross-attention layers to U-Net
3. Implement classifier-free guidance
4. Generate images from custom text prompts

**Expected Output:**
- Model generating images from text descriptions
- Demonstration of guidance scale effects (w = 1, 3, 7)
- Comparison of conditional vs unconditional samples
- Gallery of diverse generations from varied prompts

**Estimated Time:** 10-15 hours

**Advanced Extension:**
- Fine-tune on custom dataset (e.g., specific art style)
- Implement inpainting or outpainting
- Add ControlNet for spatial control

---

### Project 5: Latent Diffusion Implementation (Part 6)
**Objective:** Build a memory-efficient latent diffusion model

**Tasks:**
1. Train or load pre-trained VAE encoder/decoder
2. Adapt diffusion model to operate in latent space
3. Measure memory and speed improvements
4. Generate high-resolution images (512×512 or higher)

**Expected Output:**
- Comparison: pixel-space vs latent-space diffusion
- Memory consumption analysis (VRAM usage)
- Timing benchmarks (training and sampling)
- High-resolution generated samples

**Estimated Time:** 12-16 hours

**Computing Requirements:**
- Minimum 16GB GPU memory for 512×512 images
- Can use gradient checkpointing to reduce memory
- Consider using mixed precision (fp16) training

---

## 📖 Additional Resources

### Foundational Papers:
1. **DDPM (2020):** "Denoising Diffusion Probabilistic Models" - Ho et al.
   - https://arxiv.org/abs/2006.11239
   - Original paper establishing the modern diffusion framework

2. **Improved DDPM (2021):** "Improved Denoising Diffusion Probabilistic Models" - Nichol & Dhariwal
   - https://arxiv.org/abs/2102.09672
   - Learned variance, cosine schedule improvements

3. **DDIM (2020):** "Denoising Diffusion Implicit Models" - Song et al.
   - https://arxiv.org/abs/2010.02502
   - Accelerated sampling with deterministic process

4. **Classifier-Free Guidance (2022):** "Classifier-Free Diffusion Guidance" - Ho & Salimans
   - https://arxiv.org/abs/2207.12598
   - Breakthrough technique for conditional generation

5. **Latent Diffusion (2022):** "High-Resolution Image Synthesis with Latent Diffusion Models" - Rombach et al.
   - https://arxiv.org/abs/2112.10752
   - Foundation of Stable Diffusion

### Implementations and Codebases:
- **Hugging Face Diffusers:** https://github.com/huggingface/diffusers
  - Well-documented library with multiple model implementations
  - Easy-to-use pipeline APIs
  
- **Stable Diffusion:** https://github.com/CompVis/stable-diffusion
  - Official implementation of latent diffusion
  
- **Annotated Diffusion:** https://huggingface.co/blog/annotated-diffusion
  - Step-by-step tutorial with code annotations

### Video Lectures and Tutorials:
- **Lil'Log Blog:** "What are Diffusion Models?" by Lilian Weng
  - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  
- **DeepMind Tutorial:** "Diffusion Models" lecture series
  - Comprehensive mathematical treatment
  
- **Yannic Kilcher YouTube:** Paper explanations and discussions
  - Visual intuitions and implementation details

### Datasets:
- **MNIST:** 70,000 handwritten digits (28×28) - Good for initial experiments
- **CIFAR-10:** 60,000 images (32×32) - 10 classes, moderate complexity
- **CelebA-HQ:** High-quality face images (256×256, 1024×1024)
- **LAION-5B:** 5 billion image-text pairs - Used for training large models
- **ImageNet:** 1.4M images, 1000 classes - Standard benchmark

### Tools and Frameworks:
- **Weights & Biases:** Experiment tracking and visualization
- **TensorBoard:** Training monitoring and metric logging
- **Gradio:** Quick demo interface creation
- **CompVis/Taming-Transformers:** VQVAE implementation for latent diffusion

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Sequential Learning:** Work through Parts 1-7 in order, completing readings before hands-on projects
2. **Active Practice:** Implement concepts immediately after learning theory
3. **Experimentation:** Modify hyperparameters and observe effects on training/generation
4. **Community Engagement:** Join Discord/Reddit communities for discussions
   - r/MachineLearning, r/StableDiffusion
   - Hugging Face Discord server

### For Instructors:
**Suggested Course Structure (12-week semester):**
- **Weeks 1-2:** Parts 1-2 (Introduction, Forward Process)
- **Weeks 3-5:** Part 3 (Reverse Process) + Project 2
- **Weeks 6-7:** Part 4 (Sampling) + Project 3
- **Weeks 8-9:** Part 5 (Architecture) + Project 4
- **Weeks 10-11:** Part 6 (Advanced Techniques) + Project 5
- **Week 12:** Part 7 (Applications) + Final presentations

**Lecture Format:**
- 90-minute sessions: 60 min lecture + 30 min live coding/Q&A
- Use slideshow.html for presentation (keyboard navigation enabled)
- Pause after each part for discussion and clarification

**Lab Sessions:**
- 2-hour guided implementation sessions
- Provide starter code to reduce boilerplate
- Focus on conceptual understanding over debugging syntax

### For Researchers:
- **Quick Reference:** Use slides as concise reference for mathematical formulations
- **Implementation Guide:** Follow architecture slides for paper reproduction
- **Baseline Comparison:** Use techniques as baselines for novel methods
- **Citation:** Cite course materials when using for workshops or tutorials

### Interactive Slideshow:
The included `Lecture16_slideshow.html` provides:
- Keyboard navigation (← → Arrow keys or Space)
- Progress tracking with visual progress bar
- Slide counter showing current position (X/34)
- Part labels for easy navigation
- Responsive design for different screen sizes

**Keyboard Shortcuts:**
- `→` or `Space`: Next slide
- `←`: Previous slide  
- `Home`: Jump to first slide
- `End`: Jump to last slide

---

## 📊 Assessment Suggestions

### Theoretical Understanding (40%):
**Problem Sets (20%):**
- Derive forward process variance from first principles
- Prove properties of the cumulative α formulation
- Show equivalence between diffusion and score-based models
- Analyze convergence properties of different noise schedules

**Midterm Exam (20%):**
- Multiple choice: Conceptual understanding (10 questions)
- Short answer: Explain key algorithms and trade-offs (5 questions)
- Derivation: Mathematical proofs and formula manipulation (2 problems)

### Practical Implementation (40%):
**Project Milestones:**
- Project 2 (Training): 15% - Working model with convergence
- Project 3 (DDIM): 10% - Speed improvements demonstrated
- Project 4 (Text-to-Image): 15% - Quality conditional generation

**Evaluation Criteria:**
- Code quality and documentation (25%)
- Experimental rigor and analysis (35%)
- Results quality and visualization (30%)
- Innovation and extensions (10%)

### Final Project (20%):
**Options:**
1. **Research Extension:** Implement recent paper technique not covered in course
2. **Novel Application:** Apply diffusion to new domain (audio, 3D, molecular)
3. **Optimization Study:** Comprehensive analysis of hyperparameter effects
4. **Reproduction Study:** Replicate major paper results and analyze findings

**Deliverables:**
- Written report (8-10 pages, conference format)
- Code repository with documentation
- 15-minute presentation with demo
- Peer review of 2 other projects

### Participation (Optional 5% Bonus):
- Active discussion in lectures and forums
- Helpful responses to classmate questions
- Contribution to shared code repository
- Blog post or tutorial explaining concepts

### Grading Rubric for Projects:

| Criterion | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (<70%) |
|-----------|---------------------|---------------|----------------------|--------------------------|
| **Implementation** | Fully working, efficient, follows best practices | Working with minor issues, reasonable efficiency | Basic implementation, some bugs or inefficiencies | Incomplete or non-functional |
| **Experimentation** | Comprehensive experiments, insightful analysis | Good experiments, clear results | Basic experiments, limited analysis | Minimal experimentation |
| **Documentation** | Clear, thorough, reproducible | Good explanation, mostly reproducible | Basic documentation | Sparse or unclear |
| **Results** | High-quality outputs, strong quantitative metrics | Good quality, decent metrics | Acceptable quality | Poor results |

---

## 📝 Notes for Implementation

### Computing Resources:

**Minimum Requirements (Training):**
- GPU: NVIDIA RTX 3060 (12GB) or equivalent
- RAM: 16GB system memory
- Storage: 100GB SSD (for datasets and checkpoints)
- OS: Ubuntu 20.04+ or Windows 10+ with WSL2

**Recommended Setup (Research-grade):**
- GPU: NVIDIA A100 (40GB) or RTX 4090 (24GB)
- RAM: 64GB system memory
- Storage: 500GB NVMe SSD
- Multi-GPU support for parallel training

**Cloud Alternatives:**
- **Google Colab Pro+:** $50/month, A100 access, 50 compute units/month
- **Vast.ai:** On-demand GPU rental, ~$0.30-1.50/hour depending on GPU
- **Lambda Labs:** Dedicated GPU instances, $1.10/hour for A100
- **Paperspace Gradient:** $8/month + compute costs, integrated notebooks

### Memory Optimization Strategies:

**For Limited GPU Memory (8GB):**
1. Reduce batch size to 1-4
2. Use gradient accumulation to simulate larger batches
3. Enable gradient checkpointing (trade speed for memory)
4. Use mixed precision training (FP16/BF16)
5. Train at lower resolutions (64×64 or 128×128)

**Example Configuration:**
```python
# Memory-efficient training config
config = {
    'batch_size': 2,
    'gradient_accumulation_steps': 8,  # effective batch = 16
    'mixed_precision': 'fp16',
    'gradient_checkpointing': True,
    'resolution': 128,
    'num_workers': 4
}
```

### Debugging Tips:

**Common Issues:**
1. **NaN losses:** Check learning rate (try 1e-4 to 1e-5), gradient clipping
2. **No convergence:** Verify noise schedule, check data normalization [-1, 1]
3. **Poor sample quality:** Increase training iterations, check architecture
4. **Out of memory:** Reduce batch size, enable gradient checkpointing
5. **Slow training:** Use DataLoader with multiple workers, check I/O bottlenecks

**Validation Checks:**
```python
# Verify forward process
assert torch.allclose(noisy_images.mean(), torch.zeros(1), atol=0.5)
assert torch.allclose(noisy_images.std(), torch.ones(1), atol=0.5)

# Verify model output shapes
assert predicted_noise.shape == input_noise.shape
```

### Experiment Tracking:

**Weights & Biases Setup:**
```python
import wandb

wandb.init(
    project="diffusion-models",
    config={
        "learning_rate": 1e-4,
        "batch_size": 64,
        "timesteps": 1000,
        "architecture": "unet",
    }
)

# Log during training
wandb.log({
    "loss": loss.item(),
    "samples": wandb.Image(generated_samples)
})
```

### Reproducibility:

**Set Random Seeds:**
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### Performance Benchmarking:

**Key Metrics to Track:**
- **Training:** Loss curve, samples per second, GPU utilization
- **Generation:** Sampling time per image, FID score, IS (Inception Score)
- **Quality:** Visual inspection, LPIPS (perceptual similarity), user studies
- **Efficiency:** Memory usage (peak VRAM), model size (parameters, checkpoint MB)

**Expected Performance Targets:**
- CIFAR-10: FID < 5.0 (state-of-the-art)
- ImageNet 256×256: FID < 10.0
- Sampling speed: 1-2 images/second on RTX 3090 (DDIM 50 steps)
- Training time: 1-3 days on single GPU for small datasets

---

## 🙏 Credits

**Lecturer:** Ho-min Park  
**Affiliation:** Ghent University  
**Contact:**  
- Email: homin.park@ghent.ac.kr
- Email: powersimmani@gmail.com

**Course Development:**  
This lecture series was developed for advanced graduate courses in computer vision and deep learning. The materials synthesize knowledge from foundational papers, state-of-the-art implementations, and practical experience with diffusion models.

**Acknowledgments:**  
- Foundational work by researchers at OpenAI, DeepMind, Google Research, and UC Berkeley
- Open-source community contributions to diffusers, stable-diffusion, and related libraries
- Students and colleagues providing feedback on course materials

**License:**  
These educational materials are provided for academic use. Please cite appropriately when using in courses or workshops.

**Recommended Citation:**
```
Park, H. (2024). Lecture 16: Generative Models - Diffusion. 
Advanced Topics in Computer Vision, Ghent University.
```

---

**Last Updated:** November 2025  
**Course Version:** 1.0  
**Slide Count:** 34 slides across 7 parts

For questions, suggestions, or corrections, please contact the instructor directly via email.