# Lecture 15 - Part 1: GAN Introduction and Mathematical Foundations

## Overview
This part introduces Generative Adversarial Networks (GANs), their intuition, and the mathematical foundations of adversarial training.

## Key Topics

### 1. Review of Generative Models
- **Generative vs Discriminative**:
  - Discriminative: P(y|x) - classify inputs
  - Generative: P(x) - model data distribution
- **Types of Generative Models**:
  - Explicit density: VAE, Flow models
  - Implicit density: GANs
- **Goal**: Generate realistic samples from learned distribution

### 2. Why GANs?
- **No explicit density**: Don't need to model P(x) directly
- **Sharp samples**: Produce high-quality images
- **Flexibility**: Various architectures possible
- **Impact**: Revolutionized image generation

### 3. Intuitive Understanding - Counterfeiter Analogy
- **Generator (G)**: Counterfeiter making fake money
- **Discriminator (D)**: Detective identifying fakes
- **Training**: Both improve through competition
- **Equilibrium**: Generator produces indistinguishable fakes

### 4. Probability Distribution Perspective
- **Real Data**: Samples from P_data(x)
- **Generated Data**: G(z) where z ~ P_z (noise)
- **Generator Goal**: Make P_G(x) ≈ P_data(x)
- **Discriminator Goal**: Distinguish P_G from P_data

### 5. Mathematical Definition of GAN
**Min-Max Game**:
```
min_G max_D V(D,G) = E_x~P_data[log D(x)] + E_z~P_z[log(1-D(G(z)))]
```

**Components**:
- D(x): Probability that x is real
- G(z): Generated sample from noise z
- First term: D maximized when real data → 1
- Second term: D maximized when fake → 0, G minimized when fake → 1

### 6. Value Function Analysis
**Discriminator Objective** (maximize):
```
max_D E[log D(x)] + E[log(1-D(G(z)))]
```
- Binary cross-entropy
- Real samples labeled 1, fake labeled 0

**Generator Objective** (minimize):
```
min_G E[log(1-D(G(z)))]
```
- Make discriminator output high for fakes

### 7. Optimal Discriminator
**Theorem**: For fixed G, optimal D is:
```
D*_G(x) = P_data(x) / (P_data(x) + P_G(x))
```

**At Optimum**: D*(x) = 0.5 when P_data = P_G

### 8. Global Optimum
**Theorem**: Global minimum of V(G,D) achieved when:
```
P_G = P_data
```

**Value at Optimum**: V(G*, D*) = -log(4)

**Interpretation**: Generator perfectly matches data distribution

### 9. Jensen-Shannon Divergence
With optimal D, the objective becomes:
```
V(G, D*) = 2·JSD(P_data || P_G) - log(4)
```

**JSD Properties**:
- Symmetric: JSD(P||Q) = JSD(Q||P)
- Non-negative: JSD ≥ 0
- Zero iff P = Q

## Important Takeaways
1. GANs use adversarial training between generator and discriminator
2. The counterfeiter-detective analogy captures the core idea
3. Mathematical formulation is a min-max game
4. Optimal discriminator has a closed-form solution
5. Global optimum achieved when generator matches data distribution
6. Training minimizes Jensen-Shannon divergence
7. These foundations explain why GAN training can be unstable

