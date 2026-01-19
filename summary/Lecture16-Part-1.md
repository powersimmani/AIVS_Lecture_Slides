# Lecture 16 - Part 1: Diffusion Models - Introduction and Forward Process

## Overview
This part introduces diffusion models, comparing them with GANs, and explains the forward diffusion process that gradually adds noise to data.

## Key Topics

### 1. Evolution of Generative Models
- **VAEs (2013)**: Variational autoencoders, latent space
- **GANs (2014)**: Adversarial training, sharp images
- **Flow Models**: Exact likelihood, invertible
- **Diffusion (2020)**: Denoising, state-of-the-art quality

### 2. Intuitive Understanding
- **Forward Process**: Gradually add noise to image
- **Reverse Process**: Learn to denoise step by step
- **Analogy**: Spilling ink (forward), reconstructing (reverse)

**Key Insight**: Easier to learn small denoising steps than one big generation

### 3. Comparison with GANs
| Aspect | GANs | Diffusion |
|--------|------|-----------|
| Training | Adversarial (unstable) | Denoising (stable) |
| Sampling | Single forward pass | Many iterations |
| Quality | High | Higher |
| Diversity | Mode collapse risk | Full diversity |
| Likelihood | Implicit | Can compute |

### 4. Forward Process Definition
**Add Noise Gradually**:
```
q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
```
- xₜ: Noisy image at step t
- βₜ: Noise schedule (variance)
- T steps: x₀ → x₁ → ... → xₜ

**Noise Schedule**:
- β₁, β₂, ..., βₜ increasing
- Typical: Linear, cosine schedules
- Start small (~0.0001), end larger (~0.02)

### 5. Cumulative Effect
**Key Property**: Can sample xₜ directly from x₀
```
q(xₜ|x₀) = N(xₜ; √ᾱₜx₀, (1-ᾱₜ)I)
```
Where:
```
αₜ = 1 - βₜ
ᾱₜ = Π_{s=1}^{t} αₛ
```

**Sampling**:
```
xₜ = √ᾱₜx₀ + √(1-ᾱₜ)ε, where ε ~ N(0,I)
```

### 6. Mathematical Properties
- **Markov Process**: Each step depends only on previous
- **Gaussian Transitions**: Analytically tractable
- **Fixed Process**: No learning required
- **End State**: xₜ ≈ N(0,I) (pure noise)

### 7. Forward Process Visualization
```
x₀ (clean) → x₁₀₀ (slight noise) → x₅₀₀ (noisy) → x₁₀₀₀ (pure noise)
```
- Image gradually becomes unrecognizable
- Information destroyed systematically
- Final state: Standard Gaussian

### 8. Why is this Process Necessary?
**Training**:
- Provides corrupted inputs at all noise levels
- Model learns denoising at each level
- Random t sampled during training

**Inference**:
- Start from pure noise
- Reverse the process step by step
- Reconstruct clean image

**Benefit**: Breaks hard generation into easy steps

## Important Takeaways
1. Diffusion models gradually add noise (forward) and remove it (reverse)
2. Forward process is fixed Gaussian, no learning needed
3. Key formula: xₜ = √ᾱₜx₀ + √(1-ᾱₜ)ε
4. T steps transform clean image to pure noise
5. This decomposition makes learning tractable
6. More stable training than GANs
7. Foundation for state-of-the-art generation

