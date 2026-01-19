# Lecture 16 - Part 4: Sampling

## Overview
This part covers sampling methods for generating images from trained diffusion models.

## Key Topics

### 1. DDPM Sampling
**Denoising Diffusion Probabilistic Models**

**Process**: Start from noise, iteratively denoise
```python
x_T = torch.randn_like(x_0)  # Pure noise
for t in reversed(range(T)):
    z = torch.randn_like(x_T) if t > 0 else 0
    x_t = (1/√αₜ) * (x_t - (1-αₜ)/√(1-ᾱₜ) * ε_θ(x_t, t)) + σₜ * z
```

**Characteristics**:
- T steps (typically 1000)
- Slow but high quality
- Each step removes small noise

### 2. DDIM - Fast Sampling
**Denoising Diffusion Implicit Models**

**Key Insight**: Deterministic sampling possible
```
x_{t-1} = √ᾱ_{t-1} * x̂_0 + √(1-ᾱ_{t-1}) * direction
```

**Benefits**:
- Fewer steps (50-100 vs 1000)
- Deterministic (same noise → same image)
- Interpolation in latent space

**Trade-off**: Slight quality reduction for speed

### 3. Conditional Generation
**Class-Conditional**:
```
ε_θ(x_t, t, y)  # y is class label
```
- Add class embedding to model
- Generate specific categories

**Text-Conditional**:
```
ε_θ(x_t, t, text_embedding)
```
- CLIP text encoder
- Cross-attention to text
- Text-to-image generation

### 4. Classifier-Free Guidance
**Problem**: Need guidance for better conditional generation

**Solution**: Mix conditional and unconditional predictions
```
ε̃ = ε_θ(x_t, t, ∅) + s * (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
```
- s > 1: Stronger conditioning
- s = 1: Standard conditional
- s < 1: Weaker conditioning

**Benefits**:
- No separate classifier needed
- Better sample quality
- Controllable strength

### 5. Sampling Strategies
**Temperature/Guidance Scale**:
- Higher: More condition-adherent
- Lower: More diversity

**Truncation**:
- Limit noise range
- Trade diversity for quality

**Ancestral Sampling**:
- Add noise at each step
- More stochastic

## Important Takeaways
1. DDPM sampling iterates T denoising steps
2. DDIM enables much faster deterministic sampling
3. Conditional generation adds control
4. Classifier-free guidance improves quality
5. Guidance scale controls condition strength
6. Various strategies trade quality vs diversity

