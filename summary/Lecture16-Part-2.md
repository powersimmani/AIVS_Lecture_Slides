# Lecture 16 - Part 2: Reverse Process and Training

## Overview
This part covers the reverse diffusion process, how neural networks learn to denoise, and the training objective for diffusion models.

## Key Topics

### 1. Reverse Process Goal
- **Forward**: x₀ → x₁ → ... → xₜ (add noise)
- **Reverse**: xₜ → xₜ₋₁ → ... → x₀ (remove noise)
- **Challenge**: q(xₜ₋₁|xₜ) intractable without x₀
- **Solution**: Learn with neural network

### 2. Neural Network Parameterization
**Reverse Distribution**:
```
p_θ(xₜ₋₁|xₜ) = N(xₜ₋₁; μ_θ(xₜ,t), Σ_θ(xₜ,t))
```

**Three Parameterization Options**:
1. Predict mean μ directly
2. Predict x₀ from xₜ
3. **Predict noise ε** (most common)

**Noise Prediction**:
```
ε_θ(xₜ, t) ≈ ε (the noise added)
```

### 3. Deep Dive into Score Function
**Score Function**: ∇_x log p(x)
- Points toward high probability regions
- Diffusion models learn score

**Connection**:
```
∇_xₜ log q(xₜ|x₀) = -ε / √(1-ᾱₜ)
```
- Predicting noise ≈ predicting score

### 4. Denoising Objective
**Simple Loss**:
```
L = E_{x₀,t,ε} [||ε - ε_θ(xₜ, t)||²]
```

**Training Process**:
1. Sample x₀ from data
2. Sample t uniformly from [1, T]
3. Sample ε ~ N(0,I)
4. Compute xₜ = √ᾱₜx₀ + √(1-ᾱₜ)ε
5. Predict ε̂ = ε_θ(xₜ, t)
6. Loss = ||ε - ε̂||²

### 5. Training Algorithm (DDPM)
```python
for x₀ in dataloader:
    t = randint(1, T)                    # Random timestep
    ε = randn_like(x₀)                   # Random noise
    xₜ = sqrt(α_bar[t])*x₀ + sqrt(1-α_bar[t])*ε
    ε_pred = model(xₜ, t)                # Predict noise
    loss = MSE(ε, ε_pred)
    loss.backward()
    optimizer.step()
```

### 6. DDPM Sampling
**Generation Process**:
```python
xₜ = randn(shape)  # Start from noise
for t in range(T, 0, -1):
    ε_pred = model(xₜ, t)
    μ = (xₜ - (1-αₜ)/√(1-ᾱₜ) * ε_pred) / √αₜ
    xₜ₋₁ = μ + σₜ * z  # z ~ N(0,I) if t > 1
return x₀
```

**Steps**: Typically 1000 (slow!)

### 7. DDIM - Fast Sampling
**Denoising Diffusion Implicit Models**:
- Non-Markovian process
- Deterministic sampling possible
- Same trained model, different sampler

**Formula**:
```
xₜ₋₁ = √ᾱₜ₋₁ x̂₀ + √(1-ᾱₜ₋₁-σ²) ε_θ + σε
```

**Benefits**:
- 10-100× fewer steps (50-100 vs 1000)
- Deterministic when σ=0
- Trade quality for speed

### 8. Conditional Generation
**Classifier Guidance**:
```
ε̃ = ε_θ(xₜ) - √(1-ᾱₜ) * s * ∇_xₜ log p(y|xₜ)
```
- Requires separate classifier
- s: Guidance scale

**Classifier-Free Guidance**:
```
ε̃ = ε_θ(xₜ, ∅) + s * (ε_θ(xₜ, c) - ε_θ(xₜ, ∅))
```
- No separate classifier needed
- Train with conditional and unconditional
- s: Guidance scale (typically 5-15)

### 9. Classifier-Free Guidance Details
**Training**:
- Randomly drop condition c → ∅ (10-20%)
- Model learns both conditional and unconditional

**Sampling**:
- Interpolate between conditional and unconditional
- Higher s: More aligned with condition, less diversity

## Important Takeaways
1. Neural network learns to predict noise added at each step
2. Simple MSE loss on noise prediction
3. Sampling reverses diffusion step by step
4. DDIM enables much faster sampling
5. Classifier-free guidance controls conditional generation
6. Guidance scale trades diversity for quality
7. This foundation enables text-to-image models like Stable Diffusion

