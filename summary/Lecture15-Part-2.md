# Lecture 15 - Part 2: GAN Training and Challenges

## Overview
This part covers the GAN training algorithm, gradient flow, common challenges like mode collapse, and techniques to address training instability.

## Key Topics

### 1. Training Process Overview
**Alternating Updates**:
1. **Train D**: Fix G, update D to better distinguish
2. **Train G**: Fix D, update G to fool D
3. Repeat

**Training Loop**:
```python
for epoch in epochs:
    # Update Discriminator
    real_loss = BCE(D(real_data), ones)
    fake_loss = BCE(D(G(noise)), zeros)
    d_loss = real_loss + fake_loss
    d_loss.backward()

    # Update Generator
    g_loss = BCE(D(G(noise)), ones)
    g_loss.backward()
```

### 2. Detailed Algorithm
1. Sample minibatch of m noise vectors {z¹,...,zᵐ}
2. Sample minibatch of m real examples {x¹,...,xᵐ}
3. Update D by ascending gradient:
   ```
   ∇_θd [1/m Σ log D(xⁱ) + log(1-D(G(zⁱ)))]
   ```
4. Sample new minibatch of m noise vectors
5. Update G by descending gradient:
   ```
   ∇_θg [1/m Σ log(1-D(G(zⁱ)))]
   ```

### 3. Gradient Flow
**For Discriminator**:
- Clear gradients from both real and fake
- Standard classification gradients

**For Generator**:
- Gradients flow through D (frozen)
- Only G parameters updated

### 4. Non-Saturating Loss
**Problem**: Original G loss saturates early
- When D is confident: log(1-D(G(z))) ≈ 0
- Gradient vanishes

**Solution**: Non-saturating loss
```
Original: min_G E[log(1-D(G(z)))]
Non-saturating: max_G E[log D(G(z))]
```
- Same optimum, better gradients
- Standard practice

### 5. Mode Collapse
**Definition**: Generator produces limited variety
- Outputs only few types of samples
- "Forgets" modes of data distribution

**Symptoms**:
- Generated samples look similar
- Poor diversity
- Cyclic behavior during training

**Causes**:
- Generator finds "safe" outputs
- D can be fooled by single mode
- Local equilibrium

### 6. Training Instability
**Oscillation**: D and G alternate winning
**Divergence**: One network dominates
**Vanishing Gradients**: D too strong

**Symptoms**:
- Loss doesn't decrease
- Quality doesn't improve
- Mode collapse

### 7. Evaluation Difficulties
**Challenge**: No objective metric for generation quality

**Metrics**:
- **Inception Score (IS)**: Quality and diversity
- **FID (Fréchet Inception Distance)**: Compare feature distributions
- **Human Evaluation**: Gold standard but expensive

**FID** (lower is better):
```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))
```

### 8. Vanishing Gradient Details
**When D is optimal**:
- D(G(z)) ≈ 0 for generated samples
- log(1-D(G(z))) ≈ 0
- Gradient to G vanishes

**Disjoint Supports**:
- Real and fake distributions don't overlap
- D can perfectly separate
- No useful gradient signal

### 9. Common Failure Patterns
1. **D too strong**: G gets no learning signal
2. **G too strong**: D can't discriminate, G stops improving
3. **Mode collapse**: G covers few modes
4. **Oscillation**: Neither converges
5. **Divergence**: Losses explode

## Important Takeaways
1. GANs train through alternating D and G updates
2. Non-saturating loss provides better gradients
3. Mode collapse is a major challenge
4. Training instability is inherent to adversarial training
5. FID is the standard evaluation metric
6. Multiple failure modes require careful monitoring
7. These challenges motivated many GAN improvements

