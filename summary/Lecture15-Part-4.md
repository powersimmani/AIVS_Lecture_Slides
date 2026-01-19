# Lecture 15 - Part 4: Key Challenges

## Overview
This part covers the main challenges in training GANs including mode collapse, training instability, and evaluation difficulties.

## Key Topics

### 1. Mode Collapse
**Problem**: Generator produces limited variety of outputs
- Learns to fool discriminator with few samples
- Ignores parts of the real distribution

**Symptoms**:
- Generated images look similar
- Low diversity despite random input
- Discriminator can't distinguish within mode

**Solutions**:
- Minibatch discrimination
- Unrolled GANs
- Feature matching
- Multiple discriminators

### 2. Training Instability
**The Balancing Act**:
- Discriminator too strong → Generator gets no gradient
- Generator too strong → Discriminator useless
- Need careful balance

**Manifestations**:
- Loss oscillation
- Divergence
- Sudden quality drops

**Solutions**:
- Two-timescale update rule (TTUR)
- Spectral normalization
- Progressive growing
- Better architectures (DCGAN)

### 3. Evaluation Difficulties
**Why Hard to Evaluate?**
- No explicit likelihood
- Quality is subjective
- Diversity hard to measure

**Common Metrics**:
- **Inception Score (IS)**: Quality and diversity using classifier
- **FID**: Fréchet Inception Distance
- **Human evaluation**: Gold standard but expensive

### 4. Vanishing Gradients
**Problem**: When discriminator is too good
```
D(G(z)) ≈ 0 for all generated samples
log(1 - D(G(z))) ≈ 0 (flat gradient)
```

**Original GAN Loss Issue**:
- Optimal discriminator → zero gradients
- Generator stops learning

**Solutions**:
- Non-saturating loss: -log(D(G(z)))
- Wasserstein loss: -D(G(z))
- Label smoothing

### 5. Common Failure Patterns
**Discriminator Win**:
- Perfect discrimination
- Generator loss explodes
- No learning happens

**Generator Win (Temporary)**:
- Fooling with one trick
- Discriminator catches on
- Cycle repeats

**Oscillation**:
- Neither converges
- Quality fluctuates
- Training never stabilizes

**Checkerboard Artifacts**:
- From transposed convolutions
- Visual grid patterns

## Important Takeaways
1. Mode collapse limits output diversity
2. Training requires careful balancing
3. Evaluation lacks perfect metrics
4. Vanishing gradients stop generator learning
5. Multiple failure modes exist
6. Solutions often architecture-specific

