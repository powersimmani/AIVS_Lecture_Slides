# Lecture 15 - Part 5: Improvement Techniques

## Overview
This part covers major GAN variants and techniques developed to address training challenges.

## Key Topics

### 1. DCGAN (2015)
**Deep Convolutional GAN**

**Architecture Guidelines**:
- Replace pooling with strided convolutions
- Use batch normalization (except D input, G output)
- Remove fully connected layers
- Use ReLU in generator (except output: Tanh)
- Use LeakyReLU in discriminator

**Impact**: Made GAN training more stable

### 2. Conditional GAN (cGAN)
**Adding Labels to Generation**

**Architecture**:
```
G(z, y) → Image of class y
D(x, y) → Is x real image of class y?
```

**Applications**:
- Class-conditional image generation
- Image-to-image translation (Pix2Pix)
- Text-to-image synthesis

### 3. Wasserstein GAN (WGAN)
**Better Loss Function**

**Earth Mover (Wasserstein) Distance**:
```
W(Pr, Pg) = inf E[||x - y||]
```
- More meaningful than JS divergence
- Provides gradients even when distributions don't overlap

**WGAN Training**:
- Remove log from loss
- Clip discriminator weights (or gradient penalty)
- Train discriminator more iterations

**Benefits**:
- More stable training
- Meaningful loss metric
- Reduced mode collapse

### 4. Progressive Growing GAN
**Training Strategy**:
1. Start at low resolution (4×4)
2. Gradually add layers
3. Increase resolution step by step
4. Smooth transition with fade-in

**Benefits**:
- Stable high-resolution training
- Better quality at large scales
- Faster overall training

### 5. Other Important Improvements
**Spectral Normalization**:
- Normalize discriminator weights
- Controls Lipschitz constant
- More stable training

**Self-Attention GAN (SAGAN)**:
- Add self-attention layers
- Capture long-range dependencies
- Better global coherence

**BigGAN**:
- Large-scale training
- Class embeddings
- Truncation trick for quality/diversity trade-off

**StyleGAN**:
- Style-based generator
- Progressive growing
- High-quality face generation

## Important Takeaways
1. DCGAN established architectural best practices
2. Conditional GANs enable controlled generation
3. WGAN provides more stable training via better loss
4. Progressive growing enables high-resolution generation
5. Spectral normalization stabilizes discriminator
6. Self-attention improves global coherence

