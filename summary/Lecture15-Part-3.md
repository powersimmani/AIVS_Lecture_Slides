# Lecture 15 - Part 3: GAN Improvements and Applications

## Overview
This part covers major improvements to the original GAN including architectural advances, loss modifications, and practical applications.

## Key Topics

### 1. DCGAN (2015)
**Deep Convolutional GAN Architecture Guidelines**:
- Replace pooling with strided convolutions
- Use batch normalization (except D first layer, G output)
- Remove fully connected layers
- ReLU in G (except output: tanh), LeakyReLU in D

**Architecture**:
```
Generator: z → FC → Reshape → ConvT → ConvT → ConvT → Image
Discriminator: Image → Conv → Conv → Conv → FC → Real/Fake
```

**Impact**: First stable deep convolutional GAN

### 2. Conditional GAN (cGAN)
**Concept**: Condition generation on additional information
```
G(z, c) → Image of class c
D(x, c) → Real/Fake given class c
```

**Objective**:
```
min_G max_D E[log D(x|c)] + E[log(1-D(G(z|c)|c))]
```

**Applications**:
- Class-conditional image generation
- Image-to-image translation
- Text-to-image generation

### 3. Wasserstein GAN (WGAN)
**Problem**: JS divergence issues with disjoint supports
**Solution**: Use Wasserstein distance (Earth Mover's Distance)

**WGAN Objective**:
```
min_G max_D E[D(x)] - E[D(G(z))]
```
- D is a critic (no sigmoid), outputs unbounded values
- Requires Lipschitz constraint on D

**Lipschitz Constraint Methods**:
- Weight clipping (original WGAN)
- Gradient penalty (WGAN-GP):
  ```
  λ E[(||∇D(x̂)||₂ - 1)²]
  ```

**Benefits**:
- More stable training
- Meaningful loss (correlates with quality)
- No mode collapse

### 4. Spectral Normalization
```
W_SN = W / σ(W)
```
- σ(W): Largest singular value
- Ensures Lipschitz continuity
- Applied to discriminator
- Simpler than gradient penalty

### 5. Other Improvements
**Progressive GAN**:
- Grow resolution gradually
- 4×4 → 8×8 → ... → 1024×1024
- Stabilizes high-resolution generation

**Self-Attention GAN (SAGAN)**:
- Add self-attention layers
- Capture long-range dependencies
- Better global structure

**StyleGAN**:
- Style-based generator
- Progressive growing
- State-of-the-art face generation
- Controllable attributes

### 6. GAN Applications
**Image Generation**:
- Face generation (StyleGAN)
- Art creation
- Photorealistic images

**Image-to-Image Translation**:
- Pix2Pix: Paired translation
- CycleGAN: Unpaired translation
- Style transfer

**Super Resolution**:
- SRGAN, ESRGAN
- Enhance image resolution

**Data Augmentation**:
- Generate training data
- Handle class imbalance

**Other**:
- Video generation
- 3D object generation
- Text-to-image (older methods)

### 7. Hands-on Implementation Tips
**Architecture**:
```python
# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        self.fc = nn.Linear(latent_dim, 256*4*4)
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
```

**Training Tips**:
- Use learning rate 0.0002
- Adam with β1=0.5
- Label smoothing for real labels
- Two timescale update rule (D faster than G)

### 8. Current State and Future
**GANs Strengths**:
- High-quality, sharp images
- Fast sampling
- Good for specific domains (faces)

**Limitations**:
- Training instability
- Mode collapse
- Limited diversity

**Replaced by Diffusion** for many tasks:
- Better diversity
- More stable training
- State-of-the-art quality

**Still Relevant**:
- Real-time applications
- Domain-specific generation
- Discriminator useful for other tasks

## Important Takeaways
1. DCGAN established architectural best practices
2. Conditional GANs enable controlled generation
3. WGAN uses Wasserstein distance for stable training
4. Spectral normalization ensures Lipschitz constraint
5. Progressive training enables high-resolution generation
6. GANs have wide applications in image generation
7. Diffusion models have largely superseded GANs

