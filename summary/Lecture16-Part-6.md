# Lecture 16 - Part 6: Advanced Techniques

## Overview
This part covers advanced diffusion model techniques including latent diffusion and noise schedule improvements.

## Key Topics

### 1. Latent Diffusion (Stable Diffusion)
**Problem**: Diffusion in pixel space is expensive
- 512×512×3 = 786K dimensions
- Memory and compute intensive

**Solution**: Diffusion in latent space
1. Train VAE encoder/decoder
2. Compress image: 512×512 → 64×64 latent
3. Run diffusion in latent space
4. Decode back to pixels

**Architecture**:
```
Image → VAE Encoder → Latent → Diffusion ← Text
Latent' → VAE Decoder → Image'
```

**Benefits**:
- 4-8× spatial compression
- Much faster training/inference
- Similar quality

### 2. VAE for Latent Diffusion
**Training**:
- Reconstruct images
- KL regularization
- Perceptual loss for quality

**Latent Space Properties**:
- Semantically meaningful
- Smooth interpolation
- Lower dimensional

### 3. Noise Schedule Improvements
**Linear Schedule** (Original):
```
βₜ = β₁ + (β_T - β₁) * t / T
```
- Simple but not optimal

**Cosine Schedule**:
```
ᾱₜ = cos²((t/T + s)/(1+s) * π/2)
```
- Smoother noise addition
- Better image quality

**Learned Schedule**:
- Optimize schedule during training
- Task-specific adaptation

### 4. Other Improvement Techniques
**v-Prediction**:
- Predict velocity instead of noise
- v = √ᾱₜ * ε - √(1-ᾱₜ) * x₀
- More stable training

**Offset Noise**:
- Add global noise offset
- Better very dark/bright images

**Min-SNR Weighting**:
- Weight loss by signal-to-noise ratio
- Balance learning across timesteps

### 5. ControlNet
**Concept**: Add spatial control to diffusion
```
Frozen Stable Diffusion + Trainable Copy
                    ↓
              Control signal (edge, pose, depth)
```

**Applications**:
- Edge-guided generation
- Pose-controlled people
- Depth-conditioned scenes
- Scribble to image

## Important Takeaways
1. Latent diffusion dramatically reduces computation
2. VAE compresses images to efficient latent space
3. Noise schedules affect quality significantly
4. v-prediction can improve training stability
5. ControlNet adds fine-grained spatial control
6. Many techniques combine for state-of-the-art models

