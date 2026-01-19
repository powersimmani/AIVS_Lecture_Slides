# Lecture 16 - Part 3: Architecture and Advanced Techniques

## Overview
This part covers the neural network architecture for diffusion models, including U-Net, attention mechanisms, and advanced techniques like latent diffusion.

## Key Topics

### 1. U-Net Structure
**Architecture**:
```
Input → Encoder → Bottleneck → Decoder → Output
         ↓_______skip connections_______↑
```

**Components**:
- **Encoder**: Downsample, increase channels
- **Decoder**: Upsample, decrease channels
- **Skip Connections**: Preserve spatial details
- **Timestep Embedding**: Injected throughout

### 2. U-Net for Diffusion
**Modifications**:
- Residual blocks
- Self-attention at multiple resolutions
- Group normalization
- Timestep conditioning

**Timestep Embedding**:
```python
t_emb = sinusoidal_embedding(t)
t_emb = MLP(t_emb)  # Project to model dim
# Add to each residual block
```

### 3. Attention in Diffusion Models
**Self-Attention**: Capture long-range dependencies
```
Attention in 16×16 and 8×8 feature maps
```

**Cross-Attention**: For conditioning (text, etc.)
```
Q from image features
K, V from text embeddings
```

**Placement**: Middle layers (lower resolutions)

### 4. Condition Injection Methods
**Concatenation**: Add condition to input
```
[xₜ; condition] → UNet
```

**Addition**: Add to timestep embedding
```
t_emb + condition_emb → residual blocks
```

**Cross-Attention**: Most powerful
```
Image features attend to condition embeddings
```

**Text Conditioning**: CLIP or T5 text encoder

### 5. Latent Diffusion (Stable Diffusion)
**Problem**: Pixel-space diffusion is expensive
**Solution**: Diffusion in latent space

**Architecture**:
```
Image → VAE Encoder → Latent z₀
z₀ → Diffusion in latent space → z'₀
z'₀ → VAE Decoder → Image
```

**Benefits**:
- 4-8× spatial compression
- Much faster training/sampling
- Similar quality

### 6. Latent Diffusion Components
**VAE**:
- Encoder: Image → Latent (64×64 → 8×8×4)
- Decoder: Latent → Image

**U-Net**: Operates on latents only

**Text Encoder**: CLIP for text conditioning

**Stable Diffusion**:
- Open-source, widely used
- 860M U-Net parameters
- Trained on LAION-5B

### 7. Noise Schedule Improvements
**Linear Schedule** (Original):
```
βₜ = β₁ + (t-1)/(T-1) × (βₜ - β₁)
```

**Cosine Schedule** (Improved):
```
ᾱₜ = cos²((t/T + s)/(1+s) × π/2)
```
- Less noise early
- More gradual transition
- Better image quality

**v-prediction**: Alternative parameterization
```
v = √ᾱₜε - √(1-ᾱₜ)x₀
```

### 8. Other Improvements
**Classifier-Free Guidance**: Standard for conditional generation

**Negative Prompts**: Guide away from unwanted attributes

**ControlNet**: Add spatial control (pose, edges, depth)

**LoRA**: Efficient fine-tuning for styles

**Textual Inversion**: Learn new concepts from few images

### 9. Applications and Extensions
**Text-to-Image**:
- Stable Diffusion
- DALL-E 2/3
- Midjourney
- Imagen

**Image Editing**:
- Inpainting
- Outpainting
- SDEdit

**Other Modalities**:
- Video generation
- 3D generation
- Audio synthesis

**Current State**:
- Best image generation quality
- Very flexible and controllable
- Active research area

## Important Takeaways
1. U-Net with skip connections is the standard architecture
2. Attention captures long-range spatial relationships
3. Cross-attention enables text conditioning
4. Latent diffusion provides massive speedup
5. Cosine noise schedule improves quality
6. Classifier-free guidance is essential for good results
7. Many extensions enable precise control over generation

