# Lecture 16 - Part 5: Architecture

## Overview
This part covers the neural network architectures used in diffusion models, primarily U-Net and attention mechanisms.

## Key Topics

### 1. U-Net Structure
**Why U-Net?**
- Input and output same size
- Skip connections preserve details
- Multi-scale processing

**Architecture**:
```
Encoder (downsample):
[64] → [128] → [256] → [512]

Bottleneck:
[512]

Decoder (upsample):
[512] → [256] → [128] → [64]

Skip connections: encoder → decoder at each scale
```

### 2. U-Net Components
**Residual Blocks**:
```python
class ResBlock(nn.Module):
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)  # Time conditioning
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h  # Residual
```

**Time Embedding**:
```python
t_emb = sinusoidal_embedding(t)
t_emb = MLP(t_emb)  # [batch, dim]
```
- Sinusoidal like positional encoding
- Project to residual block dimensions

### 3. Attention Mechanism
**Self-Attention in U-Net**:
- Applied at lower resolutions (computation)
- Captures global dependencies
- Often at 16×16, 8×8 levels

**Implementation**:
```python
class SelfAttention(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, H*W)
        k = self.key(x).view(B, C, H*W)
        v = self.value(x).view(B, C, H*W)

        attn = (q.transpose(-2,-1) @ k) / sqrt(C)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v.transpose(-2,-1)).view(B, C, H, W)
        return out
```

### 4. Condition Injection Methods
**Time Conditioning**:
- Add to residual blocks
- Scale and shift features
- AdaGN (Adaptive Group Norm)

**Class Conditioning**:
- Embedding lookup
- Add to time embedding
- Or separate conditioning path

**Text Conditioning** (Cross-attention):
```python
Q = self.query(image_features)
K = self.key(text_features)
V = self.value(text_features)
attention = softmax(Q @ K.T / √d) @ V
```

### 5. Modern Architecture Improvements
**DiT (Diffusion Transformer)**:
- Replace U-Net with Transformer
- Patch-based processing
- Scales better with compute

**Efficient U-Net**:
- Fewer channels at high resolution
- More at low resolution
- Better compute allocation

## Important Takeaways
1. U-Net provides multi-scale processing with skip connections
2. Time embedding conditions all layers
3. Attention captures global dependencies
4. Cross-attention enables text conditioning
5. Condition injection through various methods
6. Transformer architectures emerging as alternative

