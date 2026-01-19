# Lecture 13 - Part 5: Transformer Architecture

## Overview
This part covers the complete Transformer architecture including encoder, decoder, and how all components work together.

## Key Topics

### 1. Overall Transformer Structure
```
ENCODER:
Source Input → Embedding + Positional Encoding
            → [Encoder Layer] × N
            → Encoder Output

DECODER:
Target Input → Embedding + Positional Encoding
            → [Decoder Layer] × N (with encoder attention)
            → Linear → Softmax → Output Probabilities
```

### 2. Encoder Architecture
**Single Encoder Layer**:
```
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm (Residual + LayerNorm)
  ↓
Feed-Forward Network
  ↓
Add & Norm (Residual + LayerNorm)
  ↓
Output
```

**Stack**: N identical layers (typically N=6)

### 3. Detailed Analysis of the Encoder
**Self-Attention**:
- Bidirectional: Each position attends to all positions
- Captures dependencies regardless of distance
- Computes representations with global context

**Feed-Forward Network (FFN)**:
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
      = ReLU(xW₁ + b₁)W₂ + b₂
```
- Applied to each position independently
- Expands then contracts dimension
- d_ff = 4 × d_model (typical)

**Residual Connections**:
```
output = LayerNorm(x + SubLayer(x))
```
- Enable gradient flow through deep networks
- Allow model to learn identity mappings

### 4. Decoder Architecture
**Single Decoder Layer**:
```
Input
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Multi-Head Cross-Attention (to encoder)
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output
```

### 5. Detailed Analysis of the Decoder
**Masked Self-Attention**:
- Causal masking: Cannot see future positions
- Position i can only attend to positions ≤ i
- Ensures autoregressive property

**Causal Mask Matrix**:
```
[1, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 0]
[1, 1, 1, 1]
```

**Cross-Attention**:
- Query: From decoder
- Key, Value: From encoder output
- Connects decoder to source sequence
- Each decoder position attends to entire source

### 6. Feed-Forward Network & Layer Normalization
**FFN Details**:
```python
def ffn(x):
    hidden = F.relu(self.linear1(x))  # d_model → d_ff
    output = self.linear2(hidden)      # d_ff → d_model
    return output
```
- Modern variant uses GELU instead of ReLU
- Applied position-wise (same for each position)

**Layer Normalization**:
```
LayerNorm(x) = γ × (x - μ) / (σ + ε) + β
```
- μ, σ: Mean and std across features (not batch)
- γ, β: Learned scale and shift

**Pre-LN vs Post-LN**:
- Original (Post-LN): Normalize after residual add
- Modern (Pre-LN): Normalize before sublayer

### 7. Training vs Inference
**Training**:
- Teacher forcing: Use ground truth target
- All decoder positions computed in parallel
- Single forward pass for entire sequence

**Inference**:
- Autoregressive: Generate one token at a time
- Use previous predictions as input
- KV-cache: Store computed K, V for efficiency

**KV-Cache**:
```python
# Avoid recomputing K, V for previous positions
if cached_kv is not None:
    K = torch.cat([cached_kv['K'], new_K], dim=1)
    V = torch.cat([cached_kv['V'], new_V], dim=1)
```

### 8. Complete Forward Pass
**Encoder**:
1. Embed source tokens
2. Add positional encoding
3. Pass through N encoder layers
4. Output: Encoder representations

**Decoder**:
1. Embed target tokens (shifted right)
2. Add positional encoding
3. Pass through N decoder layers (with cross-attention)
4. Linear projection to vocabulary size
5. Softmax for probabilities

## Important Takeaways
1. Encoder uses bidirectional self-attention
2. Decoder uses masked (causal) self-attention
3. Cross-attention connects decoder to encoder
4. FFN adds non-linearity at each position
5. Residual connections enable deep stacking
6. LayerNorm stabilizes training
7. Training is parallel, inference is sequential

