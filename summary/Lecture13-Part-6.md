# Lecture 13 - Part 6: Implementation Tips

## Overview
This part covers practical implementation considerations for Transformers including masking, training stabilization, and hyperparameters.

## Key Topics

### 1. Masking Implementation
**Padding Mask**:
```python
# Mask padded positions
padding_mask = (input_ids == pad_token_id)  # [batch, seq_len]

# Expand for attention: [batch, 1, 1, seq_len]
padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
```

**Causal Mask (Decoder)**:
```python
seq_len = target.size(1)
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, device=target.device),
    diagonal=1
).bool()
# Upper triangular = True (masked)
```

**Applying Masks**:
```python
# Set masked positions to -inf before softmax
scores = scores.masked_fill(mask, float('-inf'))
weights = F.softmax(scores, dim=-1)
```

### 2. Learning Stabilization Techniques
**Learning Rate Schedule (Warmup)**:
```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```
- Warmup: Gradually increase LR (4000 steps typical)
- Then: Inverse square root decay
- Prevents early training instability

**Implementation**:
```python
def get_lr(step, d_model, warmup_steps):
    return d_model ** (-0.5) * min(
        step ** (-0.5),
        step * warmup_steps ** (-1.5)
    )
```

**Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Dropout Placement**:
- After attention weights
- After FFN activation
- After embedding
- After residual addition (before LayerNorm in Post-LN)

**Label Smoothing**:
```python
# Instead of [0, 0, 1, 0, 0]
# Use [0.02, 0.02, 0.92, 0.02, 0.02]
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 3. Hyperparameter Guide
| Parameter | Base | Large | Description |
|-----------|------|-------|-------------|
| d_model | 512 | 1024 | Hidden dimension |
| d_ff | 2048 | 4096 | FFN inner dim |
| num_heads | 8 | 16 | Attention heads |
| num_layers | 6 | 12+ | Encoder/decoder layers |
| dropout | 0.1 | 0.1 | Dropout rate |
| warmup_steps | 4000 | 4000 | LR warmup steps |
| batch_size | 25k tokens | 25k+ | Tokens per batch |

**Scaling Rules**:
- d_ff = 4 × d_model
- d_k = d_v = d_model / num_heads
- More parameters → more data needed

### 4. Common Implementation Issues
**Memory Issues**:
- O(n²) attention for sequence length n
- Use gradient checkpointing
- Reduce batch size or sequence length
- Use mixed precision

**Numerical Stability**:
- Use float32 for attention scores
- Scale scores before softmax
- Careful with mask values (-1e9 not -inf)

**Common Bugs**:
- Forgetting positional encoding
- Wrong mask dimensions
- Not shifting targets for teacher forcing
- Incorrect attention score scaling

### 5. Weight Initialization
**Linear Layers**:
```python
nn.init.xavier_uniform_(self.linear.weight)
nn.init.zeros_(self.linear.bias)
```

**Embeddings**:
```python
nn.init.normal_(self.embedding.weight, mean=0, std=d_model ** -0.5)
```

**Output Projection**: Often tied to input embeddings

## Important Takeaways
1. Proper masking is critical for correct behavior
2. Learning rate warmup essential for stable training
3. Gradient clipping prevents explosion
4. Label smoothing improves generalization
5. Standard hyperparameters follow clear patterns
6. Memory is the main implementation bottleneck
7. Many subtle details affect training success

