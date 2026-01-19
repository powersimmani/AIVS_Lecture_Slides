# Lecture 12 - Part 6: Practical Implementation

## Overview
This part covers practical considerations for implementing sequence-to-sequence models, including batching, masking, and best practices.

## Key Topics

### 1. Batching for Sequences
- **Challenge**: Variable length sequences in a batch
- **Solution**: Padding + Masking
- **Process**:
  1. Determine max length in batch
  2. Pad shorter sequences with <PAD> token
  3. Create mask to track real vs padded positions

### 2. Padding Strategies
- **Right Padding** (most common):
  ```
  [the, cat, sat, <PAD>, <PAD>]
  ```
- **Left Padding** (for some decoders):
  ```
  [<PAD>, <PAD>, the, cat, sat]
  ```
- **Dynamic Padding**: Pad to batch's max length
- **Fixed Padding**: Pad to global max (wasteful)

### 3. Masking Types
**Padding Mask**: Ignore padded positions
```python
mask = (input_ids != pad_token_id)  # [batch, seq_len]
```

**Attention Mask**: Prevent attending to invalid positions
```python
# Set attention scores to -inf before softmax
scores = scores.masked_fill(mask == 0, -1e9)
```

**Causal Mask**: Prevent looking at future (decoder)
```
[1, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 0]
[1, 1, 1, 1]
```

### 4. Applying Masks
**In Loss Computation**:
```python
loss = criterion(predictions, targets)  # Per-token loss
masked_loss = (loss * mask).sum() / mask.sum()  # Only real tokens
```

**In Attention**:
```python
scores = query @ keys.T  # Attention scores
scores = scores.masked_fill(mask == 0, -1e9)  # Mask padding
weights = F.softmax(scores, dim=-1)  # Safe softmax
```

### 5. Bucketing and Sorting
- **Problem**: Padding wastes computation
- **Bucketing**: Group similar-length sequences
- **Sorting**: Sort by length within batch
```python
# PyTorch packed sequences
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

packed = pack_padded_sequence(x, lengths, enforce_sorted=False)
output, _ = lstm(packed)
output, _ = pad_packed_sequence(output)
```

### 6. Practical Checklist
**Data Preparation**:
- [ ] Define vocabulary with special tokens
- [ ] Proper tokenization strategy
- [ ] Handle out-of-vocabulary words
- [ ] Filter very long sequences

**Model Setup**:
- [ ] Embedding layer with padding_idx
- [ ] Proper weight initialization
- [ ] Gradient clipping (max_norm=1.0)

**Training**:
- [ ] Masking in loss computation
- [ ] Teacher forcing schedule
- [ ] Learning rate warmup
- [ ] Early stopping on validation

**Inference**:
- [ ] Beam search implementation
- [ ] Length normalization
- [ ] Repetition penalty (if needed)

## Important Takeaways
1. Proper padding and masking are essential
2. Multiple mask types serve different purposes
3. Bucketing improves training efficiency
4. Many subtle details affect performance
5. Debugging sequence models requires specific strategies
6. These concepts transfer directly to Transformers

