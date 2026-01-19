# Lecture 12 - Part 4: Teacher Forcing

## Overview
This part covers teacher forcing, a training strategy for sequence models, along with its problems and solutions.

## Key Topics

### 1. What is Teacher Forcing?
- **Definition**: Using ground truth as decoder input during training
- **Analogy**: Teacher providing correct answers step by step
- **Purpose**: Stabilize and accelerate training

### 2. Teacher Forcing vs Autoregressive
**Autoregressive (Without Teacher Forcing)**:
```
ŷ₁ = Decoder(s₀, <START>)
ŷ₂ = Decoder(s₁, ŷ₁)      # Uses predicted ŷ₁
ŷ₃ = Decoder(s₂, ŷ₂)      # Uses predicted ŷ₂
```

**With Teacher Forcing**:
```
ŷ₁ = Decoder(s₀, <START>)
ŷ₂ = Decoder(s₁, y₁)      # Uses ground truth y₁
ŷ₃ = Decoder(s₂, y₂)      # Uses ground truth y₂
```

### 3. Comparison
| Aspect | Teacher Forcing | Autoregressive |
|--------|----------------|----------------|
| Decoder Input | Ground truth | Own predictions |
| Training Speed | Fast | Slow |
| Training Stability | Very stable | Can diverge |
| Parallelizable | Yes | No |
| Train/Test Match | Mismatch | Perfect match |

### 4. Teacher Forcing Problems
- **Exposure Bias**:
  - Model never sees own mistakes during training
  - At test time, errors are new experience
- **Error Accumulation**:
  - One mistake feeds into next step
  - Errors compound through sequence
- **Distribution Mismatch**:
  - Training: Perfect inputs
  - Testing: Imperfect predictions

### 5. Solutions to Exposure Bias

**1. Scheduled Sampling**:
- Gradually reduce teacher forcing ratio
- Start with 100% ground truth, decrease over time
```python
# Linear decay
teacher_forcing_ratio = max(0.1, 1.0 - epoch * 0.1)

# Exponential decay
teacher_forcing_ratio = 0.9 ** epoch

# During training
if random.random() < teacher_forcing_ratio:
    decoder_input = ground_truth
else:
    decoder_input = predicted
```

**2. Curriculum Learning**:
- Start with easy (short) examples
- Gradually increase difficulty
- Build up model's capability

**3. Beam Search at Inference**:
- Maintain multiple hypotheses
- Reduces impact of single wrong token
- Finds better overall sequences

**4. Reinforcement Learning**:
- REINFORCE algorithm
- Train on actual generated sequences
- Optimize sequence-level metrics (BLEU)

## Important Takeaways
1. Teacher forcing uses ground truth as decoder input
2. Makes training faster and more stable
3. Creates exposure bias: train/test mismatch
4. Scheduled sampling gradually removes teacher forcing
5. Multiple strategies exist to address the problem
6. Understanding this trade-off is crucial for seq2seq training

