# Lecture 12 - Part 5: Attention Mechanism

## Overview
This part introduces the attention mechanism, which revolutionized sequence-to-sequence learning by addressing the fixed context vector bottleneck.

## Key Topics

### 1. Limitations of Basic Seq2Seq
- **Fixed Context Vector**: All information in one vector
- **Long Sequences**: Early information lost
- **No Alignment**: Decoder doesn't know which input parts are relevant
- **Bottleneck**: Encoder must compress everything

### 2. Core Ideas of Attention
- **Key Insight**: Decoder should look at relevant parts of input
- **Instead of**: One fixed context for all steps
- **Do**: Dynamic context, different for each decoder step
- **Intuition**: Like human attention - focus on relevant parts

### 3. Attention Mechanism Structure
```
Query (Q):  Current decoder state sₜ
Keys (K):   All encoder hidden states [h₁, h₂, ..., hₙ]
Values (V): Same encoder hidden states

Process:
1. Compare query with each key → scores
2. Normalize scores → attention weights (αᵢ)
3. Weighted sum of values → context vector
```

### 4. Attention Formula
**Step 1: Compute Attention Scores**
```
eᵢ = score(sₜ, hᵢ)  for all encoder positions i
```

**Step 2: Normalize with Softmax**
```
αᵢ = softmax(eᵢ) = exp(eᵢ) / Σⱼ exp(eⱼ)
```

**Step 3: Compute Context Vector**
```
cₜ = Σᵢ αᵢ × hᵢ
```

**Step 4: Combine for Output**
```
output = f([sₜ; cₜ])  # Concatenate decoder state and context
```

### 5. Attention Score Functions
**Dot Product (Simple)**:
```
score(s, h) = sᵀh
```

**Scaled Dot Product (Transformer uses this)**:
```
score(s, h) = sᵀh / √d
```

**Additive / Bahdanau (Original)**:
```
score(s, h) = vᵀ tanh(Ws·s + Wh·h)
```

**Multiplicative / Luong**:
```
score(s, h) = sᵀWh
```

### 6. Effects of Attention
**Performance**:
- Significant improvement on long sequences
- Better handling of rare words
- More accurate translations

**Interpretability**:
- Attention weights show what model focuses on
- Visualize alignment between input/output
- Debug and understand model behavior

**Flexibility**:
- Different context for each output position
- Learns soft alignment automatically
- Handles word reordering naturally

### 7. Attention Visualization
```
Source: The cat sat on the mat
Target: Le chat était assis sur le tapis

Attention Matrix:
        The   cat   sat   on   the   mat
Le     [0.9  0.0   0.0  0.0  0.0   0.1]
chat   [0.0  0.9   0.0  0.0  0.0   0.1]
était  [0.0  0.1   0.8  0.1  0.0   0.0]
...
```

## Important Takeaways
1. Attention addresses the fixed context bottleneck
2. Each decoder step gets a dynamically computed context
3. Context is weighted sum of all encoder states
4. Multiple score functions exist with similar performance
5. Attention weights provide interpretability
6. This was a major breakthrough leading to Transformers
7. Same Q-K-V concept extends to self-attention

