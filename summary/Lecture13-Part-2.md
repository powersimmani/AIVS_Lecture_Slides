# Lecture 13 - Part 2: Self-Attention Mechanism

## Overview
This part explains the self-attention mechanism, the core innovation of the Transformer architecture.

## Key Topics

### 1. RNN Attention vs Self-Attention
**RNN Attention (Cross-Attention)**:
- Query: Decoder state
- Keys/Values: Encoder states
- Attends across two different sequences
- Still relies on RNN for processing

**Self-Attention**:
- Query, Key, Value: All from same sequence
- Each position attends to all positions in same sequence
- No recurrence needed
- Fully parallelizable

### 2. Query, Key, and Value Concepts
**Intuition - Information Retrieval**:
- **Query (Q)**: What am I looking for?
- **Key (K)**: What information is available? (index)
- **Value (V)**: What is the actual content?

**Computation**:
```
Q = X × WQ
K = X × WK
V = X × WV
```
- X: Input embeddings [n × d]
- WQ, WK, WV: Learned projection matrices

### 3. Self-Attention Computation Process (Step 1-2)
**Step 1: Linear Projections**
```
Q = XWQ  →  [n × dq]
K = XWK  →  [n × dk]
V = XWV  →  [n × dv]
```
- Typically dq = dk for dot product
- dv can be different

**Step 2: Compute Attention Scores**
```
scores = Q × Kᵀ  →  [n × n]
```
- Each position gets score with every other position
- High score = high relevance

### 4. Self-Attention Computation Process (Step 3-4)
**Step 3: Scale and Softmax**
```
scaled_scores = scores / √dk
weights = softmax(scaled_scores)  →  [n × n]
```
- Scaling prevents softmax saturation
- Each row sums to 1

**Step 4: Weighted Sum of Values**
```
output = weights × V  →  [n × dv]
```
- Each position gets weighted combination of all values
- Weights determined by attention scores

### 5. Complete Self-Attention Formula
```
Attention(Q, K, V) = softmax(QKᵀ / √dk) × V
```

**Why Scale by √dk?**:
- Without scaling: Dot products grow with dimension
- Large values → Softmax becomes very peaked
- Peaked softmax → Small gradients
- Scaling keeps variance approximately 1

### 6. Self-Attention Matrix Operations
**Key Properties**:
- **Parallelizable**: All positions computed simultaneously (matrix multiply)
- **Global Context**: Every position sees every other position
- **Quadratic Complexity**: O(n²) for sequence length n
- **No Position Info**: Permutation invariant (problem!)

**Visualization**:
```
Position 1 → attends to → [1, 2, 3, 4, ...]
Position 2 → attends to → [1, 2, 3, 4, ...]
...
```

## Important Takeaways
1. Self-attention computes attention within the same sequence
2. Q, K, V are learned projections of input embeddings
3. Attention score = dot product of query and key
4. Scaling by √dk prevents gradient problems
5. Output is weighted sum of values
6. Fully parallelizable unlike RNNs
7. O(n²) complexity is the main limitation

