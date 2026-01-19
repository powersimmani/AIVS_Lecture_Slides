# Lecture 13 - Part 3: Multi-Head Attention

## Overview
This part explains multi-head attention, which allows the model to attend to information from different representation subspaces.

## Key Topics

### 1. Why is Multi-Head Necessary?
**Single Head Limitation**:
- One set of Q, K, V projections
- Captures only one type of relationship
- May miss important patterns

**Different Types of Relationships**:
- Syntactic: Subject-verb agreement
- Semantic: Word meanings
- Positional: Nearby words
- Coreference: Pronoun resolution

**Solution**: Multiple attention heads in parallel

### 2. Multi-Head Attention Architecture
**Concept**:
- Run multiple attention operations in parallel
- Each head has its own Q, K, V projections
- Concatenate all head outputs
- Final linear projection

**Formula**:
```
head_i = Attention(QWᵢQ, KWᵢK, VWᵢV)
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × WO
```

### 3. Multi-Head Parameters
**Typical Configuration**:
- h = 8 heads (or 16 for larger models)
- d_model = 512 (total dimension)
- d_k = d_v = d_model / h = 64 per head

**Parameter Matrices**:
- WᵢQ: [d_model × d_k] for each head
- WᵢK: [d_model × d_k] for each head
- WᵢV: [d_model × d_v] for each head
- WO: [h × d_v × d_model] output projection

### 4. Multi-Head Operation Example
**Configuration**:
- d_model = 512
- h = 8 heads
- d_k = d_v = 64

**Process**:
1. Project input to 8 sets of Q, K, V (each 64-dim)
2. Compute 8 attention outputs (each 64-dim)
3. Concatenate: 8 × 64 = 512
4. Project output: 512 → 512

**Efficiency**:
- Same total computation as one 512-dim head
- But captures 8 different attention patterns

### 5. What Different Heads Learn
**Empirical Observations**:
- Some heads: Local attention (nearby words)
- Some heads: Syntactic patterns
- Some heads: Rare word attention
- Some heads: Positional patterns

**Visualization**: Attention patterns vary dramatically between heads

### 6. Multi-Head Attention Implementation
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        # Concatenate and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(output)
```

### 7. Implementation Points
**Efficient Computation**:
- Single matrix multiply for all heads
- Reshape instead of separate computations
- GPU-efficient implementation

**Key Dimensions**:
- Input: [batch, seq_len, d_model]
- Reshaped: [batch, num_heads, seq_len, d_k]
- Attention: [batch, num_heads, seq_len, seq_len]
- Output: [batch, seq_len, d_model]

## Important Takeaways
1. Multi-head attention captures different relationship types
2. Each head has independent Q, K, V projections
3. Heads operate in parallel (same computational cost)
4. Outputs are concatenated and projected
5. Different heads learn different attention patterns
6. Standard: 8 heads for base, 16+ for larger models
7. Per-head dimension = model_dim / num_heads

