# Lecture 13 - Part 4: Positional Encoding

## Overview
This part explains positional encoding methods that inject sequence order information into the Transformer.

## Key Topics

### 1. The Need for Positional Information
**Problem**: Self-attention is permutation invariant
```
Attention("cat sat mat") = Attention("mat cat sat")
```
- Order doesn't affect attention computation
- "Dog bites man" vs "Man bites dog" would be same!

**In RNNs**: Position implicit from sequential processing
**In Transformers**: Must explicitly inject position

### 2. Positional Encoding Methods
**Two Main Approaches**:
1. **Sinusoidal** (Original Transformer)
2. **Learned** (BERT, GPT)

**Key Requirement**: Each position gets unique representation

### 3. Sinusoidal Positional Encoding
**Formula**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- pos: Position in sequence (0, 1, 2, ...)
- i: Dimension index (0, 1, 2, ... d_model/2)
- d_model: Embedding dimension

**How It Works**:
- Even dimensions: sine function
- Odd dimensions: cosine function
- Different frequencies for different dimensions

### 4. Why Sinusoidal Works
**Properties**:
1. **Unique**: Each position has unique encoding
2. **Bounded**: Values between -1 and 1
3. **Relative Position**: PE(pos+k) can be represented as linear function of PE(pos)
4. **Extrapolation**: Can handle sequences longer than training

**Frequency Intuition**:
- Low dimensions: Slow variation (like days)
- High dimensions: Fast variation (like seconds)
- Combined: Unique timestamp for each position

### 5. Positional Encoding Visualization
```
Position 0: [sin(0), cos(0), sin(0), cos(0), ...]
Position 1: [sin(1/f), cos(1/f), sin(1/f²), cos(1/f²), ...]
Position 2: [sin(2/f), cos(2/f), sin(2/f²), cos(2/f²), ...]
```
- f varies with dimension (10000^(2i/d))
- Creates wave patterns at different frequencies

### 6. Learned Positional Embeddings
**Alternative Approach** (Used in BERT, GPT):
```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)
```
- Trainable embedding for each position
- Learned during training
- Often works as well as sinusoidal

**Pros**:
- Can learn task-specific patterns
- Simple implementation

**Cons**:
- Limited to maximum training length
- Cannot extrapolate to longer sequences

### 7. Adding Positional Encoding
**How to Apply**:
```python
# Word embedding + positional encoding
x = token_embedding(input_ids) + positional_encoding
```

**Why Addition (not Concatenation)?**:
- Preserves dimensionality
- Works empirically
- Efficient computation

### 8. Implementation
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## Important Takeaways
1. Positional encoding is essential for Transformers
2. Without it, model cannot distinguish word order
3. Sinusoidal encoding uses sin/cos at different frequencies
4. Learned embeddings are simpler but limited in length
5. Position info is added to (not concatenated with) embeddings
6. Sinusoidal allows extrapolation to longer sequences
7. Both approaches work well in practice

