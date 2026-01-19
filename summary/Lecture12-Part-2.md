# Lecture 12 - Part 2: Bidirectional RNN

## Overview
This part covers bidirectional RNN architecture, which processes sequences in both forward and backward directions to capture full context.

## Key Topics

### 1. Why Bidirectional?
- **Motivation**: Many tasks benefit from future context
- **Examples**:
  - "I saw a bat flying" vs "I saw a bat and ball"
  - Named entity recognition: context on both sides
  - Sentiment analysis: whole sentence matters
- **Key Insight**: Meaning often depends on surrounding context

### 2. Bidirectional RNN Architecture
- **Structure**: Two RNNs processing in opposite directions
  ```
  Forward RNN:  h→₁ → h→₂ → h→₃ → h→₄
  Input:        x₁    x₂    x₃    x₄
  Backward RNN: h←₁ ← h←₂ ← h←₃ ← h←₄
  ```
- **Output**: Concatenation at each position
  ```
  hₜ = [h→ₜ; h←ₜ]
  ```
- **Parameters**: Separate weights for each direction

### 3. BiRNN Formulas and Operations
- **Forward Pass**:
  ```
  h→ₜ = f(Wₓ→xₜ + Wₕ→h→ₜ₋₁ + b→)
  ```
- **Backward Pass**:
  ```
  h←ₜ = f(Wₓ←xₜ + Wₕ←h←ₜ₊₁ + b←)
  ```
- **Combination Options**:
  - Concatenation: hₜ = [h→ₜ; h←ₜ] (most common)
  - Addition: hₜ = h→ₜ + h←ₜ
  - Average: hₜ = (h→ₜ + h←ₜ) / 2

### 4. BiRNN Implementation
```python
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        # output: [batch, seq_len, 2*hidden_size]
        return output
```

### 5. BiRNN Pros and Cons
**Advantages**:
- Full context (past and future)
- Better performance on many NLP tasks
- State-of-the-art for sequence labeling

**Disadvantages**:
- Cannot use for real-time/streaming applications
- 2× parameters and computation
- Requires full sequence upfront

### 6. Applications
- **Named Entity Recognition**: [Paris]LOC is the capital
- **Part-of-Speech Tagging**: Identify word roles
- **Sentiment Analysis**: Understand full review
- **Machine Translation Encoder**: Capture source fully
- **Speech Recognition**: Acoustic model

## Important Takeaways
1. BiRNN processes sequences in both directions
2. Concatenates forward and backward hidden states
3. Provides full context at each position
4. Cannot be used for streaming/real-time tasks
5. Standard choice for sequence encoding tasks

