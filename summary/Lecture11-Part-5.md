# Lecture 11 - Part 5: Advanced Topics

## Overview
This part covers advanced sequence modeling topics including encoder-decoder architectures, training strategies, and decoding methods.

## Key Topics

### 1. Sequence-to-Sequence Models
- **Problem**: Input and output sequences have different lengths
- **Examples**:
  - Machine translation (English → French)
  - Text summarization (long → short)
  - Speech recognition (audio → text)
- **Solution**: Encoder-Decoder architecture

### 2. Encoder-Decoder Architecture
- **Encoder**: Processes input, produces context vector
  ```
  h₁, h₂, ..., hₙ = Encoder(x₁, x₂, ..., xₙ)
  context = hₙ (or pooled hidden states)
  ```
- **Decoder**: Generates output from context
  ```
  y₁ = Decoder(context, <START>)
  y₂ = Decoder(context, y₁)
  ...
  ```
- **Components**:
  - Encoder RNN/LSTM/GRU
  - Context vector (bottleneck)
  - Decoder RNN/LSTM/GRU
- **Limitation**: Fixed context vector bottleneck

### 3. Teacher Forcing
- **Concept**: Use ground truth as decoder input during training
- **Without Teacher Forcing** (Autoregressive):
  ```
  ŷₜ = Decoder(ŷₜ₋₁)  # Uses own predictions
  ```
- **With Teacher Forcing**:
  ```
  ŷₜ = Decoder(yₜ₋₁)  # Uses ground truth
  ```
- **Benefits**:
  - Faster convergence
  - More stable training
  - Better gradient flow
- **Problem**: Exposure bias (train/test mismatch)
- **Solution**: Scheduled sampling (gradual reduction)

### 4. Beam Search
- **Problem**: Greedy decoding may miss best sequence
- **Greedy**: Pick best token at each step
- **Beam Search**: Maintain k best candidates
- **Algorithm**:
  1. Start with k most likely first tokens
  2. For each candidate, expand with k best next tokens
  3. Keep top k overall sequences
  4. Repeat until end tokens
- **Beam Width (k)**: Typically 4-10
- **Trade-off**: Quality vs computation
- **Scoring**: Often length-normalized log probability

### 5. CTC Loss (Connectionist Temporal Classification)
- **Problem**: Alignment between input and output unknown
- **Use Case**: Speech recognition, OCR
- **Key Idea**: Sum over all valid alignments
- **Blank Token**: Allows repetition handling
  ```
  "hello" could be: "--hh-e-ll-oo--"
  ```
- **Forward-Backward Algorithm**: Efficient computation
- **Benefits**:
  - No pre-segmented data needed
  - End-to-end training

### 6. Practical Implementation Tips
- **Padding**: Handle variable lengths with padding + masking
- **Bucketing**: Group similar lengths for efficiency
- **Gradient Clipping**: Prevent exploding gradients
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
  ```
- **Learning Rate**: Start lower for RNNs
- **Initialization**: Orthogonal for recurrent weights
- **Layer Normalization**: Stabilizes training
- **Dropout**: Apply between layers, not timesteps

### 7. Applications and Next Steps
- **Current Applications**:
  - Machine translation
  - Speech recognition
  - Text summarization
  - Chatbots and dialogue
  - Video captioning
- **Limitations**:
  - Sequential computation (slow)
  - Still struggles with very long sequences
  - Attention mechanism helps (next lecture)
- **Modern Evolution**: Transformers replace RNNs

## Important Takeaways
1. Encoder-decoder handles different input/output lengths
2. Teacher forcing speeds training but creates exposure bias
3. Beam search improves decoding quality over greedy
4. CTC loss enables training without alignment
5. Practical tips crucial for successful training
6. RNN-based seq2seq laid foundation for modern NLP
7. Attention mechanism (next lecture) addresses key limitations

