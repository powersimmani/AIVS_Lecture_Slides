# Lecture 12 - Part 3: Sequence-to-Sequence (Seq2Seq)

## Overview
This part introduces the encoder-decoder architecture for handling sequence-to-sequence tasks where input and output lengths differ.

## Key Topics

### 1. The Need for Seq2Seq
- **Problem**: Different input/output lengths
  - Translation: "Hello" → "Bonjour"
  - Summarization: Long document → Short summary
  - Question answering: Question → Answer
- **Cannot Use Standard RNN**:
  - Assumes same length input/output
  - No natural way to handle length difference
- **Solution**: Encoder-Decoder architecture

### 2. Encoder-Decoder Architecture
- **Two-Stage Process**:
  1. **Encoder**: Read and understand input
  2. **Decoder**: Generate output from understanding
- **Information Flow**:
  ```
  Input → Encoder → Context Vector → Decoder → Output
  ```
- **Key Concept**: Separate reading from generation

### 3. Encoder Details
- **Function**: Process input sequence, create representation
- **Architecture**: RNN/LSTM/GRU (often bidirectional)
  ```
  h₁, h₂, ..., hₙ = Encoder(x₁, x₂, ..., xₙ)
  context = hₙ (or processed version)
  ```
- **Output Options**:
  - Last hidden state: context = hₙ
  - Concatenate last states: context = [h→ₙ; h←₁]
  - Average all states: context = mean(h₁, ..., hₙ)

### 4. Decoder Details
- **Function**: Generate output sequence from context
- **Initialization**: From encoder's context vector
  ```
  s₀ = context
  ```
- **Generation Process**:
  ```
  y₁ = Decoder(s₀, <START>)
  y₂ = Decoder(s₁, y₁)
  ...
  yₜ = Decoder(sₜ₋₁, yₜ₋₁)
  ```
- **Termination**: Stop at <END> token or max length

### 5. Overall Seq2Seq Process
1. **Tokenize**: Convert input to token IDs
2. **Encode**: Process source, get context
   ```python
   encoder_outputs, hidden = encoder(source)
   ```
3. **Initialize Decoder**: Set state from context
   ```python
   decoder_hidden = hidden
   ```
4. **Decode Loop**:
   ```python
   input_token = <START>
   outputs = []
   while input_token != <END>:
       output, decoder_hidden = decoder(input_token, decoder_hidden)
       input_token = output.argmax()
       outputs.append(input_token)
   ```

### 6. Training vs Inference
**Training**:
- Input: Ground truth tokens (teacher forcing)
- Parallel: All timesteps computed together
- Loss: Cross-entropy on each position

**Inference**:
- Input: Own predicted tokens
- Sequential: One token at a time
- Strategy: Greedy or beam search

### 7. Context Vector Bottleneck
- **Problem**: All information in single fixed vector
- **Consequences**:
  - Information loss for long sequences
  - Early input tokens poorly represented
  - Decoder sees compressed view only
- **Solution**: Attention mechanism (next part)

## Important Takeaways
1. Seq2Seq handles variable input/output lengths
2. Encoder reads and compresses input
3. Decoder generates output from context
4. Context vector is the information bridge
5. This architecture enabled machine translation breakthrough
6. Bottleneck limitation led to attention mechanism

