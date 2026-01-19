# Lecture 11 - Part 4: Deep Learning for Sequences

## Overview
This part covers deep learning approaches for sequence modeling, including CNNs, RNNs, LSTMs, GRUs, and bidirectional models.

## Key Topics

### 1. High-Dimensional Sequences Challenge
- **Problem**: Traditional methods can't handle:
  - High-dimensional inputs (images, text)
  - Complex non-linear patterns
  - Very long sequences
- **Solution**: Deep learning with automatic feature learning

### 2. CNN for Sequences
- **1D Convolution**: Sliding filters over sequence
  ```
  y[t] = Σ w[k] × x[t+k]
  ```
- **Benefits**:
  - Captures local patterns
  - Parameter sharing
  - Parallel computation
- **Architecture**: Conv1D → Pool → Conv1D → ... → FC
- **Use Cases**: Text classification, signal processing
- **Limitation**: Fixed receptive field

### 3. RNN (Recurrent Neural Network)
- **Core Idea**: Hidden state carries information through time
- **Formula**:
  ```
  hₜ = tanh(Wₕₕhₜ₋₁ + Wₓₕxₜ + b)
  yₜ = Wₕᵧhₜ
  ```
- **Unrolled View**: Chain of repeated modules
- **Benefits**:
  - Variable length sequences
  - Memory of past
  - Parameter sharing across time
- **Limitations**:
  - Vanishing/exploding gradients
  - Difficulty with long-range dependencies
  - Sequential computation (slow)

### 4. LSTM (Long Short-Term Memory)
- **Key Innovation**: Cell state + gating mechanism
- **Three Gates**:
  1. **Forget Gate**: What to discard
     ```
     fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
     ```
  2. **Input Gate**: What to store
     ```
     iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
     C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
     ```
  3. **Output Gate**: What to output
     ```
     oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
     hₜ = oₜ × tanh(Cₜ)
     ```
- **Cell State Update**:
  ```
  Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ
  ```
- **Benefits**:
  - Long-range dependencies
  - Mitigates vanishing gradient
  - State-of-the-art for many years

### 5. GRU (Gated Recurrent Unit)
- **Simplified LSTM**: Two gates instead of three
- **Gates**:
  1. **Reset Gate**:
     ```
     rₜ = σ(Wr·[hₜ₋₁, xₜ])
     ```
  2. **Update Gate**:
     ```
     zₜ = σ(Wz·[hₜ₋₁, xₜ])
     ```
- **Hidden State**:
  ```
  h̃ₜ = tanh(W·[rₜ×hₜ₋₁, xₜ])
  hₜ = (1-zₜ)×hₜ₋₁ + zₜ×h̃ₜ
  ```
- **Comparison to LSTM**:
  - Fewer parameters
  - Faster training
  - Similar performance
  - No separate cell state

### 6. Bidirectional RNNs
- **Concept**: Process sequence in both directions
- **Architecture**:
  ```
  Forward:  h→ₜ = RNN→(xₜ, h→ₜ₋₁)
  Backward: h←ₜ = RNN←(xₜ, h←ₜ₊₁)
  Output:   hₜ = [h→ₜ; h←ₜ]
  ```
- **Benefits**:
  - Access to future context
  - Better for classification tasks
- **Limitation**: Cannot be used for real-time/streaming
- **Use Cases**: NLP, speech recognition

## Important Takeaways
1. CNNs capture local patterns efficiently but have limited receptive field
2. RNNs have memory but suffer from vanishing gradients
3. LSTMs use gating to handle long-range dependencies
4. GRUs simplify LSTMs with similar performance
5. Bidirectional processing improves context understanding
6. Choice depends on task: CNN for local, LSTM/GRU for long-range
7. Modern Transformers are replacing RNNs in many applications

