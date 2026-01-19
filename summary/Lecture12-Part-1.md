# Lecture 12 - Part 1: Introduction & Review

## Overview
This part reviews previous lecture content, introduces today's learning objectives, and explores the limitations of standard RNNs.

## Key Topics

### 1. Review of Last Lesson
- **Sequence Modeling Basics**: Why specialized approaches needed
- **RNN Architecture**: Hidden state carries information through time
- **LSTM/GRU**: Gating mechanisms to handle long-range dependencies
- **Key Concepts**:
  - Vanishing/exploding gradients
  - Sequential computation constraints
  - Variable length processing

### 2. Today's Learning Objectives
- Understand bidirectional RNN architecture
- Learn encoder-decoder (Seq2Seq) framework
- Master teacher forcing training strategy
- Comprehend attention mechanism fundamentals
- Apply practical implementation techniques

### 3. Exploring the Limitations of RNNs
- **Unidirectional Processing**: Only sees past context
  - "I saw a bat" - animal or sports equipment?
  - Context from future tokens often critical
- **Information Bottleneck**:
  - Context compressed to single vector
  - Long sequences lose early information
- **Long-range Dependencies**: Still challenging despite LSTM/GRU
- **Sequential Computation**:
  - Cannot parallelize training
  - Slow for long sequences
- **Fixed Input/Output Length**: Standard RNN assumes same length

### 4. Motivation for This Lecture
- **Bidirectional**: Access both past and future context
- **Seq2Seq**: Handle variable input/output lengths
- **Attention**: Overcome information bottleneck
- **Practical Skills**: Implement these architectures effectively

## Important Takeaways
1. RNNs have fundamental limitations for certain tasks
2. Unidirectional processing misses important context
3. Fixed context vector creates bottleneck
4. This lecture introduces solutions: BiRNN, Seq2Seq, Attention
5. These concepts are foundation for modern NLP architectures

