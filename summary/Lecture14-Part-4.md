# Lecture 14 - Part 4: GPT - Decoder-based Models

## Overview
This part covers GPT and decoder-only models, which excel at text generation and have led to modern large language models.

## Key Topics

### 1. Introduction to GPT Series
**GPT**: Generative Pre-trained Transformer (OpenAI, 2018)

**Key Characteristics**:
- Decoder-only architecture
- Autoregressive language modeling
- Left-to-right text generation

### 2. GPT Pre-training
**Objective**: Predict next token
```
P(xₜ | x₁, x₂, ..., xₜ₋₁)
Loss = -Σ log P(xₜ | x<t)
```

**Training**:
- No masks needed (causal attention built-in)
- Simple objective, powerful results
- Natural for generation tasks

**Architecture**:
- Transformer decoder stack
- Causal (masked) self-attention
- No cross-attention (no encoder)

### 3. GPT-3 and Few-shot Learning
**GPT-3 Scale** (2020):
- 175 billion parameters
- 300 billion training tokens
- 96 attention layers

**Few-shot Learning**:
```
Translate English to French:
sea otter → loutre de mer
peppermint → menthe poivrée
cheese →
```
- No gradient updates
- Just provide examples in prompt
- Model generalizes pattern

**Types of In-Context Learning**:
- **Zero-shot**: Task description only
- **One-shot**: One example
- **Few-shot**: Multiple examples

### 4. GPT Family Development
| Model | Year | Parameters | Key Feature |
|-------|------|------------|-------------|
| GPT | 2018 | 117M | Generative pre-training |
| GPT-2 | 2019 | 1.5B | Zero-shot capabilities |
| GPT-3 | 2020 | 175B | Few-shot learning |
| InstructGPT | 2022 | 175B | RLHF alignment |
| GPT-4 | 2023 | ~1.7T? | Multimodal, reasoning |

### 5. GPT vs BERT
| Aspect | GPT | BERT |
|--------|-----|------|
| Architecture | Decoder | Encoder |
| Direction | Left-to-right | Bidirectional |
| Objective | Next token | Masked token |
| Strength | Generation | Understanding |
| Use Case | Text generation, chat | Classification, NER |

### 6. Modern GPT Applications
**Text Generation**:
- Story writing, creative content
- Code generation (Codex)
- Email drafting

**Conversational AI**:
- ChatGPT
- Customer service bots
- Virtual assistants

**Reasoning Tasks**:
- Chain-of-thought prompting
- Code execution
- Mathematical problem solving

## Important Takeaways
1. GPT uses autoregressive (left-to-right) generation
2. Decoder-only architecture with causal attention
3. Few-shot learning emerged at large scale
4. No fine-tuning needed for many tasks (prompting)
5. GPT-style models dominate text generation
6. Scaling leads to emergent capabilities

