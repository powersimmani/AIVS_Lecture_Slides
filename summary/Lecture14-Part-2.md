# Lecture 14 - Part 2: Pre-training Concepts

## Overview
This part explains the fundamentals of pre-training, including objectives, data requirements, and why scale matters.

## Key Topics

### 1. What is Pre-training?
**Definition**: Training a model on a large corpus to learn general representations before task-specific training

**Key Characteristics**:
- Self-supervised: Labels derived from data itself
- Large-scale: Billions of tokens
- General knowledge: Not task-specific

### 2. Language Modeling Objective Functions

**Autoregressive LM (GPT-style)**:
```
P(x₁, x₂, ..., xₙ) = Π P(xᵢ | x₁, ..., xᵢ₋₁)
Loss = -Σ log P(xᵢ | x₁, ..., xᵢ₋₁)
```
- Predict next token given previous
- Left-to-right generation
- Natural for text generation

**Masked LM (BERT-style)**:
```
Randomly mask 15% of tokens
Loss = -Σ log P(xₘₐₛₖₑₐ | context)
```
- Predict masked tokens from context
- Bidirectional understanding
- Better for classification/NLU

**Denoising (T5-style)**:
```
Corrupt input text (drop, shuffle, mask spans)
Loss = Reconstruct original
```
- Text-to-text framework
- Unified approach for all tasks

### 3. Pre-training Data
**Common Corpora**:
- Books (BookCorpus, Books3)
- Web text (Common Crawl, C4)
- Wikipedia
- Code (GitHub)
- Scientific papers

**Data Size Examples**:
| Model | Training Tokens |
|-------|-----------------|
| BERT | 3.3B |
| GPT-2 | 8B |
| GPT-3 | 300B |
| LLaMA | 1.4T |

### 4. The Importance of Scale
**Scaling Laws** (Kaplan et al., 2020):
- Performance improves predictably with:
  - More parameters (N)
  - More data (D)
  - More compute (C)

**Empirical Observations**:
- Larger models are more sample-efficient
- Emergent abilities appear at certain scales
- Compute-optimal scaling exists

### 5. Pre-training Infrastructure
**Requirements**:
- Massive compute (thousands of GPUs)
- Efficient distributed training
- Optimized data pipelines
- Months of training time

**Cost Examples**:
- GPT-3: Estimated $4.6M
- PaLM: Estimated $8M+

## Important Takeaways
1. Pre-training learns general language representations
2. Different objectives suit different model types
3. Scale of data and model matters significantly
4. Self-supervised learning enables using unlabeled data
5. Infrastructure requirements are substantial
6. Scaling laws guide model development

