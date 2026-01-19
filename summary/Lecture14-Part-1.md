# Lecture 14 - Part 1: Introduction & Paradigm Shift

## Overview
This part introduces pre-trained language models and explains the paradigm shift from task-specific models to pre-train then fine-tune approach.

## Key Topics

### 1. Review of Previous Lessons
- **Transformers**: Self-attention, multi-head attention
- **Architecture**: Encoder, decoder, encoder-decoder
- **Key Concepts**: Positional encoding, masking, parallelization

### 2. Paradigm Shift in AI
**Traditional Approach** (Before 2018):
- Train from scratch for each task
- Task-specific architectures
- Limited by labeled data availability
- No transfer learning

**New Paradigm** (Pre-train & Fine-tune):
1. Pre-train on massive unlabeled text
2. Fine-tune on specific task with small labeled data
3. Transfer learned knowledge across tasks

### 3. Why This Shift Matters
**Benefits of Pre-training**:
- Leverage billions of words of text
- Learn general language understanding
- Reduce need for task-specific labeled data
- Better generalization

**Analogy**: Like teaching someone to read before asking them to do specific tasks

### 4. Timeline of Key Models
| Year | Model | Innovation |
|------|-------|------------|
| 2017 | Transformer | Attention is all you need |
| 2018 | ELMo | Contextualized embeddings |
| 2018 | BERT | Bidirectional pre-training |
| 2018 | GPT | Autoregressive pre-training |
| 2019 | GPT-2 | Scaling, zero-shot |
| 2020 | GPT-3 | Few-shot learning |
| 2022 | ChatGPT | Instruction following |
| 2023+ | GPT-4, Claude | Advanced reasoning |

### 5. The Foundation Model Concept
**Foundation Models**:
- Large models trained on broad data
- Adapted to wide range of tasks
- Single model, many applications

**Characteristics**:
- Scale (billions of parameters)
- Emergent capabilities
- Multi-task generalization

## Important Takeaways
1. AI shifted from task-specific to pre-train + fine-tune
2. Pre-training leverages massive unlabeled data
3. Knowledge transfers across tasks
4. Foundation models serve as base for many applications
5. This paradigm enabled modern LLMs

