# Lecture 14 - Part 5: Encoder-Decoder Models

## Overview
This part covers encoder-decoder pre-trained models like T5 and BART that combine understanding and generation capabilities.

## Key Topics

### 1. T5 - Text-to-Text Framework
**T5**: Text-to-Text Transfer Transformer (Google, 2019)

**Key Innovation**: Unified text-to-text format
```
Classification:  "classify: I love this!" → "positive"
Translation:     "translate English to French: Hello" → "Bonjour"
Summarization:   "summarize: [long text]" → "[summary]"
QA:              "question: [Q] context: [C]" → "[answer]"
```

**Benefits**:
- Same architecture for all tasks
- Same loss function (cross-entropy)
- Easy to add new tasks

### 2. T5 Architecture and Training
**Architecture**: Full encoder-decoder Transformer

**Pre-training Objective**: Span corruption
```
Input:  "The [X] brown fox [Y] over the lazy dog"
Output: "[X] quick [Y] jumps"
```
- Mask random spans (not individual tokens)
- Predict only masked spans (efficient)

**Sizes**:
| Model | Parameters |
|-------|------------|
| T5-Small | 60M |
| T5-Base | 220M |
| T5-Large | 770M |
| T5-3B | 3B |
| T5-11B | 11B |

### 3. BART and Other Models
**BART** (Facebook): Bidirectional and Auto-Regressive Transformers

**Pre-training**: Denoising autoencoder
- Multiple corruption strategies:
  - Token masking
  - Token deletion
  - Sentence permutation
  - Document rotation

**Strengths**:
- Excellent at summarization
- Strong on generation tasks
- Combines BERT's encoder + GPT's decoder

### 4. Other Encoder-Decoder Models
**mT5** (Multilingual T5):
- 101 languages
- Cross-lingual transfer

**FLAN-T5**:
- Instruction-tuned T5
- Better zero-shot performance

**UL2** (Google):
- Unified Language Learning
- Mixture of denoisers

### 5. When to Use Each Architecture
| Task Type | Best Architecture |
|-----------|-------------------|
| Classification | Encoder (BERT) |
| NER, Tagging | Encoder (BERT) |
| Text Generation | Decoder (GPT) |
| Translation | Encoder-Decoder (T5) |
| Summarization | Encoder-Decoder (BART) |
| Question Answering | Any (task-dependent) |
| Dialogue | Decoder (GPT) |

### 6. Encoder-Decoder Advantages
**Compared to Encoder-only**:
- Can generate variable-length output
- Natural for seq2seq tasks

**Compared to Decoder-only**:
- Full bidirectional encoding of input
- Can attend to complete input
- Often better for conditioned generation

## Important Takeaways
1. T5 unifies all tasks as text-to-text
2. Encoder-decoder combines understanding + generation
3. BART uses denoising pre-training
4. Architecture choice depends on task type
5. Text-to-text format simplifies multi-task learning
6. Encoder-decoder excels at translation and summarization

