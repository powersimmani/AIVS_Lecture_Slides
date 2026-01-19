# Lecture 14 - Part 3: BERT - Encoder-based Models

## Overview
This part covers BERT and encoder-only pre-trained models, which excel at understanding and classification tasks.

## Key Topics

### 1. Introduction to BERT
**BERT**: Bidirectional Encoder Representations from Transformers (Google, 2018)

**Key Innovation**:
- Bidirectional context (unlike GPT's left-to-right)
- Masked Language Modeling (MLM)
- Pre-train deep bidirectional Transformer

**Impact**: State-of-the-art on 11 NLP benchmarks

### 2. BERT Pre-training Objectives
**Masked Language Modeling (MLM)**:
```
Input:  [CLS] The cat [MASK] on the mat [SEP]
Output: Predict "sat" at [MASK] position
```
- Randomly mask 15% of tokens
- 80% → [MASK], 10% → random, 10% → unchanged
- Learn bidirectional representations

**Next Sentence Prediction (NSP)**:
```
Input:  [CLS] Sentence A [SEP] Sentence B [SEP]
Output: IsNextSentence? Yes/No
```
- Learn sentence relationships
- Later found less important

### 3. BERT Architecture
**Configuration**:
| | Base | Large |
|--|------|-------|
| Layers | 12 | 24 |
| Hidden | 768 | 1024 |
| Heads | 12 | 16 |
| Params | 110M | 340M |

**Input Format**:
```
[CLS] token₁ token₂ ... tokenₙ [SEP] (sentence₂) [SEP]
```
- [CLS]: Classification token
- [SEP]: Separator token
- Segment embeddings: A vs B

### 4. BERT Fine-tuning
**Classification**:
```
[CLS] embedding → Linear → Softmax → Label
```

**Token Classification (NER)**:
```
Each token embedding → Linear → Label
```

**Question Answering**:
```
Learn start/end positions of answer span
```

**Sentence Pair Tasks**:
```
[CLS] Sentence A [SEP] Sentence B [SEP]
→ Classify relationship
```

### 5. BERT Family Models
**RoBERTa** (Facebook):
- Removed NSP, dynamic masking
- More data, longer training
- Better performance

**ALBERT** (Google):
- Parameter sharing across layers
- Factorized embeddings
- Smaller but competitive

**DistilBERT**:
- Knowledge distillation
- 40% smaller, 60% faster
- 97% of BERT's performance

**DeBERTa** (Microsoft):
- Disentangled attention
- Enhanced mask decoder
- State-of-the-art on many benchmarks

## Important Takeaways
1. BERT introduced bidirectional pre-training
2. MLM allows using all context for prediction
3. [CLS] token used for classification tasks
4. Fine-tuning adapts pre-trained model to tasks
5. Many variants improve on original BERT
6. Encoder-only models excel at understanding tasks

