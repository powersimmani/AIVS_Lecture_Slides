# Lecture 14 - Part 6: Fine-tuning Strategies

## Overview
This part covers strategies for adapting pre-trained models to downstream tasks, from full fine-tuning to parameter-efficient methods.

## Key Topics

### 1. Full Fine-tuning vs Transfer Learning
**Full Fine-tuning**:
- Update all model parameters
- Risk of catastrophic forgetting
- Requires storing separate model per task

**Feature Extraction**:
- Freeze pre-trained weights
- Only train task-specific head
- Faster but less performant

**The Trade-off**:
```
Full Fine-tuning: Best performance, high cost
Feature Extraction: Lower performance, low cost
PEFT: Good performance, moderate cost
```

### 2. Parameter-Efficient Fine-tuning (PEFT)
**Motivation**:
- Full fine-tuning expensive for large models
- Storage: Separate 175B model per task?
- Compute: Update billions of parameters?

**Key PEFT Methods**:
1. Adapters
2. LoRA
3. Prefix Tuning
4. Prompt Tuning

### 3. Adapter Layers
**Concept**: Insert small trainable modules
```
Original:     Attention → FFN
With Adapter: Attention → Adapter → FFN → Adapter
```

**Adapter Structure**:
```
x → Down-project → Activation → Up-project → + x (residual)
```
- Down: d_model → r (r << d_model)
- Up: r → d_model
- Only adapters trained (1-5% of parameters)

### 4. LoRA (Low-Rank Adaptation)
**Key Idea**: Decompose weight updates
```
W' = W + ΔW = W + BA
```
- W: Frozen original weights [d × k]
- B: Trainable [d × r]
- A: Trainable [r × k]
- r << min(d, k) (e.g., r=8)

**Benefits**:
- No inference latency (merge weights)
- Very few trainable parameters
- Easy to switch between tasks

**Implementation**:
```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8):
        self.original = original_linear  # Frozen
        self.A = nn.Parameter(torch.randn(r, original_linear.in_features))
        self.B = nn.Parameter(torch.zeros(original_linear.out_features, r))

    def forward(self, x):
        return self.original(x) + x @ self.A.T @ self.B.T
```

### 5. Prefix Tuning and Prompt Tuning
**Prefix Tuning**:
- Prepend learnable vectors to K, V
- Like continuous prompts in attention layers

**Prompt Tuning**:
- Prepend learnable embeddings to input
- Simpler, fewer parameters
- "Soft prompts"

```
Original: [CLS] token₁ token₂ ... [SEP]
Prompt:   [P₁][P₂]...[Pₙ] token₁ token₂ ... [SEP]
```

### 6. Fine-tuning Practical Tips
**Learning Rate**:
- Lower than pre-training (1e-5 to 5e-5)
- Warmup important
- Different rates for different layers possible

**Data**:
- Even small datasets can work (hundreds of examples)
- More data generally helps but diminishing returns

**Regularization**:
- Dropout (inherited from pre-training)
- Early stopping
- Weight decay

**Evaluation**:
- Use validation set
- Monitor for overfitting
- Task-specific metrics

## Important Takeaways
1. Full fine-tuning updates all parameters
2. PEFT methods reduce trainable parameters significantly
3. LoRA decomposes updates into low-rank matrices
4. Adapters insert small modules into layers
5. Prompt tuning learns soft prompts
6. Lower learning rates for fine-tuning
7. PEFT enables efficient task adaptation

