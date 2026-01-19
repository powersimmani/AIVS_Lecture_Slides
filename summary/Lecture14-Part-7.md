# Lecture 14 - Part 7: Prompting & In-Context Learning

## Overview
This part covers prompting techniques that allow using language models without gradient updates, including prompt engineering and advanced methods.

## Key Topics

### 1. Prompt Engineering Basics
**What is Prompting?**
- Conditioning the model with text
- No parameter updates needed
- Model "understands" task from prompt

**Components of a Prompt**:
```
[System Message/Instruction]
[Context/Examples]
[Input]
[Output Indicator]
```

**Example**:
```
Classify the sentiment of the following review.

Review: "This movie was fantastic!"
Sentiment:
```

### 2. Types of Prompting
**Zero-shot**:
```
Translate to French: "Hello"
```
- No examples, just instruction
- Relies on pre-trained knowledge

**One-shot**:
```
Translate to French:
"Hello" → "Bonjour"
"Goodbye" →
```
- One example provided

**Few-shot**:
```
Translate to French:
"Hello" → "Bonjour"
"Thank you" → "Merci"
"Goodbye" →
```
- Multiple examples (typically 3-10)

### 3. Advanced Prompting Techniques
**Chain-of-Thought (CoT)**:
```
Q: What is 15% of 80?
A: Let me think step by step.
   15% means 15/100 = 0.15
   0.15 × 80 = 12
   The answer is 12.
```
- Elicits reasoning steps
- Dramatically improves math/logic

**Self-Consistency**:
- Sample multiple CoT paths
- Majority vote on final answer
- More robust reasoning

**Tree-of-Thought (ToT)**:
- Explore multiple reasoning branches
- Evaluate and backtrack
- Complex problem solving

### 4. Prompt Templates
**Classification**:
```
Given the text: "{text}"
Classify it as one of: {labels}
Classification:
```

**Summarization**:
```
Summarize the following article in 3 sentences:
{article}
Summary:
```

**Code Generation**:
```
# Function to calculate fibonacci numbers
# Input: n (positive integer)
# Output: nth fibonacci number
def fibonacci(n):
```

### 5. Fine-tuning vs Prompting
| Aspect | Fine-tuning | Prompting |
|--------|-------------|-----------|
| Parameter Updates | Yes | No |
| Data Required | Hundreds+ | 0-few |
| Compute Cost | High | Low |
| Flexibility | Task-specific | General |
| Performance | Often higher | Good for many tasks |

**When to Use Prompting**:
- Quick prototyping
- Limited labeled data
- Many diverse tasks
- Inference-only access

**When to Fine-tune**:
- Maximum performance needed
- Sufficient labeled data
- Specific domain/task

### 6. Limitations of Prompting
- **Sensitivity**: Small changes big effects
- **Context Length**: Limited prompt space
- **Reliability**: Can be inconsistent
- **Complex Tasks**: May need fine-tuning

## Important Takeaways
1. Prompting enables task completion without training
2. Few-shot examples improve performance
3. Chain-of-thought improves reasoning
4. Prompt engineering is crucial skill
5. Trade-off between prompting and fine-tuning
6. Advanced techniques like ToT for complex problems

