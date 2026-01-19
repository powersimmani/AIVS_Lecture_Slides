# Lecture 14 - Part 8: Current Trends

## Overview
This part covers current trends in large language models including RLHF, instruction tuning, and emerging capabilities.

## Key Topics

### 1. RLHF (Reinforcement Learning from Human Feedback)
**Problem**: Pre-trained models don't follow instructions well

**RLHF Process**:
1. **Supervised Fine-tuning**: Train on demonstration data
2. **Reward Model**: Train to predict human preferences
3. **RL Optimization**: Use PPO to maximize reward

**How Reward Model Works**:
```
Input: Prompt + Two responses
Human: Labels which is better
Model: Learns to score responses
```

**PPO Training**:
- Generate responses
- Score with reward model
- Update policy to increase scores
- KL penalty to prevent divergence

### 2. Instruction Tuning
**Concept**: Fine-tune on diverse instruction-following data

**FLAN (Fine-tuned Language Net)**:
- Many tasks phrased as instructions
- Improves zero-shot performance

**Examples**:
```
"Summarize this article: {text}"
"Translate to French: {text}"
"Answer this question: {question}"
```

**Self-Instruct**: Generate instructions with LLM itself

### 3. Constitutional AI
**Approach**: Train AI to follow principles
```
Principles:
1. Be helpful
2. Be harmless
3. Be honest
```

**Process**:
- Generate response
- Critique against principles
- Revise response
- Train on revised responses

### 4. Emergence and Scaling
**Emergent Capabilities**:
- Abilities that appear at certain scales
- Not predicted from smaller models
- Examples: arithmetic, code, reasoning

**Scaling Observations**:
- Larger models = new capabilities
- Threshold effects (sudden improvements)
- Debate on what's truly "emergent"

### 5. Present and Future
**Current Capabilities**:
- Advanced reasoning
- Multi-turn dialogue
- Code generation and execution
- Multi-modal (vision + language)

**Active Research Areas**:
- Efficiency (smaller, faster models)
- Alignment (safety, honesty)
- Reasoning (formal, mathematical)
- Multimodal (image, video, audio)
- Agents (tool use, planning)

### 6. Open vs Closed Models
**Closed Models**:
- GPT-4, Claude, Gemini
- API access only
- Cutting-edge performance

**Open Models**:
- LLaMA, Mistral, Falcon
- Weights available
- Community improvements

**Trade-offs**:
- Open: Transparency, customization
- Closed: Safety controls, support

## Important Takeaways
1. RLHF aligns models with human preferences
2. Instruction tuning improves following directions
3. Constitutional AI encodes principles
4. Emergent capabilities appear at scale
5. Active research on alignment and safety
6. Both open and closed models advancing

