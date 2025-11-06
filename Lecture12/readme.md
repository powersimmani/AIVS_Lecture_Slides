# Lecture 12: Advanced Sequence Models
## Deep Learning for Natural Language Processing

---

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Total Slides:** 34 slides  
**Lecture Duration:** ~120 minutes (recommended)  
**Difficulty Level:** Intermediate to Advanced  
**Course Level:** Graduate/Advanced Undergraduate

This lecture provides comprehensive coverage of advanced sequence modeling architectures including Bidirectional RNNs, Sequence-to-Sequence models, Teacher Forcing strategies, and Attention Mechanisms. Students will gain both theoretical understanding and practical implementation knowledge of modern NLP architectures.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Analyze RNN Limitations**: Understand and articulate the fundamental limitations of standard RNNs including vanishing gradients, information bottleneck, and sequential processing constraints

2. **Implement Bidirectional RNNs**: Design and implement BiRNN architectures that capture both past and future context for improved sequence understanding

3. **Build Seq2Seq Models**: Construct encoder-decoder architectures for variable-length sequence transformation tasks such as machine translation and text summarization

4. **Apply Attention Mechanisms**: Integrate attention mechanisms to solve the information bottleneck problem and enable dynamic context selection

5. **Master Training Strategies**: Utilize advanced training techniques including teacher forcing, scheduled sampling, batching, and masking for production-ready sequence models

---

## 📚 Lecture Structure

### **Part 1: Introduction & Review** (Slides 3-6)

#### Topics Covered:
- **Slide 4 - Review of Last Lesson**
  - Recurrent Neural Networks fundamentals
  - Hidden state mechanics and memory propagation
  - Gradient problems (vanishing/exploding)
  - LSTM and GRU gate mechanisms
  - Gate control flow (forget, input, output gates)
  - Key applications in NLP and time series

- **Slide 5 - Today's Learning Objectives**
  - Understanding RNN limitations
  - Mastering Bidirectional RNN architecture
  - Implementing Seq2Seq models with attention
  - Applying advanced training techniques

- **Slide 6 - Exploring the Limitations of RNNs**
  - Vanishing Gradient Problem: exponential gradient decay over time steps
  - Long-term Dependencies: information decay across sequences
  - Exploding Gradient Problem: numerical instability requiring gradient clipping
  - Sequential Processing Bottleneck: inability to parallelize computation

#### Key Concepts:
- **Gradient Flow Issues**: Understanding why gradients diminish/explode in deep RNN architectures
- **Memory Constraints**: Recognition of hidden state capacity limitations
- **Parallelization Barriers**: Sequential dependency preventing GPU acceleration
- **Information Bottleneck**: Single hidden state compression problem

#### Learning Outcomes:
- Diagnose failure modes in RNN training
- Identify scenarios where standard RNNs underperform
- Justify the need for architectural improvements

---

### **Part 2: Bidirectional RNNs** (Slides 7-11)

#### Topics Covered:
- **Slide 8 - Why Bidirectional?**
  - Context importance in language understanding
  - Comparison: Unidirectional vs Bidirectional processing
  - Example: Word disambiguation using future context ("bank" in "bank of the river")
  - Forward-only limitations vs full-context advantages

- **Slide 9 - Bidirectional RNN Architecture**
  - Dual processing streams: forward (→) and backward (←)
  - Hidden state computation at each time step
  - Concatenation of forward and backward states
  - Output generation from combined representations

- **Slide 10 - BiRNN Formulas and Operations**
  - Forward pass: h→_t = f(W_fh h→_{t-1} + W_fx x_t + b_f)
  - Backward pass: h←_t = f(W_bh h←_{t+1} + W_bx x_t + b_b)
  - Output: y_t = g(W_y [h→_t; h←_t] + b_y)
  - Parameter doubling implications

- **Slide 11 - BiRNN Pros and Cons & Applications**
  - **Advantages**: Full context capture, better representations, improved accuracy, disambiguation
  - **Disadvantages**: Cannot stream, 2× computation, not for generation, slower training
  - **Applications**: Named Entity Recognition, POS tagging, Sentiment Analysis, Slot Filling, Dependency Parsing, Speech Recognition

#### Key Concepts:
- **Bidirectional Processing**: Simultaneous forward and backward information flow
- **Context Vector Enrichment**: Enhanced representations from dual-direction encoding
- **Trade-offs**: Accuracy gains vs computational cost and streaming limitations
- **Task Suitability**: When to use BiRNN (classification) vs unidirectional (generation)

#### Learning Outcomes:
- Implement BiRNN layers in PyTorch/TensorFlow
- Calculate memory requirements (2× parameters)
- Select appropriate architectures for specific NLP tasks
- Understand why BiRNN cannot be used for autoregressive generation

---

### **Part 3: Sequence-to-Sequence Models** (Slides 12-17)

#### Topics Covered:
- **Slide 13 - The Need for Seq2Seq**
  - Variable-length input → variable-length output challenge
  - Examples: Machine Translation, Text Summarization, Question Answering
  - Encoder-Decoder solution architecture
  - Context vector as information bridge

- **Slide 14 - Encoder-Decoder Architecture**
  - Encoder: Compress input sequence into fixed context vector
  - Context Vector (c): Fixed-size semantic representation
  - Decoder: Generate output sequence from context
  - End-to-end learning paradigm

- **Slide 15 - Encoder Details**
  - Input embedding and processing
  - RNN cell operations (typically LSTM/GRU)
  - Final hidden state as context vector
  - Handling variable-length inputs

- **Slide 16 - Decoder Details**
  - Context initialization
  - Autoregressive generation process
  - Special tokens: <START>, <END>
  - Beam search vs greedy decoding

- **Slide 17 - Overall Seq2Seq Process**
  - Complete forward pass walkthrough
  - Training vs inference mode differences
  - Loss computation (cross-entropy)
  - Teacher forcing integration

#### Key Concepts:
- **Encoder-Decoder Paradigm**: Separation of comprehension and generation
- **Context Vector**: Fixed-size bottleneck representation
- **Autoregressive Decoding**: Sequential output generation
- **Training-Inference Gap**: Different behavior during training vs deployment

#### Learning Outcomes:
- Build complete Seq2Seq models from scratch
- Implement encoder and decoder components
- Handle variable-length sequences with padding
- Debug common Seq2Seq training issues

---

### **Part 4: Teacher Forcing** (Slides 18-21)

#### Topics Covered:
- **Slide 19 - What is Teacher Forcing?**
  - Definition: Using ground-truth tokens as decoder inputs during training
  - Comparison with autoregressive (using predictions)
  - Visual demonstration of both approaches
  - Training acceleration benefits

- **Slide 20 - Teacher Forcing vs. Autoregressive**
  - Training mode: Ground-truth input feeding
  - Inference mode: Model prediction feedback
  - Error accumulation in autoregressive mode
  - Convergence speed comparison

- **Slide 21 - Teacher Forcing Problems and Solutions**
  - **Problems**:
    - Exposure bias: Model never sees its own errors
    - Train-test mismatch: Different input distributions
    - Error recovery inability
    - Overconfidence on ground-truth
  - **Solutions**:
    - Scheduled Sampling: Gradual mixing (ε increases from 0.1 → 0.9)
    - Mixed Training: Random alternation (p=0.5)
    - Curriculum Learning: Progressive difficulty increase

#### Key Concepts:
- **Exposure Bias**: Distribution mismatch between training and inference
- **Scheduled Sampling**: Curriculum-based mixing strategy
- **Training Stability**: Balance between convergence speed and robustness
- **Error Propagation**: Compounding mistakes in autoregressive generation

#### Learning Outcomes:
- Implement teacher forcing in training loops
- Apply scheduled sampling strategies
- Diagnose and fix exposure bias issues
- Balance training speed vs inference robustness

---

### **Part 5: Attention Mechanism Basics** (Slides 22-27)

#### Topics Covered:
- **Slide 23 - Limitations of Seq2Seq**
  - Information Bottleneck: Fixed-size context vector constraint
  - Gradient Vanishing: Long-range dependency failure
  - Equal Weighting: No selective focus mechanism
  - No Alignment: Inability to map input-output correspondence

- **Slide 24 - Core Ideas of Attention**
  - Dynamic context generation for each output step
  - Query-Key-Value paradigm
  - Alignment scores computation
  - Weighted combination of encoder states

- **Slide 25 - Attention Mechanism Structure**
  - Query: Current decoder state
  - Keys: All encoder hidden states
  - Values: Encoder hidden states (same as keys in basic attention)
  - Attention weights via softmax normalization

- **Slide 26 - Attention Formula**
  - Score computation: e_tj = score(s_t, h_j)
  - Softmax normalization: α_tj = exp(e_tj) / Σ_k exp(e_tk)
  - Context vector: c_t = Σ_j α_tj h_j
  - Output generation: y_t = f(s_t, c_t, y_{t-1})

- **Slide 27 - Effects of Attention**
  - **Before Attention**: Uniform weights (0.25 each), static context
  - **With Attention**: Dynamic weights (0.05-0.7), focused context
  - **Benefits**:
    - Better alignment (15-20% BLEU improvement)
    - Long sequence handling (30-40% improvement on 50+ tokens)
    - Interpretability through attention visualization
  - Gradient flow stabilization

#### Key Concepts:
- **Attention Weights**: Learned importance scores for each input position
- **Dynamic Context**: Different context vectors for each decoding step
- **Alignment Learning**: Automatic input-output correspondence discovery
- **Interpretability**: Visualization of model focus patterns

#### Learning Outcomes:
- Implement attention mechanisms from scratch
- Calculate attention weights and context vectors
- Visualize attention distributions
- Understand attention's role in solving information bottleneck
- Achieve 15-20% BLEU score improvements in translation tasks

---

### **Part 6: Practical Implementation Tips** (Slides 28-31)

#### Topics Covered:
- **Slide 29 - Batching**
  - Variable-length sequence challenge
  - Padding strategy: Extend to max batch length
  - Sequential vs Parallel processing comparison (3× speedup)
  - Memory-speed trade-offs
  - Typical batch sizes: 32, 64, 128

- **Slide 30 - Masking**
  - **Padding Mask**: Ignore PAD tokens in attention
    - Implementation: `mask = (input != PAD_TOKEN)`
    - Application: `scores.masked_fill(~mask, -∞)`
  - **Look-ahead Mask**: Prevent future token attention (causal masking)
    - Lower triangular matrix structure
    - Implementation: `torch.tril(torch.ones(seq_len, seq_len))`
  - Combining multiple masks with logical AND

- **Slide 31 - Practical Checklist**
  - Gradient clipping (prevents explosion)
  - Learning rate scheduling
  - Dropout regularization
  - Beam search for inference
  - BLEU/ROUGE evaluation metrics
  - Checkpoint saving strategies

#### Key Concepts:
- **Efficient Batching**: GPU parallelization through padding
- **Masking Strategies**: Preventing invalid attention
- **Training Stability**: Gradient clipping and regularization
- **Production Readiness**: Checkpointing and evaluation

#### Learning Outcomes:
- Implement efficient batched training loops
- Apply padding and masking correctly
- Achieve 3× training speedup through parallelization
- Debug masking-related errors
- Deploy production-ready sequence models

---

### **Part 7: Conclusion & Next Steps** (Slides 32-34)

#### Topics Covered:
- **Slide 33 - Summary and Preview of Next Lecture**
  - **Today's Topics**:
    1. Seq2Seq Architecture (Encoder-Decoder, Context vector)
    2. RNN Variants (LSTM, GRU, Gradient flow)
    3. Bidirectional RNN (Forward+Backward, Better encoding)
    4. Teacher Forcing (Training strategy, Exposure bias)
    5. Attention Mechanism (Query-Key-Value, Dynamic context)
    6. Implementation (Batching, Padding, Masking)
  - **Next Lecture Preview**: Transformers & Modern Architectures
    - Self-Attention mechanisms
    - Transformer architecture
    - Multi-Head Attention
    - BERT & GPT models

- **Slide 34 - Thank You**
  - Course completion acknowledgment
  - Instructor contact information

#### Key Concepts:
- **Architecture Evolution**: From RNNs → Attention → Transformers
- **Key Takeaway**: Attention revolutionized sequence modeling by solving information bottleneck
- **Future Directions**: Self-attention and transformer-based models

#### Learning Outcomes:
- Synthesize all lecture concepts into coherent understanding
- Prepare for transformer architecture study
- Connect attention mechanisms to modern NLP models

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Deep Learning Fundamentals**
  - Neural network basics (forward/backward propagation)
  - Gradient descent optimization
  - Backpropagation through time (BPTT)
  
- **Recurrent Neural Networks**
  - Basic RNN architecture and operations
  - LSTM and GRU cells
  - Hidden state concepts
  - Sequence modeling fundamentals
  
- **Mathematics**
  - Linear algebra (matrix operations, vector spaces)
  - Calculus (derivatives, chain rule)
  - Probability (softmax, distributions)
  
- **Programming**
  - Python proficiency
  - NumPy for numerical operations
  - PyTorch or TensorFlow basics

### Recommended Prior Courses:
- Introduction to Deep Learning
- Neural Networks and Backpropagation
- Lecture 11: Recurrent Neural Networks (previous lecture)

### Software Requirements:
```bash
# Python 3.8+
python >= 3.8

# Deep Learning Frameworks
torch >= 1.12.0
tensorflow >= 2.10.0  # Alternative to PyTorch

# Scientific Computing
numpy >= 1.21.0
scipy >= 1.7.0

# Visualization
matplotlib >= 3.5.0
seaborn >= 0.11.0

# NLP Tools
nltk >= 3.6
spacy >= 3.2.0

# Utilities
jupyter >= 1.0.0
tensorboard >= 2.10.0
```

### Hardware Recommendations:
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 8GB+ VRAM (NVIDIA RTX 3060 or better)
- **Optimal**: Multi-GPU setup with 16GB+ VRAM per GPU

---

## 💻 Hands-on Components

### Project 1: Bidirectional RNN for Named Entity Recognition

**Objective**: Implement a BiRNN for NER task on CoNLL-2003 dataset

**Implementation Steps**:
1. Data preprocessing and tokenization
2. BiRNN architecture implementation
   ```python
   class BiRNN_NER(nn.Module):
       def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, embedding_dim)
           self.birnn = nn.LSTM(embedding_dim, hidden_dim, 
                               bidirectional=True, batch_first=True)
           self.fc = nn.Linear(hidden_dim * 2, num_tags)
   ```
3. Training with proper masking
4. Evaluation using F1 score

**Expected Results**:
- F1 Score: 85-90% on CoNLL-2003 test set
- Training time: ~30 minutes on GPU

**Deliverables**:
- Working BiRNN implementation
- Training/validation curves
- Confusion matrix and error analysis

---

### Project 2: Seq2Seq Machine Translation

**Objective**: Build English→German translation system with attention

**Dataset**: Multi30k (30,000 sentence pairs)

**Architecture Components**:
1. **Encoder**: 2-layer LSTM (hidden_dim=512)
2. **Attention**: Additive (Bahdanau) attention
3. **Decoder**: 2-layer LSTM with attention integration
4. **Implementation**:
   ```python
   class Seq2SeqWithAttention(nn.Module):
       def __init__(self, encoder, attention, decoder):
           super().__init__()
           self.encoder = encoder
           self.attention = attention
           self.decoder = decoder
       
       def forward(self, src, trg, teacher_forcing_ratio=0.5):
           # Encoder
           encoder_outputs, hidden = self.encoder(src)
           
           # Decoder with attention
           outputs = []
           input = trg[0]
           for t in range(1, trg.size(0)):
               # Attention
               context = self.attention(hidden, encoder_outputs)
               # Decoder step
               output, hidden = self.decoder(input, hidden, context)
               outputs.append(output)
               
               # Teacher forcing
               use_teacher_forcing = random.random() < teacher_forcing_ratio
               input = trg[t] if use_teacher_forcing else output.argmax(1)
           
           return torch.stack(outputs)
   ```

**Training Strategies**:
- Initial teacher forcing ratio: 0.8
- Scheduled sampling: Linear decay from 0.8 → 0.3 over 20 epochs
- Gradient clipping: max_norm=1.0
- Optimizer: Adam with learning rate 0.001
- Batch size: 64

**Expected Results**:
- BLEU Score: 25-30 (baseline: ~15 without attention)
- Training time: 2-3 hours on GPU
- 15-20% improvement with attention vs without

**Deliverables**:
- Complete Seq2Seq model with attention
- Attention visualization for sample translations
- BLEU score comparison (with/without attention)
- Error analysis with examples

---

### Project 3: Teacher Forcing Experiments

**Objective**: Compare training strategies and analyze exposure bias

**Experiments**:
1. **Pure Teacher Forcing** (baseline)
   - Teacher forcing ratio: 1.0 throughout training
   - Fastest convergence but poor inference

2. **Scheduled Sampling**
   ```python
   def get_teacher_forcing_ratio(epoch, total_epochs):
       # Linear decay
       return 1.0 - (epoch / total_epochs) * 0.7
   ```

3. **Mixed Training**
   ```python
   def get_teacher_forcing_ratio(epoch, total_epochs):
       # Random sampling with decaying probability
       return 0.5  # Fixed 50% throughout
   ```

**Metrics to Compare**:
- Training convergence speed
- Validation loss curves
- BLEU scores at inference
- Error distribution analysis

**Expected Findings**:
- Pure TF: Fastest training, ~5% lower BLEU at inference
- Scheduled Sampling: Best balance, ~2% lower training speed
- Mixed Training: Most robust, slightly slower convergence

---

### Project 4: Attention Visualization

**Objective**: Implement and visualize attention mechanisms

**Implementation**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(sentence, translation, attention_weights):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(attention_weights, 
                xticklabels=sentence.split(),
                yticklabels=translation.split(),
                cmap='YlOrRd', ax=ax)
    plt.xlabel('Source Sentence')
    plt.ylabel('Target Sentence')
    plt.title('Attention Weights Heatmap')
    plt.tight_layout()
    plt.show()
```

**Analysis Tasks**:
1. Visualize attention for different sentence lengths
2. Identify alignment patterns (monotonic vs non-monotonic)
3. Analyze attention for ambiguous words
4. Compare attention patterns before/after convergence

**Deliverables**:
- Attention heatmap generation code
- Gallery of 10+ attention visualizations
- Analysis report on alignment patterns

---

### Project 5: Production-Ready Implementation

**Objective**: Build complete training and inference pipeline

**Components**:
1. **Data Pipeline**
   - Efficient data loading with DataLoader
   - Dynamic batching with padding
   - Proper masking implementation

2. **Training Script**
   ```python
   def train_epoch(model, dataloader, optimizer, criterion, clip):
       model.train()
       epoch_loss = 0
       
       for batch in tqdm(dataloader):
           src, trg = batch.src, batch.trg
           
           # Create masks
           src_mask = (src != PAD_IDX)
           trg_mask = (trg != PAD_IDX)
           
           optimizer.zero_grad()
           output = model(src, trg, src_mask, trg_mask)
           
           # Loss with masking
           loss = criterion(output[trg_mask], trg[trg_mask])
           loss.backward()
           
           torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
           optimizer.step()
           
           epoch_loss += loss.item()
       
       return epoch_loss / len(dataloader)
   ```

3. **Inference Pipeline**
   - Beam search implementation (beam_size=5)
   - Length normalization
   - Post-processing and detokenization

4. **Evaluation Suite**
   - BLEU score computation
   - Perplexity calculation
   - Inference speed benchmarking

**Performance Targets**:
- Training throughput: >100 sentences/second on GPU
- Inference speed: <100ms per sentence
- Memory efficiency: <8GB VRAM for batch_size=32

**Deliverables**:
- Complete training script with checkpointing
- Inference script with beam search
- Evaluation suite
- Performance benchmarking report

---

## 📖 Additional Resources

### Foundational Papers:
1. **Sequence to Sequence Learning with Neural Networks** (Sutskever et al., 2014)
   - Original Seq2Seq paper
   - URL: https://arxiv.org/abs/1409.3215

2. **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2015)
   - Introduction of attention mechanism
   - URL: https://arxiv.org/abs/1409.0473

3. **Effective Approaches to Attention-based Neural Machine Translation** (Luong et al., 2015)
   - Alternative attention mechanisms
   - URL: https://arxiv.org/abs/1508.04025

4. **Scheduled Sampling for Sequence Prediction** (Bengio et al., 2015)
   - Teacher forcing solutions
   - URL: https://arxiv.org/abs/1506.03099

### Textbooks:
- **Speech and Language Processing** (Jurafsky & Martin, 3rd ed.)
  - Chapter 9: RNNs and LSTMs
  - Chapter 10: Encoder-Decoder Models
  
- **Deep Learning** (Goodfellow, Bengio, Courville)
  - Chapter 10: Sequence Modeling
  
- **Neural Network Methods for Natural Language Processing** (Goldberg)
  - Part II: Sequence Models

### Online Courses:
- **CS224N: Natural Language Processing with Deep Learning** (Stanford)
  - Lectures 5-8 cover RNNs, Seq2Seq, and Attention
  
- **Fast.ai NLP Course**
  - Practical implementation-focused approach

### Code Repositories:
1. **PyTorch Seq2Seq Tutorial**
   - URL: https://github.com/bentrevett/pytorch-seq2seq
   - Excellent step-by-step implementations

2. **OpenNMT**
   - Production-ready NMT toolkit
   - URL: https://github.com/OpenNMT/OpenNMT-py

3. **Annotated Transformer**
   - URL: http://nlp.seas.harvard.edu/annotated-transformer/
   - Line-by-line implementation guide

### Datasets:
- **Multi30k**: English-German translation (30K pairs)
- **WMT14**: Large-scale translation dataset
- **CoNLL-2003**: Named Entity Recognition
- **SQuAD**: Question Answering

### Visualization Tools:
- **TensorBoard**: Training metrics visualization
- **Weights & Biases**: Experiment tracking
- **BertViz**: Attention visualization (adaptable for RNN attention)

---

## 🔗 How to Use These Materials

### For Self-Study:

1. **Pre-Lecture Preparation** (30 minutes)
   - Review RNN basics from Lecture 11
   - Read slides 1-6 (Introduction)
   - List questions about RNN limitations

2. **Main Study Session** (2-3 hours)
   - Go through all slides sequentially
   - Take notes on key formulas
   - Draw architecture diagrams by hand
   - Pause at each "Key Concepts" to self-test

3. **Post-Lecture Review** (1 hour)
   - Summarize each part in your own words
   - Recreate attention formula from memory
   - Review the practical checklist

4. **Hands-on Practice** (4-6 hours per project)
   - Start with Project 1 (BiRNN NER)
   - Progress to Project 2 (Seq2Seq)
   - Complete all 5 projects over 2-3 weeks

### For Classroom Instruction:

**Week 1: Session 1 (90 minutes)**
- Parts 1-2: Introduction and Bidirectional RNNs
- Live coding: Simple BiRNN implementation
- Assignment: Project 1 (BiRNN NER)

**Week 1: Session 2 (90 minutes)**
- Part 3: Sequence-to-Sequence Models
- Live coding: Basic Seq2Seq without attention
- Discussion: Information bottleneck problem

**Week 2: Session 1 (90 minutes)**
- Parts 4-5: Teacher Forcing and Attention Mechanisms
- Live coding: Attention implementation
- In-class activity: Attention visualization

**Week 2: Session 2 (90 minutes)**
- Part 6: Practical Implementation
- Live coding: Complete training pipeline
- Assignment: Projects 2-5

### Recommended Study Flow:

```
Day 1: Slides 1-11 (BiRNN) + Start Project 1
Day 2: Complete Project 1 + Review
Day 3: Slides 12-17 (Seq2Seq) + Read attention paper
Day 4: Slides 18-21 (Teacher Forcing) + Experiments
Day 5: Slides 22-27 (Attention) + Start Project 2
Day 6-7: Complete Project 2
Day 8: Slides 28-31 (Implementation) + Projects 3-4
Day 9-10: Project 5 + Complete review
```

### Tips for Maximum Learning:
- **Don't just read**: Implement each concept in code
- **Visualize everything**: Draw architectures, plot attention
- **Debug actively**: Intentionally break code to understand errors
- **Compare implementations**: PyTorch vs TensorFlow
- **Read papers**: Supplement slides with original research
- **Join communities**: Discuss on forums (r/MachineLearning, Papers with Code)

---

## 📊 Assessment Suggestions

### Formative Assessment (During Learning):

1. **Concept Quizzes** (After each part)
   - Part 1: RNN limitations (5 questions)
   - Part 2: BiRNN architecture (5 questions)
   - Part 3: Seq2Seq mechanics (5 questions)
   - Part 4: Teacher forcing (5 questions)
   - Part 5: Attention mechanisms (10 questions)
   - Part 6: Implementation (5 questions)

2. **Code Reviews**
   - Peer review of Project 1 implementations
   - Group debugging sessions
   - Code quality checklist

3. **Attention Visualization Competition**
   - Most informative visualization
   - Best error analysis
   - Creative presentation

### Summative Assessment:

#### **Option 1: Comprehensive Exam (100 points)**

**Part A: Theoretical Understanding (40 points)**
1. Explain the information bottleneck problem in Seq2Seq models (10 pts)
2. Derive attention weight computation step-by-step (10 pts)
3. Compare BiRNN with unidirectional RNN for NER task (10 pts)
4. Analyze exposure bias and propose solutions (10 pts)

**Part B: Implementation (40 points)**
1. Implement attention mechanism from scratch (20 pts)
2. Debug provided buggy Seq2Seq code (10 pts)
3. Design masking strategy for variable-length batches (10 pts)

**Part C: Analysis (20 points)**
1. Interpret given attention heatmaps (10 pts)
2. Propose architecture improvements for specific task (10 pts)

#### **Option 2: Project Portfolio (100 points)**

**Project 1: BiRNN NER (15 points)**
- Implementation correctness (8 pts)
- F1 score performance (4 pts)
- Code quality and documentation (3 pts)

**Project 2: Seq2Seq Translation (30 points)**
- Architecture implementation (10 pts)
- Attention mechanism integration (10 pts)
- BLEU score achievement (5 pts)
- Error analysis quality (5 pts)

**Project 3: Teacher Forcing Experiments (15 points)**
- Experimental design (5 pts)
- Results analysis (5 pts)
- Conclusions and insights (5 pts)

**Project 4: Attention Visualization (10 points)**
- Visualization quality (5 pts)
- Interpretation depth (5 pts)

**Project 5: Production Pipeline (30 points)**
- Complete implementation (15 pts)
- Performance benchmarks (5 pts)
- Code quality and documentation (5 pts)
- Deployment readiness (5 pts)

#### **Option 3: Research Paper Critique (100 points)**

Select one of:
- Bahdanau et al. (2015) - Attention mechanism
- Luong et al. (2015) - Attention variants
- Bengio et al. (2015) - Scheduled sampling

**Critique Structure (15-20 pages)**:
1. Summary and motivation (15 pts)
2. Technical approach analysis (25 pts)
3. Experimental validation (20 pts)
4. Reproduction attempts (25 pts)
5. Limitations and improvements (15 pts)

### Grading Rubric for Projects:

**Excellent (90-100%)**:
- Flawless implementation
- Exceeds performance benchmarks
- Exceptional code quality
- Insightful analysis
- Creative improvements

**Good (80-89%)**:
- Correct implementation
- Meets performance benchmarks
- Clean, documented code
- Solid analysis
- Minor improvements

**Satisfactory (70-79%)**:
- Mostly correct implementation
- Approaches performance benchmarks
- Acceptable code quality
- Basic analysis
- Follows instructions

**Needs Improvement (<70%)**:
- Incomplete or incorrect implementation
- Below performance benchmarks
- Poor code quality
- Superficial analysis
- Missing components

---

## 📝 Notes for Implementation

### Technical Requirements:

#### **Environment Setup**:
```bash
# Create conda environment
conda create -n seq2seq python=3.9
conda activate seq2seq

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy scipy matplotlib seaborn
pip install nltk spacy tensorboard
pip install jupyter ipywidgets

# Download language models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

#### **Memory Optimization**:
- **Gradient Checkpointing**: Reduce memory by 50% at cost of 20% speed
  ```python
  from torch.utils.checkpoint import checkpoint
  
  def forward(self, x):
      return checkpoint(self.layer, x)
  ```

- **Mixed Precision Training**: 2× speedup, 50% memory reduction
  ```python
  from torch.cuda.amp import autocast, GradScaler
  
  scaler = GradScaler()
  with autocast():
      output = model(input)
      loss = criterion(output, target)
  scaler.scale(loss).backward()
  ```

- **Gradient Accumulation**: Simulate larger batches
  ```python
  accumulation_steps = 4
  for i, batch in enumerate(dataloader):
      loss = model(batch) / accumulation_steps
      loss.backward()
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```

#### **Common Issues and Solutions**:

1. **Out of Memory Error**:
   - Reduce batch size: 64 → 32 → 16
   - Enable gradient checkpointing
   - Use mixed precision training
   - Reduce model size (hidden_dim: 512 → 256)

2. **Slow Training**:
   - Increase batch size (if memory allows)
   - Use DataLoader with num_workers=4
   - Pin memory: `pin_memory=True`
   - Move data preprocessing to GPU

3. **NaN Loss**:
   - Apply gradient clipping: `max_norm=1.0`
   - Reduce learning rate: 0.001 → 0.0001
   - Check for division by zero in attention
   - Initialize weights properly

4. **Poor Translation Quality**:
   - Increase model capacity (layers, hidden_dim)
   - Train longer (20 → 50 epochs)
   - Use beam search (beam_size=5)
   - Apply dropout (0.3)
   - Try different attention mechanisms

5. **Attention Weights Not Interpretable**:
   - Check softmax temperature
   - Increase encoder hidden size
   - Visualize at different training stages
   - Use attention regularization

### Performance Benchmarks:

#### **Training Speed** (Multi30k, batch_size=64, GPU: RTX 3090):
- BiRNN NER: ~500 sentences/sec
- Seq2Seq without attention: ~200 sentence pairs/sec
- Seq2Seq with attention: ~150 sentence pairs/sec
- Expected training time: 2-3 hours for 30 epochs

#### **Memory Usage**:
- BiRNN NER: ~2GB VRAM
- Seq2Seq (hidden_dim=512): ~6GB VRAM
- Seq2Seq (hidden_dim=256): ~3GB VRAM
- Batch size impact: Linear scaling (~100MB per batch)

#### **Quality Metrics**:
- BiRNN NER F1: 85-90%
- Seq2Seq BLEU (no attention): 15-20
- Seq2Seq BLEU (with attention): 25-30
- Attention visualization: >90% alignment accuracy

### Debugging Checklist:

```python
# Shape debugging
def debug_shapes(name, tensor):
    print(f"{name}: {tensor.shape}")

# Gradient flow check
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm()}")

# Attention weight sanity check
def verify_attention(attn_weights):
    assert torch.allclose(attn_weights.sum(dim=-1), 
                         torch.ones_like(attn_weights.sum(dim=-1)))
    assert (attn_weights >= 0).all()
    assert (attn_weights <= 1).all()
```

### Deployment Considerations:

1. **Model Export**:
   ```python
   # Save for inference
   torch.save({
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'config': config,
       'vocab': vocab
   }, 'checkpoint.pt')
   
   # Export to ONNX
   torch.onnx.export(model, dummy_input, "model.onnx")
   ```

2. **Inference Optimization**:
   - Use `model.eval()` and `torch.no_grad()`
   - Batch inference requests
   - Cache encoder outputs
   - Quantize model for CPU deployment

3. **API Integration**:
   ```python
   from flask import Flask, request, jsonify
   
   app = Flask(__name__)
   model = load_model('checkpoint.pt')
   
   @app.route('/translate', methods=['POST'])
   def translate():
       text = request.json['text']
       translation = model.translate(text)
       return jsonify({'translation': translation})
   ```

---

## 🙏 Credits

**Lecturer**: Ho-min Park  
**Affiliation**: Ghent University (Korea Campus)  
**Contact**: 
- Email: homin.park@ghent.ac.kr
- Email: powersimmani@gmail.com

**Course**: Deep Learning for Natural Language Processing  
**Lecture Number**: 12  
**Topic**: Advanced Sequence Models

**Acknowledgments**:
- Slide design inspired by modern pedagogical principles
- Code examples adapted from PyTorch tutorials
- Attention visualizations based on Bahdanau et al. (2015)

**License**: Educational use only. Please cite appropriately when using these materials.

**Version**: 1.0  
**Last Updated**: 2024

---

## 📌 Quick Reference Card

### Key Formulas:

**BiRNN Output**:
```
h→_t = RNN_forward(x_t, h→_{t-1})
h←_t = RNN_backward(x_t, h←_{t+1})
y_t = f([h→_t; h←_t])
```

**Attention Mechanism**:
```
e_tj = score(s_t, h_j)
α_tj = softmax(e_tj)
c_t = Σ_j α_tj * h_j
```

**Teacher Forcing**:
```
# Training
input_t = ground_truth_{t-1}

# Inference
input_t = prediction_{t-1}
```

### Important Hyperparameters:

| Parameter | Typical Range | Recommended |
|-----------|---------------|-------------|
| hidden_dim | 128-1024 | 512 |
| num_layers | 1-4 | 2 |
| dropout | 0.1-0.5 | 0.3 |
| learning_rate | 1e-4 to 1e-3 | 0.001 |
| batch_size | 16-128 | 64 |
| max_seq_len | 50-200 | 100 |
| teacher_forcing_ratio | 0.5-1.0 | 0.8→0.3 |

### Command Cheatsheet:

```bash
# Training
python train.py --model seq2seq --attention --batch-size 64 --epochs 30

# Inference
python translate.py --checkpoint best_model.pt --input "Hello world"

# Evaluation
python evaluate.py --checkpoint best_model.pt --test-data test.txt

# Visualization
python visualize_attention.py --checkpoint best_model.pt --sentence "example"
```

---

**End of README.md**

*For questions, clarifications, or additional materials, please contact the instructor.*