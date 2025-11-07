# Lecture 13: Transformer Architecture

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Total Slides:** 34  
**Difficulty Level:** Advanced (Graduate Level)  
**Estimated Duration:** 3-4 hours (lecture + practice)

This lecture provides a comprehensive exploration of the Transformer architecture, which revolutionized deep learning and became the foundation for modern AI systems including GPT, BERT, and multimodal models. Students will gain both theoretical understanding and practical implementation knowledge of self-attention mechanisms, multi-head attention, and complete Transformer models.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand and implement self-attention mechanisms** from scratch, including the Query-Key-Value framework and attention score computation
2. **Design and configure multi-head attention systems** with appropriate head dimensions and parallelization strategies
3. **Apply positional encoding techniques** to preserve sequential information in parallel architectures
4. **Construct complete Transformer encoder-decoder architectures** with proper layer normalization, residual connections, and masking
5. **Optimize Transformer training** using learning rate warmup, gradient clipping, and hyperparameter tuning strategies

---

## 📚 Lecture Structure

### Part 1/7: Introduction & Motivation (Slides 3-5)

**Topics Covered:**
- **Review of Last Lesson** (Slide 4)
  - Evolution from RNN → LSTM/GRU → Attention mechanisms
  - Problems: vanishing gradients, sequential processing limitations
  - Remaining issues: parallelization constraints, long-range dependencies
  
- **The Emergence and Impact of Transformers** (Slide 5)
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - Core innovations: eliminated recurrence, enabled massive parallelization
  - Impact: Foundation for GPT, BERT, ViT, and modern LLMs
  - Applications beyond NLP: computer vision, audio processing, multimodal AI

**Key Concepts:**
- Sequential processing bottlenecks in RNNs
- Attention mechanism evolution
- Paradigm shift from recurrence to pure attention
- Foundation model revolution

**Learning Outcomes:**
- Identify limitations of sequential models
- Explain the historical context leading to Transformers
- Articulate why Transformers became dominant in modern AI

---

### Part 2/7: Self-Attention Mechanism (Slides 6-11)

**Topics Covered:**
- **RNN Attention vs. Self-Attention** (Slide 7)
  - RNN Attention: Query from decoder, Keys/Values from encoder, sequential processing
  - Self-Attention: All from same sequence, parallel processing, O(n²) complexity
  - Advantages: better gradient flow, captures all position relationships
  
- **Query, Key, and Value Concepts** (Slide 8)
  - Query (Q): "What information am I looking for?"
  - Key (K): "What information do I contain?"
  - Value (V): "The actual information I store"
  - Database analogy: Query matches Keys to retrieve Values
  - Linear projections: Q = XW_Q, K = XW_K, V = XW_V
  - Numerical example with 5 tokens × 3 dimensions

- **Self-Attention Computation Process** (Slides 9-10)
  - Step-by-step computation with concrete examples
  - Attention score calculation: QK^T / √d_k
  - Softmax normalization for attention weights
  - Weighted sum of Values
  
- **Self-Attention Matrix Operations** (Slide 11)
  - Complete matrix formulation
  - Attention(Q, K, V) = softmax(QK^T / √d_k)V
  - Efficient batch computation strategies

**Key Concepts:**
- Query-Key-Value framework
- Scaled dot-product attention
- Attention score normalization
- Parallel computation advantages
- O(n²) complexity trade-offs

**Learning Outcomes:**
- Implement self-attention from scratch using matrix operations
- Calculate attention weights for given input sequences
- Explain the role of scaling factor √d_k
- Compare computational complexity of RNN vs. self-attention

---

### Part 3/7: Multi-Head Attention (Slides 12-16)

**Topics Covered:**
- **Why is Multi-Head Necessary?** (Slide 13)
  - Single head limitations: limited representation capacity, single attention pattern
  - Multi-head advantages: different heads learn different patterns (syntax vs. semantics)
  - Analogy: Similar to multiple CNN filters capturing different features
  - Empirical performance improvements
  
- **Multi-Head Attention Architecture** (Slide 14)
  - Typical configuration: h = 8 heads
  - Head dimension: d_k = d_model / h (e.g., 512/8 = 64)
  - Parallel head processing with separate W_Q, W_K, W_V for each head
  - Concatenation and final linear projection W_O
  - Formula: MultiHead(Q,K,V) = Concat(head₁,...,head_h)W_O

- **Multi-Head Operation Example** (Slide 15)
  - Concrete numerical example with 8 heads
  - Dimension preservation throughout pipeline
  
- **Implementation Points** (Slide 16)
  - Initialization: appropriate scaling for projection matrices
  - Regularization: dropout after attention weights and final output
  - Layer normalization placement strategies
  - Optimization: linear attention variants for long sequences
  - Trade-off: more heads increase capacity but require higher computational cost

**Key Concepts:**
- Multi-head parallelization
- Representation diversity
- Head dimension calculation
- Concatenation and projection
- Computational trade-offs

**Learning Outcomes:**
- Configure multi-head attention with appropriate dimensions
- Implement parallel head processing
- Explain why multiple heads improve performance
- Calculate parameter counts for multi-head systems

---

### Part 4/7: Positional Encoding (Slides 17-20)

**Topics Covered:**
- **The Need for Positional Information** (Slide 18)
  - Problem: Self-attention is permutation invariant
  - "cat ate mouse" = "mouse ate cat" without position info
  - RNN approach: inherent position through sequential processing
  - Transformer approach: needs explicit positional encoding

- **Positional Encoding Methods** (Slide 19)
  - **Sinusoidal** (Original Transformer):
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    - Advantages: works for unseen sequence lengths, deterministic pattern
  - **Learned** (BERT approach):
    - Trainable embedding matrix: PE ∈ ℝ^(max_len × d_model)
    - Better for fixed-length tasks, task-specific optimization
  - **Relative** (T5, Transformer-XL):
    - Encodes relative distances: distance = pos_i - pos_j
    - Captures relationships better, position-invariant patterns

- **Positional Encoding Visualization** (Slide 20)
  - Visual patterns of sinusoidal encodings
  - How different frequencies encode position information

**Key Concepts:**
- Permutation invariance problem
- Sinusoidal vs. learned encodings
- Relative positional representations
- Frequency-based position encoding
- Generalization to unseen lengths

**Learning Outcomes:**
- Implement sinusoidal positional encoding
- Compare different positional encoding methods
- Explain why position information is critical
- Choose appropriate encoding method for different tasks

---

### Part 5/7: Transformer Architecture (Slides 21-26)

**Topics Covered:**
- **Overall Transformer Structure** (Slide 22)
  - Encoder-decoder architecture overview
  - Input embeddings + positional encoding
  - N = 6 stacked layers for both encoder and decoder
  - Final linear layer and softmax for output

- **Detailed Analysis of the Encoder** (Slide 23)
  - Single encoder layer structure:
    1. Multi-head self-attention + residual + layer norm
    2. Feed-forward network + residual + layer norm
  - 6 stacked layers with different parameters
  - Dimension preservation: d_model = 512 throughout

- **Detailed Analysis of the Decoder** (Slide 24)
  - Three sub-layers per decoder layer:
    1. Masked multi-head self-attention (prevents attending to future)
    2. Encoder-decoder cross-attention (Q from decoder, K & V from encoder)
    3. Feed-forward network
  - All with residual connections and layer normalization
  - 6 stacked decoder layers

- **Feed-Forward Network & Layer Normalization** (Slide 25)
  - FFN structure: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
  - Dimension flow: d_model (512) → 4×d_model (2048) → d_model (512)
  - Two linear transformations with ReLU activation
  - Position-wise application (same FFN applied to each position)
  - Layer normalization: normalizes across feature dimension
  - Stabilizes training and speeds convergence

- **Training vs. Inference** (Slide 26)
  - Training: teacher forcing with parallel processing
  - Inference: autoregressive generation (one token at a time)
  - Masking differences between training and inference
  - Computational trade-offs

**Key Concepts:**
- Encoder-decoder architecture
- Residual connections and layer normalization
- Masked self-attention for causality
- Cross-attention mechanism
- Feed-forward network expansion
- Training vs. inference modes

**Learning Outcomes:**
- Construct complete Transformer encoder-decoder from components
- Implement proper residual connections and normalization
- Explain the role of each sub-layer
- Configure appropriate hyperparameters (layers, dimensions, heads)
- Understand training vs. inference differences

---

### Part 6/7: Implementation Tips (Slides 27-30)

**Topics Covered:**
- **Masking Implementation** (Slide 28)
  - **Padding Mask**: Ignores padding tokens, shape (batch_size, 1, 1, seq_len)
  - **Look-Ahead Mask (Causal)**: Upper triangular matrix, prevents attending to future
  - Application timing: before softmax in attention calculation
  - Mask values: use -inf for masked positions (becomes 0 after softmax)
  - Combined masks: element-wise OR operation

- **Learning Stabilization Techniques** (Slide 29)
  - **Learning Rate Warmup**: lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
  - **Label Smoothing**: reduces overconfidence (ε = 0.1 typical)
  - **Gradient Clipping**: prevents exploding gradients
  - **Dropout**: applied to attention weights and FFN outputs
  - **Weight Initialization**: Xavier or He initialization
  - **Mixed Precision Training**: FP16 for faster computation and reduced memory

- **Hyperparameter Guide** (Slide 30)
  - **Base Model**: d_model=512, N=6 layers, h=8 heads, d_ff=2048, dropout=0.1
  - **Large Model**: d_model=1024, N=12+ layers, h=16 heads, d_ff=4096
  - **Warmup Steps**: 4000 steps (typical)
  - **Batch Size**: as large as GPU memory allows
  - **Adam Optimizer**: β₁=0.9, β₂=0.98, ε=10⁻⁹

**Key Concepts:**
- Proper masking for causality
- Training stability techniques
- Learning rate scheduling
- Regularization strategies
- Hyperparameter scaling principles
- Mixed precision optimization

**Learning Outcomes:**
- Implement correct masking for encoder and decoder
- Configure learning rate warmup schedules
- Apply appropriate regularization techniques
- Tune hyperparameters for different model sizes
- Stabilize training for deep Transformers

---

### Part 7/7: Applications & Next Steps (Slides 31-33)

**Topics Covered:**
- **Transformer Applications** (Slide 32)
  - **NLP**: GPT (text generation), BERT (understanding), T5 (text-to-text)
  - **Machine Translation**: Near human-level performance
  - **Computer Vision**: ViT (Vision Transformers), DINO, CLIP
  - **Speech Processing**: Whisper (transcription), audio generation
  - **Multimodal AI**: DALL-E, GPT-4, Flamingo
  - **Time Series**: Temporal Fusion Transformer, forecasting
  - **Code Generation**: GitHub Copilot, AlphaCode
  - **Biology & Science**: AlphaFold2 for protein structure prediction

- **Summary and Next Steps** (Slide 33)
  - Key concepts recap: self-attention, multi-head, positional encoding
  - Advantages: parallelization, long-range dependencies, flexibility
  - Challenges: O(n²) complexity, large memory requirements
  - Next steps:
    1. Advanced topics: sparse attention, linear attention
    2. Efficient Transformers: Reformer, Linformer, Performer
    3. Practice: implement mini-Transformer from scratch
    4. Explore libraries: Hugging Face Transformers
  - Recommended reading: "Attention Is All You Need" (Vaswani et al., 2017)

**Key Concepts:**
- Domain-agnostic architecture
- Transfer learning and foundation models
- Efficient Transformer variants
- Practical implementation frameworks
- Future research directions

**Learning Outcomes:**
- Identify appropriate Transformer applications for different domains
- Understand the breadth of Transformer impact across fields
- Navigate modern Transformer libraries and frameworks
- Recognize efficiency challenges and solutions
- Plan next steps for advanced Transformer study

---

## 🛠️ Prerequisites

### Required Background Knowledge:
- **Linear Algebra**: Matrix multiplication, dot products, eigenvalues
- **Deep Learning Fundamentals**: 
  - Neural network basics (forward/backward propagation)
  - Activation functions (ReLU, softmax)
  - Loss functions and optimization (SGD, Adam)
- **Sequence Models**: 
  - RNN, LSTM, GRU architectures
  - Basic attention mechanisms (encoder-decoder attention)
- **Python Programming**: 
  - NumPy for numerical operations
  - Basic PyTorch or TensorFlow knowledge

### Software Requirements:
- **Python** 3.8+ 
- **PyTorch** 1.12+ or **TensorFlow** 2.8+
- **NumPy** 1.21+
- **Matplotlib** for visualization
- **Jupyter Notebook** or **Google Colab**
- **GPU**: NVIDIA GPU with CUDA 11.0+ (8GB+ VRAM recommended)
  - For CPU-only: reduce batch size and model dimensions

### Recommended (but not required):
- **Hugging Face Transformers** library experience
- **Understanding of NLP basics** (tokenization, embeddings)
- **Experience with tensor operations** and automatic differentiation

---

## 💻 Hands-on Components

### Project 1: Implementing Self-Attention (2-3 hours)
**Objective**: Build a self-attention layer from scratch using only NumPy or PyTorch tensors.

**Tasks**:
1. Implement Query, Key, Value projection matrices
2. Calculate attention scores with proper scaling (√d_k)
3. Apply softmax normalization
4. Compute weighted sum of Values
5. Test with sample sequences (5 tokens × 512 dimensions)
6. Visualize attention weights as heatmaps

**Expected Outputs**:
- Working self-attention function with shape validation
- Attention weight visualization showing which tokens attend to which
- Comparison of attention patterns for different input sequences

**Success Criteria**:
- Correct output shapes for all intermediate steps
- Attention weights sum to 1.0 for each query position
- Runtime < 0.1 seconds for sequences up to 100 tokens

---

### Project 2: Multi-Head Attention Module (3-4 hours)
**Objective**: Extend single-head attention to multi-head attention with 8 parallel heads.

**Tasks**:
1. Configure 8 attention heads with d_k = 64 (for d_model = 512)
2. Implement parallel head processing
3. Concatenate head outputs and apply final linear projection W_O
4. Add residual connections and layer normalization
5. Implement dropout (rate = 0.1) for regularization
6. Compare learned attention patterns across different heads

**Expected Outputs**:
- Multi-head attention module matching Transformer paper specifications
- Per-head attention pattern visualizations
- Performance comparison: single-head vs. multi-head on sample task

**Success Criteria**:
- Parameter count matches theoretical calculation: 4 × d_model × d_model
- Different heads learn distinct attention patterns
- Gradient flow verification through backpropagation test

---

### Project 3: Complete Transformer Encoder (4-6 hours)
**Objective**: Build a complete 6-layer Transformer encoder for sequence classification.

**Tasks**:
1. Implement positional encoding (sinusoidal method)
2. Stack 6 encoder layers with multi-head attention + FFN
3. Add layer normalization after each sub-layer
4. Implement residual connections properly
5. Create padding mask for variable-length sequences
6. Train on sentiment classification task (e.g., IMDB or SST-2)

**Expected Outputs**:
- Encoder achieving >80% validation accuracy on sentiment classification
- Training curves showing stable convergence
- Attention pattern analysis for sample sentences

**Success Criteria**:
- Model converges within 10 epochs
- Validation accuracy within 5% of reported benchmarks
- Proper handling of sequences from 10 to 512 tokens

---

### Project 4: Encoder-Decoder for Translation (6-8 hours)
**Objective**: Build full Transformer for machine translation (English → French).

**Tasks**:
1. Implement 6-layer encoder and 6-layer decoder
2. Add masked self-attention in decoder (causal mask)
3. Implement encoder-decoder cross-attention
4. Create look-ahead mask for training
5. Implement beam search for inference (beam width = 5)
6. Train on WMT English-French dataset (can use subset)
7. Evaluate with BLEU score

**Expected Outputs**:
- Translation model with BLEU score >25 on validation set
- Comparison of greedy decoding vs. beam search
- Attention visualization showing alignment between source and target
- Analysis of common translation errors

**Success Criteria**:
- Model generates grammatically correct translations
- Attention weights show reasonable source-target alignment
- Beam search improves BLEU by at least 2 points vs. greedy
- Inference speed: >10 sentences/second on GPU

---

### Project 5: Hyperparameter Optimization (3-4 hours)
**Objective**: Explore impact of different hyperparameters on Transformer performance.

**Tasks**:
1. Vary number of heads: {4, 8, 16} and measure accuracy/speed trade-offs
2. Test different learning rate schedules with warmup steps {1000, 4000, 10000}
3. Experiment with dropout rates {0.0, 0.1, 0.3}
4. Compare label smoothing values {0.0, 0.1, 0.2}
5. Analyze effect of model depth (4 vs. 6 vs. 8 layers)

**Expected Outputs**:
- Comprehensive hyperparameter sensitivity report
- Training time and memory usage measurements
- Optimal configuration recommendation for your task
- Visualization of hyperparameter impact on convergence

**Success Criteria**:
- At least 15 different configurations tested
- Clear documentation of trade-offs (accuracy vs. speed vs. memory)
- Reproducible experiments with fixed random seeds

---

## 📖 Additional Resources

### Essential Papers:
1. **"Attention Is All You Need"** - Vaswani et al., 2017 (NeurIPS)
   - Original Transformer paper - must read
   - https://arxiv.org/abs/1706.03762

2. **"BERT: Pre-training of Deep Bidirectional Transformers"** - Devlin et al., 2018
   - Encoder-only architecture for language understanding
   - https://arxiv.org/abs/1810.04805

3. **"Language Models are Unsupervised Multitask Learners"** - Radford et al., 2019 (GPT-2)
   - Decoder-only architecture for generation
   - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

4. **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** - Dosovitskiy et al., 2020 (ViT)
   - Vision Transformers extending architecture to images
   - https://arxiv.org/abs/2010.11929

### Video Lectures:
- **Stanford CS224N**: "Transformers and Self-Attention" by Chris Manning
- **MIT 6.S191**: "Recurrent Neural Networks and Transformers"
- **DeepMind x UCL**: "Deep Learning Lecture Series on Transformers"
- **The Illustrated Transformer** by Jay Alammar (visual walkthrough)

### Code Repositories:
- **Harvard NLP - The Annotated Transformer**: 
  - Line-by-line implementation with explanations
  - http://nlp.seas.harvard.edu/annotated-transformer/

- **Hugging Face Transformers**:
  - Production-ready implementations of 100+ Transformer models
  - https://github.com/huggingface/transformers

- **nanoGPT by Andrej Karpathy**:
  - Minimalist GPT implementation for learning
  - https://github.com/karpathy/nanoGPT

- **fairseq by Meta AI**:
  - Research-oriented Transformer toolkit
  - https://github.com/facebookresearch/fairseq

### Textbooks:
- **"Speech and Language Processing"** (3rd ed.) - Jurafsky & Martin, Chapter 10 (Transformers)
- **"Dive into Deep Learning"** - Zhang et al., Chapter 11 (Attention Mechanisms)
- **"Deep Learning"** - Goodfellow et al., Chapter 12 (Sequence Modeling)

### Online Courses:
- **Fast.ai - Practical Deep Learning** (Part 2 covers Transformers in detail)
- **DeepLearning.AI - Natural Language Processing Specialization**
- **Coursera - Attention Models** by Andrew Ng

### Tools & Frameworks:
- **Weights & Biases**: Experiment tracking and hyperparameter tuning
- **TensorBoard**: Visualization of training metrics and attention patterns
- **Comet.ml**: Model tracking and comparison
- **Optuna**: Automated hyperparameter optimization

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Week 1**: Study Parts 1-2 (Introduction and Self-Attention)
   - Watch lecture slides 1-11
   - Implement Project 1 (Self-Attention from scratch)
   - Read "Attention Is All You Need" sections 3.1-3.2

2. **Week 2**: Study Parts 3-4 (Multi-Head and Positional Encoding)
   - Review slides 12-20
   - Complete Project 2 (Multi-Head Attention)
   - Experiment with different positional encoding methods

3. **Week 3**: Study Part 5 (Full Architecture)
   - Analyze slides 21-26
   - Build Project 3 (Complete Encoder)
   - Compare with Harvard Annotated Transformer implementation

4. **Week 4**: Study Parts 6-7 (Implementation and Applications)
   - Review slides 27-33
   - Tackle Project 4 (Encoder-Decoder Translation)
   - Conduct Project 5 (Hyperparameter Optimization)

### For Instructors:
- **Lecture Format**: 3-hour session with 10-minute break after Part 4
- **Interactive Elements**: 
  - Live coding of self-attention mechanism (15 mins)
  - Class discussion on multi-head necessity (10 mins)
  - Whiteboard derivation of attention formulas (20 mins)
  
- **Assignment Sequence**:
  - Week 1: Project 1 (Self-Attention) - Due in 1 week
  - Week 2: Project 2 (Multi-Head) - Due in 1 week
  - Week 3-4: Project 3 or 4 (Encoder or Translation) - Due in 2 weeks
  - Optional: Project 5 for extra credit

- **Flipped Classroom Option**:
  - Students watch slide recordings before class
  - Class time for hands-on implementation and debugging
  - Instructor circulates to help with coding issues

### Study Tips:
- **Don't skip the math**: Work through attention score calculations by hand
- **Visualize everything**: Plot attention weights, positional encodings, training curves
- **Start small**: Test with toy examples (5 tokens) before scaling up
- **Debug systematically**: Check shapes at every step, verify gradients flow
- **Compare implementations**: Cross-reference your code with Harvard Annotated Transformer

---

## 📊 Assessment Suggestions

### Formative Assessment (During Learning):
1. **Conceptual Quizzes** (5-10 questions per part):
   - Why is the scaling factor √d_k necessary in attention?
   - What would happen if we removed positional encoding?
   - How does multi-head attention differ from ensemble learning?
   
2. **Code Review Checkpoints**:
   - Instructor reviews Project 1 implementation for correctness
   - Peer review of attention weight visualizations
   - Unit tests for each module (provided test cases)

3. **Interactive Debugging Sessions**:
   - Common bugs workshop (shape mismatches, gradient issues)
   - Live coding challenge: fix broken attention implementation

### Summative Assessment Options:

#### Option 1: Comprehensive Project (40% of grade)
**Build Transformer from Scratch for Custom Task**
- Choose application: translation, summarization, or question answering
- Implement complete encoder-decoder (no pre-built libraries for core attention)
- Achieve minimum performance benchmarks
- Write 5-page report analyzing design decisions

**Grading Rubric**:
- Implementation Correctness (40%): Passes all unit tests, correct shapes
- Performance (20%): Meets benchmark (e.g., BLEU >25 for translation)
- Code Quality (15%): Documentation, modularity, efficiency
- Analysis (25%): Insights on hyperparameter tuning, error analysis

#### Option 2: Theory + Implementation Exam (30% of grade)
**Part A: Theory (45 minutes)**
- Derive attention formula from first principles
- Explain multi-head attention mathematically
- Analyze computational complexity of Transformer vs. RNN
- Design positional encoding for novel use case

**Part B: Coding (90 minutes)**
- Implement masked multi-head attention module
- Debug provided broken Transformer code
- Optimize attention for long sequences
- Answer questions about code behavior

**Grading**:
- Theory correctness (50%)
- Code functionality (40%)
- Code efficiency (10%)

#### Option 3: Research Paper Replication (35% of grade)
**Replicate Key Experiments from Transformer Paper**
- Implement WMT English-German translation model
- Reproduce Table 1 results (BLEU scores within ±2 points)
- Conduct ablation studies:
  - Effect of removing multi-head attention
  - Impact of different positional encoding methods
  - Performance vs. number of layers
- 8-page report in NeurIPS format

**Grading**:
- Replication accuracy (40%)
- Ablation study design and execution (30%)
- Analysis and insights (20%)
- Report quality (10%)

### Continuous Assessment Components:
1. **Weekly Implementation Milestones** (20%):
   - Project 1 submission (5%)
   - Project 2 submission (5%)
   - Project 3 or 4 submission (10%)

2. **Class Participation** (10%):
   - Answering conceptual questions
   - Helping peers in debugging sessions
   - Contributing to discussion forums

3. **Final Presentation** (10%):
   - 10-minute presentation of final project
   - Demo of working implementation
   - Q&A on design decisions

### Recommended Grade Weights:
- **Implementation Projects**: 40% (Projects 1-4)
- **Theory Exam or Paper Replication**: 30%
- **Final Project/Presentation**: 20%
- **Participation and Quizzes**: 10%

### Alternative Assessment (Portfolio-Based):
Students maintain a GitHub repository with:
- All project implementations
- Jupyter notebooks with experiments
- README documenting learning journey
- Blog posts explaining key concepts

**Assessed on**:
- Code quality and documentation (40%)
- Breadth of experiments (30%)
- Depth of understanding in explanations (30%)

---

## 📝 Notes for Implementation

### Technical Requirements:

#### Minimum Hardware:
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 16GB minimum (32GB recommended for large models)
- **Storage**: 50GB free space for datasets and model checkpoints
- **GPU** (strongly recommended):
  - NVIDIA GTX 1060 (6GB VRAM) - minimum for base model
  - NVIDIA RTX 3060 (12GB VRAM) - recommended
  - NVIDIA A100 (40GB VRAM) - for large-scale experiments

#### If GPU is unavailable:
- Use **Google Colab** (free tier provides T4 GPU)
- Use **Kaggle Notebooks** (30 hours/week free GPU)
- Reduce model size: d_model=256, N=3 layers, h=4 heads
- Use gradient accumulation to simulate larger batches

### Common Implementation Pitfalls:

1. **Shape Mismatches in Attention**:
   ```python
   # Wrong: (batch, seq, d_model) @ (d_model, d_k)
   # Correct: (batch, seq, d_model) @ (d_model, d_model)
   # Then split into heads: (batch, heads, seq, d_k)
   ```

2. **Forgetting to Scale Attention Scores**:
   ```python
   # Must divide by sqrt(d_k) before softmax!
   scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
   ```

3. **Incorrect Masking**:
   ```python
   # Causal mask must be applied BEFORE softmax
   scores = scores.masked_fill(mask == 0, float('-inf'))
   attention_weights = torch.softmax(scores, dim=-1)
   ```

4. **Positional Encoding Not Added**:
   ```python
   # Must add (not concatenate) to input embeddings
   x = token_embeddings + positional_encoding
   ```

5. **Layer Normalization Placement**:
   - Original paper: Post-LN (after residual addition)
   - Modern practice: Pre-LN (before sub-layer) for stability
   - Choose one consistently!

### Memory Optimization Tips:

1. **Gradient Checkpointing**: Trade computation for memory
   ```python
   # PyTorch
   from torch.utils.checkpoint import checkpoint
   output = checkpoint(encoder_layer, x)
   ```

2. **Mixed Precision Training**:
   ```python
   # Use automatic mixed precision (AMP)
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       output = model(input)
   ```

3. **Batch Size Adjustment**:
   - Start with batch_size = 32 and reduce if OOM error
   - Use gradient accumulation for effective larger batches

4. **Attention Optimization**:
   - For sequences >512: consider Flash Attention or Memory-Efficient Attention
   - Use `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+)

### Debugging Checklist:

- [ ] All tensor shapes are correct at each step
- [ ] Attention weights sum to 1.0 along the correct dimension
- [ ] Positional encoding is added, not concatenated
- [ ] Masks are applied before softmax with -inf values
- [ ] Learning rate warmup is implemented correctly
- [ ] Gradient norms are reasonable (not NaN or exploding)
- [ ] Layer normalization is applied consistently
- [ ] Dropout is disabled during evaluation

### Performance Benchmarks (for validation):

**Base Model (d_model=512, N=6) on WMT En-De Translation:**
- Training time: ~12 hours on single V100 GPU
- Memory usage: ~8GB VRAM with batch_size=32
- BLEU score: ~27.3 (should be within ±1.5)

**Inference Speed:**
- Encoder: ~100 sentences/second (batch_size=1)
- Decoder: ~10 sentences/second (beam_size=5)

If your implementation is significantly slower, profile with:
```python
import torch.profiler
with torch.profiler.profile() as prof:
    model(input)
print(prof.key_averages().table())
```

### Dataset Recommendations:

**For Practice (Small Datasets)**:
- IMDB Sentiment: 25k reviews, binary classification
- SST-2: Stanford Sentiment Treebank, 67k sentences
- SNLI: 570k sentence pairs for entailment

**For Translation**:
- WMT14 En-De: ~4.5M sentence pairs
- WMT14 En-Fr: ~36M sentence pairs
- Use sacrebleu library for standardized evaluation

**For Language Modeling**:
- WikiText-2: 2M tokens (small, fast experiments)
- WikiText-103: 103M tokens (larger scale)

### Version Control and Reproducibility:

```bash
# Save exact environment
pip freeze > requirements.txt

# Log hyperparameters
# Use wandb, mlflow, or sacred for experiment tracking

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

---

## 🙏 Credits

**Instructor**: Ho-min Park  
**Affiliation**: Ghent University  
**Contact**: 
- Email: homin.park@ghent.ac.kr
- Email: powersimmani@gmail.com

**Course**: Advanced Deep Learning (Graduate Level)  
**Academic Year**: 2024-2025

### Acknowledgments:
This lecture material is based on the seminal work "Attention Is All You Need" by Vaswani et al. (2017) and draws inspiration from:
- The Annotated Transformer by Harvard NLP
- Stanford CS224N course materials
- Original Transformer implementation by Google Research

**License**: These educational materials are provided for academic use. Please cite this course if you adapt these materials for your own teaching.

---

**Last Updated**: November 2024  
**Version**: 1.0  
**Slide Count**: 34 slides across 7 comprehensive parts

For questions, clarifications, or suggestions for improvement, please contact the instructor via email.