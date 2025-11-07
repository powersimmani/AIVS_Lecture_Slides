# Lecture 14: Pre-trained Language Models & LLM Era
## Deep Learning for Natural Language Processing

---

## 📋 Overview

**Course**: Deep Learning for Natural Language Processing  
**Lecture Number**: 14  
**Topic**: Pre-trained Language Models & LLM Era  
**Instructor**: Ho-min Park  
**Email**: homin.park@ghent.ac.kr, powersimmani@gmail.com  
**Total Slides**: 36  
**Difficulty Level**: Advanced  
**Estimated Duration**: 3-4 hours

This lecture provides a comprehensive exploration of pre-trained language models, marking the paradigm shift in NLP from task-specific models to foundation models. The course covers the entire spectrum from BERT to GPT, including modern techniques like RLHF and prompt engineering.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand the Paradigm Shift**: Explain the transformation from task-specific models to foundation models and articulate the advantages of transfer learning in NLP
2. **Distinguish Model Architectures**: Compare and contrast encoder-based (BERT), decoder-based (GPT), and encoder-decoder (T5, BART) architectures, understanding when to use each
3. **Apply Fine-tuning Strategies**: Implement various fine-tuning approaches including full fine-tuning, parameter-efficient methods (LoRA, Adapters), and select appropriate strategies based on resource constraints
4. **Master Prompt Engineering**: Design effective prompts using advanced techniques such as Chain-of-Thought reasoning, few-shot learning, and understand the trade-offs between prompting and fine-tuning
5. **Evaluate Modern Techniques**: Critically assess current trends including RLHF, instruction tuning, multimodal capabilities, and ethical considerations in deploying large language models

---

## 📚 Lecture Structure

### Part 1: Introduction & Paradigm Shift (Slides 3-5)

#### Topics Covered:

**Slide 4: Review of Previous Lessons**
- Evolution timeline: Rule-Based Systems → RNN & LSTM → Seq2Seq Models → Word Embeddings → Attention Mechanism
- Limitations of previous approaches: shallow representations, task-specific training, heavy feature engineering
- The need for better context understanding and transfer learning capabilities

**Slide 5: Paradigm Shift in AI**
- **Old Paradigm**: Task-specific models trained from scratch, requires labeled data for each task, no knowledge transfer
- **New Paradigm**: General-purpose foundation models, pre-train on massive text then adapt, self-supervised learning, emergent capabilities at scale, democratization of AI

#### Key Concepts:
- Transfer learning revolution in NLP
- Foundation models vs. task-specific models
- Self-supervised learning from unlabeled data
- Emergent capabilities through scaling
- Democratization of AI technology

#### Learning Outcomes:
- Students can articulate the limitations of pre-transformer NLP approaches
- Students can explain why pre-training on large corpora enables better downstream performance
- Students understand the economic and accessibility implications of the foundation model paradigm

---

### Part 2: Pre-training Concepts (Slides 6-9)

#### Topics Covered:

**Slide 7: What is Pre-training?**
- Two-stage learning process: Pre-training → Fine-tuning
- Self-supervised learning objective: predict masked/next words from massive text corpora
- Benefits: captures syntax & semantics, learns world knowledge, creates universal feature extractors, reduces downstream data requirements

**Slide 8: Language Modeling Objective Function**
- **Autoregressive LM (Causal/Left-to-Right)**:
  - Predicts next token: P(w_t | w_1, ..., w_{t-1})
  - Causal masking (unidirectional)
  - Optimal for generation tasks
  - Used in GPT-style models
- **Masked LM (Bidirectional Context)**:
  - Predicts masked tokens: P(w_masked | context)
  - Bidirectional context (left + right)
  - Better for understanding tasks
  - Used in BERT-style models

**Slide 9: The Importance of Scale**
- Scaling Laws: Performance improves predictably with model size
- More parameters capture more patterns and knowledge
- Larger datasets provide richer knowledge coverage
- Emergent abilities appear at scale thresholds (100B+ parameters)
- Few-shot learning capabilities dramatically improve with scale
- Trade-off: Performance vs. Efficiency & Accessibility

#### Key Concepts:
- Self-supervised pre-training objectives
- Autoregressive vs. masked language modeling
- Scaling laws and emergent capabilities
- Transfer learning efficiency
- The importance of data scale and quality

#### Learning Outcomes:
- Students can implement basic pre-training objectives (MLM, CLM)
- Students can predict which objective function suits different downstream tasks
- Students understand the relationship between model scale and capability emergence
- Students can calculate computational requirements for different model sizes

---

### Part 3: BERT - Encoder-based Models (Slides 10-14)

#### Topics Covered:

**Slide 11: Introduction to BERT**
- Bidirectional Encoder Representations from Transformers
- Transformer encoder architecture (12-24 layers)
- Bidirectional context understanding
- Pre-training on Wikipedia + BookCorpus (3.3B words)

**Slide 12: BERT Pre-training**
- **Task 1: Masked Language Model (MLM)**
  - 15% token masking strategy
  - Uses bidirectional context
  - Example: "The cat [MASK] on the mat" → predict "sat"
- **Task 2: Next Sentence Prediction (NSP)**
  - Binary classification task
  - [CLS] token for sentence-level representation
  - Example: Sentence A + Sentence B → IsNext or NotNext
- Tokenization: WordPiece with 30K vocabulary
- Special tokens: [CLS], [SEP], [MASK], [PAD]
- Training: 4 days on 4-16 Cloud TPUs

**Slide 13: BERT Fine-tuning**
- **Sequence Classification**: Uses [CLS] token representation (sentiment analysis, text classification)
- **Token Classification**: Uses each token's output (NER, POS tagging)
- **Question Answering**: Predicts start & end spans (SQuAD, reading comprehension)
- End-to-end fine-tuning: All parameters updated with task-specific data
- Fast training: Only few epochs needed
- Transfer learning: Enables few-shot and zero-shot capabilities

**Slide 14: BERT Family Models**
- **Optimization Branch**:
  - RoBERTa: No NSP, dynamic masking, larger batches
  - ALBERT: Parameter sharing, factorized embeddings
  - DistilBERT: 6 layers, 40% smaller, 97% performance
- **Innovation Branch**:
  - ELECTRA: Discriminative pre-training, more efficient
  - DeBERTa: Disentangled attention, enhanced mask decoder
- **Domain-Specific**: BioBERT, SciBERT, ClinicalBERT
- **Multilingual**: mBERT, XLM-R (100+ languages)

#### Key Concepts:
- Bidirectional context understanding through masked language modeling
- Multi-task pre-training (MLM + NSP)
- Transfer learning through fine-tuning
- Model compression and efficiency improvements
- Domain adaptation strategies

#### Learning Outcomes:
- Students can implement BERT fine-tuning for classification tasks
- Students can select appropriate BERT variants for specific use cases
- Students understand the trade-offs between model size and performance
- Students can apply domain-specific BERT models to specialized tasks

---

### Part 4: GPT - Decoder-based Models (Slides 15-19)

#### Topics Covered:

**Slide 16: Introduction to the GPT Series**
- Generative Pre-trained Transformer architecture
- Autoregressive language modeling approach
- Unidirectional (left-to-right) attention
- Focus on generation capabilities

**Slide 17: GPT Pre-training**
- Causal language modeling: Predict next token autoregressively
- Training data: Diverse internet text (Common Crawl, WebText)
- BPE (Byte-Pair Encoding) tokenization
- Unidirectional attention enables efficient generation
- Pattern learning: facts, reasoning, world knowledge from data
- No explicit fine-tuning needed for GPT-3+

**Slide 18: GPT-3 and Few-shot Learning**
- Model scale: 175B parameters, 300B tokens
- **Zero-shot**: Task description only, no examples
- **One-shot**: Single example provided
- **Few-shot**: Multiple examples (typically 3-10)
- In-context learning: Learn from context without weight updates
- Emergent capabilities: arithmetic, translation, reasoning
- Democratization: Natural language interfaces for all users

**Slide 19: GPT Family Development**
- **InstructGPT**: RLHF for instruction following
- **ChatGPT**: Conversation-optimized with safety alignment
- **GPT-4**: Multimodal, improved reasoning & reliability
- **Codex**: Code generation, powers GitHub Copilot
- Open source alternatives: GPT-Neo, GPT-J, Bloom, LLaMA
- Continuous improvements: safety, accuracy, API access

#### Key Concepts:
- Autoregressive generation and causal masking
- In-context learning without parameter updates
- Few-shot learning capabilities at scale
- Instruction following through RLHF
- Multimodal understanding and generation

#### Learning Outcomes:
- Students can design effective few-shot prompts for various tasks
- Students understand the relationship between model scale and in-context learning
- Students can compare generation quality across different GPT versions
- Students can implement basic prompt-based applications using GPT APIs

---

### Part 5: Encoder-Decoder Models (Slides 20-22)

#### Topics Covered:

**Slide 21: T5 - Text-to-Text Framework**
- Full Transformer architecture: Encoder + Decoder
- Unified framework: All NLP tasks → Text generation
- Task prefixes: "translate English to German:", "summarize:", "sentiment:", etc.
- Pre-training: C4 corpus (750GB), span corruption objective
- Scalability: T5-Small → T5-Base → T5-Large → T5-3B → T5-11B
- Strong performance across diverse NLP tasks

**Slide 22: BART and Other Models**
- **BART**: Denoising autoencoder (BERT encoder + GPT decoder)
- Corruption schemes: token masking, deletion, span infilling, sentence permutation, document rotation
- **mBART**: Multilingual BART for 50+ languages
- **PEGASUS**: Gap sentence generation for summarization
- **Marian**: Efficient neural machine translation
- Best suited for sequence-to-sequence tasks: translation, summarization, QA

#### Key Concepts:
- Encoder-decoder architecture advantages
- Text-to-text unified framework
- Denoising autoencoder pre-training
- Span corruption objectives
- Multilingual and domain-specific variants

#### Learning Outcomes:
- Students can select appropriate architecture (encoder-only, decoder-only, encoder-decoder) for specific tasks
- Students can implement T5 fine-tuning for custom text-to-text tasks
- Students understand the corruption strategies used in BART pre-training
- Students can evaluate trade-offs between different seq2seq architectures

---

### Part 6: Fine-tuning Strategies (Slides 23-26)

#### Topics Covered:

**Slide 24: Full Fine-tuning vs. Transfer Learning**
- **Full Fine-tuning**: Update all model parameters on downstream task
  - Advantages: Maximum performance, task-specific optimization
  - Disadvantages: High computational cost, requires significant GPU memory
- **Transfer Learning Approaches**:
  - Feature extraction: Freeze base model, train only task-specific layers
  - Partial fine-tuning: Freeze early layers, fine-tune later layers
  - Progressive unfreezing: Gradually unfreeze layers during training

**Slide 25: Parameter-Efficient Fine-tuning (PEFT)**
- **LoRA (Low-Rank Adaptation)**:
  - Freeze original weights, inject trainable low-rank matrices
  - Only 0.1-1% of parameters trainable
  - Maintains performance with minimal memory overhead
- **Adapters**: Small bottleneck layers inserted between transformer layers
- **Prefix Tuning**: Learn task-specific prefix embeddings
- **Prompt Tuning**: Learn soft prompt embeddings (similar to prefix tuning)
- Benefits: Memory efficient, modular task switching, prevent catastrophic forgetting

**Slide 26: Fine-tuning Practical Tips**
- **Learning Rate**: Start with 1e-5 to 5e-5 range
- **Warmup Steps**: Use warmup to stabilize training
- **Learning Rate Schedule**: Warmup → Main training → Decay
- **Monitoring**: Track validation loss to prevent overfitting
- **Early Stopping**: Stop based on validation metrics
- **Data Quality**: Quality > Quantity for better results
- **Batch Size**: Affects convergence & generalization
- **Multi-task Learning**: Consider for related tasks
- **Best Practice**: Save checkpoints, test multiple configurations

#### Key Concepts:
- Parameter-efficient fine-tuning methods
- Low-rank adaptation (LoRA) mathematics
- Learning rate scheduling strategies
- Overfitting detection and prevention
- Checkpoint management and model versioning

#### Learning Outcomes:
- Students can implement LoRA fine-tuning for large models
- Students can design appropriate learning rate schedules
- Students can detect and prevent overfitting through validation monitoring
- Students understand memory-performance trade-offs in different fine-tuning approaches
- Students can select optimal batch sizes based on available resources

---

### Part 7: Prompting & In-Context Learning (Slides 27-30)

#### Topics Covered:

**Slide 28: Prompt Engineering Basics**
- **Bad Prompts**: Vague instructions, missing context, no format specification
- **Good Prompts**: Clear role assignment, specific format requirements, complete examples
- Key Principles:
  - Clear Instructions: Explicit & precise
  - Context: Provide background information
  - Format: Define output structure
  - Role: "You are an expert in..."
- Iterative refinement: Test and improve prompts
- Consistency: Maintain format across examples

**Slide 29: Advanced Prompting Techniques**
- **Chain-of-Thought (CoT)**: "Let's think step by step"
  - Breaks down reasoning into explicit steps
  - Significantly improves mathematical and logical reasoning
  - Example: Multi-step math problems with intermediate calculations
- **Few-shot Learning**: Provide multiple input-output examples
- **Self-consistency**: Sample multiple outputs, vote for most consistent answer
- **Tree of Thoughts**: Explore multiple reasoning paths simultaneously
- **ReAct**: Reasoning and acting in interleaved manner
- **Auto Prompt Optimization**: Use evolutionary methods to improve prompts
- **Meta-prompting**: Prompts that generate prompts for different tasks

**Slide 30: Fine-tuning vs. Prompting**
- **Prompting Advantages**:
  - Quick: No training needed
  - Flexible: Easy to iterate
  - API-based: Simple access
  - Cost: Pay per API call
  - Ideal for: Prototyping, few examples, varied tasks, quick experiments
- **Fine-tuning Advantages**:
  - Better performance: Task-specific optimization
  - Consistent: Reliable outputs
  - Private data: Full control
  - Infrastructure: Requires training resources
  - Ideal for: Production, large datasets, low latency, custom domains
- **Decision Factors**: Requirements (task complexity), data (size & quality), resources (cost & infrastructure)
- **Hybrid Approach**: Prompt-based fine-tuning (instruction tuning)
- **Future Trend**: Prompting becoming more powerful with larger models

#### Key Concepts:
- Prompt design principles and best practices
- Chain-of-thought reasoning mechanisms
- In-context learning without parameter updates
- Comparison of prompting vs. fine-tuning approaches
- Advanced prompting techniques for complex tasks

#### Learning Outcomes:
- Students can design effective prompts for various task types
- Students can implement Chain-of-Thought reasoning for complex problems
- Students can evaluate when to use prompting vs. fine-tuning
- Students understand the cost-benefit analysis of different approaches
- Students can apply advanced techniques like self-consistency and Tree of Thoughts

---

### Part 8: Current Trends (Slides 31-33)

#### Topics Covered:

**Slide 32: RLHF and Instruction Tuning**
- **Three-Stage RLHF Pipeline**:
  - Stage 1: Supervised Fine-tuning on high-quality demonstrations
  - Stage 2: Reward Model training from human preference comparisons
  - Stage 3: PPO (Proximal Policy Optimization) training using reward model
- Benefits:
  - Helpfulness: Better at following instructions
  - Truthfulness: More accurate and reliable
  - Safety: Reduces harmful outputs
- **Instruction Tuning**: Train on diverse task instructions to improve generalization
- **Constitutional AI**: Self-improvement through principles and guidelines
- Key to Success: Fundamental to ChatGPT and assistant models

**Slide 33: Present and Future**
- **Current Trends**:
  - Multimodal: Text, image, audio, video integration
  - Mixture of Experts: Sparse activation for efficiency
  - Long Context: 100K+ tokens processing
  - Better Reasoning: Math, logic, commonsense improvements
- **Retrieval Augmentation**: Combining LMs with knowledge bases for accurate information
- **Smaller Efficient Models**: On-device and edge deployment capabilities
- **Enhanced Capabilities**: Mathematical reasoning, logical thinking, commonsense understanding
- **Context Processing**: 100K+ tokens enables handling entire books and documents
- **Specialized Domain Models**: Medical, Legal, Scientific, Finance, Engineering
- **AGI Considerations**: Focus on capabilities, safety, and alignment

#### Key Concepts:
- Reinforcement learning from human feedback (RLHF)
- Reward model training from preference data
- Proximal Policy Optimization (PPO)
- Constitutional AI and self-improvement
- Multimodal learning and integration
- Mixture of Experts architectures
- Retrieval-augmented generation (RAG)

#### Learning Outcomes:
- Students can explain the RLHF training pipeline
- Students understand how human preferences are incorporated into model training
- Students can identify current frontier capabilities in LLMs
- Students can discuss safety and alignment challenges
- Students recognize emerging trends in model efficiency and deployment

---

### Part 9: Ethics and Practice (Slides 34-35)

#### Topics Covered:

**Slide 35: Summary and Practical Considerations**
- **Key Takeaway**: Pre-trained LLMs revolutionized the NLP landscape
- **Architecture Choice**: Select based on task (Encoder: BERT-style, Decoder: GPT-style, Both: T5-style)
- **Ethical Concerns**:
  - Bias & Fairness: Models can perpetuate societal biases
  - Privacy: Training data may contain sensitive information
  - Misinformation: Risk of generating false or misleading content
  - Environment: Training emissions and energy consumption
- **Best Practices**:
  - Diverse Evaluation: Test on diverse datasets and scenarios
  - Document Limitations: Clearly state model capabilities and constraints
  - Transparency: Open about model behavior and decisions
  - Human Oversight: Maintain human-in-the-loop for critical tasks
- **Environmental Cost**: Training emissions and energy consumption considerations
- **Accessibility**: Balance between democratization and resource concentration
- **Continuous Learning**: Field evolves rapidly, stay updated
- **Responsible Deployment**: Transparency, safety measures, human oversight in all deployments

#### Key Concepts:
- Ethical implications of large language models
- Bias detection and mitigation strategies
- Environmental impact of model training
- Responsible AI deployment practices
- Model transparency and documentation

#### Learning Outcomes:
- Students can identify potential biases in language models
- Students understand the environmental cost of training large models
- Students can implement responsible deployment practices
- Students can evaluate models for fairness across different demographics
- Students recognize the importance of human oversight in AI systems

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Deep Learning Fundamentals**:
  - Neural network architectures (feedforward, convolutional, recurrent)
  - Backpropagation and gradient descent optimization
  - Loss functions and evaluation metrics
- **NLP Basics**:
  - Tokenization and text preprocessing
  - Word embeddings (Word2Vec, GloVe)
  - Sequence modeling with RNNs and LSTMs
- **Transformer Architecture**:
  - Self-attention mechanism
  - Multi-head attention
  - Positional encoding
  - Encoder-decoder structure
- **Python Programming**:
  - Proficiency in Python 3.7+
  - NumPy and pandas for data manipulation
  - Experience with Jupyter notebooks
- **Mathematical Background**:
  - Linear algebra (matrices, vectors, matrix multiplication)
  - Probability and statistics
  - Calculus (derivatives, chain rule)
  - Understanding of optimization algorithms

### Required Software:
- **Python Environment**: Python 3.8+ with pip or conda
- **Deep Learning Frameworks**:
  - PyTorch 2.0+ or TensorFlow 2.x
  - Hugging Face Transformers library (>=4.30.0)
  - Hugging Face Datasets library
- **Development Tools**:
  - Jupyter Notebook or JupyterLab
  - Git for version control
  - Visual Studio Code or PyCharm (recommended)
- **Optional but Recommended**:
  - CUDA 11.8+ and cuDNN for GPU acceleration
  - Weights & Biases for experiment tracking
  - Docker for reproducible environments

### Hardware Requirements:
- **Minimum**: 16GB RAM, multi-core CPU
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **For Fine-tuning**: 24GB+ GPU VRAM (RTX 3090, A5000, or cloud GPU instances)
- **Cloud Alternatives**: Google Colab Pro, AWS SageMaker, Azure ML Studio

---

## 💻 Hands-on Components

### Project 1: BERT Fine-tuning for Text Classification
**Objective**: Fine-tune a pre-trained BERT model on a sentiment analysis task

**Dataset**: IMDb Movie Reviews (25,000 training samples)

**Implementation Steps**:
1. Load pre-trained `bert-base-uncased` from Hugging Face
2. Prepare dataset with appropriate tokenization
3. Add classification head (linear layer on [CLS] token)
4. Implement training loop with:
   - Learning rate: 2e-5
   - Batch size: 16
   - Epochs: 3
   - Warmup steps: 500
5. Evaluate on test set and calculate accuracy, F1 score
6. Analyze predictions and error cases

**Expected Outcomes**:
- Achieve >90% accuracy on test set
- Understand fine-tuning hyperparameters
- Learn to use Hugging Face Transformers API

**Time Estimate**: 2-3 hours

---

### Project 2: GPT-based Text Generation with Prompting
**Objective**: Explore prompt engineering techniques with GPT models

**Tasks**:
1. **Zero-shot Classification**: Design prompts for sentiment analysis without examples
2. **Few-shot Learning**: Create 3-shot prompts for custom classification tasks
3. **Chain-of-Thought**: Implement CoT prompting for math word problems
4. **Creative Generation**: Generate stories with controlled style and content

**Implementation**:
- Use OpenAI API (GPT-3.5-turbo) or open alternatives (GPT-Neo, GPT-J)
- Experiment with temperature, top-p, and max tokens
- Compare different prompt formulations systematically
- Document prompt variations and output quality

**Deliverables**:
- Collection of effective prompts for different tasks
- Quantitative comparison of prompting strategies
- Analysis of failure cases and limitations

**Time Estimate**: 3-4 hours

---

### Project 3: Parameter-Efficient Fine-tuning with LoRA
**Objective**: Implement LoRA fine-tuning for efficient model adaptation

**Model**: GPT-2 or LLaMA-7B

**Dataset**: Custom instruction dataset (Alpaca format)

**Implementation**:
1. Install PEFT library from Hugging Face
2. Configure LoRA parameters:
   - Rank (r): 8
   - Alpha: 16
   - Target modules: query and value projection layers
3. Fine-tune on instruction-following task
4. Compare:
   - Full fine-tuning vs. LoRA (memory usage, training time)
   - Different LoRA ranks (r=4, 8, 16, 32)
5. Evaluate instruction-following quality

**Metrics**:
- Training time per epoch
- GPU memory consumption
- Perplexity on validation set
- Qualitative evaluation of generated responses

**Expected Results**:
- 90%+ reduction in trainable parameters
- Similar or better performance than full fine-tuning
- 3-5x reduction in GPU memory usage

**Time Estimate**: 4-5 hours

---

### Project 4: Building a RAG System
**Objective**: Create a Retrieval-Augmented Generation system for question answering

**Components**:
1. **Document Store**: Vector database (FAISS or Chroma)
2. **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
3. **LLM**: GPT-3.5-turbo or open-source alternative
4. **Dataset**: Wikipedia articles or custom knowledge base

**Implementation Pipeline**:
1. Chunk documents into 512-token segments
2. Generate embeddings and store in vector database
3. Implement retrieval:
   - Query embedding
   - Top-k similarity search (k=3-5)
4. Construct prompt with retrieved context
5. Generate answer using LLM
6. Implement citation tracking

**Evaluation**:
- Answer accuracy on factual questions
- Citation precision (are sources relevant?)
- Latency measurements
- Compare RAG vs. zero-shot LLM performance

**Time Estimate**: 5-6 hours

---

### Optional Advanced Project: RLHF Implementation
**Objective**: Implement a simplified RLHF pipeline

**Stages**:
1. **Supervised Fine-tuning**: Train on high-quality demonstrations
2. **Reward Model**: Train classifier on preference pairs
3. **RL Fine-tuning**: Use PPO to optimize policy (simplified version)

**Dataset**: Anthropic HH-RLHF dataset

**Implementation Guidance**:
- Use TRL (Transformer Reinforcement Learning) library
- Start with small model (GPT-2) for feasibility
- Focus on understanding each stage rather than perfect results
- Document challenges and insights

**Time Estimate**: 8-10 hours

---

## 📖 Additional Resources

### Essential Papers:
1. **BERT**: Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. **GPT-3**: Brown et al. (2020) - "Language Models are Few-Shot Learners"
3. **T5**: Raffel et al. (2019) - "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
4. **LoRA**: Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"
5. **InstructGPT**: Ouyang et al. (2022) - "Training language models to follow instructions with human feedback"

### Recommended Textbooks:
- **"Speech and Language Processing" (3rd ed.)** by Jurafsky & Martin - Chapters 10-11 on transformers and pre-training
- **"Natural Language Processing with Transformers"** by Tunstall, von Werra, & Wolf - Practical Hugging Face guide
- **"Dive into Deep Learning"** by Zhang et al. - Chapter 16 on NLP pre-training (free online)

### Online Courses:
- **Hugging Face Course**: https://huggingface.co/course - Free comprehensive course on transformers
- **Stanford CS224N**: Natural Language Processing with Deep Learning (available on YouTube)
- **DeepLearning.AI**: "ChatGPT Prompt Engineering for Developers"
- **Fast.ai**: Practical Deep Learning for Coders - NLP modules

### Tutorials and Blogs:
- **Jay Alammar's Blog**: http://jalammar.github.io/ - Visual explanations of BERT, GPT, transformers
- **Lil'Log**: https://lilianweng.github.io/ - In-depth technical posts on LLMs
- **Hugging Face Blog**: https://huggingface.co/blog - Latest developments and techniques
- **Andrej Karpathy's YouTube**: Neural network walkthroughs and LLM tutorials

### Tools and Libraries:
- **Hugging Face Transformers**: https://github.com/huggingface/transformers
- **PEFT (Parameter-Efficient Fine-Tuning)**: https://github.com/huggingface/peft
- **TRL (Transformer Reinforcement Learning)**: https://github.com/huggingface/trl
- **LangChain**: Framework for developing LLM applications
- **LlamaIndex**: Data framework for LLM applications

### Datasets:
- **GLUE Benchmark**: General Language Understanding Evaluation
- **SuperGLUE**: More challenging language understanding tasks
- **SQuAD**: Question answering dataset
- **IMDb Reviews**: Sentiment analysis
- **WikiText**: Language modeling
- **C4**: Colossal Clean Crawled Corpus (T5 pre-training data)

### Model Repositories:
- **Hugging Face Model Hub**: https://huggingface.co/models - 100,000+ pre-trained models
- **TensorFlow Hub**: https://tfhub.dev/ - TensorFlow model repository
- **ONNX Model Zoo**: https://github.com/onnx/models - Interoperable models

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Sequential Learning**: Follow slides in order from Part 1 to Part 9 for comprehensive understanding
2. **Time Allocation**: Allocate 3-4 hours for initial lecture review, 10-15 hours for hands-on projects
3. **Active Learning**: Pause at each slide to implement concepts in code before moving forward
4. **Project-Based**: Complete at least Projects 1-3 to solidify understanding
5. **Supplement**: Read cited papers and blog posts for deeper understanding
6. **Practice**: Use `lecture14_slideshow.html` for presentation mode review

### For Instructors:
1. **Lecture Delivery**: Use slideshow HTML for in-class presentation (estimated 2-3 hour lecture)
2. **Pacing**: 
   - Week 1: Parts 1-3 (Introduction, Pre-training, BERT) + Project 1
   - Week 2: Parts 4-5 (GPT, Encoder-Decoder) + Project 2
   - Week 3: Parts 6-7 (Fine-tuning, Prompting) + Project 3
   - Week 4: Parts 8-9 (Trends, Ethics) + Project 4
3. **Interactive Elements**: 
   - Live coding demonstrations for fine-tuning
   - Group prompt engineering competitions
   - Ethical case study discussions
4. **Assessment**: Use projects as graded assignments with provided rubrics
5. **Customization**: Slides are HTML-based and easily modifiable for specific course needs

### Viewing the Slideshow:
1. Open `lecture14_slideshow.html` in a modern web browser
2. Navigate using:
   - Arrow keys (←/→) or Space bar
   - On-screen navigation buttons
   - Keyboard shortcuts: Home (first slide), End (last slide)
3. Progress bar and slide counter show current position
4. Full-screen mode recommended for presentations

---

## 📊 Assessment Suggestions

### Formative Assessment:

**1. Concept Check Quizzes** (After each Part)
- Multiple choice questions on key concepts
- Short answer questions on architecture differences
- Example: "Explain when to use encoder-only vs. decoder-only architectures"

**2. Code Review Exercises**
- Identify bugs in provided fine-tuning code
- Optimize inefficient prompt designs
- Debug common errors in Hugging Face pipelines

**3. Discussion Forums**
- Weekly discussion on ethical implications
- Peer review of prompt designs
- Sharing of successful fine-tuning strategies

### Summative Assessment:

**1. Project Submissions** (60% of grade)
- Project 1 (BERT Fine-tuning): 15%
- Project 2 (Prompt Engineering): 15%
- Project 3 (LoRA): 15%
- Project 4 (RAG System): 15%

**Grading Rubric for Projects**:
- Code Quality (30%): Correctness, efficiency, documentation
- Results (30%): Performance metrics, comparative analysis
- Understanding (20%): Written explanations, design choices
- Creativity (20%): Novel approaches, extensions beyond requirements

**2. Final Exam** (30% of grade)
- **Part A: Theoretical** (50%)
  - Architecture comparisons
  - Pre-training objective explanations
  - Fine-tuning strategy selection
  - Ethical considerations
- **Part B: Practical** (50%)
  - Code writing: Implement fine-tuning loop
  - Prompt design: Create effective prompts for given tasks
  - Debugging: Fix provided buggy code
  - Analysis: Interpret training curves and results

**3. Final Project** (10% of grade)
- Open-ended application of learned techniques
- Proposal, implementation, and presentation
- Examples:
  - Custom domain chatbot with RAG
  - Specialized fine-tuned model for domain task
  - Novel prompting technique evaluation
  - Ethical audit of existing LLM system

### Peer Assessment:
- Peer review of prompt designs (formative)
- Code review in pairs during lab sessions
- Group discussion facilitation on ethical topics

---

## 📝 Notes for Implementation

### Technical Requirements:

**1. Computing Resources**
- **For Basic Exercises**: CPU-only environment sufficient (Google Colab free tier)
- **For Fine-tuning Projects**: 
  - GPU with 8GB+ VRAM required
  - Cloud options: Colab Pro ($10/month), AWS g4dn.xlarge (~$0.50/hour), Paperspace Gradient
- **For LoRA and Large Models**: 
  - 16-24GB VRAM recommended
  - Consider using mixed-precision training (fp16) to reduce memory
  - Gradient checkpointing can reduce memory by 30-40%

**2. Dataset Access**
- Most datasets available through Hugging Face `datasets` library
- Some datasets require signup (e.g., The Pile)
- Ensure compliance with dataset licenses for academic use
- Consider data privacy regulations when using real user data

**3. API Access**
- OpenAI API requires paid account (~$5 credit sufficient for course)
- Alternative free options: Hugging Face Inference API (rate-limited)
- Consider hosting open-source models locally for unlimited access

**4. Library Versions**
```bash
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.39.0  # For quantization
scipy>=1.10.0
scikit-learn>=1.2.0
```

### Common Pitfalls and Solutions:

**1. Out-of-Memory Errors**
- Solution: Reduce batch size, use gradient accumulation, enable gradient checkpointing
- Example: `model.gradient_checkpointing_enable()`

**2. Slow Training**
- Solution: Use mixed-precision training, optimize data loading with multiple workers
- Example: Use `torch.cuda.amp.autocast()` for automatic mixed precision

**3. Poor Fine-tuning Results**
- Solution: Adjust learning rate (try 2e-5, 3e-5, 5e-5), increase epochs, check data quality
- Monitor validation loss to detect overfitting early

**4. Inconsistent Prompt Results**
- Solution: Set temperature=0 for deterministic output, increase temperature for creativity
- Use multiple samples and self-consistency for important tasks

### Reproducibility:
```python
# Set random seeds for reproducibility
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### Time Management:
- **Lecture Review**: 3-4 hours
- **Reading Papers**: 5-6 hours
- **Hands-on Projects**: 15-20 hours total
- **Final Project**: 10-15 hours
- **Total Course Time**: 35-45 hours

### Recommended Learning Path:
1. Watch lecture slides (use slideshow mode)
2. Read corresponding paper sections
3. Follow along with code tutorials
4. Complete hands-on project
5. Participate in discussion forum
6. Repeat for next part

---

## 🙏 Credits

**Instructor**: Ho-min Park  
**Email**: homin.park@ghent.ac.kr, powersimmani@gmail.com  
**Institution**: Ghent University

**Course**: Deep Learning for Natural Language Processing  
**Lecture**: 14 - Pre-trained Language Models & LLM Era

This lecture material comprehensively covers the evolution and current state of pre-trained language models, from the introduction of BERT in 2018 to the latest developments in GPT-4, RLHF, and instruction tuning. The curriculum balances theoretical understanding with practical implementation skills, preparing students for both research and industry applications in modern NLP.

The hands-on projects use industry-standard tools (Hugging Face Transformers, PyTorch) and cover essential techniques from fine-tuning to prompt engineering to parameter-efficient methods. Ethical considerations and responsible AI deployment are integrated throughout the course, preparing students to develop and deploy language models responsibly.

For questions, clarifications, or additional resources, please contact the instructor at the provided email addresses.

---

**Last Updated**: November 2025  
**Version**: 1.0  
**License**: Educational Use