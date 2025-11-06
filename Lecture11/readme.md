# Lecture 11: Sequence Models

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com  
**Total Slides:** 33  
**Difficulty Level:** Intermediate to Advanced  
**Estimated Duration:** 3-4 hours  
**Course Type:** Deep Learning / Machine Learning

This comprehensive lecture covers the fundamentals and advanced concepts of sequence modeling, progressing from traditional statistical methods to modern deep learning architectures. Students will gain both theoretical understanding and practical implementation skills for handling sequential data across multiple domains.

---

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand the unique characteristics** of sequence data and explain why specialized modeling approaches are necessary for temporal and spatial dependencies
2. **Apply traditional statistical methods** (MA, AR, ARMA, ARIMA) for time series analysis and understand their limitations
3. **Design and implement deep learning architectures** (RNN, LSTM, GRU) for sequence modeling tasks
4. **Evaluate and select appropriate architectures** (CNN, RNN, Bidirectional RNNs) based on specific sequence modeling requirements
5. **Implement advanced techniques** including encoder-decoder architectures, teacher forcing, beam search, and CTC loss for complex sequence-to-sequence tasks

---

## 📚 Lecture Structure

### Part 1/5: Sequence Modeling

**Topics Covered:**

#### Slide 1: What is Sequence Data?
- **Definition:** Data points ordered by time, position, or logical sequence
- **Key Property:** Order matters – shuffling changes meaning
- **Characteristics:**
  - Dependencies: Temporal or spatial dependencies between elements
  - Feature Types: Univariate (single) or multivariate (multiple features)
  - Length: Fixed or variable across samples
  - Structure: Sequential ordering preserves critical information
- **Examples with visualizations:**
  - Stock Prices: $100 → $105 → $103 → $108 → $110
  - DNA Sequences: A → T → G → C → T → A
  - Sentences: "I love machine learning"
  - Video Frames: Frame₁ → Frame₂ → Frame₃ → Frame₄

#### Slide 2: Why Special Modeling Is Needed?
- **Traditional ML Limitations:**
  - Assumes independence between samples
  - No memory of past events
  - Feedforward networks cannot capture sequential patterns
  - Ignores temporal/spatial dependencies
  - Context from previous elements is not utilized
- **Sequence Models Advantages:**
  - Recognize dependencies between data points
  - Maintain memory of historical information
  - Designed to capture sequential patterns
  - Model temporal/spatial relationships
  - Use context to influence predictions
- **Key Insight:** Memory of past events is crucial for accurate forecasting

#### Slide 3: Features of Sequences
- **Temporal Ordering:** Position in sequence is meaningful and carries information
- **Variable Length:** Sequences can have different durations or sizes
- **Local Patterns:** Nearby elements often correlate with each other
- **Long-Range Dependencies:** Distant elements may influence each other across time
- **Special Properties:**
  - Trend & Seasonality in time series data
  - Hierarchical Structure in language and biology

#### Slide 4: Feature Engineering for Sequences
- Traditional approaches to extracting features from sequences
- Handcrafted features vs. learned representations
- Domain-specific engineering techniques

**Key Concepts:**
- Sequential data fundamentals and ordering importance
- Temporal and spatial dependencies
- Memory requirements in sequence modeling
- Variable-length sequence handling

**Learning Outcomes:**
- Students can identify sequence data characteristics
- Students can explain why traditional ML fails on sequences
- Students can recognize different types of sequential dependencies

---

### Part 2/5: Types of Sequence Data

**Topics Covered:**

#### Slide 5: Time Series Data
- **Application Domains:**
  - 💰 Financial: Stock prices, trading volumes, market indices
  - 📡 IoT Sensors: Temperature, humidity, energy consumption
  - 🥼 Medical: Heart rate, blood pressure, EEG signals
  - 🌤️ Weather: Temperature, precipitation, wind speed over time
  - 📊 Business Metrics: Sales, revenue, customer traffic
- **Key Characteristics:**
  - Continuous values
  - Regular/Irregular sampling intervals

#### Slide 6: Text Data
- **Sequential Tokens:** Words, Characters, Subwords
- **Key Characteristics:**
  - Natural language structure
  - Variable length sequences
  - Grammatical structure
  - Context-dependent meaning
- **Applications:**
  - Machine Translation (e.g., "안녕하세요" → "Hello")
  - Text Summarization
  - Sentiment Analysis (e.g., Amazon reviews)
- **Challenges:**
  - Large vocabulary size
  - Rare words handling
  - Ambiguity (polysemy, homonyms)

#### Slide 7: Speech/Audio Data & Other Sequences
- **Audio Waveforms:** Speech recognition, music generation
- **Video Sequences:** Action recognition, video prediction
- **Biological Sequences:** DNA, RNA, protein structures
- **User Behavior:** Clickstreams, purchase history
- **Network Traffic:** Packet sequences, anomaly detection
- **Note:** Each domain has unique characteristics and challenges

**Key Concepts:**
- Domain-specific sequence characteristics
- Sampling rates and temporal resolution
- Multimodal sequence data

**Learning Outcomes:**
- Students can categorize different types of sequence data
- Students can identify domain-specific challenges
- Students can select appropriate preprocessing techniques

---

### Part 3/5: Statistical Approaches

**Topics Covered:**

#### Slide 8: Moving Average (MA)
- **Types:**
  - Simple MA: Equal weights for recent observations
  - Weighted MA: Different weights by recency
  - Exponential Smoothing: Exponentially decreasing weights
- **Benefits:**
  - Smoothing noisy data
  - Trend identification
  - No look-ahead bias for forecasting
- **Limitations:**
  - Cannot capture complex patterns
  - Cannot model seasonality
- **Detailed Example:**
  - Setup: Time series [10, 12, 15, 14, 18]
  - 3-Period MA Calculation:
    - t=3: MA₃ = (10 + 12 + 15) / 3 = 12.33
    - t=4: MA₃ = (12 + 15 + 14) / 3 = 13.67
    - t=5: MA₃ = (15 + 14 + 18) / 3 = 15.67
  - Trend: 12.33 → 13.67 → 15.67 (Upward)
  - Forecast for t=6: ŷ₆ = 15.67

#### Slide 9: Autoregressive (AR)
- Definition and mathematical formulation
- Order selection and parameter estimation
- Applications in forecasting

#### Slide 10: ARMA & ARIMA
- Combined Moving Average and Autoregressive components
- Integration for non-stationary series
- Model identification and diagnostics

#### Slide 11: Regression-Based Extensions
- **ARIMAX:** ARIMA with exogenous variables for external predictors
- **Vector Autoregression (VAR):** Handles multivariate time series simultaneously
- **State Space Models:** Uses Kalman filtering for latent state estimation
- **Structural Time Series:** Models underlying components explicitly
- **Benefits:**
  - Incorporates exogenous variables
  - Handles multivariate relationships

#### Slide 12: Limitations of Traditional Methods
- **Six Fundamental Constraints:**
  - 📏 Linear Assumptions: Assume linear relationships between observations
  - 🛠️ Manual Engineering: Require manual feature engineering
  - 🧩 Complex Patterns: Struggle with high-dimensional or complex patterns
  - ⏳ Long-Term Dependencies: Limited capacity for long-term dependencies
  - 🖼️ Unstructured Data: Cannot handle raw data (images, text)
  - 🧠 Representation Learning: Lack automatic representation learning

**Key Concepts:**
- Statistical modeling fundamentals
- Stationarity and differencing
- Model order selection (p, d, q parameters)
- Linear assumption limitations

**Learning Outcomes:**
- Students can implement MA and AR models
- Students can diagnose time series stationarity
- Students can identify when statistical methods are insufficient

---

### Part 4/5: Deep Learning for Sequences

**Topics Covered:**

#### Slide 13: High-Dimensional Sequences Challenge
- **Challenges:**
  - Curse of dimensionality in sequence modeling
  - Exponential growth in parameter space
  - Need for automatic feature extraction
  - Raw data (pixels, waveforms) vs handcrafted features
- **Deep Learning Solution:**
  - Hierarchical representation learning
  - End-to-end training from raw sequences
  - Automatic feature learning at multiple levels
  - Direct processing of raw data

#### Slide 14: CNN for Sequences
- 1D Convolutional operations for sequences
- Temporal feature extraction
- Local pattern detection
- Limitations for long-range dependencies

#### Slide 15: RNN (Recurrent Neural Network)
- **Core Architecture:** x_t → h_t ↻ h_{t+1}
- **Hidden State Update Formula:** 
  - h_t = tanh(W_hh × h_{t-1} + W_xh × x_t)
- **Features:**
  - Hidden state maintains sequence memory
  - Weight sharing: same weights at each timestep
  - Bidirectional RNN: forward & backward processing
  - Sequential processing: models temporal dependencies
- **Training Challenges:**
  - Vanishing/exploding gradients
  - Difficult to train for long sequences
- **Detailed Calculation Example:**
  - Setup: Hidden dim = 2, Input dim = 3
  - W_xh = [[0.5, 0.3], [0.2, 0.4], [0.1, 0.6]] (3×2)
  - W_hh = [[0.7, 0.2], [0.3, 0.8]] (2×2)
  - Input: x₀ = [1.0, 0.5, 0.8], x₁ = [1.2, 0.6, 0.9]
  - Step 1 (t=0):
    - h₀ = [0, 0]
    - W_hh × h₀ = [0, 0]
    - W_xh × x₀ = [0.68, 0.88]
    - h₁ = tanh([0.68, 0.88]) = [0.59, 0.71]
  - Step 2 (t=1):
    - W_hh × h₁ = [0.55, 0.75]
    - W_xh × x₁ = [0.78, 1.02]
    - h₂ = tanh([1.33, 1.77]) = [0.87, 0.95]
  - Trend: [0, 0] → [0.59, 0.71] → [0.87, 0.95]

#### Slide 16: LSTM (Long Short-Term Memory)
- **Solution:** Solves vanishing gradient problem with gating mechanism
- **Three Gates:**
  - 🚪 Forget Gate: Decides what to remove from cell state
  - ➕ Input Gate: Controls new information addition
  - 📤 Output Gate: Determines hidden state output
- **Memory Components:**
  - Cell State: Long-term memory pathway
  - Hidden State: Short-term working memory
- **Comparison with RNN:**
  - **RNN Issues:**
    - Vanishing gradient problem
    - Cannot capture long-term dependencies
    - Simple architecture with single state
    - Gradient diminishes exponentially over time
  - **LSTM Solutions:**
    - Gating mechanism prevents gradient vanishing
    - Maintains long-term memory via cell state
    - Complex architecture with gates
    - Controlled information flow preserves gradients
- **Gradient Flow Example:**
  - RNN: ∂L/∂h₀ = ∂L/∂h_T × (∏ᵢ₌₁ᵀ ∂hᵢ/∂hᵢ₋₁) → vanishes when T is large
  - LSTM: Forget gate + Input gate + Cell state = Stable gradient pathway
- **Success:** Widely successful for various sequence tasks across domains

#### Slide 17: GRU (Gated Recurrent Unit)
- Simplified alternative to LSTM
- Update and reset gates
- Computational efficiency benefits

#### Slide 18: Bidirectional RNNs
- **Dual-Direction Architecture:**
  - Forward →: x₁ → x₂ → x₃ → x₄
  - ← Backward: x₄ → x₃ → x₂ → x₁
  - Combination: ⊕ Concatenate or Average
- **Features:**
  - Captures future context for current prediction
  - Full sequence required before processing
- **Best For:**
  - Named Entity Recognition
  - POS Tagging
  - Sentiment Analysis
- **Cannot Be Used For:** Real-time streaming applications where future context is unavailable

**Key Concepts:**
- Recurrent connections and hidden states
- Gradient flow in deep networks
- Gating mechanisms (forget, input, output gates)
- Bidirectional processing

**Learning Outcomes:**
- Students can implement RNN architectures
- Students can explain the vanishing gradient problem
- Students can compare LSTM vs GRU tradeoffs
- Students can choose between unidirectional and bidirectional RNNs

---

### Part 5/5: Advanced Topics

**Topics Covered:**

#### Slide 19: Sequence-to-Sequence Models
- Variable-length input to variable-length output
- Machine translation paradigm
- Attention mechanisms preview

#### Slide 20: Encoder-Decoder Architecture
- **Three Components:**
  - 📥 Encoder: Converts input sequence to context vector (Typically LSTM/GRU)
  - 🎯 Context Vector: Fixed-size representation of entire input
  - 📤 Decoder: Generates output conditioned on context (Typically LSTM/GRU)
- **Training:** Teacher forcing for faster convergence
- **Inference:** Autoregressive generation
- **Implementation:** Both encoder and decoder typically use LSTM or GRU cells

#### Slide 21: Teacher Forcing
- Training strategy for sequence generation
- Ground truth feeding during training
- Exposure bias discussion

#### Slide 22: Beam Search
- Approximate decoding strategy
- Beam width selection
- Comparison with greedy search

#### Slide 23: CTC Loss
- Connectionist Temporal Classification
- Alignment-free training
- Applications in speech recognition

#### Slide 24: Practical Implementation Tips
- **⚡ Gradient Clipping (Stability):** Prevent exploding gradients during training
- **📏 Batch/Layer Normalization (Normalization):** Stabilize training and improve convergence
- **🛡️ Dropout (Regularization):** Apply to non-recurrent connections only
- **📈 Learning Rate Scheduling (Training):** Use warmup + decay strategy
- **🔄 Data Augmentation (Augmentation):** Time warping, noise injection
- **📊 Validation Metrics (Monitoring):** Track to prevent overfitting

#### Slide 25: Applications and Next Steps
- **Current Applications:**
  - 💬 NLP: Transformers, BERT, GPT for language understanding
  - 📈 Time Series: Forecasting with attention mechanisms
  - 🎤 Speech: End-to-end ASR with CTC and attention
  - 🎬 Video: Action recognition, video captioning
  - 🤖 Reinforcement Learning: Sequential decision making in dynamic environments
- **Future Directions:**
  - 🔄 Transformers replacing RNNs
  - 🕸️ Temporal Graph Networks

**Key Concepts:**
- Encoder-decoder paradigm
- Training vs. inference strategies
- Alignment and loss functions for sequences
- Production deployment considerations

**Learning Outcomes:**
- Students can implement seq2seq models
- Students can apply teacher forcing and beam search
- Students can optimize models for production
- Students can identify future research directions

---

## 🛠️ Prerequisites

### Required Knowledge:
- **Mathematics:**
  - Linear algebra (matrix operations, vector spaces)
  - Calculus (derivatives, chain rule, backpropagation)
  - Probability and statistics (distributions, expectations)
  - Basic optimization (gradient descent)
- **Machine Learning:**
  - Supervised learning fundamentals
  - Neural network basics (feedforward networks)
  - Backpropagation algorithm
  - Loss functions and optimization
- **Programming:**
  - Python proficiency (NumPy, pandas)
  - Basic understanding of deep learning frameworks

### Recommended Software:
```
- Python 3.8+
- PyTorch 1.12+ or TensorFlow 2.8+
- NumPy, pandas, matplotlib
- Jupyter Notebook or JupyterLab
- (Optional) CUDA-enabled GPU for faster training
```

### Recommended Prior Courses:
- Introduction to Machine Learning
- Neural Networks Fundamentals
- Linear Algebra
- Probability and Statistics

---

## 💻 Hands-on Components

### Project 1: Time Series Forecasting with Statistical Methods
**Objective:** Implement MA, AR, and ARIMA models for stock price prediction

**Dataset:** Historical stock prices (S&P 500 or similar)
- Time range: 5+ years of daily data
- Features: Open, High, Low, Close, Volume

**Tasks:**
1. Data preprocessing and stationarity testing (ADF test)
2. Implement 3-period, 7-period, and 30-period Moving Averages
3. Build AR(p) models with order selection using ACF/PACF plots
4. Implement ARIMA(p,d,q) with parameter tuning
5. Compare forecast accuracy (RMSE, MAE, MAPE)
6. Visualize predictions vs. actual values
7. Document limitations encountered

**Expected Output:** Jupyter notebook with implementations, visualizations, and performance comparison report

**Estimated Time:** 4-6 hours

---

### Project 2: Text Classification with RNN/LSTM
**Objective:** Build sentiment analysis system using recurrent architectures

**Dataset:** IMDB Movie Reviews or Twitter Sentiment Dataset
- Training samples: 20,000+
- Classes: Positive/Negative (binary) or Multi-class
- Preprocessing: Tokenization, vocabulary building

**Tasks:**
1. Text preprocessing pipeline (tokenization, padding)
2. Build word embeddings (Word2Vec or GloVe)
3. Implement vanilla RNN classifier
4. Implement LSTM classifier
5. Implement GRU classifier
6. Compare architectures:
   - Training time per epoch
   - Convergence speed
   - Final accuracy
   - Memory usage
7. Analyze vanishing gradient through gradient norms
8. Visualize attention weights (if applicable)

**Expected Output:** 
- Training scripts with all three architectures
- Performance comparison report with graphs
- Analysis of gradient flow

**Estimated Time:** 8-10 hours

---

### Project 3: Sequence-to-Sequence Translation
**Objective:** Build neural machine translation system with attention

**Dataset:** Translation dataset (e.g., English-French, WMT dataset)
- Sentence pairs: 50,000+
- Maximum sequence length: 50-100 tokens

**Tasks:**
1. Data preprocessing and vocabulary building
2. Implement encoder-decoder architecture with LSTM
3. Add attention mechanism
4. Implement teacher forcing during training
5. Implement beam search for inference (beam width = 3, 5, 10)
6. Evaluate using BLEU score
7. Compare with/without attention
8. Analyze attention weights visualization
9. Test edge cases (very long sentences, rare words)

**Expected Output:**
- Complete translation system
- BLEU score comparison report
- Attention visualization examples
- Discussion of failure cases

**Estimated Time:** 12-15 hours

---

### Project 4: Speech Recognition with CTC Loss
**Objective:** Build end-to-end automatic speech recognition system

**Dataset:** LibriSpeech or TIMIT
- Audio format: WAV, 16kHz sampling rate
- Transcriptions: Character or phoneme level

**Tasks:**
1. Audio feature extraction (MFCC or mel-spectrogram)
2. Build bidirectional LSTM acoustic model
3. Implement CTC loss function
4. Train with CTC decoding
5. Implement beam search decoder
6. Evaluate Word Error Rate (WER)
7. Compare with language model integration
8. Analyze common errors (substitutions, insertions, deletions)

**Expected Output:**
- Working ASR system
- WER performance report
- Error analysis document
- Audio samples with predictions

**Estimated Time:** 15-20 hours

---

## 📖 Additional Resources

### Textbooks:
1. **"Deep Learning" by Goodfellow, Bengio, and Courville**
   - Chapter 10: Sequence Modeling: Recurrent and Recursive Nets
   - Available online: https://www.deeplearningbook.org/

2. **"Speech and Language Processing" by Jurafsky and Martin**
   - Chapter 9: Sequence Processing with Recurrent Networks
   - Draft available: https://web.stanford.edu/~jurafsky/slp3/

3. **"Time Series Analysis and Its Applications" by Shumway and Stoffer**
   - Chapters 3-4: ARIMA models

### Research Papers:
1. **LSTM Original Paper:**
   - Hochreiter & Schmidhuber (1997), "Long Short-Term Memory"
   - Neural Computation 9(8): 1735-1780

2. **GRU Paper:**
   - Cho et al. (2014), "Learning Phrase Representations using RNN Encoder-Decoder"
   - EMNLP 2014

3. **Sequence-to-Sequence:**
   - Sutskever et al. (2014), "Sequence to Sequence Learning with Neural Networks"
   - NIPS 2014

4. **Attention Mechanism:**
   - Bahdanau et al. (2014), "Neural Machine Translation by Jointly Learning to Align and Translate"
   - ICLR 2015

5. **CTC Loss:**
   - Graves et al. (2006), "Connectionist Temporal Classification"
   - ICML 2006

### Online Tutorials:
- Stanford CS224N: Natural Language Processing with Deep Learning
- MIT 6.S191: Introduction to Deep Learning (RNN lectures)
- fast.ai: Practical Deep Learning for Coders
- TensorFlow/PyTorch official RNN tutorials

### Datasets:
- **Time Series:** UCR Time Series Archive, Yahoo S5 benchmark
- **Text:** IMDB, SST, AG News, WikiText
- **Speech:** LibriSpeech, TIMIT, Common Voice
- **Translation:** WMT, IWSLT, Tatoeba
- **Video:** UCF101, Kinetics, ActivityNet

### Code Repositories:
```
# PyTorch Examples
https://github.com/pytorch/examples/tree/main/time_sequence_prediction
https://github.com/pytorch/examples/tree/main/word_language_model

# TensorFlow Examples
https://www.tensorflow.org/tutorials/text/nmt_with_attention
https://www.tensorflow.org/tutorials/structured_data/time_series

# Sequence Models Zoo
https://github.com/tensorflow/models/tree/master/research/seq2seq
```

---

## 🔗 How to Use These Materials

### For Self-Study:
1. **Start with the slideshow:** Open `Lecture11_Slideshow.html` in a web browser for interactive navigation
2. **Follow the sequence:** Progress through Parts 1-5 in order
3. **Pause for comprehension:** Spend extra time on mathematical formulations
4. **Implement examples:** Code the detailed calculation examples (RNN, LSTM)
5. **Complete projects:** Work through hands-on projects sequentially
6. **Review limitations:** Understand when each method fails

### For Classroom Teaching:
1. **Part 1 (30 min):** Fundamentals and motivation with real-world examples
2. **Part 2 (20 min):** Data types with domain-specific case studies
3. **Part 3 (40 min):** Statistical methods with live coding demonstrations
4. **Break (10 min)**
5. **Part 4 (60 min):** Deep learning architectures with architectural diagrams on whiteboard
6. **Part 5 (40 min):** Advanced topics and practical tips
7. **Q&A and Project Introduction (20 min)**

### Navigation Controls:
- **Arrow Keys:** ← Previous slide, → Next slide
- **Space:** Next slide
- **Home/End:** First/Last slide
- **Menu Button:** Show slide list for quick navigation
- **Fullscreen Button:** Toggle fullscreen mode
- **Progress Bar:** Visual indication of lecture progress

### Recommended Pacing:
- **Week 1:** Parts 1-2 + Project 1 assigned
- **Week 2:** Part 3 + Project 1 due, Project 2 assigned
- **Week 3:** Part 4 (Slides 13-16) + Project 2 due
- **Week 4:** Part 4 (Slides 17-18) + Part 5 + Projects 3-4 assigned
- **Week 5-6:** Project presentations and discussions

---

## 📊 Assessment Suggestions

### Formative Assessment:
1. **Quick Quizzes (10%):**
   - After each part, 5-question multiple choice
   - Concepts: sequence properties, model architectures, gradient flow
   - Auto-graded via LMS

2. **Concept Maps (5%):**
   - Students create visual maps connecting:
     - MA → AR → ARMA → ARIMA → Limitations → Deep Learning
     - RNN → Vanishing Gradient → LSTM → GRU
   - Submission: Digital diagrams with explanations

3. **Code Reviews (10%):**
   - Peer review of Project 1 and 2 implementations
   - Rubric: Code quality, documentation, correctness

### Summative Assessment:
1. **Project Portfolio (50%):**
   - Project 1: 10% (Statistical methods mastery)
   - Project 2: 15% (RNN/LSTM implementation)
   - Project 3: 12% (Seq2seq translation)
   - Project 4: 13% (Advanced CTC application)
   - Grading criteria:
     - Implementation correctness (40%)
     - Performance metrics (30%)
     - Documentation and analysis (20%)
     - Code quality (10%)

2. **Technical Report (15%):**
   - 8-10 pages comparing all architectures
   - Must include:
     - Theoretical background
     - Experimental setup
     - Results with tables/graphs
     - Critical analysis of trade-offs
     - Future work suggestions
   - Due: End of course

3. **Final Presentation (10%):**
   - 15-minute presentation of best project
   - 5-minute Q&A
   - Evaluation: Clarity, depth, technical accuracy, visual aids

### Grading Rubric Example (Project 2):

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|-------------------|------------------|
| Implementation | All 3 models correctly implemented, efficient code | 2-3 models correct, minor bugs | 1-2 models work, significant issues | Non-functional or incomplete |
| Performance | Accuracy >85%, proper train/val split | Accuracy 75-85%, some issues | Accuracy 60-75%, methodology flaws | Accuracy <60% or no evaluation |
| Analysis | Deep insights on gradient flow, comprehensive comparison | Good comparison, some analysis | Basic comparison, limited depth | Little to no analysis |
| Documentation | Excellent comments, README, reproducible | Good documentation, mostly clear | Basic documentation, unclear parts | Poor or missing documentation |

---

## 📝 Notes for Implementation

### Technical Requirements:

**Minimum Hardware:**
- CPU: 4+ cores (Intel i5 or equivalent)
- RAM: 8GB (16GB recommended)
- Storage: 10GB free space for datasets
- GPU: Optional for faster training (NVIDIA with CUDA support)

**GPU Recommendations:**
- Project 1-2: CPU sufficient
- Project 3: GPU recommended (4GB VRAM+)
- Project 4: GPU strongly recommended (6GB VRAM+)

### Environment Setup:

```bash
# Create virtual environment
python -m venv sequence_models_env
source sequence_models_env/bin/activate  # Linux/Mac
# OR
sequence_models_env\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn statsmodels
pip install jupyter notebook
pip install tensorboard

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

### Common Issues and Solutions:

**Issue 1: Out of Memory (OOM)**
- Solution: Reduce batch size, use gradient accumulation
- Code: `batch_size = 32 // accumulation_steps`

**Issue 2: Vanishing Gradients in Deep RNNs**
- Solution: Use LSTM/GRU, gradient clipping
- Code: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

**Issue 3: Slow Training**
- Solution: Use GPU, reduce sequence length, optimize data loading
- Code: `DataLoader(dataset, num_workers=4, pin_memory=True)`

**Issue 4: Overfitting on Small Datasets**
- Solution: Dropout, early stopping, data augmentation
- Code: `nn.Dropout(p=0.5)` in non-recurrent connections only

### Reproducibility Checklist:
- [ ] Set random seeds: `torch.manual_seed(42)`
- [ ] Document hyperparameters in config file
- [ ] Version control with git
- [ ] Save model checkpoints regularly
- [ ] Log training curves to TensorBoard
- [ ] Include requirements.txt with exact versions

### Performance Benchmarks:

**Expected Training Times (on GPU):**
- Project 1: N/A (statistical methods are fast)
- Project 2: 15-30 min/epoch (LSTM on IMDB)
- Project 3: 2-4 hours total (50k sentence pairs)
- Project 4: 8-12 hours total (LibriSpeech subset)

**Expected Accuracy Ranges:**
- Project 2 Sentiment: 85-90% (LSTM), 82-87% (vanilla RNN)
- Project 3 Translation: BLEU 15-25 (without attention), 25-35 (with attention)
- Project 4 Speech: WER 15-25% (character-level CTC)

### Debugging Tips:
1. **Start small:** Test on 100 samples before full dataset
2. **Overfit single batch:** Ensure model capacity is sufficient
3. **Visualize embeddings:** Use t-SNE for word vectors
4. **Check gradient flow:** Plot gradient norms per layer
5. **Inspect predictions:** Manually review errors for patterns

---

## 🙏 Credits

**Lecture Materials Created By:**  
**Ho-min Park, Ph.D.**  
Department of Artificial Intelligence  
Ghent University Global Campus

**Contact:**
- Email: homin.park@ghent.ac.kr
- Email: powersimmani@gmail.com

**Acknowledgments:**
- Slide design inspired by modern educational presentation standards
- Examples adapted from classical deep learning literature
- Calculation walkthroughs created for pedagogical clarity

**License:**
These materials are intended for educational purposes. Please contact the instructor for permission to use in other courses or redistribute.

**Version:** 1.0  
**Last Updated:** November 2025  
**Lecture Series:** Deep Learning Fundamentals

---

**Total Lecture Components:**
- 📊 33 Interactive HTML slides
- 📝 4 Comprehensive hands-on projects
- 🎯 5 Major topic areas
- 📚 25+ Key concepts covered
- ⏱️ 3-4 hours of content delivery
- 💻 20-50 hours of hands-on practice

**Course Completion Criteria:**
Students who successfully complete all components will have mastery of sequence modeling from traditional statistics through modern deep learning, with practical implementation experience across multiple domains (time series, text, speech).