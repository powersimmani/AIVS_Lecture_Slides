# Lecture 07 - Part 2: Traditional Feature Extraction

## Overview
This part covers traditional (non-deep learning) methods for extracting features from different data modalities, including text, images, audio, and time series.

## Key Topics

### 1. Feature Engineering Principles
- **Goal**: Transform raw data into meaningful features for ML models
- **Key Principles**:
  - Domain knowledge integration
  - Feature relevance to task
  - Computational efficiency
  - Feature independence (reduce redundancy)
- **Feature Types**:
  - Numerical: Continuous values
  - Categorical: Discrete categories
  - Binary: Yes/No
- **Feature Scaling**: Normalization, standardization

### 2. Text - Bag of Words (BoW) and TF-IDF
- **Bag of Words (BoW)**:
  - Document as word frequency vector
  - Ignores word order
  - Vocabulary size = vector dimension
  - Sparse representation
- **TF-IDF (Term Frequency - Inverse Document Frequency)**:
  ```
  TF-IDF(t,d) = TF(t,d) × log(N / DF(t))
  ```
  - TF: Word frequency in document
  - IDF: Inverse of documents containing word
  - Downweights common words
- **Implementation**:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(documents)
  ```

### 3. Text - N-gram and POS
- **N-gram**: Sequence of N consecutive words
  - Unigram (N=1): Individual words
  - Bigram (N=2): Word pairs
  - Trigram (N=3): Word triplets
  - Captures local word context
- **Character N-grams**: For spelling, subword patterns
- **POS (Part-of-Speech) Tagging**:
  - Label words with grammatical category
  - Noun, verb, adjective, etc.
  - Useful for syntactic analysis
- **Named Entity Recognition (NER)**: Identify entities (person, location, organization)

### 4. Image - Edge and Corner Detection
- **Edge Detection**:
  - Identifies boundaries in images
  - **Sobel**: Gradient-based, horizontal/vertical edges
  - **Canny**: Multi-stage, robust edge detector
  - Preprocessing step for many algorithms
- **Corner Detection**:
  - Finds points with high curvature
  - **Harris Corner**: Eigenvalue analysis
  - **Shi-Tomasi**: Good features to track
  - Stable under viewpoint changes
- **Applications**: Feature matching, object detection

### 5. Image - SIFT, SURF, HOG
- **SIFT (Scale-Invariant Feature Transform)**:
  - Keypoint detection and description
  - Invariant to scale, rotation
  - 128-dimensional descriptor
  - Computationally expensive
- **SURF (Speeded Up Robust Features)**:
  - Faster approximation of SIFT
  - Uses integral images
  - 64-dimensional descriptor
- **HOG (Histogram of Oriented Gradients)**:
  - Local gradient orientation histograms
  - Divides image into cells
  - Popular for pedestrian detection
  - Dense features (every location)

### 6. Audio - FFT and Spectrogram
- **FFT (Fast Fourier Transform)**:
  - Time domain → Frequency domain
  - Shows which frequencies are present
  - Fixed window analysis
- **Spectrogram**:
  - Time-frequency representation
  - STFT (Short-Time Fourier Transform)
  - 2D image of audio
  - X: Time, Y: Frequency, Color: Magnitude
- **Mel Spectrogram**:
  - Spectrogram on mel scale
  - Perceptually motivated
  - Standard input for audio ML

### 7. Audio - MFCC and Chroma
- **MFCC (Mel-Frequency Cepstral Coefficients)**:
  - Compact representation of spectral envelope
  - ~13-20 coefficients typically
  - Standard for speech recognition
  - Process: Audio → FFT → Mel filter → Log → DCT
- **Chroma Features**:
  - 12 dimensions (one per semitone)
  - Captures harmonic content
  - Invariant to octave
  - Used for music analysis
- **Other Features**: Zero-crossing rate, spectral centroid, spectral rolloff

### 8. Time Series - Statistical Features
- **Basic Statistics**:
  - Mean, median, mode
  - Variance, standard deviation
  - Skewness, kurtosis
  - Min, max, range
- **Temporal Features**:
  - Trend (linear regression coefficient)
  - Autocorrelation
  - Seasonality indicators
- **Frequency Domain**:
  - Dominant frequency
  - Spectral energy
  - Entropy
- **Libraries**: tsfresh, tsfel for automated feature extraction

## Important Takeaways
1. TF-IDF improves upon BoW by weighting term importance
2. N-grams capture local word context and phrases
3. SIFT/SURF provide robust image keypoints; HOG provides dense features
4. Spectrograms convert audio to 2D image-like representation
5. MFCC is the standard feature for speech/audio tasks
6. Statistical features summarize time series patterns
7. Domain knowledge guides effective feature engineering
