# Lecture 11 - Part 2: Types of Sequence Data

## Overview
This part covers different types of sequence data including time series, text, and speech/audio data, along with their characteristics and preprocessing approaches.

## Key Topics

### 1. Time Series Data
- **Definition**: Numerical values recorded at regular or irregular time intervals
- **Characteristics**:
  - Numerical values over time
  - Regular or irregular intervals
  - Often univariate or multivariate
- **Components**:
  - Trend: Long-term direction
  - Seasonality: Repeating patterns
  - Noise: Random fluctuations
- **Applications**:
  - Stock price prediction
  - Weather forecasting
  - Anomaly detection
  - Demand forecasting
- **Challenges**:
  - Non-stationarity
  - Missing values
  - Multiple temporal scales

### 2. Text Data
- **Definition**: Discrete sequences of tokens (words, characters, subwords)
- **Characteristics**:
  - Discrete tokens (words, characters)
  - Rich semantic content
  - Variable vocabulary size
  - Context-dependent meaning
- **Preprocessing Steps**:
  - Tokenization
  - Lowercasing, stemming/lemmatization
  - Stop word removal
  - Embedding (Word2Vec, GloVe)
- **Applications**:
  - NLP tasks
  - Sentiment analysis
  - Machine translation
  - Text classification
- **Challenges**:
  - Out-of-vocabulary words
  - Context understanding
  - Long-range dependencies

### 3. Speech/Audio Data
- **Definition**: 1D waveform signals representing sound over time
- **Characteristics**:
  - 1D waveform signals
  - High sample rates (16-44.1kHz)
  - Frequency content important
  - Temporal structure
- **Representations**:
  - Raw waveform
  - Spectrogram
  - MFCC (Mel-frequency cepstral coefficients)
  - Mel spectrogram
- **Applications**:
  - Speech recognition (ASR)
  - Speaker identification
  - Music analysis
  - Emotion detection
- **Challenges**:
  - High dimensionality
  - Noise and variability
  - Speaker differences

### 4. Other Sequence Types
- **Video**: Sequences of image frames
- **Biological Sequences**: DNA, RNA, protein sequences
- **Event Sequences**: Clickstreams, transactions
- **Sensor Data**: IoT readings, accelerometer data

## Important Takeaways
1. Time series: numerical data with temporal dependencies
2. Text: discrete tokens requiring embedding representations
3. Audio: high-dimensional signals needing spectral analysis
4. Each type requires specific preprocessing approaches
5. Understanding data type guides model architecture choice
6. Common challenges: variable length, noise, dependencies

