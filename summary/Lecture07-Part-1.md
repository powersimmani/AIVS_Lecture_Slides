# Lecture 07 - Part 1: Understanding Data Modalities

## Overview
This part introduces different types of data modalities in machine learning, their unique characteristics, and considerations for processing each type.

## Key Topics

### 1. Overview of Data Modalities
- **Data Modality**: Type/form of data (text, image, audio, etc.)
- **Common Modalities**:
  - Text (natural language)
  - Image (visual data)
  - Audio/Speech
  - Video (temporal visual)
  - Time Series
  - Graph/Network
  - Tabular (structured)
- **Multi-modal**: Combining multiple modalities
- **Importance**: Different modalities require different processing approaches

### 2. Structured vs Unstructured Data
- **Structured Data**:
  - Organized in tables (rows/columns)
  - Fixed schema
  - Examples: Databases, spreadsheets, CSV
  - Easy to query and analyze
- **Unstructured Data**:
  - No predefined format
  - Examples: Text, images, audio, video
  - Requires feature extraction
  - ~80% of enterprise data
- **Semi-structured**: JSON, XML (flexible schema)

### 3. Text Data Characteristics
- **Properties**:
  - Sequential/temporal nature
  - Variable length
  - Discrete tokens (words, characters)
  - Rich semantic content
- **Challenges**:
  - Ambiguity (same word, different meanings)
  - Context dependency
  - Multiple languages
  - Informal text (social media)
- **Preprocessing**: Tokenization, lowercasing, stopword removal, stemming/lemmatization

### 4. Image Data Characteristics
- **Properties**:
  - Grid of pixels (height × width × channels)
  - Spatial relationships
  - Color information (RGB, grayscale)
  - Scale/rotation variance
- **Common Formats**: JPEG, PNG, BMP, TIFF
- **Challenges**:
  - High dimensionality (1000×1000×3 = 3M values)
  - Viewpoint variations
  - Occlusion
  - Lighting conditions
- **Preprocessing**: Resizing, normalization, augmentation

### 5. Audio/Speech Data Characteristics
- **Properties**:
  - 1D signal over time
  - Sample rate (Hz): e.g., 16kHz, 44.1kHz
  - Amplitude values
  - Frequency content
- **Representations**:
  - Raw waveform
  - Spectrogram (time-frequency)
  - Mel-spectrogram
- **Challenges**:
  - Background noise
  - Speaker variability
  - Temporal alignment
- **Applications**: Speech recognition, music classification, speaker identification

### 6. Video Data Characteristics
- **Properties**:
  - Sequence of images (frames)
  - Temporal dimension
  - Frame rate (fps)
  - Motion information
- **Dimensionality**: Time × Height × Width × Channels
- **Challenges**:
  - Very high dimensionality
  - Temporal consistency
  - Action localization
  - Computational cost
- **Approaches**: 3D CNNs, frame sampling, optical flow

### 7. Graph/Network Data
- **Properties**:
  - Nodes (entities)
  - Edges (relationships)
  - May have node/edge attributes
  - Non-Euclidean structure
- **Types**:
  - Directed vs undirected
  - Weighted vs unweighted
  - Homogeneous vs heterogeneous
- **Applications**: Social networks, molecules, knowledge graphs
- **Challenges**: Variable size, permutation invariance
- **Methods**: Graph Neural Networks (GNNs)

### 8. Multimodal Data
- **Definition**: Data combining multiple modalities
- **Examples**:
  - Video + Audio (movies)
  - Image + Text (social media posts)
  - Text + Table (documents)
- **Challenges**:
  - Alignment across modalities
  - Different representations
  - Missing modalities
- **Fusion Strategies**: Early, late, hybrid fusion
- **Applications**: Visual question answering, video captioning, multimodal sentiment

## Important Takeaways
1. Each data modality has unique characteristics requiring specific processing
2. Unstructured data dominates but requires feature extraction
3. Text requires understanding of language structure and semantics
4. Images have spatial structure that CNNs can exploit
5. Audio/video add temporal dimension to processing
6. Graphs model relationships and require specialized architectures
7. Multimodal learning combines strengths of different modalities
