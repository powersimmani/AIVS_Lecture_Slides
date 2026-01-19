# Lecture 11 - Part 1: Sequence Modeling

## Overview
This part introduces sequence data, explains why specialized modeling approaches are needed, and covers basic feature engineering for sequential data.

## Key Topics

### 1. What is Sequence Data?
- **Definition**: Ordered data where position/order matters
- **Key Characteristic**: Temporal or sequential dependencies
- **Examples**:
  - Time series (stock prices, sensor readings)
  - Text (sentences, documents)
  - Audio/speech (waveforms)
  - Video (frame sequences)
  - DNA sequences

### 2. Why Special Modeling is Needed
- **Order Dependency**: Meaning changes with order
  - "Dog bites man" vs "Man bites dog"
- **Variable Length**: Sequences have different lengths
- **Long-range Dependencies**: Early elements affect later ones
- **Traditional ML Limitations**:
  - Fixed input size requirement
  - No built-in memory
  - Cannot capture temporal patterns

### 3. Features of Sequences
- **Temporal Correlation**: Adjacent elements are related
- **Varying Length**: Different sequences have different lengths
- **Context Dependency**: Element meaning depends on context
- **Directionality**: Past → Present → Future
- **Periodicity**: Many sequences have cycles/patterns

### 4. Feature Engineering for Sequences
- **Windowing**: Sliding window approach
  ```
  [x_t-n, ..., x_t-1, x_t] → y_t
  ```
- **Lag Features**: Previous values as features
- **Rolling Statistics**:
  - Moving average
  - Rolling standard deviation
  - Rolling min/max
- **Temporal Features**:
  - Hour of day, day of week
  - Trend indicators
  - Seasonal components
- **Limitations**: Manual features may miss complex patterns

## Important Takeaways
1. Sequence data is ordered data where position matters
2. Traditional ML struggles with variable length and dependencies
3. Feature engineering can help but has limitations
4. Understanding data characteristics guides model selection
5. Temporal dependencies are the key challenge in sequence modeling

