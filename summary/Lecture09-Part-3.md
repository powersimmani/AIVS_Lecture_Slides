# Lecture 09 - Part 3: Regularization and Generalization

## Overview
This part covers regularization techniques to improve model generalization, including dropout variants, data augmentation strategies, early stopping, and ensemble methods.

## Key Topics

### 1. Principles of Dropout
- **Concept**: Randomly set neurons to zero during training
- **Formula**:
  - Training: h = f(x) × mask, where mask ~ Bernoulli(1-p)
  - Inference: h = f(x) × (1-p) (scaling)
- **Dropout Rate**: Probability of dropping (typically 0.2-0.5)
- **Benefits**:
  - Prevents co-adaptation of neurons
  - Acts like ensemble of sub-networks
  - Strong regularization effect
- **Placement**: Typically after fully connected layers
- **Not Used**: After batch normalization (redundant regularization)

### 2. Dropout Variants
- **DropConnect**:
  - Drops connections (weights) instead of activations
  - More fine-grained than dropout
- **DropBlock**:
  - Drops contiguous regions in feature maps
  - Better for CNNs (spatial correlation)
- **Spatial Dropout**:
  - Drops entire feature map channels
  - Preserves spatial structure
- **DropPath (Stochastic Depth)**:
  - Drops entire layers/paths
  - For residual networks
- **Comparison**:
  | Variant | Drops | Best For |
  |---------|-------|----------|
  | Dropout | Neurons | FC layers |
  | DropConnect | Weights | Any |
  | DropBlock | Regions | CNNs |
  | Spatial | Channels | CNNs |
  | DropPath | Layers | ResNets |

### 3. Stochastic Depth
- **Concept**: Randomly skip entire residual blocks during training
- **Survival Probability**: Decreases with depth
  ```
  p_l = 1 - l/L × (1 - p_L)
  ```
  - l: Layer index
  - L: Total layers
  - p_L: Final survival probability
- **Training**: Skip layer with probability 1-p_l
- **Inference**: Use all layers (scaled)
- **Benefits**:
  - Enables training of very deep networks (1000+ layers)
  - Reduces training time
  - Implicit ensemble effect
  - Better gradient flow

### 4. Data Augmentation Strategies
- **Geometric Transforms**:
  - Random crop
  - Horizontal/vertical flip
  - Rotation
  - Scale/zoom
  - Shear
- **Color/Intensity**:
  - Brightness adjustment
  - Contrast adjustment
  - Saturation change
  - Hue shift
  - Color jittering
- **Advanced Techniques**:
  - **Cutout**: Randomly mask square regions
  - **Random Erasing**: Randomly erase rectangles
  - **GridMask**: Structured masking patterns
- **Benefits**:
  - Increases effective dataset size
  - Improves generalization
  - Reduces overfitting
  - Makes model invariant to transformations

### 5. Mixup and CutMix
- **Mixup**:
  ```
  x_mix = λ × x_i + (1-λ) × x_j
  y_mix = λ × y_i + (1-λ) × y_j
  ```
  - λ ~ Beta(α, α), typically α = 0.2-0.4
  - Linearly interpolate images and labels
- **Benefits**:
  - Smoother decision boundaries
  - Better calibration
  - Reduces memorization
- **CutMix**:
  - Cut and paste image regions
  - Labels proportional to area
  ```
  x_mix = M ⊙ x_i + (1-M) ⊙ x_j
  ```
  - M: Binary mask
- **CutMix Benefits**:
  - Preserves more local features
  - Often outperforms Mixup
  - Better for localization tasks
- **Comparison**:
  | Method | Image Mixing | Label Mixing |
  |--------|--------------|--------------|
  | Mixup | Blend entire | Weighted |
  | CutMix | Patch swap | Area-weighted |

### 6. Early Stopping
- **Concept**: Stop training when validation performance stops improving
- **Procedure**:
  1. Monitor validation loss/metric
  2. Track best validation score
  3. Stop if no improvement for patience epochs
  4. Restore best model weights
- **Patience**: Number of epochs to wait (typically 5-20)
- **Implementation**:
  ```python
  if val_loss < best_loss:
      best_loss = val_loss
      patience_counter = 0
      save_model()
  else:
      patience_counter += 1
      if patience_counter >= patience:
          stop_training()
  ```
- **Benefits**:
  - Simple and effective
  - Prevents overfitting
  - Saves computation
  - No hyperparameter for regularization strength

### 7. Ensemble Methods
- **Averaging**:
  - Average predictions from multiple models
  - Reduces variance
  ```
  ŷ = (1/M) Σ f_m(x)
  ```
- **Voting**:
  - Majority vote for classification
  - Hard voting (class labels) or soft voting (probabilities)
- **Stacking**:
  - Train meta-model on base model predictions
  - Learns optimal combination
- **Typical Ensemble Size**: 3-10 models
- **Sources of Diversity**:
  - Different random seeds
  - Different architectures
  - Different hyperparameters
  - Different training subsets
- **Benefits**:
  - Reduces variance
  - Improves robustness
  - Often 1-3% accuracy improvement
- **Trade-off**: Increased computational cost

## Important Takeaways
1. Dropout prevents co-adaptation and acts as implicit ensemble
2. Dropout variants (DropBlock, Spatial) are better suited for CNNs
3. Stochastic depth enables training of extremely deep networks
4. Data augmentation is essential for good generalization
5. Mixup and CutMix provide strong regularization through label smoothing
6. Early stopping is simple but effective against overfitting
7. Ensembles trade computational cost for improved accuracy
8. Combine multiple regularization techniques for best results

