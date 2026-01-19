# Lecture 18 - Part 1: Self-Supervised and Contrastive Learning

## Overview
This part covers self-supervised learning approaches, particularly contrastive learning methods like SimCLR, MoCo, and BYOL for learning visual representations.

## Key Topics

### 1. What is Self-Supervised Learning?
- **Definition**: Learn representations from unlabeled data using pretext tasks
- **Pretext Tasks**: Artificially created supervision
- **Goal**: Learn useful features that transfer to downstream tasks
- **Contrast with Supervised**: No human labels needed

### 2. Types of Self-Supervised Tasks
**Pretext Tasks**:
- Rotation prediction
- Jigsaw puzzle solving
- Colorization
- Inpainting

**Contrastive Learning**:
- Learn by comparing positive and negative pairs
- Similar items close, different items far
- Most successful approach

### 3. Contrastive Learning Principles
**Core Idea**: Pull positives together, push negatives apart

**Loss Function (InfoNCE)**:
```
L = -log(exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ))
```
- z_i, z_j: Positive pair embeddings
- z_k: All samples (including negatives)
- τ: Temperature parameter

### 4. SimCLR Algorithm
**Simple Framework for Contrastive Learning**:

**Process**:
1. Take image x
2. Apply two random augmentations → x̃_i, x̃_j
3. Encode both → h_i, h_j
4. Project → z_i, z_j
5. Contrastive loss

**Augmentations**:
- Random crop + resize
- Color distortion
- Gaussian blur
- Random flip

**Key Findings**:
- Composition of augmentations crucial
- Larger batch sizes help
- Projection head improves representations

### 5. MoCo and BYOL
**MoCo (Momentum Contrast)**:
- Maintain queue of negative samples
- Momentum encoder for consistency
- Memory efficient (doesn't need huge batches)
```
θ_k = m * θ_k + (1-m) * θ_q  # Momentum update
```

**BYOL (Bootstrap Your Own Latent)**:
- No negative samples needed!
- Two networks: online and target
- Target updated with momentum
- Prevents collapse through asymmetry

### 6. Self-Supervised Pretraining Pipeline
```
1. Pretrain on large unlabeled data (ImageNet)
2. Learn general visual features
3. Fine-tune on downstream task with few labels
4. Achieve strong performance
```

**Benefits**:
- Leverage abundant unlabeled data
- Learn transferable features
- Reduce labeling cost

### 7. Hands-on: Image Representation Learning
```python
# SimCLR-style augmentations
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Two views of same image
x1 = augmentation(image)
x2 = augmentation(image)

# Positive pair: (x1, x2)
# Negative pairs: other images in batch
```

### 8. Modern Self-Supervised Methods
- **DINO**: Self-distillation with no labels
- **MAE**: Masked Autoencoder (reconstruct patches)
- **SwAV**: Swapped predictions between views
- **Barlow Twins**: Redundancy reduction

**Trend**: Moving away from explicit negatives

### 9. Applications and Impact
**Applications**:
- Transfer learning for vision
- Medical imaging (limited labels)
- Video representation
- Multi-modal learning

**Impact**:
- Closing gap with supervised learning
- Foundation for large vision models
- Reducing annotation requirements

## Important Takeaways
1. Self-supervised learning uses pretext tasks for supervision
2. Contrastive learning pulls positives together, pushes negatives apart
3. SimCLR showed strong augmentations are key
4. MoCo uses momentum queue for memory efficiency
5. BYOL eliminates need for negative samples
6. Pre-trained representations transfer well
7. Foundation for modern computer vision

