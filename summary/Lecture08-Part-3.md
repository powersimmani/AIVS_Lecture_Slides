# Lecture 08 - Part 3: Learning Rate Scheduling

## Overview
This part covers learning rate scheduling strategies that dynamically adjust the learning rate during training to improve convergence and final model performance.

## Key Topics

### 1. Importance of Learning Rate
- **Critical Hyperparameter**: Most important hyperparameter in deep learning
- **Effects**:
  - Too high: Divergence, oscillation, overshooting
  - Too low: Slow convergence, stuck in local minima
  - Just right: Fast convergence to good solution
- **Challenge**: Optimal learning rate changes during training
- **Solution**: Learning rate scheduling

### 2. Fixed vs Adaptive Learning Rate
- **Fixed Learning Rate**:
  - Simple to implement
  - Constant throughout training
  - May not be optimal at all stages
- **Adaptive Learning Rate**:
  - Changes during training
  - Higher early (exploration)
  - Lower later (fine-tuning)
  - Better convergence properties
- **Recommendation**: Use scheduling for better results

### 3. Step Decay
- **Formula**:
  ```
  LR = initial_lr × decay_factor^(epoch // step_size)
  ```
- **Parameters**:
  - decay_factor: Typically 0.1
  - step_size: Epochs between drops
- **Example**:
  - Initial LR: 0.1
  - Drop by 10× every 30 epochs
  - LR at epoch 60: 0.001
- **Use Case**: Standard training schedules (ResNet, VGG)
- **Implementation**:
  ```python
  scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
  ```

### 4. Exponential Decay
- **Formula**:
  ```
  LR = initial_lr × decay_rate^(epoch / decay_steps)
  ```
  or
  ```
  LR = initial_lr × e^(-k·epoch)
  ```
- **Properties**:
  - Smooth, continuous decrease
  - Gradual reduction
  - No sudden changes
- **Parameters**:
  - decay_rate: Decay per step (e.g., 0.95)
  - k: Decay constant
- **Best For**: When smooth transitions are preferred

### 5. Cosine Annealing
- **Formula**:
  ```
  LR = LR_min + (LR_max - LR_min) × (1 + cos(π × t/T)) / 2
  ```
  - t: Current epoch
  - T: Total epochs
- **Properties**:
  - Smooth cosine curve
  - Slow decay at start and end
  - Faster in middle
- **Benefits**:
  - No hyperparameter tuning for decay schedule
  - Natural warm restart points
  - Good empirical results
- **Cosine Annealing with Warm Restarts (SGDR)**:
  - Periodically restart learning rate
  - Enables exploration of different minima
  - Can improve generalization

### 6. Warm-up and Linear Schedule
- **Warm-up Phase**:
  - Start with very low LR
  - Gradually increase to target LR
  - Duration: 5-10% of total training
- **Linear Warm-up**:
  ```
  LR = initial_lr × (epoch / warmup_epochs)
  ```
- **Why Warm-up**:
  - Stabilizes training at start
  - Prevents large updates with random weights
  - Essential for large batch training
  - Critical for Transformers
- **Combined Schedule**:
  1. Warm-up: 0 → max_lr
  2. Decay: max_lr → min_lr
- **Implementation**:
  ```python
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=warmup_steps,
      num_training_steps=total_steps
  )
  ```

### 7. Cyclical Learning Rates
- **Concept**: Learning rate oscillates between bounds
- **Triangular Policy**:
  - Linear increase then decrease
  - Cycles between min_lr and max_lr
- **Triangular2**: Range halves each cycle
- **Exponential Range**: Exponential variation within cycle
- **Parameters**:
  - min_lr, max_lr: Learning rate bounds
  - step_size: Half cycle length
- **1cycle Policy**:
  - Single cycle with warm-up and cool-down
  - Often achieves best results fastest
  - Used in fast.ai methodology
- **Benefits**:
  - Escape saddle points
  - Explore loss landscape
  - Better generalization
  - Faster convergence
  - Regularization effect

### 8. Practical Guidelines
- **Starting Point**:
  - Try lr=1e-3 with Adam
  - Try lr=0.1 with SGD+momentum
- **Learning Rate Finder**:
  - Gradually increase LR
  - Plot loss vs LR
  - Choose LR where loss decreases fastest
- **Common Schedules**:
  | Task | Recommended Schedule |
  |------|---------------------|
  | Image Classification | Step decay or Cosine |
  | NLP/Transformers | Warm-up + Linear decay |
  | Quick experiments | 1cycle |
  | Research | Cosine with restarts |
- **PyTorch Schedulers**:
  - `StepLR`, `MultiStepLR`
  - `ExponentialLR`
  - `CosineAnnealingLR`
  - `OneCycleLR`
  - `ReduceLROnPlateau`

## Important Takeaways
1. Learning rate is the most critical hyperparameter in deep learning
2. Static learning rates are suboptimal - use scheduling
3. Step decay is simple and effective for many vision tasks
4. Cosine annealing provides smooth decay with good results
5. Warm-up is essential for large batch training and Transformers
6. Cyclical learning rates can improve both speed and generalization
7. 1cycle policy often achieves best results in shortest time
8. Use learning rate finder to identify optimal range
9. Different tasks benefit from different scheduling strategies

