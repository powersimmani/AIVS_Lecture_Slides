# Lecture 01 - Part 2: Memory and ML Model Execution

## Overview
This part covers memory fundamentals, how ML models are loaded and executed, and optimization techniques for efficient model training and inference.

## Key Topics

### 1. Memory Hierarchy - RAM, VRAM, Cache
- **Memory Speed Hierarchy** (fastest to slowest):
  - Registers: ~1 cycle, KB-level capacity
  - L1 Cache: ~4 cycles, ~32KB per core
  - L2 Cache: ~10 cycles, ~256KB per core
  - L3 Cache: ~40 cycles, ~30MB shared
  - RAM: ~100 cycles, 16-512GB
  - Storage (SSD): ~10,000 cycles, TB-level
- **Key Insight**: Data should be kept as close to the processor as possible
- **VRAM vs RAM**: VRAM is dedicated GPU memory with higher bandwidth (~900 GB/s for HBM3 vs ~50 GB/s for DDR5)

### 2. ML Models Loading and Memory Management
- **Loading Flow**: Disk → RAM → VRAM (GPU Memory)
- **Challenge**: Large models like LLaMA-70B require 140GB (FP16), exceeding single GPU VRAM
- **Solutions for Large Models**:
  1. **Model Sharding**: Distribute across multiple GPUs (tensor parallelism)
  2. **Offloading**: Keep parameters in RAM, load when needed
  3. **Quantization**: Reduce memory footprint (8-bit, 4-bit)
- **Key Insights**:
  - Memory bandwidth limits loading speed
  - Disk I/O is typically not the bottleneck
  - Use memory-mapped files (mmap) for efficiency
- **Popular Tools**: Hugging Face Accelerate, DeepSpeed ZeRO

### 3. Batch Size and Memory Usage Calculation
- **Memory Usage Components**: Model Params + Optimizer States + Gradients + Activations
- **Activation Memory** ∝ Batch Size × Sequence Length
- **Example**: BERT-base with batch=32, seq=512 → ~8GB activation memory
- **Techniques to Reduce Memory**:
  - **Gradient Accumulation**: Split batch into micro-batches to save memory
  - **Gradient Checkpointing**: Recompute activations (trade compute for memory)
- **Rule of Thumb**: OOM Error? Reduce batch size by 50%
- **Monitoring Tools**: `nvidia-smi`, `torch.cuda.memory_allocated()`

### 4. Python Bytecode and ML Frameworks
- **Python Execution Flow**: Source Code (.py) → Bytecode (.pyc) → PVM Execution
- **ML Framework Modes**:
  - **Eager Mode** (PyTorch default): Execute immediately, easy debugging
  - **Graph Mode** (TensorFlow default): Build computation graph, optimized execution
- **JIT Compilation**: TorchScript, XLA for optimization
- **Key Trade-off**: Flexibility (eager) vs Performance (graph)

### 5. Memory Layout of NumPy/PyTorch Tensors
- **Tensors**: Multi-dimensional arrays stored in contiguous memory
- **Storage Orders**:
  - **Row-Major (C-style)**: Default in NumPy/PyTorch - last dimension varies fastest
  - **Column-Major (Fortran)**: First dimension varies fastest
- **Stride**: Number of bytes to jump to next element in each dimension
- **Contiguous vs Non-Contiguous**:
  - Contiguous: Faster operations, better cache locality, optimized GPU performance
  - Non-contiguous: After transpose(), view() - may need `.contiguous()`
- **Check Methods**: `.is_contiguous()`, `.stride()`

### 6. GPU Memory Management
- **GPU Memory Allocation**: PyTorch uses cached allocator for efficiency
- **Free Unused Cache**: `torch.cuda.empty_cache()`
- **Best Practices**:
  - Allocate tensors at beginning
  - Reuse tensors when possible
  - Minimize CPU↔GPU transfers
- **Avoid**:
  - Creating tensors in loops
  - Frequent CPU↔GPU transfers
  - Memory fragmentation
- **Transfer Speed**: Within GPU (fast) vs Between GPUs (slow)
- **Tools**: CUDA Unified Memory, PyTorch Profiler, NVIDIA Nsight Systems

### 7. Mixed Precision Training - FP16 and FP32
- **Concept**: Use FP16 for speed, FP32 for stability
- **Benefits**: 50% memory reduction, 2-3x speedup on Tensor Cores
- **Loss Scaling**: Prevents underflow in FP16 gradients
- **Automatic Mixed Precision (AMP)**:
  ```python
  with autocast():
      output = model(input)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  ```
- **Best for**: Tensor Core operations (matmul, conv2d)
- **Not recommended for**: Small models, numerical precision-critical tasks

### 8. Hands-on: Resource Monitoring Tools
- **GPU Monitoring**: `nvidia-smi`, `gpustat`
- **CPU/Memory**: `htop`, `top`
- **PyTorch Tools**:
  - `torch.cuda.memory_allocated()`: Currently allocated memory
  - `torch.cuda.memory_reserved()`: Total reserved memory
  - `torch.cuda.max_memory_allocated()`: Peak memory usage
- **Profiling**: PyTorch Profiler, TensorBoard integration

## Important Takeaways
1. Understanding memory hierarchy is crucial for optimizing ML performance
2. Large models require specialized techniques (sharding, offloading, quantization)
3. Batch size directly impacts memory usage - adjust based on available VRAM
4. Contiguous memory layout significantly improves tensor operation speed
5. Mixed precision training offers substantial performance gains with minimal accuracy loss
6. Regular monitoring with nvidia-smi and PyTorch tools helps identify bottlenecks
