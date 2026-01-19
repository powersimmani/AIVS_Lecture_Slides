# Lecture 01 - Part 1: Data Representation and ML Hardware Fundamentals

## Overview
This part covers the foundational concepts of how data is represented in computers and the hardware fundamentals essential for machine learning practitioners.

## Key Topics

### 1. ML Workflow and Computer Architecture
- The ML pipeline consists of: **Data → Preprocessing → Training → Inference → Deployment**
- Computer architecture impacts every stage of ML workflow
- Key bottlenecks include:
  - Data loading efficiency
  - Computation speed
  - Memory bandwidth limits

### 2. Bits and Bytes - Understanding ML Data Types
- **Fundamental unit**: 1 Byte = 8 Bits
- Common ML data types:
  - **FP32** (32-bit float): 4 bytes - PyTorch/TensorFlow default for training
  - **FP16** (16-bit float): 2 bytes - Half precision, 50% memory savings
  - **INT8** (8-bit integer): 1 byte - Quantized, 75% memory savings
- **Memory calculation example**: 1 billion parameters × FP32 = 4GB
- Byte representations: Signed (-128 to 127) vs Unsigned (0 to 255)
- Hexadecimal and ASCII encoding fundamentals
- RGB color codes: 3 bytes per pixel (each channel 0-255)

### 3. Number Representation Methods
- **Fixed Point**: Integer with implicit decimal position
  - Fast computation, limited range, fixed precision
- **Floating Point**: Sign + Exponent + Mantissa
  - Flexible representation, wide range, slower processing
- FP32 structure: 1 sign bit + 8 exponent bits + 23 mantissa bits (range: ±3.4×10³⁸)
- FP16 structure: 1 sign bit + 5 exponent bits + 10 mantissa bits (range: ±6.5×10⁴)
- **Key trade-off**: Precision ⇄ Memory ⇄ Speed
- Basic logic gates (AND, OR, XOR, NOT, NAND, NOR) and adder circuits

### 4. Quantization Principles and Memory Efficiency
- **Quantization**: Reducing precision for efficiency (FP32 → INT8)
- Memory reduction: 4x smaller (75% savings)
- Methods: Post-training quantization, Quantization-aware training
- Accuracy impact: Typically <1-2% degradation with proper quantization
- Applications: Mobile deployment, Edge devices, Faster inference

### 5. CPU vs GPU - Architectural Comparison
- **CPU**: Few powerful cores (4-64), optimized for sequential tasks, better for control flow
- **GPU**: Thousands of simple cores (1000s-10000s), massive parallel processing, 10-100x faster for matrix operations
- Common components: Registers, ALU, Control Unit, Cache Memory
- GPU advantages: More cores, more registers per core, optimized for parallel tasks
- Modern ML workflow: GPU for training, CPU for preprocessing/serving

### 6. GPU Cores and CUDA
- **CUDA**: NVIDIA's parallel computing platform (launched 2007)
- GPU Architecture Hierarchy: GPU → Streaming Multiprocessors (SMs) → CUDA Cores
- Example (RTX 4090): 128 SMs, 16,384 CUDA cores
- **SIMT** (Single Instruction, Multiple Threads) execution model
- PyTorch & TensorFlow automatically leverage CUDA for GPU acceleration

### 7. FLOPS and ML Model Performance Metrics
- **FLOPS** (uppercase S): Floating Point Operations Per Second - hardware capability metric
- **FLOPs** (lowercase s): Floating Point Operations - model complexity metric
- GPU Performance comparison:
  - NVIDIA A100: 312 TFLOPS
  - NVIDIA H100: 1000 TFLOPS (3.2x faster)
- Training time calculation: Model FLOPs / Hardware FLOPS / GPU Count
- Example: GPT-3 training requires ~3.14×10²³ FLOPs

### 8. Tensor Operations and Hardware Optimization
- **Tensor**: Multi-dimensional array (foundation of deep learning)
- Common parallelizable operations: matmul, conv2d, attention
- Memory access patterns:
  - **Coalesced access**: Sequential reads - fast & efficient
  - **Uncoalesced access**: Random reads - slow & inefficient
- Optimization strategies:
  - Batch operations together
  - Maintain contiguous memory
  - Minimize CPU-GPU transfers
- **Tensor Cores**: Specialized hardware for mixed-precision matrix multiplication
- GPU parallelization enables simultaneous computation of all output elements

## Important Takeaways
1. Understanding hardware fundamentals is crucial for optimizing ML performance
2. Data type selection significantly impacts memory usage and computation speed
3. GPUs excel at parallel operations that dominate ML workloads
4. Quantization offers substantial memory savings with minimal accuracy loss
5. Efficient memory access patterns are key to maximizing GPU utilization
