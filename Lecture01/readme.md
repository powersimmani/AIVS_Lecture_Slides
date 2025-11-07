# Lecture 1: Computer Structure and Networks for ML

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com

This lecture covers the fundamentals of computer architecture and networks for machine learning. Learn key concepts of data representation, hardware architecture, memory management, and distributed training.

---

## 🎯 Learning Objectives

1. **Understanding data representation** - Differences between fixed-point and floating-point, memory optimization through quantization
2. **Analyzing hardware architecture** - Comparing CPU and GPU architectures, selecting appropriate hardware for ML tasks
3. **Optimizing memory usage** - Calculating model memory requirements, understanding batch size impact
4. **Implementing distributed training** - Remote server connections, file transfer, deploying models on distributed systems
5. **Applying practical skills** - GPU resource monitoring, Mixed Precision Training, cloud computing utilization

---

## 📚 Key Topics

**Data representation and ML hardware**
- **Data types**: INT8, FP16, FP32 memory and precision tradeoffs
- **Quantization**: 75% memory savings converting FP32 to INT8
- **CPU vs GPU**:
  - CPU: 4-64 powerful cores (sequential processing)
  - GPU: Thousands of simple cores (parallel processing, 10-100x speedup)
- **CUDA**: NVIDIA GPU parallel processing platform
- **FLOPS**: Hardware performance metric (A100: 312 TFLOPS, H100: 1000 TFLOPS)
- **Tensor Operations**: Fundamental deep learning operations (matmul, conv2d, attention)

**Memory and ML model execution**
- **Memory hierarchy**: Cache (fastest) → RAM → VRAM → Disk (slowest)
- **Model loading**: Disk → RAM → VRAM flow
- **Large model challenges**: LLaMA-70B = 140GB in FP16
  - Solutions: Model sharding, offloading, quantization, mmap
- **Batch size and memory**:
  - Memory components: Model parameters + optimizer states + gradients + activations
  - Activation memory scales with batch size × sequence length
  - Example: BERT-base (batch=32, seq=512) → ~8GB
- **Mixed Precision Training**:
  - FP16 compute, FP32 stability
  - 2x memory reduction, 2-3x speed boost (with Tensor Cores)
  - Loss scaling prevents gradient underflow/overflow
- **Memory layout**: Row-major vs Column-major, Stride, Contiguous tensors

**Networks and distributed ML**
- **SSH connection**:
  - Secure remote server access
  - SSH key authentication, config file setup
  - Persistent sessions with tmux/screen
- **File transfer**: SCP, SFTP, rsync
- **Distributed training paradigms**:
  - **Data Parallelism**: Distribute data across multiple GPUs (most common)
  - **Model Parallelism**: Distribute model across multiple GPUs
  - **Pipeline Parallelism**: Distribute layers across multiple GPUs
- **Communication backends**: NCCL (NVIDIA), Gloo, MPI
- **Network bandwidth**:
  - InfiniBand (100+ Gbps) vs Ethernet (1-10 Gbps)
  - Bottleneck in multi-node training
- **Cloud GPU services**: AWS (p3, p4), GCP (A100, TPU), Azure (NDv2), Colab Pro

---

## 💡 Key Concepts

- GPU parallel processing accelerates ML workloads by 10-100x
- Quantization reduces model size by 75% with minimal accuracy loss
- Batch size is the primary determinant of memory usage
- Mixed Precision provides speedup and memory savings
- Contiguous memory layout is crucial for GPU performance
- SSH is the standard for remote server access
- Data Parallelism is the most common distributed training approach
- Network bandwidth can become a bottleneck in multi-node training

---

## 🛠️ Prerequisites

- Basic Python programming
- Understanding of previous lecture content
- Basic machine learning concepts

---

## 📖 Additional Resources

For detailed code examples, practice materials, and slides, please refer to the original lecture files.
Lecture materials: HTML-based interactive slides provided
