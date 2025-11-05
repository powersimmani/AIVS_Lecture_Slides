# Lecture 1: Computer Structure and Networks for ML

## 📋 Overview

- **Instructor**: Ho-min Park
- **Contact**: homin.park@ghent.ac.kr, powersimmani@gmail.com
- **Total Slides**: 31 slides (including title and closing)
- **Format**: HTML-based interactive slides
- **Level**: Intermediate to Advanced

## 🎯 Learning Objectives

By the end of this lecture, students will be able to:

1. **Understand data representation in ML systems** - Explain the differences between fixed-point and floating-point representations, and understand how quantization reduces memory usage
2. **Analyze hardware architecture for ML** - Compare CPU and GPU architectures, and identify when to use each for different ML tasks
3. **Optimize memory usage** - Calculate memory requirements for ML models, understand batch size implications, and apply memory management best practices
4. **Implement distributed training** - Set up remote server connections, transfer files securely, and deploy ML models on distributed systems
5. **Apply practical skills** - Monitor GPU resources, use mixed precision training, and leverage cloud computing for ML workloads

## 📚 Lecture Structure

### Part 1: Data Representation and ML Hardware Fundamentals
**Slides**: 3-11 (8 topics)

**Topics Covered:**
1. **Lecture Introduction - ML Workflow and Computer Architecture**
   - Overview of ML pipeline stages
   - How computer architecture impacts each stage
   - Critical bottlenecks identification

2. **Bits and Bytes - Understanding ML Data Types**
   - Fundamental data representation
   - Common ML data types (INT8, FP16, FP32)
   - Memory implications of different types

3. **Number Representation Methods - Fixed Point vs. Floating Point**
   - Fixed-point representation and characteristics
   - Floating-point formats (FP32, FP16)
   - Bit structure breakdown (sign, exponent, mantissa)
   - Trade-offs: precision vs. memory vs. speed

4. **Quantization Principles and Memory Efficiency**
   - Quantization definition and benefits
   - FP32 to INT8 conversion example
   - Memory reduction strategies (4x smaller, 75% savings)
   - Post-training vs. quantization-aware training
   - Real-world applications (mobile, edge devices)

5. **CPU vs. GPU - Architectural Comparison**
   - Core architecture differences
   - CPU: Few powerful cores (4-64) for sequential tasks
   - GPU: Thousands of simple cores for parallel processing
   - Performance comparison for matrix operations (10-100x speedup)
   - Modern ML workflow allocation

6. **GPU Cores and CUDA - Understanding Parallel Processing**
   - NVIDIA CUDA platform introduction
   - GPU architecture hierarchy (GPU → SMs → CUDA Cores)
   - Example: RTX 4090 specifications (128 SMs, 16,384 cores)
   - SIMT execution model
   - Automatic integration with PyTorch/TensorFlow

7. **FLOPS and ML Model Performance Metrics**
   - FLOPS (hardware capability) vs. FLOPs (model complexity)
   - Performance metrics explanation
   - GPU comparison: A100 (312 TFLOPS) vs. H100 (1000 TFLOPS)
   - Training time calculation formula
   - GPT-3 training example (~3.14×10²³ FLOPs)

8. **Tensor Operations and Hardware Optimization**
   - Tensor definition and importance
   - Common operations (matmul, conv2d, attention)
   - Memory access patterns (coalesced vs. uncoalesced)
   - Optimization strategies
   - Tensor Cores for mixed-precision operations

**Key Concepts:**
- Data type hierarchy: INT8 < FP16 < FP32 (memory and precision trade-offs)
- GPU parallelism enables 10-100x speedup for ML workloads
- Quantization can reduce model size by 75% with minimal accuracy loss
- Tensor operations are the foundation of deep learning computations

**Learning Outcomes:**
- Students can select appropriate data types for different ML scenarios
- Students can calculate memory requirements based on data types
- Students can explain why GPUs are preferred for ML training
- Students can identify optimization opportunities in tensor operations

### Part 2: Memory and ML Model Execution
**Slides**: 12-20 (9 topics)

**Topics Covered:**
1. **Part 2 Introduction - Memory and ML Model Execution**
   - Overview of 8 memory-related topics
   - Connection between memory and model performance

2. **Memory Hierarchy - RAM, VRAM, Cache**
   - Memory hierarchy levels and characteristics
   - Speed vs. capacity trade-offs
   - Cache locality importance
   - Memory bandwidth concepts

3. **ML Models Loading and Memory Management**
   - Loading flow: Disk → RAM → VRAM
   - Large model challenges (e.g., LLaMA-70B = 140GB in FP16)
   - Solutions: Model sharding, offloading, quantization
   - Memory-mapped files (mmap)
   - Tools: Hugging Face Accelerate, DeepSpeed ZeRO

4. **Batch Size and Memory Usage Calculation**
   - Memory components: Model params + optimizer states + gradients + activations
   - Activation memory scales with batch size × sequence length
   - Example: BERT-base with batch=32, seq=512 → ~8GB
   - Techniques: Gradient accumulation, gradient checkpointing
   - Monitoring tools: nvidia-smi, torch.cuda.memory_allocated()

5. **Python Bytecode and ML Frameworks**
   - Python execution model
   - How ML frameworks optimize beyond Python
   - JIT compilation and graph optimization
   - Framework-level optimizations

6. **Memory Layout of NumPy/PyTorch Tensors**
   - Row-major (C) vs. Column-major (Fortran) storage
   - Stride concept and calculation
   - Contiguous vs. non-contiguous tensors
   - Performance implications
   - Methods: .is_contiguous(), .stride(), .contiguous()

7. **GPU Memory Management**
   - Cached allocator mechanism (PyTorch)
   - Best practices: allocate early, reuse tensors, minimize CPU↔GPU transfers
   - Transfer speed comparison
   - Unified Memory (CUDA)
   - Profiling tools: PyTorch Profiler, NVIDIA Nsight Systems

8. **Mixed Precision Training - FP16 and FP32**
   - Mixed precision concept: FP16 for compute, FP32 for stability
   - Benefits: 2x memory reduction, 2-3x speed boost (with Tensor Cores)
   - Challenge: FP16 range limitations and gradient underflow/overflow
   - Solution: Loss scaling (1000-10000x)
   - Master weights strategy
   - Tools: torch.cuda.amp, TF mixed_precision
   - Minimal accuracy impact (<0.1%)

9. **Hands-on: Resource Monitoring Tools**
   - GPU monitoring: nvidia-smi, watch -n 1 nvidia-smi
   - CPU/System: htop, top
   - PyTorch tools: torch.cuda.memory_summary(), memory_allocated()
   - Advanced tools: TensorBoard, NVIDIA Nsight, W&B, MLflow
   - Practice workflow: Monitor → Identify bottlenecks → Optimize

**Key Concepts:**
- Memory hierarchy: Cache (fastest) → RAM → VRAM → Disk (slowest)
- Batch size directly impacts activation memory requirements
- Mixed precision training provides significant speedup with minimal accuracy loss
- Contiguous memory layout is crucial for GPU performance
- Proper memory management prevents OOM errors and improves efficiency

**Learning Outcomes:**
- Students can calculate memory requirements for a given model and batch size
- Students can implement mixed precision training in PyTorch
- Students can diagnose and resolve memory-related issues
- Students can monitor GPU utilization and memory usage effectively
- Students understand tensor memory layout and its performance implications

### Part 3: Network and Distributed ML
**Slides**: 21-30 (10 topics)

**Topics Covered:**
1. **Part 3 Introduction - Network and Distributed ML**
   - Overview of 9 networking and distribution topics
   - Importance of network infrastructure for scaling ML

2. **IP Addresses and Ports - Server Connection Basics**
   - IPv4 (32-bit) vs. IPv6 (128-bit)
   - Public vs. Private IP addresses
   - Common ports: SSH (22), HTTP (80), HTTPS (443)
   - ML server connection example
   - Port forwarding for remote Jupyter access
   - Security best practices

3. **SSH and Remote Server Connection Practice**
   - SSH (Secure Shell) encrypted connection
   - Basic connection: ssh username@hostname
   - SSH key authentication setup (3 steps)
   - SSH config file for convenience (~/.ssh/config)
   - Persistent sessions: tmux, screen
   - Practical applications for ML training

4. **File Transfer - Using SCP and SFTP**
   - SCP (Secure Copy) for file transfer over SSH
   - Upload: local → remote
   - Download: remote → local
   - Recursive transfer with -r flag
   - Alternative tools: SFTP, rsync (efficient sync)
   - Best practices: compression for large datasets, cloud storage considerations

5. **HTTP API and Model Serving**
   - REST API for ML model serving
   - Request-response flow: Client → Model → Response
   - Popular frameworks: Flask, FastAPI, TensorFlow Serving
   - Benefits: language-agnostic, easy integration, scalable
   - Considerations: latency (~10-100ms), throughput, batching
   - Production tools: Docker + Kubernetes
   - Monitoring: response time, error rates, model drift

6. **Distributed Training Overview**
   - Why distributed: Single GPU insufficient for large models/datasets
   - Data parallelism: same model, different data batches
   - Model parallelism: split model across GPUs
   - Synchronous vs. asynchronous training
   - PyTorch implementation: DistributedDataParallel (DDP)
   - Communication: all-reduce for gradient synchronization
   - Efficiency considerations: ideal linear speedup vs. overhead

7. **Network Bandwidth and Learning Speed**
   - Network bandwidth critical for multi-node training
   - Single node (NVLink/PCIe) vs. Multi-node (Ethernet/InfiniBand)
   - Interconnect comparison:
     - NVLink: 600 GB/s (GPU-to-GPU direct)
     - PCIe 4.0 x16: 32 GB/s (standard)
     - PCIe 5.0 x16: 64 GB/s (latest)
   - Communication bottleneck: gradient sync time > computation time
   - Optimizations: gradient compression, reduce frequency, ZeRO (DeepSpeed)
   - Design rule: network bandwidth should match GPU compute capability

8. **Docker Basics - ML Environment Containerization**
   - [Note: File not fully visible in uploaded materials]
   - Container concept and benefits
   - Docker for reproducible ML environments
   - Best practices for ML containers

9. **Leveraging Cloud GPUs**
   - Major cloud providers: AWS (EC2 P4/P5), GCP (A2/A3), Azure (NC/ND)
   - Pricing models:
     - On-demand: $$$$ (e.g., $32/hr for A100) - flexible but expensive
     - Spot instances: 70-90% off - cheaper but can be interrupted
   - Managed services: SageMaker (AWS), Vertex AI (GCP), Azure ML
   - GPU marketplaces: Lambda Labs, RunPod, Vast.ai
   - Cost optimization: small GPUs for debug, scale up for training
   - Free credits for research/education
   - Critical: Always shut down instances when not in use

10. **Hands-on Project - Training ML Models on Remote Server**
    - Project goal: Train image classifier on remote GPU server
    - Workflow steps:
      1. SSH & Setup: Connect and setup environment (conda/docker)
      2. Transfer Dataset: Use scp or download directly on server
      3. Write Training Script: PyTorch/TensorFlow with logging
      4. Run in tmux: Persistent session
      5. Monitor Training: nvidia-smi and TensorBoard (port forwarding)
      6. Download & Evaluate: Get trained model and evaluate locally
    - Deliverables: Working model, training logs, results presentation

**Key Concepts:**
- SSH is the standard for secure remote server access
- Data parallelism is the most common distributed training approach
- Network bandwidth can become a bottleneck in multi-node training
- Cloud GPUs offer flexibility but require careful cost management
- Proper monitoring and persistent sessions are essential for remote training

**Learning Outcomes:**
- Students can connect to and manage remote ML servers via SSH
- Students can transfer files securely between local and remote systems
- Students can set up and monitor distributed training jobs
- Students understand the cost-benefit trade-offs of cloud GPU services
- Students can complete an end-to-end remote training project
- Students can identify and address network bottlenecks in distributed training

## 🛠️ Prerequisites

**Required Knowledge:**
- Basic Python programming
- Familiarity with machine learning concepts (neural networks, training process)
- Understanding of linear algebra (matrices, vectors)
- Basic command-line interface (CLI) usage

**Recommended Background:**
- Experience with PyTorch or TensorFlow
- Basic understanding of computer architecture
- Familiarity with Linux/Unix systems (helpful for Part 3)

**Software Requirements:**
- Modern web browser (for viewing HTML slides)
- Python 3.8+ with PyTorch or TensorFlow (for hands-on exercises)
- SSH client (for Part 3 remote server exercises)
- Access to a GPU (local or cloud) for practical exercises

## 💻 Hands-on Components

### Primary Project (Part 3, Slide 30)
**Training ML Models on Remote Server**
- **Objective**: Train an image classifier on a remote GPU server
- **Skills Practiced**: SSH, file transfer, remote training, monitoring, model deployment
- **Duration**: 2-4 hours
- **Deliverables**: Trained model, training logs, performance analysis

### Additional Practical Exercises

**Part 1 Exercises:**
- Calculate memory requirements for different data types
- Compare CPU vs. GPU performance on matrix operations
- Implement quantization and measure accuracy/memory trade-offs

**Part 2 Exercises:**
- Profile memory usage during model training
- Implement mixed precision training
- Optimize batch size for available GPU memory
- Practice with nvidia-smi and PyTorch profiling tools

**Part 3 Exercises:**
- Set up SSH key authentication
- Transfer datasets using SCP
- Configure port forwarding for remote Jupyter/TensorBoard
- Deploy a simple model as a REST API
- Monitor distributed training across multiple GPUs

## 📖 Additional Resources

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### Research Papers
- "Mixed Precision Training" - Micikevicius et al., 2017
- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" - Rajbhandari et al., 2020
- "Megatron-LM: Training Multi-Billion Parameter Language Models" - Shoeybi et al., 2019

### Tools and Frameworks
- PyTorch Distributed: https://pytorch.org/tutorials/beginner/dist_overview.html
- DeepSpeed: https://www.deepspeed.ai/
- Weights & Biases: https://wandb.ai/
- TensorBoard: https://www.tensorflow.org/tensorboard

### Online Courses
- Fast.ai Practical Deep Learning
- Stanford CS231n: Convolutional Neural Networks
- CMU 11-785: Introduction to Deep Learning

## 🔗 How to Use These Materials

### For Self-Study
1. **Start with the slideshow**: Open `lecture01_slideshow.html` in a web browser
2. **Follow the three-part structure**: Complete Part 1 before moving to Part 2
3. **Take notes**: Each slide contains key concepts worth documenting
4. **Complete hands-on exercises**: Practice is essential for retention
5. **Review additional resources**: Deepen understanding of challenging topics

### For Classroom Instruction
1. **Preparation**: Review all slides and test hands-on components before class
2. **Timing**: Allocate approximately 45-60 minutes per part (3-hour total lecture)
3. **Interactive elements**: Pause for questions after each major topic
4. **Live demonstrations**: Demonstrate nvidia-smi, SSH connections, and monitoring tools
5. **Project assignment**: Assign the hands-on project as homework with 1-week deadline

### Navigation Tips
- Use **arrow keys** (← →) or **space bar** to navigate slides
- Press **Home** to return to the first slide
- Press **End** to jump to the last slide
- The progress bar at the top shows your current position
- Slide numbers are displayed in the bottom-right corner

### Individual Slide Files
Each topic is available as a standalone HTML file:
- `Lecture01_01_...html` through `Lecture01_31_...html`
- Can be opened individually for targeted review
- Useful for revisiting specific topics

## 📊 Assessment Suggestions

### Knowledge Checks (Part 1)
- Explain the trade-offs between FP32, FP16, and INT8
- Calculate memory savings from quantization
- Compare CPU and GPU for different ML workloads
- Interpret FLOPS specifications

### Practical Skills (Part 2)
- Calculate batch size given GPU memory constraints
- Implement and benchmark mixed precision training
- Profile and optimize memory usage in a training script
- Diagnose OOM errors

### Application (Part 3)
- Set up secure remote server connection
- Deploy a model as a REST API
- Configure and run distributed training
- Optimize network bandwidth usage

### Final Project
- Complete the hands-on project (Slide 30)
- Present results including:
  - Training curves and metrics
  - Resource utilization analysis
  - Performance comparison (local vs. remote)
  - Cost analysis (if using cloud resources)

## 📝 Notes for Implementation

### Technical Requirements
- All slides are HTML-based and require no special software
- Best viewed in Chrome, Firefox, or Safari
- JavaScript must be enabled for navigation features
- Screen resolution: Optimized for 960×540 (16:9 aspect ratio)

### Accessibility
- High contrast color scheme (blue #1E64C8 on white)
- Clear typography with Aptos/Segoe UI fonts
- Hover effects for interactive elements
- Keyboard navigation supported

### Customization
- Slides can be edited in any text editor
- CSS styling is embedded in each HTML file
- Colors and fonts can be modified globally
- Additional slides can be added to `lecture01_slideshow.html`

## 🙏 Credits

**Instructor**: Ho-min Park  
**Institution**: Ghent University  
**Contact**: homin.park@ghent.ac.kr, powersimmani@gmail.com

---

**Version**: 1.0  
**Last Updated**: 2024  
**License**: Educational Use

For questions, corrections, or suggestions, please contact the instructor.
