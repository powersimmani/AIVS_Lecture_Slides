# Lecture 01 - Part 3: Network and Distributed ML

## Overview
This part covers networking fundamentals, remote server access, distributed training concepts, and practical tools for deploying ML workloads on cloud infrastructure.

## Key Topics

### 1. IP Addresses and Ports - Server Connection Basics
- **IP Address**: Unique identifier for devices on a network
  - **IPv4**: 32-bit (e.g., 192.168.1.1) - ~4.3 billion addresses
  - **IPv6**: 128-bit (e.g., 2001:0db8:...) - virtually unlimited
- **Port Numbers**: Identify specific services (0-65535)
  - SSH: 22
  - HTTP: 80
  - HTTPS: 443
  - Custom ML services: Usually 5000-9000
- **Private vs Public IPs**: Private (10.x.x.x, 192.168.x.x) vs Public (internet-routable)

### 2. SSH and Remote Server Connection
- **SSH (Secure Shell)**: Encrypted protocol for remote access
- **Basic Command**: `ssh username@hostname -p port`
- **Key-based Authentication** (recommended over password):
  ```bash
  ssh-keygen -t rsa -b 4096
  ssh-copy-id user@server
  ```
- **SSH Config File** (~/.ssh/config): Simplifies connections
- **Port Forwarding**: Access remote services locally
  - `ssh -L local_port:localhost:remote_port user@server`
  - Useful for: TensorBoard, Jupyter Notebooks

### 3. File Transfer - Using SCP and SFTP
- **SCP (Secure Copy)**:
  - Upload: `scp local_file user@server:/path/`
  - Download: `scp user@server:/path/file local_path/`
  - Recursive: `scp -r folder/ user@server:/path/`
- **SFTP**: Interactive file transfer with FTP-like commands
- **rsync**: Efficient incremental transfers
  - `rsync -avz --progress source/ user@server:/dest/`
- **For Large Datasets**: Consider wget/curl directly on server

### 4. HTTP API and Model Serving
- **REST API Basics**: HTTP methods (GET, POST, PUT, DELETE)
- **Model Serving Frameworks**:
  - Flask/FastAPI: Simple Python APIs
  - TensorFlow Serving: Production-grade serving
  - TorchServe: PyTorch model serving
- **Common Patterns**:
  - Input: JSON with base64-encoded images or text
  - Output: JSON with predictions
- **Considerations**: Batching, async processing, load balancing

### 5. Distributed Training Overview
- **Data Parallelism**: Same model, different data on each GPU
  - Each GPU processes a mini-batch
  - Gradients are synchronized across GPUs
  - **DDP (Distributed Data Parallel)**: PyTorch's recommended approach
- **Model Parallelism**: Model split across GPUs
  - For models too large for single GPU
  - Pipeline parallelism for sequential layers
  - Tensor parallelism for splitting operations
- **Synchronous vs Asynchronous**:
  - Synchronous: All GPUs sync at each step (consistent but slower)
  - Asynchronous: GPUs update independently (faster but less stable)

### 6. Network Bandwidth and Training Speed
- **Common Interconnects**:
  - NVLink: 600 GB/s (within node)
  - PCIe 4.0: 32 GB/s
  - PCIe 5.0: 64 GB/s
  - InfiniBand: 200-400 Gb/s (across nodes)
  - Ethernet: 1-100 Gb/s
- **Gradient Compression**: Reduce communication overhead
- **Ring-AllReduce**: Efficient gradient aggregation algorithm
- **Key Insight**: Network bandwidth often becomes bottleneck in distributed training

### 7. Docker Basics - ML Environment Containerization
- **Docker Concepts**:
  - **Image**: Blueprint/template (read-only)
  - **Container**: Running instance of an image
  - **Dockerfile**: Recipe to build images
- **Key Commands**:
  ```bash
  docker build -t myimage .
  docker run -it myimage
  docker run --gpus all nvidia/cuda:12.0-base  # GPU access
  ```
- **Volumes**: Persist data with `-v host_path:container_path`
- **Benefits**: Reproducible environments, isolation, easy deployment
- **NVIDIA Container Toolkit**: Enable GPU access in containers

### 8. Leveraging Cloud GPUs
- **Major Providers**:
  - AWS EC2: P4d (A100), P3 (V100), G5 (A10G)
  - Google Cloud: A2 (A100), N1 + T4/V100
  - Azure: NC-series (A100, V100, T4)
- **Spot/Preemptible Instances**: 60-90% cheaper, may be interrupted
- **Managed Services**: SageMaker, Vertex AI, Azure ML
- **Cost Optimization**:
  - Use spot instances for fault-tolerant training
  - Right-size instances based on workload
  - Auto-shutdown idle instances

### 9. Hands-on Project - Training ML Models on Remote Server
- **Project Goal**: Train image classifier on remote GPU server
- **Workflow**:
  1. SSH into server and setup environment (conda/docker)
  2. Transfer dataset (scp or download directly)
  3. Write training script with proper logging
  4. Run in tmux (persistent session)
  5. Monitor with nvidia-smi and TensorBoard (port forwarding)
  6. Download trained model and evaluate locally
- **Project Deliverables**: Working model, training logs, results presentation

## Important Takeaways
1. SSH and SCP are essential skills for remote ML development
2. Port forwarding enables access to remote services like TensorBoard
3. Distributed training (especially Data Parallel) can significantly speed up training
4. Network bandwidth can be a bottleneck - choose appropriate interconnects
5. Docker ensures reproducible ML environments across different systems
6. Cloud GPUs offer flexibility but require cost management strategies
7. tmux is essential for long-running training jobs on remote servers
