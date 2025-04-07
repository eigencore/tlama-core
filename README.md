# **Tlama Core** 🚀
<div align="center">
  <img src="https://i.postimg.cc/R00W9YMj/Tlama-1.png" alt="EigenCore" width="150" style="margin-right: 50px;">
</div>

Welcome to **Tlama Core**, the foundational repository for the **Tlama models**! This is the heart of our mission to create scalable, efficient, and cutting-edge AI models optimized for **training** and **inference** on **GPUs**. Whether you’re an **AI researcher**, **developer**, or **GPU enthusiast**, this repository is your playground to push the limits of performance, scalability, and innovation.

We believe in **community-driven development**—together, we can shape the future of AI. Join us to revolutionize machine learning with **state-of-the-art optimizations** for large-scale, high-performance models.

## **Why Tlama Core?** 🤔

Tlama Core is the **foundation** for next-gen AI models, designed to enhance **Tlama models** with unmatched efficiency, power, and scalability. From **high-performance computing** to **deep learning** and **robotics**, we’re building the infrastructure for groundbreaking research and **production-ready solutions**.

## **Core Areas of Focus** ⚙️

We’re targeting key optimizations to make Tlama models faster and more scalable. Explore our focus areas:

1. **Custom CUDA Kernels 🔥**: Unlock hardware potential with tailored kernels for attention, matrix ops, and more.
2. **Mixed Precision Training 💎**: Leverage Tensor Cores to train larger models faster.
3. **Distributed & Multi-GPU Support 🌐**: Scale effortlessly with optimized multi-GPU training.
4. **Memory Optimizations 🧠**: Use checkpointing and dynamic allocation for efficient large-scale training.
5. **Profiling Tools 🕵️‍♂️**: Analyze and optimize performance with precision.
6. **Innovative Algorithms 💡**: Push beyond PyTorch and cuBLAS with fresh approaches.
7. **Compression Techniques 📦**: Lightweight models via quantization and pruning.
8. **Fine-tuning & Transfer Learning 🔄**: Adapt models quickly to new tasks.
9. **Reinforcement Learning Support 🎮**: Tools for RL experimentation and deployment.
10. **Research Support 🔬**: Utilities for rapid experimentation and innovation.

## **Roadmap 📅**

Join our journey to build the future of AI! Track our progress and contribute via our [Roadmap Project Board](https://github.com/orgs/eigencore/projects/1) and [Issues](https://github.com/eigencore/tlama-core/issues).

| Phases   | Focus                 | Key Goals                            | Status            |
|-----------|-----------------------|--------------------------------------|-------------------|
| 1   | Bases                 | Repo structure, docs, Migrate Tlama-124M   | [In Progress](https://github.com/orgs/eigencore/projects/1) |
| 2   | Foundations           | CUDA kernels, mixed precision, docs  | Pending           |
| 3   | Scalability           | Multi-GPU, profiler, Tlama-500M      | Pending           |
| 4   | Innovation/Robotics   | Optimized attention, RL, simulation  | Pending           |
| 5   | Consolidation         | Tlama-1B, API, hackathon             | Pending           |

## **How to Set Up Tlama Core** 💻

Get started easily on **Windows**, **Linux**, or **macOS**!

### **Prerequisites**
- **CUDA Toolkit** (GPU support)
- **Python 3.8+** (virtual env recommended)
- **PyTorch** (GPU version preferred)
- **NVIDIA Driver** (GPU-compatible)
- **CMake** (for CUDA kernels)

#### **CUDA Toolkit Setup** 🛠️
- **Windows**: Download from [NVIDIA](https://developer.nvidia.com/cuda-downloads) and install.
- **Linux**: Use `sudo apt install nvidia-cuda-toolkit` or download from [NVIDIA](https://developer.nvidia.com/cuda-downloads).
- **macOS**: CPU-only (no CUDA support).

### **Installation Steps** 🚀
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/tlama-core.git
   cd tlama-core
   ```
2. **Set Up Environment**:
   ```bash
   python3 -m venv tlama-env
   source tlama-env/bin/activate  # Linux/macOS
   .\tlama-env\Scripts\activate  # Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Install CUDA Kernels**:
   ```bash
   python setup.py install
   ```
5. **Verify**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"  # Should output True
   ```

### **Run Tlama 124M Model 🏃‍♂️**
Test it out:
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("eigencore/tlama-124M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("eigencore/tlama-124M", trust_remote_code=True)

prompt = "Once upon a time in a distant kingdom..."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
More at [Hugging Face](https://huggingface.co/eigencore/tlama-124M).

## **How to Contribute 🌟**

We’re all about collaboration! Here’s how to jump in:
1. Check out the [Roadmap Project Board](https://github.com/your-username/tlama-core/projects/1).
2. Pick an [Issue](https://github.com/your-username/tlama-core/issues) to tackle.
3. Follow our [Contribution Guidelines](https://eigen-core.gitbook.io/tlama-core-docs/contributing/how-to-contribute).
4. Join the convo on [Discord](https://discord.gg/eXyva8uR).

New to contributing? Look for `good-first-issue` tags!

## **Our Vision 🌍**

Our mission: **scalable, efficient, high-performance AI models**. We’re empowering researchers and developers to train bigger, deploy faster, and innovate freely. Learn more at [Eigen Core](https://www.eigencore.org).