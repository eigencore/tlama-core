# **How to Set Up Tlama Core** 💻

Getting started with Tlama Core is easy! Follow these steps to set up the repository and begin working on the optimizations, regardless of whether you're using **Windows**, **Linux**, or **macOS**.

## **Prerequisites**

Before diving into the setup, make sure you have the following installed:
- **CUDA Toolkit** (for GPU support)
- **Python 3.8+** (best within a virtual environment)
- **PyTorch** (GPU version recommended for performance)
- **NVIDIA Driver** (compatible with your GPU)
- **CMake** (required to build CUDA kernels)

### **CUDA Toolkit Setup** 🛠️

1. **Windows**: 
   - Download the [CUDA Toolkit for Windows](https://developer.nvidia.com/cuda-downloads) and install it.
   - Ensure that the CUDA version is compatible with your PyTorch version.
   - Install the [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx).

2. **Linux**: 
   - Install the CUDA Toolkit via your package manager:
     ```bash
     sudo apt update
     sudo apt install nvidia-cuda-toolkit
     ```
   - Alternatively, you can download the toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads) and follow the instructions.
   - Install the [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx).

3. **macOS**: 
   - **CUDA support** is unavailable for macOS, but you can still run CPU-based optimizations and models that do not rely on GPU.

## **Installation Steps** 🚀

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/tlama-core.git
   cd tlama-core
   ```

2. **Set Up the Python Environment**:
   - Create and activate a virtual environment:
     ```bash
     python3 -m venv tlama-env
     source tlama-env/bin/activate  # For Linux/macOS
     .\tlama-env\Scripts\activate  # For Windows
     ```

3. **Install Dependencies**:
   - Install the Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

4. **Install Custom CUDA Kernels**:
   - To enable PyTorch to use the custom CUDA kernels, run:
     ```bash
     python setup.py install
     ```

5. **Verify Installation**:
   - Run the following to check if everything is set up correctly:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```

   - If the output is `True`, you're good to go!