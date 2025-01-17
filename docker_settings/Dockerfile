# Use NVIDIA's base image with CUDA 12.1 support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive 

# Install essential tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install the latest PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set the CUDA architecture(s) manually
ENV TORCH_CUDA_ARCH_LIST="8.0"

# Clone and install a Github repository
WORKDIR /workspace
RUN git clone https://github.com/Fragile-azalea/zfp.git
WORKDIR /workspace/zfp
RUN mkdir build
WORKDIR /workspace/zfp/build
# Build ZFP with CUDA on A100 (sm_80), install to /usr/local
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DZFP_WITH_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    ..
RUN cmake --build . --config Release
RUN cmake --install .  # Puts headers in /usr/local/include, lib in /usr/local/lib
RUN RUN ctest --output-on-failure || true
WORKDIR /workspace

RUN git clone https://github.com/MaverickJune/ScheMoe_Custom.git
WORKDIR /workspace/ScheMoe_Custom
RUN pip install -e .

# Verify installation of nvcc
RUN nvcc --version

# Verify installation of PyTorch and CUDA compatibility
RUN python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

# Default command
CMD ["/bin/bash"]