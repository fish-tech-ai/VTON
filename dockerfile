# Build stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies, including Conda
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create Conda environment and install Python 3.9.0
RUN conda create -n catvton python=3.9.0 && \
    echo "conda activate catvton" >> ~/.bashrc
SHELL ["conda", "run", "-n", "catvton", "/bin/bash", "-c"]

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi==0.115.0 uvicorn==0.30.6 python-multipart==0.0.9 && \
    pip install --no-cache-dir torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install Detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2 && \
    cd /detectron2 && \
    pip install --no-cache-dir -e .

# Install DensePose
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# Final stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3 \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for runtime
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Copy the Conda environment from the builder
COPY --from=builder /opt/conda/envs/catvton /opt/conda/envs/catvton

# Copy application files and models
COPY serving.py .
COPY model /app/model
COPY vton_utils.py .
COPY eval.py .
COPY inference.py .
COPY preprocess_agnostic_mask.py .
COPY detectron2 .
COPY densepose .

ENV PATH="/opt/conda/envs/catvton/bin:$PATH"
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform

EXPOSE 5000
CMD ["conda", "run", "--no-capture-output", "-n", "catvton", "uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]