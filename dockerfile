# Build stage
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder
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
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2 && \
    cd /detectron2 && \
    pip install --no-cache-dir -e .

# Install DensePose
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# Final stage
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
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

# Ensure the Conda environment is activated by setting the PATH
ENV PATH="/opt/conda/envs/catvton/bin:$PATH"

# Copy application files and models
COPY serving.py .
COPY model /app/model
COPY vton_utils.py .
COPY eval.py .
COPY inference.py .
COPY preprocess_agnostic_mask.py .
COPY detectron2 .
COPY densepose .

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 5000
CMD ["conda", "run", "--no-capture-output", "-n", "catvton", "uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "5000"]