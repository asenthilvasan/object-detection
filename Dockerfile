# Dockerfile for Ray Serve Object Detection Service
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller, no NVIDIA packages)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies directly with pip (skip poetry to avoid duplicates)
RUN pip install --no-cache-dir \
    "ray[serve]>=2.44.1" \
    ultralytics \
    tqdm \
    seaborn \
    scipy \
    pillow \
    numpy \
    opencv-python-headless \
    pandas \
    gitpython \
    requests

# Copy application code
COPY src/ ./src/
COPY README.md ./
COPY run_serve.py ./

# Expose ports
EXPOSE 8000 8265

# Start Ray Serve
CMD ["python", "run_serve.py"]
