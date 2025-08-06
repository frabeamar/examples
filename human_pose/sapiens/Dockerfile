# Use PyTorch image with CUDA 12.1 and Python preinstalled
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update base system and install Python 3.10, pip, and other tools
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv curl git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade pip

# Install Python packages system-wide
RUN pip install opencv-python tqdm json-tricks

# Confirm versions
RUN python --version && pip --version && \
    python -c "import torch; print('PyTorch:', torch.__version__)"

# COPY ./cv /workspace/cv
# COPY ./det /workspace/det
# COPY ./engine /workspace/engine
WORKDIR /workspace
# ENV SAPIENS_ROOT=/workspace/sapiens
# # Install MMCV and MMDetection
# RUN cd /workspace/engine; pip install -e .
# RUN cd /workspace/cv; pip install -e .
# RUN cd /workspace/det; pip install -e .
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.1.0"
RUN mim install "mmdet>=3.1.0"

#pull on top
RUN apt install libgl1-mesa-glx -y 
RUN pip install "numpy<2.0"

# Set default command
CMD ["python"]
