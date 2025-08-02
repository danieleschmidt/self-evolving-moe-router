# Multi-stage Dockerfile for Self-Evolving MoE-Router Development Environment
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-venv \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    graphviz \
    graphviz-dev \
    htop \
    vim \
    tmux \
    tree \
    jq \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Install conda for better package management
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Update conda and install Python 3.9 environment
RUN conda update -n base -c defaults conda && \
    conda create -n moe python=3.9 -y && \
    echo "source activate moe" >> ~/.bashrc
ENV PATH="/opt/conda/envs/moe/bin:${PATH}"

# Install core ML dependencies with specific versions for compatibility
RUN conda install -n moe -c conda-forge -c pytorch -c nvidia \
    pytorch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    pytorch-cuda=11.8 \
    numpy=1.24.3 \
    scipy=1.11.2 \
    matplotlib=3.7.2 \
    seaborn=0.12.2 \
    pandas=2.0.3 \
    scikit-learn=1.3.0 \
    jupyter \
    jupyterlab \
    ipykernel \
    notebook \
    -y

# Install development and testing tools
RUN pip install --no-cache-dir \
    pytest==7.4.0 \
    pytest-cov==4.1.0 \
    pytest-xdist==3.3.1 \
    pytest-mock==3.11.1 \
    black==23.7.0 \
    isort==5.12.0 \
    flake8==6.0.0 \
    mypy==1.5.1 \
    pre-commit==3.3.3 \
    ruff==0.0.284

# Install ML and evolution-specific packages
RUN pip install --no-cache-dir \
    transformers==4.33.0 \
    datasets==2.14.4 \
    accelerate==0.22.0 \
    evaluate==0.4.0 \
    wandb==0.15.8 \
    tensorboard==2.14.0 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    optuna==3.3.0 \
    ray==2.6.3 \
    pyyaml==6.0.1 \
    tqdm==4.66.1

# Install visualization and analysis tools
RUN pip install --no-cache-dir \
    plotly==5.15.0 \
    dash==2.13.0 \
    networkx==3.1 \
    graphviz==0.20.1 \
    seaborn==0.12.2 \
    bokeh==3.2.2 \
    altair==5.0.1

# Install additional scientific computing tools
RUN pip install --no-cache-dir \
    numba==0.57.1 \
    cython==3.0.0 \
    statsmodels==0.14.0 \
    xarray==2023.7.0 \
    dask==2023.8.1 \
    distributed==2023.8.1

# Install code quality and documentation tools
RUN pip install --no-cache-dir \
    sphinx==7.1.2 \
    sphinx-rtd-theme==1.3.0 \
    myst-parser==2.0.0 \
    nbsphinx==0.9.2 \
    autopep8==2.0.2 \
    bandit==1.7.5 \
    safety==2.3.4

# Set up workspace
WORKDIR /workspace

# Create necessary directories
RUN mkdir -p /workspace/.cache/wandb \
             /workspace/.cache/huggingface \
             /workspace/.cache/torch \
             /workspace/data/datasets \
             /workspace/data/models \
             /workspace/data/experiments \
             /workspace/logs \
             /workspace/results

# Set up git configuration template
RUN git config --global init.defaultBranch main && \
    git config --global core.editor "vim" && \
    git config --global pull.rebase false

# Configure Jupyter for remote development
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py

# Install additional kernels
RUN python -m ipykernel install --user --name moe --display-name "Self-Evolving MoE"

# Set environment variables for development
ENV PYTHONPATH="/workspace/src:/workspace"
ENV JUPYTER_ENABLE_LAB=yes
ENV WANDB_CACHE_DIR="/workspace/.cache/wandb"
ENV HF_HOME="/workspace/.cache/huggingface"
ENV TORCH_HOME="/workspace/.cache/torch"

# Create development entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Activate conda environment\n\
source activate moe\n\
\n\
# Start services if requested\n\
if [ "$START_JUPYTER" = "true" ]; then\n\
    echo "Starting Jupyter Lab..."\n\
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &\n\
fi\n\
\n\
if [ "$START_TENSORBOARD" = "true" ]; then\n\
    echo "Starting TensorBoard..."\n\
    tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006 &\n\
fi\n\
\n\
# Execute the main command\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import numpy; import matplotlib; print('Health check passed')" || exit 1

# Development mode - keep container running
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# Production stage (minimal for deployment)
FROM base as production

# Remove development tools to reduce image size
RUN conda clean -afy && \
    pip cache purge && \
    apt-get autoremove -y && \
    apt-get clean

# Copy only necessary files
COPY src/ /workspace/src/
COPY configs/ /workspace/configs/
COPY pyproject.toml /workspace/

# Install the package
RUN pip install -e .

# Set production entrypoint
CMD ["python", "-m", "self_evolving_moe.cli"]