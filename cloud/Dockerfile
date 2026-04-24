# ── Open Sauce 2026 — Cloud GPU Container ──────────────────────────────────
# Runs COLMAP + Gaussian Splatting training (Inria implementation) on RunPod.
# Base: PyTorch 2.3 + CUDA 12.1 devel (pre-compiled CUDA toolkit needed for
# building the diff-gaussian-rasterization and simple-knn CUDA extensions).
#
# Build:  docker build -t opensauce2026-gpu .
# Push:   docker push your-dockerhub/opensauce2026-gpu:latest
# RunPod: Set container image to your-dockerhub/opensauce2026-gpu:latest
# ───────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    ninja-build \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── COLMAP (from Ubuntu packages — 3.8 available on 22.04 PPA) ────────────
# Using the COLMAP PPA for a recent stable version
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ethz-asl/colmap && \
    apt-get update && apt-get install -y colmap && \
    rm -rf /var/lib/apt/lists/*

# ── Node.js + splat-transform ──────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @playcanvas/splat-transform && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies (base) ─────────────────────────────────────────────
RUN pip install --no-cache-dir \
    boto3 \
    runpod \
    plyfile \
    numpy \
    tqdm \
    Pillow \
    scipy \
    imageio \
    lpips

# ── Gaussian Splatting (Inria reference implementation) ──────────────────
# Clone the repo and pre-compile the CUDA extensions so they're baked into
# the image — this avoids per-container compilation at job start.
WORKDIR /app
RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git

WORKDIR /app/gaussian-splatting

# Install pip requirements from the repo
RUN pip install --no-cache-dir -r requirements.txt

# Compile diff-gaussian-rasterization (custom CUDA extension)
RUN pip install --no-cache-dir submodules/diff-gaussian-rasterization

# Compile simple-knn (custom CUDA extension)
RUN pip install --no-cache-dir submodules/simple-knn

# ── Handler ────────────────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py /app/handler.py

# Smoke-test: verify key imports work
RUN python -c "import runpod, boto3, plyfile; print('imports OK')"
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || true

CMD ["python", "-u", "/app/handler.py"]
