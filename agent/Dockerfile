# syntax=docker/dockerfile:1

# =============================================================================
# Vesper — immagine Docker con llama.cpp accelerato via CUDA.
#
# Il CUDA toolkit serve SOLO nello stage di build (per compilare llama-cpp-python
# da source): viene compilato una volta sola qui, l'immagine finale contiene
# solo il runtime CUDA. llama-cpp-python è buildato con baseline CPU AVX2 e
# SENZA AVX512, così non va in SIGILL su CPU consumer (es. Intel i9-14900).
# =============================================================================

ARG CUDA_VERSION=12.6.3

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder: compila llama-cpp-python (CUDA) e installa le dipendenze
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

# Compute capability della GPU target (86 = RTX A2000 / Ampere).
# Override con: docker build --build-arg CUDA_ARCH=89 ...  (89 = Ada, 90 = Hopper)
ARG CUDA_ARCH=86
ARG LLAMA_CPP_VERSION=0.3.28
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Venv nella posizione attesa da main.py (/app/.venv): così l'entrypoint gira
# senza il re-exec di bootstrap.
RUN python3.12 -m venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/app/.venv"

RUN pip install --no-cache-dir --upgrade pip cmake ninja

# PyTorch CUDA + dipendenze applicative (torch separato per usare l'indice cu124)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url "${TORCH_INDEX}" \
    && pip install --no-cache-dir -r requirements.txt

# llama-cpp-python da source con CUDA. AVX2 on / AVX512 off → portabile e niente
# SIGILL su CPU senza AVX512. Arch GPU limitata a ${CUDA_ARCH} per build più rapida.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF -DLLAMA_CURL=OFF" \
    CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
    pip install --no-cache-dir "llama-cpp-python[server]==${LLAMA_CPP_VERSION}" \
        --no-binary llama-cpp-python

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime: immagine snella, solo runtime CUDA + venv già compilato
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Venv già pronto: stesso path → nessuna riscrittura degli shebang necessaria.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONUNBUFFERED=1

# Codice applicativo (data/, vault/, .venv, .env esclusi via .dockerignore:
# arrivano come volumi a runtime).
COPY . .

# Web UI / API FastAPI. LLM (8000) e reranker (8001) restano interni al container.
EXPOSE 8080

# sys.prefix == /app/.venv ⇒ main.py non fa il re-exec di bootstrap.
CMD ["/app/.venv/bin/python", "main.py"]
