"""Rilevamento della directory CUDA runtime su Linux/WSL.

Su Windows il loader DLL trova automaticamente le librerie CUDA via PATH/CUDA_PATH.
Su Linux/WSL find_cuda_lib_dir() restituisce la directory contenente libcudart.so*,
usata da server_manager per impostare LD_LIBRARY_PATH prima di avviare i subprocess.
"""
import glob as _glob
import sys

# Percorsi noti dove può trovarsi libcudart.so su Linux/WSL.
# Aggiungere qui se si aggiunge supporto per nuove installazioni CUDA.
_CANDIDATES = [
    "/usr/local/cuda/lib64",
    "/usr/local/lib/ollama/cuda_v12",
    "/usr/lib/wsl/lib",
    "/usr/lib/x86_64-linux-gnu",
]


def find_cuda_lib_dir() -> str | None:
    """Restituisce la prima directory Linux contenente libcudart.so*, o None.

    Su Windows restituisce sempre None: il loader Windows trova le DLL CUDA
    (cublas64_*.dll, cudart64_*.dll) automaticamente via PATH / CUDA_PATH.
    """
    if sys.platform == "win32":
        return None
    for d in _CANDIDATES:
        if _glob.glob(f"{d}/libcudart.so*"):
            return d
    return None
