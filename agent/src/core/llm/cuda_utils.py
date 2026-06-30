"""Rilevamento della directory CUDA runtime su Linux/WSL.

Su Windows il loader DLL trova automaticamente le librerie CUDA via PATH/CUDA_PATH.
Su Linux/WSL find_cuda_lib_dir() restituisce la directory contenente libcudart.so*,
usata da server_manager per impostare LD_LIBRARY_PATH prima di avviare i subprocess.
"""
import glob as _glob
import os as _os
import sysconfig as _sysconfig
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


def find_cuda_pip_lib_dirs() -> list[str]:
    """Restituisce le directory lib dei pacchetti pip nvidia-* nel venv corrente.

    I wheel CUDA di PyTorch e llama-cpp-python installano le librerie runtime
    (libcudart, libcublas, …) in site-packages/nvidia/<componente>/lib/. Queste
    directory non sono sul loader path di default, perciò libllama.so non riesce
    a trovare libcudart.so.12 / libcublas.so.12 senza aggiungerle a LD_LIBRARY_PATH.

    Su Windows restituisce sempre [] (il loader DLL usa PATH / CUDA_PATH).
    """
    if sys.platform == "win32":
        return []
    bases = {
        _sysconfig.get_paths().get("purelib"),
        _sysconfig.get_paths().get("platlib"),
    }
    out: list[str] = []
    seen: set[str] = set()
    for base in bases:
        if not base:
            continue
        for d in sorted(_glob.glob(_os.path.join(base, "nvidia", "*", "lib"))):
            if d not in seen and _glob.glob(f"{d}/*.so*"):
                seen.add(d)
                out.append(d)
    return out
