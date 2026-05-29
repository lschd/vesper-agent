"""
GPUManager — selezione device e interrogazione VRAM.

Usato da:
    server_manager._compute_n_gpu_layers() — interrogazione VRAM aggregata
    server_manager._compute_tensor_split() — VRAM per GPU per il tensor split
    server_manager.preflight()             — indice GPU non-LLM
    rag/indexer.get_chroma_client()        — scelta device embedding
    RerankerServerManager.start()          — main_gpu per il server re-ranker
"""
import logging
import subprocess

logger = logging.getLogger(__name__)

# Soglia minima per caricare l'embedding su GPU.
# Usata da get_embedding_device() prima che l'LLM sia avviato.
EMBEDDING_VRAM_MIB: int = 1536  # bge-m3 fp16 ~1.1 GiB + margine


# ── Interrogazione VRAM ───────────────────────────────────────────────────────

def get_vram_mib_all() -> list[tuple[int, int]] | None:
    """
    Restituisce [(total_mib, free_mib), ...] per ogni GPU rilevata.

    Restituisce None se nvidia-smi non è disponibile o nessuna GPU trovata.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().splitlines():
                if line.strip():
                    t, f = (int(x.strip()) for x in line.split(","))
                    gpus.append((t, f))
            return gpus if gpus else None
    except Exception:
        pass
    return None


def get_vram_mib(gpu_count: int = 1) -> tuple[int, int] | None:
    """
    Restituisce (total_mib, free_mib) sommando le prime `gpu_count` GPU.

    Restituisce None se nvidia-smi non è disponibile.
    Usare gpu_count=1 per verificare la VRAM di un singolo device;
    usare gpu_count=N per il budget aggregato multi-GPU dell'LLM.
    """
    vram_all = get_vram_mib_all()
    if vram_all is None:
        return None
    total = free = 0
    for t, f in vram_all[:gpu_count]:
        total += t
        free += f
    return (total, free) if total else None


# ── Topologia PCIe ────────────────────────────────────────────────────────────

def rank_gpus_by_pcie() -> list[tuple[int, int]]:
    """
    Classifica le GPU per larghezza di banda PCIe (gen × width).

    Restituisce [(gpu_index, score), ...] ordinati per score decrescente
    (prima le GPU migliori per l'LLM). Fallback a [(0, 0)] se nvidia-smi
    non è disponibile o restituisce dati non parsabili.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            scores: list[tuple[int, int]] = []
            for i, line in enumerate(result.stdout.strip().splitlines()):
                if line.strip():
                    parts = line.split(",")
                    if len(parts) >= 2:
                        gen = int(parts[0].strip())
                        width = int(parts[1].strip())
                        scores.append((i, gen * width))
            if scores:
                return sorted(scores, key=lambda x: x[1], reverse=True)
    except Exception:
        pass
    return [(0, 0)]


def get_non_llm_gpu_index() -> int:
    """
    Restituisce l'indice della GPU meno adatta per l'LLM (PCIe più lento).

    Su sistemi single-GPU restituisce sempre 0.
    Con più GPU, questa GPU viene assegnata all'embedding e al re-ranker
    mentre le GPU con PCIe più veloce vengono privilegiate per l'LLM.
    """
    ranked = rank_gpus_by_pcie()
    if len(ranked) <= 1:
        return 0
    # ranked è ordinato decrescente: l'ultimo ha il PCIe più lento
    return ranked[-1][0]


# ── Device selection ──────────────────────────────────────────────────────────

def _torch_cuda_available() -> bool:
    """Restituisce True solo se PyTorch è installato con supporto CUDA."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_embedding_device() -> str:
    """
    Restituisce il device per il modello di embedding.

    Su sistemi multi-GPU restituisce 'cuda:N' dove N è la GPU con PCIe più lento
    (riservata a embedding e re-ranker per lasciare le GPU più veloci all'LLM).
    Su sistemi single-GPU restituisce 'cuda' se la VRAM è sufficiente, altrimenti 'cpu'.

    Va chiamato dopo l'avvio del server LLM: la VRAM libera riportata riflette
    già il consumo dell'LLM, quindi il check è conservativo e corretto.
    """
    vram_all = get_vram_mib_all()
    if vram_all is None:
        logger.warning("GPU non rilevata — embedding su CPU (indicizzazione lenta possibile)")
        return "cpu"

    if not _torch_cuda_available():
        logger.warning(
            "nvidia-smi rileva GPU ma PyTorch non ha supporto CUDA — embedding su CPU. "
            "Reinstalla PyTorch con CUDA: uv pip install torch --index-url "
            "https://download.pytorch.org/whl/cu124"
        )
        return "cpu"

    non_llm_idx = get_non_llm_gpu_index()
    multi_gpu = len(vram_all) > 1

    idx = non_llm_idx if multi_gpu else 0
    if idx >= len(vram_all):
        idx = 0
    _, free_mib = vram_all[idx]

    device = f"cuda:{idx}" if multi_gpu else "cuda"

    if free_mib >= EMBEDDING_VRAM_MIB:
        logger.info(
            "Embedding → %s (%d MiB liberi, modello %d MiB)",
            device, free_mib, EMBEDDING_VRAM_MIB,
        )
        return device

    logger.warning(
        "Embedding → CPU (VRAM libera %d MiB < %d MiB richiesti) — "
        "indicizzazione lenta. Per liberare VRAM: ridurre LLM_N_GPU_LAYERS nel .env.",
        free_mib, EMBEDDING_VRAM_MIB,
    )
    return "cpu"
