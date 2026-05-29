"""Gestione del ciclo di vita del server llama-cpp quando LLM_MANAGED=true."""
import asyncio
import logging
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path

import httpx

from src.core.llm.cuda_utils import find_cuda_lib_dir
from src.core.llm.gpu_manager import (
    get_non_llm_gpu_index as _get_non_llm_gpu_index,
    get_vram_mib as _get_vram_mib,
    get_vram_mib_all as _get_vram_mib_all,
    rank_gpus_by_pcie as _rank_gpus_by_pcie,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL = 5
STARTUP_TIMEOUT = 600
LOG_INTERVAL = 30

_CTX_DEFAULT = 8192   # contesto sicuro quando VRAM non è rilevabile
_CTX_MIN     = 2048   # minimo accettabile — sotto questa soglia si usa --no-kv-offload
_CTX_MAX     = 65536  # cap hard — anche con RAM illimitata non ha senso andare oltre

# Binari nativi llama-server scaricati da install.py.
# Windows: CUDA DLL bundled → GPU senza toolkit installato.
# Linux: linkato dinamicamente a CUDA → richiede LD_LIBRARY_PATH verso /usr/lib/wsl/lib o simili.
_WIN_LLAMA_BIN   = Path(__file__).resolve().parents[3] / "data" / "bin" / "llama-server" / "llama-server.exe"
_LINUX_LLAMA_BIN = Path(__file__).resolve().parents[3] / "data" / "bin" / "llama-server" / "llama-server"


# ── Utility GPU ───────────────────────────────────────────────────────────────

def _detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 1


# ── GGUF metadata reader ──────────────────────────────────────────────────────

def _read_gguf_block_count(model_path: Path) -> int | None:
    """Legge block_count dal metadata GGUF senza caricare i pesi in memoria."""

    def _read_val(f, vtype: int):
        _FMT  = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        _SIZE = {0: 1,   1: 1,   2: 2,   3: 2,   4: 4,   5: 4,
                 6: 4,   7: 1,   10: 8,  11: 8,  12: 8}
        if vtype in _FMT:
            return struct.unpack(f"<{_FMT[vtype]}", f.read(_SIZE[vtype]))[0]
        if vtype == 8:   # string
            return f.read(struct.unpack("<Q", f.read(8))[0])
        if vtype == 9:   # array
            et = struct.unpack("<I", f.read(4))[0]
            for _ in range(struct.unpack("<Q", f.read(8))[0]):
                _read_val(f, et)
            return None
        raise ValueError(f"GGUF vtype sconosciuto: {vtype}")

    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return None
            f.read(8)  # tensor_count
            kv_count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(kv_count):
                key = f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                val = _read_val(f, vtype)
                if key.endswith(".block_count"):
                    return int(val)
    except Exception:
        pass
    return None


def _read_gguf_kv_params(model_path: Path) -> "tuple[int, int, int] | None":
    """
    Legge (n_layers, n_kv_heads, head_dim) dal GGUF per stimare la KV cache.
    Fa un singolo passaggio sul file cercando le chiavi con i suffissi noti.
    Restituisce None se il file non è leggibile o le chiavi non sono presenti.
    """
    import struct as _struct

    want: dict[str, "int | None"] = {
        "block_count":             None,
        "attention.head_count_kv": None,
        "attention.head_count":    None,
        "embedding_length":        None,
    }
    _FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    _FMT   = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
              6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            version = _struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return None
            f.read(8)  # tensor_count
            kv_count = _struct.unpack("<Q", f.read(8))[0]

            def _skip(vtype: int) -> None:
                if vtype in _FIXED:
                    f.read(_FIXED[vtype])
                elif vtype == 8:
                    f.read(_struct.unpack("<Q", f.read(8))[0])
                elif vtype == 9:
                    et = _struct.unpack("<I", f.read(4))[0]
                    for _ in range(_struct.unpack("<Q", f.read(8))[0]):
                        _skip(et)

            def _read_int(vtype: int) -> "int | None":
                if vtype in _FIXED:
                    return int(_struct.unpack(f"<{_FMT[vtype]}", f.read(_FIXED[vtype]))[0])
                _skip(vtype)
                return None

            found = 0
            for _ in range(kv_count):
                key = f.read(_struct.unpack("<Q", f.read(8))[0]).decode("utf-8", errors="replace")
                vtype = _struct.unpack("<I", f.read(4))[0]
                matched = next((k for k in want if key.endswith("." + k) and want[k] is None), None)
                if matched is not None:
                    want[matched] = _read_int(vtype)
                    found += 1
                    if found == len(want):
                        break
                else:
                    _skip(vtype)
    except Exception:
        return None

    n_layers   = want["block_count"]
    n_kv_heads = want["attention.head_count_kv"]
    n_heads    = want["attention.head_count"]
    emb_len    = want["embedding_length"]

    if n_layers is None or n_kv_heads is None:
        return None
    head_dim = (emb_len // n_heads) if (n_heads and emb_len) else 128
    return int(n_layers), int(n_kv_heads), int(head_dim)


def _ctx_from_vram(
    n_layers: int, n_kv_heads: int, head_dim: int, budget_mib: int
) -> "tuple[int, list[str]]":
    """
    Determina i flag KV cache necessari per allocare _CTX_DEFAULT token in VRAM.

    Non massimizza il contesto — usa sempre _CTX_DEFAULT come target fisso.
    Chi vuole più contesto imposta LLM_N_CTX nel .env.

    Strategia in ordine:
    1. KV f16 in VRAM  — nessun flag extra
    2. KV q4_0 in VRAM — 4× meno VRAM; aggiunge --cache-type-k/v q4_0
    3. KV in RAM       — --no-kv-offload; nessun limite VRAM per la KV cache

    Returns:
        (_CTX_DEFAULT, extra_flags)
    """
    # KV f16: K + V, per ogni layer, per ogni KV head, head_dim elementi a 2 byte ciascuno
    kv_f16_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    if kv_f16_per_token <= 0:
        return _CTX_DEFAULT, []

    # Riserva 512 MiB per CUDA context, compute buffer, framebuffers
    avail_bytes = max(0, budget_mib - 512) * 1024 * 1024
    target_bytes = kv_f16_per_token * _CTX_DEFAULT

    if target_bytes <= avail_bytes:
        logger.info(
            "llm_server: KV f16 in VRAM — ctx_size=%d (%.0f KiB/token, budget=%.1f GiB)",
            _CTX_DEFAULT, kv_f16_per_token / 1024, budget_mib / 1024,
        )
        return _CTX_DEFAULT, []

    # f16 non basta — prova q4_0 (4× meno VRAM)
    if (target_bytes // 4) <= avail_bytes:
        logger.info(
            "llm_server: VRAM limitata per KV f16 → KV q4_0 — ctx_size=%d (budget=%.1f GiB)",
            _CTX_DEFAULT, budget_mib / 1024,
        )
        return _CTX_DEFAULT, ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]

    # Anche q4 insufficiente — KV cache in RAM di sistema
    logger.warning(
        "llm_server: VRAM insufficiente per KV cache (%.1f GiB disponibili, %.0f KiB/token f16) "
        "— KV in RAM (--no-kv-offload), ctx_size=%d",
        budget_mib / 1024, kv_f16_per_token / 1024, _CTX_DEFAULT,
    )
    return _CTX_DEFAULT, ["--no-kv-offload"]


# ── Dimensione reale del re-ranker ────────────────────────────────────────────

def _reranker_size_mib() -> int:
    """Dimensione del file GGUF del re-ranker in MiB (0 se disabilitato o non trovato)."""
    from config import Config
    if not Config.reranker_enabled:
        return 0
    path = Path(Config.reranker_model_dir).resolve() / Config.reranker_model_file
    try:
        return path.stat().st_size // (1024 * 1024)
    except OSError:
        logger.warning("File re-ranker non trovato (%s) — reserve VRAM = 0 MiB", path)
        return 0


# ── Calcolo ottimale n_gpu_layers ─────────────────────────────────────────────

def _compute_n_gpu_layers(model_path: Path, n_gpu_layers_cfg: int, gpu_count: int = 1) -> int:
    """
    Se n_gpu_layers_cfg è -1 (auto), calcola il massimo numero di layer
    che entra nella VRAM disponibile.

    Va chiamato dopo il caricamento dell'embedding: la VRAM libera misurata
    riflette già il suo consumo reale. Si riserva solo lo spazio per il
    re-ranker (calcolato dalla dimensione reale del file GGUF) perché parte
    in parallelo con l'LLM e non è ancora in VRAM in questo momento.

    Se n_gpu_layers_cfg != -1, rispetta la scelta dell'utente senza modifiche.
    """
    if n_gpu_layers_cfg != -1:
        return n_gpu_layers_cfg

    vram = _get_vram_mib(gpu_count)
    if vram is None:
        logger.warning("VRAM non rilevabile (nvidia-smi assente?) — avvio con n_gpu_layers=-1")
        return -1

    total_mib, free_mib = vram
    model_size_mib = model_path.stat().st_size // (1024 * 1024)
    reranker_mib = _reranker_size_mib()
    usable_mib = free_mib - reranker_mib

    logger.info(
        "VRAM — budget pre-LLM %dx GPU:\n"
        "  VRAM totale:  %5d MiB  (%.1f GiB)\n"
        "  Libera:       %5d MiB  (%.1f GiB)  (embedding già in VRAM)\n"
        "  Re-ranker:   -%5d MiB  (da file)\n"
        "  ─────────────────────────────────────\n"
        "  Disponibile:  %5d MiB  (%.1f GiB)\n"
        "  Modello:      %5d MiB  (%.1f GiB)",
        gpu_count,
        total_mib, total_mib / 1024,
        free_mib, free_mib / 1024,
        reranker_mib,
        usable_mib, usable_mib / 1024,
        model_size_mib, model_size_mib / 1024,
    )

    if usable_mib >= model_size_mib:
        logger.info(
            "Modello LLM completo in GPU (%.1f GiB disponibili ≥ %.1f GiB modello)",
            usable_mib / 1024, model_size_mib / 1024,
        )
        return -1

    block_count = _read_gguf_block_count(model_path)
    if block_count is None:
        logger.warning(
            "VRAM insufficiente (%.1f GiB disponibili vs %.1f GiB modello) "
            "ma block_count non leggibile dal GGUF — avvio con n_gpu_layers=-1 (possibile OOM)",
            usable_mib / 1024, model_size_mib / 1024,
        )
        return -1

    n_layers = max(0, int((usable_mib / model_size_mib) * block_count))
    pct = 100 * n_layers // block_count if block_count else 0
    logger.warning(
        "VRAM insufficiente per il modello completo (%.1f GiB disponibili, %.1f GiB necessari)\n"
        "  Offload automatico: %d/%d layer in GPU (%d%%), rimanenti in RAM.\n"
        "  Per fissare manualmente: LLM_N_GPU_LAYERS=%d nel .env",
        usable_mib / 1024, model_size_mib / 1024,
        n_layers, block_count, pct, n_layers,
    )
    return n_layers


# ── CUDA backend check (Linux/WSL, fallback llama-cpp-python) ─────────────────

def _llama_has_cuda_backend() -> bool:
    """Verifica che llama-cpp-python sia installato con backend CUDA (Linux/WSL).

    Pattern in ordine:
    1a. dist-info tag: vecchio formato con +cu1 nel nome (es. llama_cpp_python-0.3.9+cu124)
    1b. direct_url.json: installato da URL con tag CUDA nel path (formato 0.3.20+)
    2.  libggml-cuda*: librerie separate nel package (0.2.x / 0.3.x early)
    3.  ldd su .so nel package: CUDA linkato dinamicamente (0.3.x+)
    """
    import glob as _glob
    import json as _json
    venv_lib = Path(__file__).resolve().parents[3] / ".venv" / "lib"

    # Pattern 1a: dist-info tag vecchio formato (llama_cpp_python-0.3.9+cu124.dist-info)
    for di in _glob.glob(str(venv_lib / "**" / "llama_cpp_python-*.dist-info"), recursive=True):
        if "cu1" in Path(di).name:
            return True

    # Pattern 1b: direct_url.json contiene tag CUDA nel path (es. .../v0.3.23-cu124/...)
    for durl in _glob.glob(
        str(venv_lib / "**" / "llama_cpp_python-*.dist-info" / "direct_url.json"), recursive=True
    ):
        try:
            data = _json.loads(Path(durl).read_text(encoding="utf-8"))
            url = data.get("url", "")
            if "-cu1" in url or "/cu1" in url:
                return True
        except Exception:
            pass

    # Pattern 2: libggml-cuda* nel package (0.2.x / 0.3.x early)
    if _glob.glob(str(venv_lib / "**" / "llama_cpp" / "**" / "libggml-cuda*"), recursive=True):
        return True

    # Pattern 3: ldd su qualsiasi .so nel package llama_cpp (0.3.x+)
    for lib in _glob.glob(str(venv_lib / "**" / "llama_cpp" / "**" / "*.so"), recursive=True):
        try:
            r = subprocess.run(["ldd", lib], capture_output=True, text=True, timeout=5)
            if any(dep in r.stdout for dep in ("libcudart", "libcublas")):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return False


# ── Log helper ────────────────────────────────────────────────────────────────

def _tail_log(log_file: Path, n: int = 20) -> str:
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        _ERROR_KEYWORDS = ("failed", "error", "Error", "unable", "CUDA error", "out of memory")
        last_n = lines[-n:]
        first_error_idx = next(
            (i for i, l in enumerate(last_n) if any(k in l for k in _ERROR_KEYWORDS)),
            0,
        )
        excerpt = "\n".join(last_n[first_error_idx:])
        return f"Ultime righe di {log_file}:\n{excerpt}"
    except Exception:
        return ""


# ── Helper condiviso: attendi che un server risponda ─────────────────────────

async def _wait_for_server_ready(
    health_url: str,
    process: subprocess.Popen,
    log_file: Path,
    timeout: int = STARTUP_TIMEOUT,
    label: str = "server",
) -> None:
    """Attende che il server risponda HTTP 200 su health_url; RuntimeError se timeout."""
    import time
    t0 = time.monotonic()
    next_log_at = t0

    async with httpx.AsyncClient() as client:
        while True:
            now = time.monotonic()
            elapsed = now - t0

            if elapsed >= timeout:
                break

            if process.poll() is not None:
                raise RuntimeError(
                    f"{label} terminato inaspettatamente (exit code {process.returncode})\n"
                    + _tail_log(log_file)
                )

            if now >= next_log_at:
                logger.info("%s in avvio, attendere...", label)
                next_log_at = now + LOG_INTERVAL

            try:
                resp = await client.get(health_url, timeout=3.0)
                if resp.status_code == 200:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"{label} terminato inaspettatamente (exit code {process.returncode}) — "
                            "la porta è probabilmente già occupata\n" + _tail_log(log_file)
                        )
                    logger.info("%s pronto e raggiungibile", label)
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            await asyncio.sleep(POLL_INTERVAL)

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    raise RuntimeError(f"{label} non è diventato disponibile entro {timeout}s")


def _build_ld_library_path(env: dict, using_native_bin: bool) -> None:
    """Imposta LD_LIBRARY_PATH su Linux/WSL con CUDA dir e, se binario nativo, la sua directory."""
    ld_parts: list[str] = []

    cuda_dir = find_cuda_lib_dir()
    if cuda_dir:
        ld_parts.append(cuda_dir)
        logger.info("CUDA runtime trovata in: %s", cuda_dir)
    elif not using_native_bin:
        logger.warning(
            "CUDA runtime non trovata — LD_LIBRARY_PATH non impostato. "
            "Se hai CUDA installato, aggiungi il path a _CANDIDATES in cuda_utils.py."
        )

    if using_native_bin:
        # La directory del binario può contenere .so bundled (libllama.so, libggml.so, …)
        ld_parts.append(str(_LINUX_LLAMA_BIN.parent))

    if ld_parts:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts + ([existing] if existing else []))


# ── LLM Server Manager ────────────────────────────────────────────────────────

_llm_server_manager: "LLMServerManager | None" = None


class LLMServerManager:

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self._gpu_count: int = 1
        self._non_llm_gpu: int = 0       # GPU per embedding+reranker (PCIe più lento)
        self._n_gpu_layers: int | None = None  # calcolato in start() dopo l'embedding
        self._use_tensor_split: bool = False   # True solo se il modello non entra nella singola GPU più veloce
        self._llm_main_gpu: int = 0            # GPU primaria per l'LLM quando non si usa tensor split
        self._computed_ctx: int = _CTX_DEFAULT     # ctx_size calcolato in start() da VRAM
        self._ctx_extra_flags: list[str] = []      # flag KV cache (q4, no-kv-offload, …)

    def preflight(self) -> None:
        """
        Controlli pre-avvio: verifica il file modello e la disponibilità della porta.

        Non campiona la VRAM: il calcolo di n_gpu_layers avviene in start(), dopo
        che l'embedding è già caricato e la misura riflette il consumo reale.
        """
        from config import Config

        model_path = Path(Config.llm_model_dir).resolve() / Config.llm_model_file
        if not model_path.exists():
            logger.error(
                "File modello non trovato: %s\n"
                "  1. Verifica LLM_MODEL_REPO e LLM_MODEL_FILE nel file .env\n"
                "  2. Scarica il modello eseguendo: python install.py",
                Config.llm_model_file,
            )
            sys.exit(1)

        port = int(Config.llm_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.settimeout(0.5)
            if _s.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"La porta {port} è già occupata da un altro processo. "
                    "Fermalo prima di avviare Vesper con LLM_MANAGED=true."
                )

        self._gpu_count = _detect_gpu_count() if Config.llm_multi_gpu else 1
        if Config.llm_multi_gpu and self._gpu_count > 1:
            self._non_llm_gpu = _get_non_llm_gpu_index()
            _llm_gpus = "+".join(
                f"GPU{i}" for i in range(self._gpu_count) if i != self._non_llm_gpu
            )
            logger.info(
                "%d GPU rilevate — embedding/reranker su GPU%d, LLM su %s",
                self._gpu_count, self._non_llm_gpu, _llm_gpus,
            )
        else:
            self._non_llm_gpu = 0

    def _compute_tensor_split(self) -> list[int]:
        """
        Calcola il tensor_split proporzionale alla VRAM utilizzabile per l'LLM su ogni GPU.

        La GPU non-LLM ha solo il re-ranker da riservare (embedding è già in VRAM e
        il suo consumo è già riflesso nel free_mib corrente). Fallback a split uguale
        se la VRAM non è rilevabile.
        """
        vram_all = _get_vram_mib_all()
        if vram_all is None or len(vram_all) < self._gpu_count:
            return [1] * self._gpu_count

        reranker_mib = _reranker_size_mib()
        ratios: list[int] = []
        for i in range(self._gpu_count):
            _, free_mib = vram_all[i]
            if i == self._non_llm_gpu:
                # embedding già consumato in free_mib; riserva solo il re-ranker
                free_mib = max(256, free_mib - reranker_mib)
            ratios.append(max(1, free_mib))

        logger.info(
            "Tensor split (MiB usabili per GPU): %s",
            " | ".join(
                f"GPU{i}={r}" + (" [embed+rerank]" if i == self._non_llm_gpu else "")
                for i, r in enumerate(ratios)
            ),
        )
        return ratios

    def _auto_ctx_size(self, model_path: Path) -> "tuple[int, list[str]]":
        """
        Stima ctx_size e flag KV in base alla VRAM disponibile dopo il caricamento del modello.

        Logica:
        - Se n_gpu_layers == 0 (CPU only): KV è già in RAM, usa _CTX_DEFAULT senza flag.
        - Altrimenti: stima la VRAM residua = free_now - model_vram_estimate, poi delega
          a _ctx_from_vram che sceglie tra KV f16, KV q4_0, o --no-kv-offload.
        """
        assert self._n_gpu_layers is not None

        if self._n_gpu_layers == 0:
            return _CTX_DEFAULT, []

        vram = _get_vram_mib(self._gpu_count)
        if vram is None:
            logger.warning("llm_server: VRAM non rilevabile — ctx_size=%d", _CTX_DEFAULT)
            return _CTX_DEFAULT, []

        _, free_mib = vram
        model_size_mib = model_path.stat().st_size // (1024 * 1024)

        if self._n_gpu_layers == -1:
            # Tutto il modello in GPU
            kv_budget_mib = max(0, free_mib - model_size_mib)
        else:
            # Offload parziale: proporzionale ai layer effettivamente in GPU
            block_count = _read_gguf_block_count(model_path) or 1
            frac = min(1.0, self._n_gpu_layers / block_count)
            kv_budget_mib = max(0, free_mib - int(model_size_mib * frac))

        params = _read_gguf_kv_params(model_path)
        if params is None:
            logger.warning("llm_server: parametri KV non leggibili dal GGUF — ctx_size=%d", _CTX_DEFAULT)
            return _CTX_DEFAULT, []

        n_layers, n_kv_heads, head_dim = params
        return _ctx_from_vram(n_layers, n_kv_heads, head_dim, kv_budget_mib)

    def _build_command(self) -> list[str]:
        from config import Config

        assert self._n_gpu_layers is not None, "n_gpu_layers non calcolato — start() non completato"
        model_path = Path(Config.llm_model_dir).resolve() / Config.llm_model_file

        # Contesto effettivo da passare al server: usa LLM_N_CTX se impostato,
        # altrimenti legge context_length dai metadati GGUF (valore nativo del modello),
        # con fallback a 8192 se il file non è leggibile.                                                                                                                                             
        # Senza --ctx-size esplicito il binario llama-server usa il proprio default (spesso 2048).                                                                                                    
        # if Config.llm_n_ctx is not None:                                                                                                                                                              
        #     ctx_size = Config.llm_n_ctx
        # else:
        #     try:
        #         from src.core.llm.client import _read_n_ctx_from_gguf
        #     except ImportError:
        #         from core.llm.client import _read_n_ctx_from_gguf  # type: ignore[no-redef]                                                                                                           
        #     ctx_size = _read_n_ctx_from_gguf(model_path) or 8192                                                                                                                                      
        #     logger.info("llm_server: ctx_size=%d (da GGUF o default)", ctx_size)

        # ctx_size e flag KV calcolati da _auto_ctx_size() in start(), o da LLM_N_CTX se impostato.
        ctx_size  = self._computed_ctx
        kv_native = self._ctx_extra_flags
        # llama_cpp.server usa underscore nei flag invece dei trattini del binario nativo
        kv_python = ["--" + f[2:].replace("-", "_") if f.startswith("--") else f
                     for f in kv_native]

        if sys.platform == "win32" and _WIN_LLAMA_BIN.exists():
            # Binario nativo Windows: CUDA DLL bundled, flag con trattino, tensor-split virgola.
            cmd = [
                str(_WIN_LLAMA_BIN),
                "--model", str(model_path),
                "--n-gpu-layers", str(self._n_gpu_layers),
                "--ctx-size", str(ctx_size),
                "--host", Config.llm_host,
                "--port", Config.llm_port,
            ]
            cmd += kv_native
            if self._use_tensor_split:
                split = ",".join(str(r) for r in self._compute_tensor_split())
                cmd += ["--tensor-split", split]
            elif self._llm_main_gpu != 0:
                cmd += ["--main-gpu", str(self._llm_main_gpu)]

        elif sys.platform != "win32" and _LINUX_LLAMA_BIN.exists():
            # Binario nativo Linux (Ubuntu CUDA): stessi flag del binario Windows.
            # CUDA linkato dinamicamente — LD_LIBRARY_PATH impostato in start().
            cmd = [
                str(_LINUX_LLAMA_BIN),
                "--model", str(model_path),
                "--n-gpu-layers", str(self._n_gpu_layers),
                "--ctx-size", str(ctx_size),
                "--host", Config.llm_host,
                "--port", Config.llm_port,
            ]
            cmd += kv_native
            if self._use_tensor_split:
                split = ",".join(str(r) for r in self._compute_tensor_split())
                cmd += ["--tensor-split", split]
            elif self._llm_main_gpu != 0:
                cmd += ["--main-gpu", str(self._llm_main_gpu)]

        else:
            # Fallback: Python server llama-cpp-python (Linux/WSL senza binario nativo o Windows).
            cmd = [
                sys.executable, "-m", "llama_cpp.server",
                "--model", str(model_path),
                "--n_gpu_layers", str(self._n_gpu_layers),
                "--n_ctx", str(ctx_size),
                "--host", Config.llm_host,
                "--port", Config.llm_port,
            ]
            cmd += kv_python
            if self._use_tensor_split:
                cmd += ["--tensor_split"] + [str(r) for r in self._compute_tensor_split()]
            elif self._llm_main_gpu != 0:
                cmd += ["--main_gpu", str(self._llm_main_gpu)]

        return cmd

    async def start(self) -> None:
        from config import Config

        if not Config.llm_managed:
            logger.info("LLM server gestito esternamente — avvio automatico disabilitato")
            return

        if self.process is not None and self.process.poll() is None:
            logger.warning("LLM server già in esecuzione (PID %d)", self.process.pid)
            return

        model_path = Path(Config.llm_model_dir).resolve() / Config.llm_model_file
        self._n_gpu_layers = _compute_n_gpu_layers(
            model_path, Config.llm_n_gpu_layers, self._gpu_count,
        )

        # Decide se usare tensor split: solo se il modello non entra nella GPU LLM più veloce.
        self._use_tensor_split = False
        if Config.llm_multi_gpu and self._gpu_count > 1:
            vram_all = _get_vram_mib_all()
            if vram_all is not None and len(vram_all) >= self._gpu_count:
                model_size_mib = model_path.stat().st_size // (1024 * 1024)
                ranked = _rank_gpus_by_pcie()
                fastest_llm_idx = next(
                    (idx for idx, _ in ranked if idx != self._non_llm_gpu),
                    ranked[0][0],
                )
                self._llm_main_gpu = fastest_llm_idx
                if fastest_llm_idx < len(vram_all):
                    _, fastest_free_mib = vram_all[fastest_llm_idx]
                    if fastest_free_mib < model_size_mib:
                        self._use_tensor_split = True
                        logger.info(
                            "Tensor split attivato: GPU%d ha %.1f GiB liberi < %.1f GiB modello",
                            fastest_llm_idx, fastest_free_mib / 1024, model_size_mib / 1024,
                        )
                    else:
                        logger.info(
                            "Tensor split disabilitato: GPU%d ha %.1f GiB liberi ≥ %.1f GiB modello "
                            "— modello su singola GPU%d",
                            fastest_llm_idx, fastest_free_mib / 1024, model_size_mib / 1024,
                            fastest_llm_idx,
                        )
            else:
                # VRAM non rilevabile: abilita tensor split per sicurezza
                self._use_tensor_split = True

        # Calcola ctx_size automaticamente dalla VRAM residua dopo il modello,
        # a meno che l'utente non abbia impostato LLM_N_CTX esplicitamente.
        if Config.llm_n_ctx is None:
            self._computed_ctx, self._ctx_extra_flags = self._auto_ctx_size(model_path)
        else:
            self._computed_ctx = Config.llm_n_ctx
            self._ctx_extra_flags = []

        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "llama_server.log"

        cmd = self._build_command()
        logger.info("Avvio llama-server: %s", " ".join(cmd))

        env = os.environ.copy()
        if sys.platform != "win32":
            using_native = _LINUX_LLAMA_BIN.exists()
            if not using_native and not _llama_has_cuda_backend():
                logger.warning(
                    "llama-cpp-python installato senza backend CUDA — il server LLM girerà su CPU.\n"
                    "  Per abilitare GPU: esegui 'python install.py' da WSL/Linux."
                )
            _build_ld_library_path(env, using_native_bin=using_native)

        log_start = log_file.stat().st_size if log_file.exists() else 0

        with open(log_file, "a") as lf:
            self.process = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)

        logger.info("llama-server avviato (PID %d) — log in %s", self.process.pid, log_file)

        health_url = f"http://localhost:{Config.llm_port}/v1/models"
        await _wait_for_server_ready(health_url, self.process, log_file, label="llama-server")
        self._check_gpu_in_log(log_file, log_start)

    def _check_gpu_in_log(self, log_file: Path, log_start: int) -> None:
        """Verifica che il log di questa sessione mostri uso GPU; altrimenti avvisa."""
        if self._n_gpu_layers == 0:
            return  # CPU esplicita — nessun warning
        try:
            with open(log_file, "rb") as f:
                f.seek(log_start)
                content = f.read().decode("utf-8", errors="replace")
            # Pattern del binario nativo e del server Python — include formati llama.cpp b5000+.
            _cuda_patterns = (
                "- CUDA",           # system_info: "Backends: BLAS = 1, CUDA = 1, ..."
                "| CUDA :",         # formato binario pre-b5000
                "ggml_cuda_init",   # funzione init CUDA (b4000–b5000)
                "CUDA0",            # llm_load_tensors: "CUDA0 model buffer size = ..."
                "CUDA_DEVICE_COUNT",
                "GGML_USE_CUDA",
            )
            if any(p in content for p in _cuda_patterns):
                return  # CUDA rilevato e attivo
            logger.warning(
                "llama-server avviato ma CUDA non rilevato nel log — inferenza probabilmente su CPU "
                "(LLM_N_GPU_LAYERS=%s).\n"
                "  Esegui 'python install.py' per installare il backend GPU.",
                self._n_gpu_layers,
            )
        except OSError:
            pass

    async def stop(self) -> None:
        from config import Config

        if not Config.llm_managed:
            return

        if self.process is None:
            return

        if self.process.poll() is not None:
            logger.info("llama-server già terminato (exit code %d)", self.process.returncode)
            self.process = None
            return

        logger.info("Arresto llama-server (PID %d)...", self.process.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
            logger.info("llama-server arrestato correttamente")
        except subprocess.TimeoutExpired:
            logger.warning("Timeout SIGTERM scaduto — invio SIGKILL")
            self.process.kill()
            self.process.wait()
            logger.info("llama-server terminato forzatamente")
        self.process = None

    async def is_running(self) -> bool:
        from config import Config

        health_url = f"http://localhost:{Config.llm_port}/v1/models"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(health_url, timeout=3.0)
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


def get_llm_server_manager() -> LLMServerManager:
    global _llm_server_manager
    if _llm_server_manager is None:
        _llm_server_manager = LLMServerManager()
    return _llm_server_manager


# ── Reranker Server Manager ───────────────────────────────────────────────────

_reranker_server_manager: "RerankerServerManager | None" = None


class RerankerServerManager:
    """
    Gestisce il ciclo di vita del server re-ranker.

    Avvia llama-server come subprocess dedicato su RERANKER_PORT con il modello
    GGUF del re-ranker. Usa il binario nativo (Windows .exe o Linux CUDA binary)
    se disponibile — nessuna dipendenza da llama-cpp-python CUDA.

    Se il binario nativo non è presente, logga un warning e il re-ranker rimane
    disabilitato (vault_search fallback su ranking vettoriale).
    """

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    async def start(self) -> None:
        from config import Config

        if not Config.reranker_enabled:
            return

        native_bin = _WIN_LLAMA_BIN if sys.platform == "win32" else _LINUX_LLAMA_BIN
        use_native = native_bin.exists()

        if not use_native:
            if sys.platform == "win32":
                logger.warning(
                    "Binario llama-server non trovato (%s) — re-ranker server disabilitato.\n"
                    "  Esegui 'python install.py' per scaricarlo.",
                    native_bin,
                )
                return
            # Linux/WSL: fallback al server Python llama-cpp-python (stesso wheel CUDA dell'LLM)
            logger.info("Binario nativo assente — re-ranker userà llama-cpp-python")

        model_path = Path(Config.reranker_model_dir).resolve() / Config.reranker_model_file
        if not model_path.exists():
            raise RuntimeError(
                f"File modello re-ranker non trovato: {model_path}\n"
                "  Verifica RERANKER_MODEL_REPO e RERANKER_MODEL_FILE nel .env\n"
                "  oppure esegui: python install.py"
            )

        port = Config.reranker_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.settimeout(0.5)
            if _s.connect_ex(("127.0.0.1", int(port))) == 0:
                raise RuntimeError(
                    f"La porta {port} (RERANKER_PORT) è già occupata da un altro processo. "
                    "Cambia RERANKER_PORT nel .env o fermalo prima di avviare Vesper."
                )

        gpu_idx = 0
        try:
            gpu_idx = _get_non_llm_gpu_index()
        except Exception:
            pass

        if use_native:
            cmd = [
                str(native_bin),
                "--model", str(model_path),
                "--n-gpu-layers", "-1",
                "--ctx-size", "4096",
                "--batch-size", "512",
                "--host", "127.0.0.1",
                "--port", port,
                "--main-gpu", str(gpu_idx),
            ]
        else:
            cmd = [
                sys.executable, "-m", "llama_cpp.server",
                "--model", str(model_path),
                "--n_gpu_layers", "-1",
                "--n_ctx", "4096",
                "--host", "127.0.0.1",
                "--port", port,
            ]

        env = os.environ.copy()
        if sys.platform != "win32":
            _build_ld_library_path(env, using_native_bin=use_native)

        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "reranker_server.log"

        with open(log_file, "a") as lf:
            self.process = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)

        logger.info(
            "Re-ranker server avviato (PID %d) — porta %s — log in %s",
            self.process.pid, port, log_file,
        )

        health_url = f"http://127.0.0.1:{port}/v1/models"
        await _wait_for_server_ready(
            health_url, self.process, log_file, timeout=120, label="re-ranker server"
        )

    async def stop(self) -> None:
        if self.process is None:
            return

        if self.process.poll() is not None:
            logger.info("Re-ranker server già terminato (exit code %d)", self.process.returncode)
            self.process = None
            return

        logger.info("Arresto re-ranker server (PID %d)...", self.process.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
            logger.info("Re-ranker server arrestato correttamente")
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            logger.info("Re-ranker server terminato forzatamente")
        self.process = None


def get_reranker_server_manager() -> RerankerServerManager:
    global _reranker_server_manager
    if _reranker_server_manager is None:
        _reranker_server_manager = RerankerServerManager()
    return _reranker_server_manager
