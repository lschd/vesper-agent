"""Configurazione centralizzata di Vesper. Carica le variabili da .env all'importazione."""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).parent / ".env"


def _read_raw_path(key: str) -> str:
    """
    Legge un path dal .env senza interpretare escape sequences.

    python-dotenv processa i valori tra virgolette come stringhe Python:
    "C:\\Users\\nome\\agent" → C:\\Users\\nome\\<chr7>gent  (\\a = bell).
    Questa funzione bypassa quel comportamento e restituisce il testo letterale.
    Gestisce anche la conversione WSL: /mnt/c/... → C:/...
    """
    if not _ENV_FILE.exists():
        return ""
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() != key:
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        # Converti path WSL /mnt/X/... → X:/...
        if sys.platform == "win32" and len(v) >= 7 and v[:5] == "/mnt/" and v[6:7] == "/":
            v = f"{v[5].upper()}:{v[6:]}"
        return v
    return ""


# Su Windows imposta i path prima di load_dotenv per evitare l'interpretazione
# dei backslash come escape Python. load_dotenv(override=False) li lascerà intatti.
if sys.platform == "win32":
    for _path_key in ("VAULT_PATH",):
        _raw = _read_raw_path(_path_key)
        if _raw:
            os.environ[_path_key] = _raw

load_dotenv(override=False)


def _require(key: str) -> str:
    """Legge una variabile obbligatoria o solleva RuntimeError con messaggio chiaro."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Variabile d'ambiente obbligatoria mancante: {key}\n"
            f"Controlla il file .env (vedi .env.example per riferimento)."
        )
    return value


def _parse_list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass
class _Config:
    # LLM — Client
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "local-model"))

    # LLM — Server
    llm_host: str = field(default_factory=lambda: os.getenv("LLM_HOST", "0.0.0.0"))
    llm_port: str = field(default_factory=lambda: os.getenv("LLM_PORT", "8000"))

    # LLM — Paths
    llm_model_repo: str = field(default_factory=lambda: os.getenv("LLM_MODEL_REPO", "unsloth/Qwen3.6-27B-GGUF"))
    llm_model_file: str = field(default_factory=lambda: os.getenv("LLM_MODEL_FILE", "Qwen3.6-27B-Q6_K.gguf"))

    @property
    def llm_model_dir(self) -> str:
        return str(Path("./data/models") / self.llm_model_repo)

    # LLM — Performance (passati a llama-server quando LLM_MANAGED=true)
    llm_n_gpu_layers: int = field(default_factory=lambda: int(os.getenv("LLM_N_GPU_LAYERS", "-1")))
    llm_n_ctx: int | None = field(
        default_factory=lambda: int(os.getenv("LLM_N_CTX")) if os.getenv("LLM_N_CTX", "").strip() else None
    )
    llm_multi_gpu: bool = field(default_factory=lambda: os.getenv("LLM_MULTI_GPU", "true").lower() == "true")

    # Gestione server LLM
    llm_managed: bool = field(default_factory=lambda: os.getenv("LLM_MANAGED", "true").lower() == "true")

    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: _require("TELEGRAM_BOT_TOKEN"))
    telegram_admin_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_ADMIN_CHAT_ID", ""))

    # Vault
    vault_path: Path = field(default_factory=lambda: Path(_require("VAULT_PATH")))

    # RAG
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./data/chroma"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "300")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    vault_watch_debounce_seconds: int = field(default_factory=lambda: int(os.getenv("VAULT_WATCH_DEBOUNCE_SECONDS", "5")))

    # Logs
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "./data/logs")))

    # Memoria conversazioni
    conversations_db_path: str = field(
        default_factory=lambda: os.getenv("CONVERSATIONS_DB_PATH", "./data/conversations.db")
    )

    # Knowledge graph
    graph_db_path: str = field(default_factory=lambda: os.getenv("GRAPH_DB_PATH", "./data/graph.db"))

    # Re-ranker
    reranker_enabled: bool = field(default_factory=lambda: os.getenv("RERANKER_ENABLED", "true").lower() == "true")
    reranker_model_repo: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL_REPO", "Mungert/Qwen3-Reranker-4B-GGUF"))
    reranker_model_file: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL_FILE", "Qwen3-Reranker-4B-Q4_K_M.gguf"))
    reranker_top_k: int = field(default_factory=lambda: int(os.getenv("RERANKER_TOP_K", "5")))
    reranker_port: str = field(default_factory=lambda: os.getenv("RERANKER_PORT", "8001"))

    # Web
    web_host: str = field(default_factory=lambda: os.getenv("WEB_HOST", "0.0.0.0"))
    web_port: int = field(default_factory=lambda: int(os.getenv("WEB_PORT", "8080")))
    web_wiki_allowed_sections: list[str] = field(default_factory=lambda: _parse_list("WEB_WIKI_ALLOWED_SECTIONS", "wiki/public"))
    web_session_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("WEB_SESSION_TTL_SECONDS", "3600")))
    web_cors_origins: list[str] = field(
        default_factory=lambda: _parse_list(
            "WEB_CORS_ORIGINS",
            "http://localhost,http://localhost:3000,http://127.0.0.1,http://127.0.0.1:3000",
        )
    )

    @property
    def embedding_model_dir(self) -> str:
        return str(Path("./data/models") / self.embedding_model)

    @property
    def reranker_model_dir(self) -> str:
        return str(Path("./data/models") / self.reranker_model_repo)


Config = _Config()
