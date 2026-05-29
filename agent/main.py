#!/usr/bin/env python3
"""Punto di ingresso principale. Avvia bot Telegram e server web in parallelo."""

import sys
import os
import subprocess
from pathlib import Path

# ── Auto-restart nel venv (solo stdlib, deve girare senza dipendenze) ─────────
_MIN_PY = (3, 10)
_root = Path(__file__).resolve().parent
_venv_py = (
    _root / ".venv" / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else _root / ".venv" / "bin" / "python"
)
if not _venv_py.exists():
    sys.exit("venv non trovato. Esegui 'python install.py' per installare il progetto.")
if Path(sys.prefix).resolve() != (_root / ".venv").resolve():
    # Su POSIX os.execv sostituisce il processo in-place (stesso PID → shell aspetta).
    # Su Windows os.execv avvia un nuovo processo ed esce subito, restituendo il prompt
    # mentre il venv Python gira in background. subprocess.run tiene vivo il padre.
    if sys.platform == "win32":
        try:
            sys.exit(subprocess.run([str(_venv_py)] + sys.argv).returncode)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
if sys.version_info[:2] < _MIN_PY:
    sys.exit(
        f"Python {sys.version.split()[0]} non supportato "
        f"(minimo richiesto: {_MIN_PY[0]}.{_MIN_PY[1]}).\n"
        f"  Ricrea il venv: python install.py"
    )
# ─────────────────────────────────────────────────────────────────────────────

import logging
import asyncio
import signal
import time
from datetime import datetime

VERSION = (Path(__file__).resolve().parent / "VERSION").read_text(encoding="utf-8").strip()

def print_banner():
    GREEN = "\033[38;2;74;222;128m"
    WHITE = "\033[97m"
    DIM   = "\033[2m"
    RESET = "\033[0m"
    
    print()
    print(f"{GREEN} ✦{RESET}  {WHITE}VESPER{RESET}")
    print(f"{DIM} ✧  Evening star. Local by design.{RESET}")
    print(f"{DIM}    lschd ~ v{VERSION}{RESET}")
    print()

print_banner()

from config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
class ColoredFormatter(logging.Formatter):
    RESET = "\033[0m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"

    COLORS = {
        "DEBUG": GRAY,      # grigio
        "INFO": "\033[92m",       # verde
        "WARNING": "\033[93m",    # giallo
        "ERROR": "\033[91m",      # rosso
        "CRITICAL": "\033[95m",   # magenta
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, self.RESET)

        # genera timestamp
        asctime = self.formatTime(record, self.datefmt)

        # salva originali
        original_name = record.name
        original_levelname = record.levelname

        # colora
        record.levelname = (
            f"{level_color}{record.levelname}{self.RESET}"
        )

        record.name = (
            f"{self.BLUE}{record.name}{self.RESET if record.levelname != "DEBUG" else self.GRAY}"
        )

        # formatta
        formatted = super().format(record)

        # colora timestamp
        formatted = formatted.replace(
            asctime,
            f"{self.GRAY}{asctime}{self.RESET}",
            1
        )

        # restore
        record.name = original_name
        record.levelname = original_levelname

        return formatted

# LOG_FORMAT = "%(asctime)s %(levelname)s: %(name)8s → %(message)s"
LOG_FORMAT = "%(asctime)s %(levelname)s: %(name)8s → %(message)s"

formatter = ColoredFormatter(
    fmt=LOG_FORMAT,
    datefmt="%m-%d %H:%M:%S",
)

# Handler console di default (StreamHandler colorato).
# Se la TUI viene inizializzata più avanti, sarà sostituito con RichHandler.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

_LOG_KEEP = 10
_log_dir = Config.log_dir
_log_dir.mkdir(parents=True, exist_ok=True)
_existing_logs = sorted(_log_dir.glob("vesper_*.log"))
for _old in _existing_logs[: max(0, len(_existing_logs) - (_LOG_KEEP - 1))]:
    _old.unlink(missing_ok=True)
_session_log = _log_dir / f"vesper_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

file_handler = logging.FileHandler(
    _session_log,
    encoding="utf-8"
)

# niente colori nel file log, altrimenti ti ritrovi gli escape ANSI nel file
file_handler.setFormatter(logging.Formatter(
    fmt=LOG_FORMAT,
    datefmt="%m-%d %H:%M:%S",
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        console_handler,
        file_handler,
    ],
)

logger = logging.getLogger("main")

logging.getLogger("main").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)



# ── TUI (status bar con stellina) ────────────────────────────────────────────
# Colore verde Vesper: RGB(74, 222, 128) — coerente con il banner.
# Specificato qui per avere un unico punto di riferimento visivo.
_TUI_COLOR = "rgb(74,222,128)"
_tui = None
if sys.stdout.isatty():
    try:
        from src.interfaces.tui import VesperTUI, _register_active_tui
        _tui = VesperTUI(color=_TUI_COLOR)
        _register_active_tui(_tui)  # espone set_tui_state() per base.py
    except Exception as e:
        logger.warning(e)
        
# ─────────────────────────────────────────────────────────────────────────────

# Se la TUI è attiva, sostituisce il console_handler con RichHandler sullo
# stesso Console del Live display: rich gestisce automaticamente la pausa/ripresa
# del display ad ogni log, tenendo la stellina fissa in basso mentre i log scorrono.
if _tui is not None and _tui.console is not None:
    from rich.logging import RichHandler
    _rich_handler = RichHandler(
        console=_tui.console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format="%m-%d %H:%M:%S",
    )
    _rich_handler.setFormatter(logging.Formatter("%(name)8s → %(message)s"))
    _root_logger = logging.getLogger()
    _root_logger.removeHandler(console_handler)
    _root_logger.addHandler(_rich_handler)
    console_handler = _rich_handler
    del _rich_handler, _root_logger


async def _check_llm_server(base_url: str) -> None:
    """Verifica che llama-server sia raggiungibile; interrompe l'avvio in caso contrario."""
    import httpx
    from pathlib import Path as _Path

    models_url = base_url.rstrip("/").removesuffix("/v1") + "/v1/models"
    try:
        async with httpx.AsyncClient() as client:
            await client.get(models_url, timeout=2.0)
        logger.info("llama-server raggiungibile: %s", models_url)
        return
    except Exception:
        pass

    # Server non raggiungibile: costruisci un messaggio utile in base allo stato locale.
    model_path = _Path(Config.llm_model_dir).resolve() / Config.llm_model_file
    if not model_path.exists():
        logger.error(
            "Server LLM non raggiungibile su %s e modello locale non trovato.\n"
            "  Opzione A — server gestito da Vesper: imposta LLM_MANAGED=true nel .env\n"
            "              e riesegui 'python install.py' per scaricare il modello.\n"
            "  Opzione B — server esterno: avvia il tuo llama-server e verifica\n"
            "              che LLM_BASE_URL punti all'endpoint corretto nel .env.",
            models_url,
        )
    else:
        logger.error(
            "Server LLM non raggiungibile su %s.\n"
            "  Modello locale trovato in: %s\n"
            "  Opzione A — avvia llama-server manualmente.\n"
            "  Opzione B — imposta LLM_MANAGED=true nel .env per avvio automatico.",
            models_url, model_path,
        )
    sys.exit(1)


async def _main() -> None:
    import time
    _startup_t0 = time.monotonic()

    # ── TUI: avvia il Live display e imposta lo stato di caricamento ──────────
    if _tui is not None:
        _tui.start()
        _tui.set_state("loading")
    # ─────────────────────────────────────────────────────────────────────────

    from src.core.llm.server_manager import get_llm_server_manager, get_reranker_server_manager

    llm_mgr = get_llm_server_manager()
    reranker_mgr = get_reranker_server_manager()

    # ── 0. Controlli rapidi (prima di caricare qualsiasi libreria pesante) ────
    if not Config.vault_path.exists():
        logger.error(
            "Vault non trovato: %s\n"
            "  Verifica VAULT_PATH nel file .env.",
            Config.vault_path,
        )
        sys.exit(1)

    if Config.llm_managed:
        try:
            llm_mgr.preflight()  # verifica modello, porta, campiona VRAM mentre è libera
        except RuntimeError as exc:
            logger.error("%s", exc)
            sys.exit(1)
    else:
        await _check_llm_server(Config.llm_base_url)
        # Server già in esecuzione: leggi subito n_ctx (non si può fare dopo l'avvio
        # perché con LLM_MANAGED=false non c'è un llm_start_task da attendere).
        from src.core.llm.client import get_llm_client as _get_client
        await _get_client().fetch_n_ctx()

    # ── 1. Import librerie pesanti ─────────────────────────────────────────────
    logger.info("Carico librerie...")

    import uvicorn
    from src.core.rag.watchdog import Watchdog
    from src.interfaces.telegram import TelegramInterface
    from src.interfaces.web import WebInterface, create_app
    from src.proactive.scheduler import VesperScheduler
    from src.core.rag.indexer import get_chroma_client, index_vault

    logger.info(
        "Config: vault=%s | web=%s:%d",
        Config.vault_path, Config.web_host, Config.web_port,
    )

    # ── 2. ChromaDB + indicizzazione (embedding occupa VRAM prima dell'LLM) ───
    logger.info("Inizializzazione ChromaDB...")
    get_chroma_client()

    logger.info("Indicizzazione vault...")
    report = await index_vault()
    logger.info(
        "Indicizzazione: %d indicizzati, %d saltati, %d errori",
        report["indexed"],
        report["skipped"],
        len(report.get("errors", [])),
    )

    # ── 3. Avvio LLM e re-ranker (VRAM residua dopo embedding; Telegram/web in parallelo) ─
    llm_start_task: asyncio.Task | None = None
    reranker_start_task: asyncio.Task | None = None
    if Config.llm_managed:
        llm_start_task = asyncio.create_task(llm_mgr.start())
        if Config.reranker_enabled:
            reranker_start_task = asyncio.create_task(reranker_mgr.start())
        await asyncio.sleep(0)  # cede il loop ai task: logga avvio e lancia i processi

    # ── 4. Watchdog ───────────────────────────────────────────────────────────
    watcher = Watchdog()
    watcher.start()

    # ── 5. Telegram ───────────────────────────────────────────────────────────
    logger.info("Inizializzazione bot Telegram...")
    tg_iface = TelegramInterface()

    # ── 6. Scheduler ──────────────────────────────────────────────────────────
    scheduler = VesperScheduler(tg_iface._app.job_queue)
    scheduler.load_tasks()

    # ── 7. FastAPI ────────────────────────────────────────────────────────────
    web_iface = WebInterface()
    fastapi_app = create_app(web_iface)

    uv_server = uvicorn.Server(
        uvicorn.Config(
            fastapi_app,
            host=Config.web_host,
            port=Config.web_port,
            log_level="warning",
            lifespan="on",
        )
    )

    # ── 8. Attendi LLM e re-ranker ────────────────────────────────────────────
    if llm_start_task is not None:
        try:
            await llm_start_task
        except RuntimeError as exc:
            logger.error("Avvio LLM fallito: %s", exc)
            sys.exit(1)

    if reranker_start_task is not None:
        try:
            await reranker_start_task
        except RuntimeError as exc:
            logger.warning("Avvio re-ranker fallito: %s — vault_search userà solo ranking vettoriale", exc)

    # ── 9. Leggi n_ctx effettivo dal server (usato per il budget dei prompt) ────
    # Con LLM_MANAGED=false è già stato fatto allo step 0 (server già attivo).
    if Config.llm_managed:
        from src.core.llm.client import get_llm_client
        await get_llm_client().fetch_n_ctx()

    logger.info("Pronto in %.1fs", time.monotonic() - _startup_t0)
    if _tui is not None:
        _tui.set_state("idle")  # startup completato — stellina statica

    # ── 10. Avvio parallelo con shutdown graceful ───────────────────────────────
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_shutdown() -> None:
        if not stop.is_set():
            logger.info("Segnale di arresto ricevuto — shutdown in corso...")
            stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_shutdown)
        except (OSError, NotImplementedError):
            signal.signal(
                sig,
                lambda _s, _f, _loop=loop, _cb=_on_shutdown: _loop.call_soon_threadsafe(_cb),
            )

    tg_app = tg_iface._app

    async def _run_telegram() -> None:
        async with tg_app:
            await tg_app.start()
            await tg_app.updater.start_polling(drop_pending_updates=True)
            logger.info("Bot Telegram in ascolto (polling)")
            await stop.wait()
            logger.info("Arresto bot Telegram...")
            await tg_app.updater.stop()
            await tg_app.stop()

    async def _run_web() -> None:
        logger.info("Server web: http://%s:%d", Config.web_host, Config.web_port)
        await uv_server.serve()

    async def _watch_stop() -> None:
        await stop.wait()
        uv_server.should_exit = True

    try:
        results = await asyncio.gather(
            _run_telegram(),
            _run_web(),
            _watch_stop(),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, (asyncio.CancelledError, SystemExit)):
                logger.error("Errore in un componente: %s", r)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop.set()
        uv_server.should_exit = True
        watcher.stop()
        await llm_mgr.stop()
        await reranker_mgr.stop()
        if _tui is not None:
            _tui.stop()
        logger.info("Sessione terminata.")


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.exit(0)
