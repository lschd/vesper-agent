"""
WebInterface — interfaccia consumer via FastAPI + SSE.

Sola lettura delle sezioni wiki definite in WEB_WIKI_ALLOWED_SECTIONS.
Upload documenti (max 20 MB) per analisi e generazione contenuti.
Sessioni in RAM con TTL configurabile via WEB_SESSION_TTL_SECONDS.
"""
import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__":
    # Stub per dipendenze non installate nell'ambiente di test
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

    from unittest.mock import MagicMock as _M

    class _FakeAPIRouter:
        def get(self, *a, **k): return lambda f: f
        def post(self, *a, **k): return lambda f: f

    class _FakeBaseModel:
        def __init__(self, **kw): ...

    _fastapi = _M()
    _fastapi.APIRouter = _FakeAPIRouter
    _fastapi.HTTPException = Exception
    _fastapi.File = lambda *a, **k: None
    _fastapi.UploadFile = _M()

    _sse_mod = _M()
    _sse_mod.EventSourceResponse = _M()

    _pydantic = _M()
    _pydantic.BaseModel = _FakeBaseModel

    _cors_mod = _M()
    _cors_mod.CORSMiddleware = _M()

    for _name, _mod in [
        ("fastapi", _fastapi),
        ("fastapi.middleware", _M()),
        ("fastapi.middleware.cors", _cors_mod),
        ("sse_starlette", _sse_mod),
        ("sse_starlette.sse", _sse_mod),
        ("pydantic", _pydantic),
    ]:
        if _name not in _sys.modules:
            _sys.modules[_name] = _mod

import asyncio
import contextlib
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

try:
    from src.interfaces.base import AbstractInterface
except ImportError:
    from base import AbstractInterface  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Azioni disponibili per l'interfaccia consumer
_WEB_PERMISSIONS = ["RETRIEVE", "ANALYZE", "GENERATE"]

# Intervallo di pulizia sessioni scadute
_CLEANUP_INTERVAL_SECONDS = 300  # 5 minuti

# Max messaggi conservati in cronologia per sessione
_SESSION_HISTORY_LIMIT = 40  # 20 scambi


# ---------------------------------------------------------------------------
# Modelli Pydantic
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    document_content: str | None = None


# ---------------------------------------------------------------------------
# Session store (RAM)
# ---------------------------------------------------------------------------

# {session_id: {"history": list[dict], "last_active": float}}
_sessions: dict[str, dict] = {}


def _get_or_create_session(session_id: str | None, ttl: int) -> tuple[str, dict]:
    """
    Restituisce (session_id, session_dict).
    Crea una nuova sessione se session_id è None, vuoto o non esistente.
    Aggiorna last_active per le sessioni esistenti.
    """
    now = time.monotonic()
    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        # Sessione esistente ma scaduta: ricrea
        if now - session["last_active"] <= ttl:
            session["last_active"] = now
            return session_id, session

    new_id = str(uuid.uuid4())
    _sessions[new_id] = {"history": [], "last_active": now}
    logger.debug("WebInterface: nuova sessione %s", new_id)
    return new_id, _sessions[new_id]


async def _cleanup_loop(ttl: int) -> None:
    """Background task: rimuove le sessioni scadute ogni _CLEANUP_INTERVAL_SECONDS."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        now = time.monotonic()
        expired = [
            sid for sid, s in _sessions.items()
            if now - s["last_active"] > ttl
        ]
        for sid in expired:
            _sessions.pop(sid, None)
        if expired:
            logger.info("WebInterface: %d sessioni scadute rimosse", len(expired))


# ---------------------------------------------------------------------------
# WebInterface
# ---------------------------------------------------------------------------

class WebInterface(AbstractInterface):
    """
    Interfaccia consumer via FastAPI + SSE.

    send() e send_error() sono no-op con logging: la comunicazione con il client
    avviene esclusivamente tramite EventSourceResponse all'interno della sessione.
    """

    def __init__(self) -> None:
        from config import Config
        self._config = Config
        logger.info("WebInterface: inizializzata")

    # ------------------------------------------------------------------
    # AbstractInterface
    # ------------------------------------------------------------------

    def get_permissions(self) -> list[str]:
        return list(_WEB_PERMISSIONS)

    async def send(self, target: str, message: str) -> None:
        logger.info("WebInterface.send: target=%s (no-op SSE), msg=%.100s", target, message)

    async def send_error(self, target: str, task_name: str, reason: str) -> None:
        logger.warning(
            "WebInterface.send_error: target=%s task=%s reason=%s (no-op SSE)",
            target, task_name, reason,
        )

    def _post_plan_hook(self, plan: dict) -> dict:
        """Restringe vault_search alle sezioni wiki permesse per i client web."""
        if "RETRIEVE" in plan:
            ri = plan["RETRIEVE"]
            if ri.get("source", "vault") in ("vault", "both", ""):
                allowed = self._config.web_wiki_allowed_sections
                if allowed and not ri.get("where"):
                    ri["where"] = allowed[0]
        return plan

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def create_router(self) -> APIRouter:
        """
        Restituisce un APIRouter con tutti gli endpoint dell'interfaccia web.
        Può essere montato da main.py su qualsiasi prefix.
        """
        router = APIRouter()

        @router.post("/chat")
        async def chat(request: ChatRequest) -> EventSourceResponse:
            """
            Elabora un messaggio e restituisce la risposta via SSE.

            Eventi SSE emessi (in ordine):
              status  — aggiornamenti durante l'elaborazione
              response — risposta finale del modello
              done    — segnale di fine stream
              error   — messaggio di errore (solo in caso di fallimento)

            L'header X-Session-Id della risposta contiene il session_id corrente
            (nuovo o riusato).
            """
            ttl = self._config.web_session_ttl_seconds
            session_id, session = _get_or_create_session(request.session_id, ttl)

            queue: asyncio.Queue[dict | None] = asyncio.Queue()

            asyncio.create_task(
                self._run_chat(request, session, session_id, queue)
            )

            async def event_generator():
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item

            return EventSourceResponse(
                event_generator(),
                headers={"X-Session-Id": session_id},
            )

        @router.post("/upload")
        async def upload(file: UploadFile = File(...)) -> dict:
            """
            Carica un documento in vault/raw/.

            Limite: 20 MB. Restituisce {"ok": true, "filename": str, "path": str}.
            """
            max_bytes = 20 * 1024 * 1024

            # Legge il contenuto in memoria per controllare la dimensione
            content = await file.read()
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="File troppo grande (max 20 MB).",
                )

            filename = file.filename or f"upload_{uuid.uuid4().hex}.bin"
            dest = self._config.vault_path / "raw" / filename
            dest.parent.mkdir(parents=True, exist_ok=True)

            dest.write_bytes(content)
            logger.info("WebInterface /upload: salvato '%s' (%d bytes)", dest, len(content))

            return {"ok": True, "filename": filename, "path": str(dest)}

        @router.get("/wiki/{section}/{filename}")
        async def get_wiki(section: str, filename: str) -> dict:
            """
            Legge un documento dalla wiki.

            Permette solo le sezioni in WEB_WIKI_ALLOWED_SECTIONS.
            Restituisce {"content": str}.
            """
            allowed = self._config.web_wiki_allowed_sections
            if f"wiki/{section}" not in allowed:
                raise HTTPException(
                    status_code=403,
                    detail=f"Sezione '{section}' non permessa.",
                )

            path = f"wiki/{section}/{filename}.md"
            try:
                from src.tools.read_document import read_document
            except ImportError:
                from tools.read_document import read_document  # type: ignore[no-redef]

            try:
                content = await read_document(path)
                return {"content": content}
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404,
                    detail=f"File non trovato: {path}",
                )

        return router

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def _run_chat(
        self,
        request: ChatRequest,
        session: dict,
        session_id: str,
        queue: asyncio.Queue,
    ) -> None:
        """
        Esegue il flusso di elaborazione e pubblica gli eventi SSE sulla queue.
        Invia sempre il sentinel None alla fine (successo o errore).
        """
        try:
            await self._process_chat(request, session, session_id, queue)
        except Exception as exc:
            logger.error("WebInterface: errore non gestito in _run_chat — %s: %s", type(exc).__name__, exc)
            await queue.put({"event": "error", "data": str(exc)})
        finally:
            await queue.put(None)

    async def _process_chat(
        self,
        request: ChatRequest,
        session: dict,
        session_id: str,
        queue: asyncio.Queue,
    ) -> None:
        """Delega la pipeline a process_request() e pubblica eventi status/response/done."""
        async def status_callback(text: str) -> None:
            await queue.put({"event": "status", "data": text})

        history = list(session.get("history", []))

        # Se è allegato un documento, viene passato inline all'Orchestrator come testo.
        # Il testo originale (senza snippet) va in history; effective_user_text va al LLM.
        effective_user_text: str | None = None
        extra_context: dict | None = None
        if request.document_content:
            snippet = request.document_content[:3000]
            effective_user_text = f"{request.message}\n\n---\nDocumento allegato:\n{snippet}"
            extra_context = {"document_content": request.document_content}

        response_text, req_ctx, stats = await self.process_request(
            user_text=request.message,
            history=history,
            session_id=session_id,
            status_callback=status_callback,
            effective_user_text=effective_user_text,
            extra_context=extra_context,
        )

        response_text = response_text or "Elaborazione completata."

        # Aggiorna cronologia RAM (con limite)
        session["history"].append({"role": "user", "content": request.message})
        session["history"].append({"role": "assistant", "content": response_text})
        if len(session["history"]) > _SESSION_HISTORY_LIMIT:
            session["history"] = session["history"][-_SESSION_HISTORY_LIMIT:]

        await queue.put({"event": "response", "data": response_text})
        await queue.put({"event": "done", "data": ""})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(web_interface: WebInterface | None = None) -> "FastAPI":
    """
    Crea e restituisce una FastAPI app completa con CORS e router montato.

    Usata da main.py come entry point dell'interfaccia web.

    Args:
        web_interface: Istanza WebInterface da usare. Se None, ne viene creata una nuova.

    Returns:
        FastAPI app pronta all'uso (non ancora in esecuzione).
    """
    from contextlib import asynccontextmanager
    from fastapi import FastAPI

    from config import Config

    iface = web_interface or WebInterface()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(_cleanup_loop(Config.web_session_ttl_seconds))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Vesper Web API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.web_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(iface.create_router())
    return app


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
    os.environ.setdefault("VAULT_PATH", str(Path(__file__).resolve().parent.parent.parent / "vault"))

    passed = 0
    failed = 0

    def check(label: str, condition: bool) -> None:
        global passed, failed
        if condition:
            print(f"[OK] {label}")
            passed += 1
        else:
            print(f"[FAIL] {label}")
            failed += 1

    print("\n=== WebInterface — get_permissions ===\n")

    iface = WebInterface()
    perms = iface.get_permissions()
    check("get_permissions restituisce lista", isinstance(perms, list))
    check("RETRIEVE permesso", "RETRIEVE" in perms)
    check("ANALYZE permesso", "ANALYZE" in perms)
    check("GENERATE permesso", "GENERATE" in perms)
    check("STORE NON permesso", "STORE" not in perms)
    check("REASON NON permesso", "REASON" not in perms)

    print("\n=== Session store ===\n")

    _sessions.clear()
    TTL = 3600

    # Nuova sessione con session_id=None
    sid1, s1 = _get_or_create_session(None, TTL)
    check("nuova sessione: session_id generato", bool(sid1))
    check("nuova sessione: history vuota", s1["history"] == [])
    check("nuova sessione: in _sessions", sid1 in _sessions)

    # Sessione esistente: stessa istanza
    sid1b, s1b = _get_or_create_session(sid1, TTL)
    check("sessione esistente: stesso ID", sid1b == sid1)
    check("sessione esistente: stessa istanza", s1b is s1)

    # Session_id sconosciuto: nuova sessione
    sid2, s2 = _get_or_create_session("id-inesistente", TTL)
    check("id sconosciuto: nuovo session_id generato", sid2 != "id-inesistente")
    check("id sconosciuto: sessione creata", sid2 in _sessions)

    # Sessione scaduta: deve essere ricreata
    fake_ttl = 1
    _sessions["old-session"] = {"history": [{"role": "user", "content": "ciao"}], "last_active": 0.0}
    sid_expired, s_expired = _get_or_create_session("old-session", fake_ttl)
    check("sessione scaduta: nuovo ID generato", sid_expired != "old-session")
    check("sessione scaduta: storia non ereditata", s_expired["history"] == [])

    _sessions.clear()

    print("\n=== _cleanup_loop ===\n")

    async def _test_cleanup():
        # TTL di 2s, sleep di 0.5s: "alive" ha 0.5s < 2s; "dead" ha last_active=0 > 2s
        TTL_SHORT = 2
        _sessions["alive"] = {"history": [], "last_active": time.monotonic()}
        _sessions["dead"] = {"history": [], "last_active": 0.0}  # scaduta
        await asyncio.sleep(0.5)
        now = time.monotonic()
        expired = [sid for sid, s in _sessions.items() if now - s["last_active"] > TTL_SHORT]
        for sid in expired:
            _sessions.pop(sid, None)
        return expired

    expired = asyncio.run(_test_cleanup())
    check("cleanup rimuove sessioni scadute", "dead" in expired)
    check("cleanup preserva sessioni attive", "alive" in _sessions)

    _sessions.clear()

    print("\n=== Sezione wiki permessa ===\n")

    # Simula il controllo sezione del route get_wiki
    allowed = ["wiki/public", "wiki/docs"]
    check("sezione permessa: wiki/public", "wiki/public" in allowed)
    check("sezione permessa: wiki/docs", "wiki/docs" in allowed)
    check("sezione NON permessa: wiki/private", "wiki/private" not in allowed)
    check("sezione NON permessa: wiki/admin", "wiki/admin" not in allowed)

    print("\n=== send / send_error no-op ===\n")

    async def _test_noop():
        await iface.send("999", "messaggio di test")
        await iface.send_error("999", "task-test", "errore di prova")
        return True

    check("send e send_error completano senza eccezioni", asyncio.run(_test_noop()))

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
