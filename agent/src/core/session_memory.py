"""
session_memory — Memoria di sessione persistente (call / store / forget).

Salva capsule strutturate per ogni turno completato con successo, permettendo
all'Orchestrator e al Generator di accedere a contenuti recuperati in turni
precedenti. Interface-agnostic: ogni interfaccia passa session_id: str.

Formato capsule:
    Richiesta: {request}
    Fonti vault: wiki/file.md (0.92), ...
    Fonti web: https://...
    Risposta: {primi 200 char dell'output Generator}

Embedding: solo il testo della richiesta (più stabile semanticamente per il recall).
"""
import json
import logging
import time

logger = logging.getLogger(__name__)

_SESSION_COLLECTION = "session_memory"
_collection = None
_embed_override = None  # può essere impostato nei test per bypassare la dipendenza reale


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        from src.core.rag.indexer import get_chroma_collection
    except ImportError:
        from core.rag.indexer import get_chroma_collection  # type: ignore[no-redef]
    _collection = get_chroma_collection(_SESSION_COLLECTION)
    return _collection


def _get_embed():
    if _embed_override is not None:
        return _embed_override
    try:
        from src.core.rag.indexer import _embed
    except ImportError:
        from core.rag.indexer import _embed  # type: ignore[no-redef]
    return _embed


async def store(session_id: str, context) -> None:
    """Salva una capsule per il turno corrente. Nessun LLM — estrazione algoritmica."""
    request = context.results.get("request", "")
    if not request:
        return

    # Salva solo se GENERATE è andato a buon fine
    generate_res = context.results.get("GENERATE", {})
    if not (isinstance(generate_res, dict) and generate_res.get("success")):
        return

    vault_sources: list[str] = []
    web_sources: list[str] = []
    for tc in context.tools_called:
        tool = tc.get("tool", "")
        if tool == "vault_search":
            for r in (tc.get("output") or []):
                p = r.get("path")
                score = r.get("score", 0.0)
                if p:
                    vault_sources.append(f"{p} ({score:.2f})")
        elif tool == "web_search":
            for r in (tc.get("output") or []):
                url = r.get("url")
                if url:
                    web_sources.append(url)

    lines = [f"Richiesta: {request}"]
    if vault_sources:
        lines.append(f"Fonti vault: {', '.join(vault_sources[:5])}")
    if web_sources:
        lines.append(f"Fonti web: {', '.join(web_sources[:3])}")

    response_text = str(generate_res.get("output", ""))
    if response_text:
        excerpt = response_text[:200]
        if len(response_text) > 200:
            excerpt += "..."
        lines.append(f"Risposta: {excerpt}")

    capsule_text = "\n".join(lines)

    try:
        embed_fn = _get_embed()
        embedding = embed_fn([request])[0]
        collection = _get_collection()
        collection.upsert(
            ids=[context.request_id],
            embeddings=[embedding],
            documents=[capsule_text],
            metadatas=[{
                "session_id": session_id,
                "request_id": context.request_id,
                "timestamp": time.time(),
                "actions_executed": json.dumps(context.actions_executed),
            }],
        )
        logger.debug("session_memory: capsule salvata session=%s req=%s", session_id, context.request_id)
    except Exception as exc:
        logger.warning("session_memory: store fallito — %s", exc)


async def query(session_id: str, query_text: str, top_k: int = 3) -> list[dict]:
    """Recupera capsule rilevanti per la sessione e la query corrente."""
    try:
        embed_fn = _get_embed()
        collection = _get_collection()

        total = collection.count()
        if total == 0:
            return []

        raw = collection.query(
            query_embeddings=[embed_fn([query_text])[0]],
            n_results=min(top_k, total),
            where={"session_id": {"$eq": session_id}},
            include=["documents", "metadatas", "distances"],
        )

        if not raw.get("ids") or not raw["ids"][0]:
            return []

        results: list[dict] = []
        for capsule_id, content, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append({
                "content": content,
                "score": round(max(0.0, 1.0 - dist), 4),
                "source": "session_memory",
                "capsule_id": capsule_id,
            })
        return results
    except Exception as exc:
        logger.warning("session_memory: query fallita — %s", exc)
        return []


async def count(session_id: str) -> int:
    """Restituisce il numero di capsule per session_id."""
    try:
        collection = _get_collection()
        res = collection.get(where={"session_id": {"$eq": session_id}}, include=[])
        return len(res.get("ids", []))
    except Exception as exc:
        logger.warning("session_memory: count fallito — %s", exc)
        return 0


async def forget(session_id: str, before_timestamp: float | None = None) -> int:
    """Elimina capsule per session_id. Con before_timestamp filtra per data."""
    try:
        collection = _get_collection()

        if before_timestamp is not None:
            where = {
                "$and": [
                    {"session_id": {"$eq": session_id}},
                    {"timestamp": {"$lt": before_timestamp}},
                ]
            }
        else:
            where = {"session_id": {"$eq": session_id}}

        res = collection.get(where=where, include=[])
        ids = res.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            logger.info("session_memory: eliminate %d capsule session=%s", len(ids), session_id)
        return len(ids)
    except Exception as exc:
        logger.warning("session_memory: forget fallito — %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

    async def _run() -> None:
        # Mock ChromaDB e embed per test senza dipendenze reali
        import src.core.session_memory as sm

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.upsert = MagicMock()
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_collection.get.return_value = {"ids": []}

        sm._collection = mock_collection
        sm._embed_override = lambda texts: [[0.1] * 4 for _ in texts]

        # Test store: turno senza GENERATE → nessun upsert
        ctx_no_gen = MagicMock()
        ctx_no_gen.results = {"request": "test"}
        ctx_no_gen.tools_called = []
        ctx_no_gen.actions_executed = []
        ctx_no_gen.request_id = "req-1"
        await sm.store("session-1", ctx_no_gen)
        check("store: nessun upsert senza GENERATE success", not mock_collection.upsert.called)

        # Test store: turno con GENERATE success → upsert
        ctx_ok = MagicMock()
        ctx_ok.results = {
            "request": "Come funziona asyncio?",
            "GENERATE": {"success": True, "output": "Asyncio è una libreria Python per I/O asincrono."},
        }
        ctx_ok.tools_called = [
            {"tool": "vault_search", "output": [{"path": "wiki/asyncio.md", "score": 0.91}]},
        ]
        ctx_ok.actions_executed = ["RETRIEVE", "GENERATE"]
        ctx_ok.request_id = "req-2"
        await sm.store("session-1", ctx_ok)
        check("store: upsert chiamato con GENERATE success", mock_collection.upsert.called)

        # Verifica struttura capsule
        call_args = mock_collection.upsert.call_args
        docs = call_args.kwargs.get("documents", []) if call_args else []
        capsule_text = docs[0] if docs else ""
        check("capsule contiene richiesta", "Come funziona asyncio?" in capsule_text)
        check("capsule contiene fonti vault", "wiki/asyncio.md" in capsule_text)
        check("capsule contiene risposta", "Asyncio" in capsule_text)

        # Test query: collezione vuota → lista vuota
        result = await sm.query("session-1", "asyncio", top_k=3)
        check("query: lista vuota su collezione vuota", result == [])

        # Test query: con risultati
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "ids": [["req-2"]],
            "documents": [["Richiesta: Come funziona asyncio?\nFonti vault: wiki/asyncio.md (0.91)"]],
            "metadatas": [[{"session_id": "session-1", "request_id": "req-2", "timestamp": 1000.0, "actions_executed": "[]"}]],
            "distances": [[0.08]],
        }
        results = await sm.query("session-1", "asyncio", top_k=3)
        check("query: restituisce risultati", len(results) == 1)
        check("query: score calcolato correttamente", results[0]["score"] == round(1.0 - 0.08, 4))
        check("query: source è session_memory", results[0]["source"] == "session_memory")

        # Test forget
        mock_collection.get.return_value = {"ids": ["req-2"]}
        n = await sm.forget("session-1")
        check("forget: elimina capsule", n == 1)
        check("forget: delete chiamato", mock_collection.delete.called)

        # Ripristina stato
        sm._embed_override = None

    asyncio.run(_run())
    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
