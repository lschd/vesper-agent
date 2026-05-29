"""
vault_search — Ricerca RAG nel vault Obsidian con graph traversal sui [[link]].

Usa ChromaDB e il modello di embedding gestiti da vault_indexer (src/core/).
La rilevanza è giudicata dal vettore — nessuna LLM call aggiuntiva.
Il re-ranking opzionale è delegato a src/core/llm/reranker.py (HTTP su RERANKER_PORT).

Funzioni esposte:
    vault_search(query, ...) -> list[dict]   — ricerca RAG + graph traversal
"""
import logging
import re
from pathlib import Path

import chromadb

try:
    from src.core.rag.indexer import COLLECTION_NAME, _embed, get_chroma_client
except ImportError:
    from core.rag.indexer import COLLECTION_NAME, _embed, get_chroma_client  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Istruzione di sistema per il re-ranker cross-encoder (Qwen3-Reranker).
# Definisce come il modello deve prioritizzare i chunk del vault Obsidian.
# Non esposta come variabile d'ambiente: è un parametro di sistema, non di configurazione utente.
_RERANKER_INSTRUCTION = (
    "Prioritize documents that directly and specifically address the query with concrete information. "
    "Prefer comprehensive, well-structured notes over brief mentions. "
    "Rank higher documents that contain actionable data, definitions, or detailed explanations. "
    "When similar content exists, favor the most complete and authoritative version."
)

# Fattore di boost applicato ai risultati di vault/synthesis/ prima del re-ranking.
# Le sintesi sono già riassunti validati → meritano priorità sui documenti grezzi della wiki.
_SYNTHESIS_BOOST = 1.2


# ---------------------------------------------------------------------------
# Risoluzione link Obsidian
# ---------------------------------------------------------------------------

def _extract_link_name(link_raw: str) -> str:
    """Estrae il nome file da un link Obsidian [[Nome#Heading|Alias]]."""
    return link_raw.split("|")[0].split("#")[0].strip()


def _resolve_link(
    link_raw: str,
    collection: chromadb.Collection,
    vault_path: Path,
) -> str | None:
    """
    Risolve un [[link]] Obsidian al document ID ChromaDB corrispondente.

    Strategie (in ordine):
    1. Match esatto: "wiki/{nome}.md"
    2. Match case-insensitive sul nome file all'interno di vault/wiki/ (ricorsivo)

    Returns:
        Document ID (path POSIX relativo a VAULT_PATH) se trovato, altrimenti None.
    """
    link_name = _extract_link_name(link_raw)
    if not link_name:
        return None

    # 1. Match esatto
    candidate_id = f"wiki/{link_name}.md"
    if collection.get(ids=[candidate_id], include=[])["ids"]:
        return candidate_id

    # 2. Match case-insensitive
    wiki_dir = vault_path / "wiki"
    if not wiki_dir.exists():
        return None

    link_lower = link_name.lower()
    for md_file in wiki_dir.rglob("*.md"):
        if md_file.stem.lower() == link_lower:
            rel_id = md_file.relative_to(vault_path).as_posix()
            if collection.get(ids=[rel_id], include=[])["ids"]:
                return rel_id

    return None


# ---------------------------------------------------------------------------
# Ricerca RAG
# ---------------------------------------------------------------------------

def _merge_adjacent_chunks(results: list[dict]) -> list[dict]:
    """
    Unisce chunk adiacenti (chunk_index consecutivo) dello stesso documento.

    Quando la ricerca vettoriale restituisce più chunk contigui dello stesso file,
    unirli ripristina la continuità semantica che il chunking a sliding window aveva
    spezzato, offrendo al re-ranker e al subagent passaggi più coerenti.

    Invarianti:
    - I risultati link_traversal non vengono toccati (non hanno chunk_index stabile).
    - Score del passaggio unito = max degli score originali.
    - L'ordine relativo tra documenti diversi è preservato (primo chunk di ogni gruppo).
    """
    to_merge = [r for r in results if r.get("source") != "link_traversal"]
    traversal = [r for r in results if r.get("source") == "link_traversal"]

    # Raggruppa per documento preservando l'ordine di prima apparizione
    seen_order: list[str] = []
    groups: dict[str, list[dict]] = {}
    for r in to_merge:
        p = r["path"]
        if p not in groups:
            seen_order.append(p)
        groups.setdefault(p, []).append(r)

    merged: list[dict] = []
    for path in seen_order:
        chunks = sorted(groups[path], key=lambda c: c.get("chunk_index", 0))
        current = chunks[0].copy()
        for nxt in chunks[1:]:
            if nxt.get("chunk_index", 0) - current.get("chunk_index", 0) <= 1:
                current["content"] = current["content"].rstrip() + "\n" + nxt["content"].lstrip()
                current["score"] = max(current["score"], nxt["score"])
                current["chunk_index"] = nxt.get("chunk_index", 0)
            else:
                merged.append(current)
                current = nxt.copy()
        merged.append(current)

    return merged + traversal


async def vault_search(
    query: str,
    where: str | None = None,
    top_k: int = 5,
    follow_links: bool = True,
    rerank: bool = False,
    instruction: str | None = None,
    merge_chunks: bool = True,
    graph_expand: bool = False,
) -> list[dict]:
    """
    Ricerca semantica nel vault con graph traversal opzionale sui [[link]] Obsidian.

    Flusso:
    1. Genera l'embedding della query.
    2. Interroga ChromaDB (con filtro su "path" se where è specificato).
    3. Se merge_chunks: unisce chunk adiacenti dello stesso documento prima del
       re-ranking, ripristinando la continuità semantica tagliata dallo sliding window.
    4. Se rerank e RERANKER_ENABLED: invia i candidati al re-ranker cross-encoder
       (POST /rerank) con l'instruction che guida il ranking. Fallback graceful
       se il server non è raggiungibile.
    5. Se follow_links: estrae i [[link]] e aggiunge i documenti collegati come
       contesto (source="link_traversal"), senza ricalcolare il ranking.
    6. Se graph_expand: espande i risultati con doc correlati dal knowledge graph
       (source="graph_traversal", score=0.0). Il Token Optimizer gestisce l'over-retrieval.

    Args:
        query: Testo della query per la ricerca semantica.
        where: Stringa di ricerca nel campo metadata "path" (es. "wiki/llm").
               Se None, cerca in tutta la collection.
        top_k: Numero massimo di risultati dalla ricerca vettoriale (post-merge).
        follow_links: Se True, segue i [[link]] Obsidian nei documenti trovati.
        rerank: Se True, recupera top_k*5 candidati vettoriali e li riordina con
                il re-ranker cross-encoder. Richiede RERANKER_ENABLED=true.
        instruction: Istruzione per il re-ranker su come prioritizzare i documenti.
                     Se None, usa Config.reranker_instruction come default.
                     Esempio: "Privilegia documenti tecnici recenti sugli altri."
        merge_chunks: Se True (default), unisce chunk adiacenti dello stesso documento
                      prima del re-ranking. Over-fetcha top_k*2 per compensare la
                      riduzione post-merge.
        graph_expand: Se True, espande i risultati con documenti correlati dal knowledge
                      graph locale (SQLite graph.db). Abilita query multi-hop.

    Returns:
        Lista di dict: [{"content": str, "path": str, "score": float, "source": str,
                          "chunk_index": int}]
        "source" è "vector" per risultati semantici, "reranked" dopo il reranking,
        "link_traversal" per documenti aggiunti via link Obsidian (score=0.0),
        "graph_traversal" per documenti aggiunti via knowledge graph (score=0.0).

    Raises:
        RuntimeError: se chiamata prima di index_vault() (collection vuota).
    """
    from config import Config

    collection = get_chroma_client()

    count = collection.count()
    if count == 0:
        logger.warning(
            "vault_search: collection '%s' vuota — esegui index_vault() prima",
            COLLECTION_NAME,
        )
        return []

    query_embedding = _embed([query])[0]

    # Over-fetch: rerank > where-filter > merge_chunks > baseline
    if rerank:
        n_fetch = min(top_k * 5, count)
    elif where:
        n_fetch = min(top_k * 4, count)
    elif merge_chunks:
        n_fetch = min(top_k * 2, count)
    else:
        n_fetch = min(top_k, count)

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_fetch,
        include=["documents", "metadatas", "distances"],
    )

    # Accumula candidati (senza limite top_k se rerank, per passarli tutti al re-ranker)
    results: list[dict] = []
    seen_ids: set[str] = set()
    _accum_limit = top_k * 2 if merge_chunks else top_k

    for doc_id, content, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        path = meta.get("path", doc_id)

        if where and where not in path:
            continue

        if not rerank and len(results) >= _accum_limit:
            break

        score = round(max(0.0, 1.0 - dist), 4)
        results.append({
            "content": content,
            "path": path,
            "score": score,
            "source": "vector",
            "chunk_index": int(meta.get("chunk_index", 0)),
        })
        seen_ids.add(doc_id)

    # Boost synthesis: i risultati da vault/synthesis/ hanno già score × _SYNTHESIS_BOOST
    # applicato prima del merge e del re-ranker, così entrambi vedono la priorità corretta.
    for r in results:
        if r["path"].startswith("synthesis/"):
            r["score"] = min(1.0, round(r["score"] * _SYNTHESIS_BOOST, 4))
    results.sort(key=lambda r: r["score"], reverse=True)

    if merge_chunks:
        before = len(results)
        results = _merge_adjacent_chunks(results)
        logger.info(
            "vault_search: chunk merging: %d → %d passaggi",
            before, len(results),
        )

    logger.info(
        "vault_search: query=%r — candidati: %d (top_k=%d, rerank=%s, merge=%s, follow_links=%s)",
        query[:60], len(results), top_k, rerank, merge_chunks, follow_links,
    )

    # ── Re-ranking ────────────────────────────────────────────────────────────────
    if rerank and Config.reranker_enabled:
        _instruction = instruction if instruction is not None else _RERANKER_INSTRUCTION
        try:
            try:
                from src.core.llm.reranker import rerank as _reranker_rerank
            except ImportError:
                from core.llm.reranker import rerank as _reranker_rerank  # type: ignore[no-redef]
            texts = [r["content"] for r in results]
            _ranked = await _reranker_rerank(query, texts, _instruction)
            _ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = [results[r["index"]] for r in _ranked[:min(Config.reranker_top_k, top_k)]]
            for r in results:
                r["source"] = "reranked"
            logger.info(
                "vault_search: reranking completato — %d risultati finali (instruction=%r)",
                len(results), _instruction[:60],
            )
        except Exception as _exc:
            logger.warning(
                "vault_search: re-ranker non raggiungibile (%s) — "
                "restituzione candidati vettoriali senza reranking",
                _exc,
            )
            results = results[:top_k]
    elif rerank:
        results = results[:top_k]
    else:
        results = results[:top_k]
    # ─────────────────────────────────────────────────────────────────────────────

    if not follow_links or not results:
        return results

    # --- Graph traversal ---
    vault_path = Config.vault_path
    traversal_ids: list[str] = []

    for item in results:
        for link_raw in _LINK_RE.findall(item["content"]):
            resolved = _resolve_link(link_raw, collection, vault_path)
            if resolved and resolved not in seen_ids:
                seen_ids.add(resolved)
                traversal_ids.append(resolved)

    if traversal_ids:
        fetched = collection.get(
            ids=traversal_ids,
            include=["documents", "metadatas"],
        )
        for doc_id, content, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        ):
            results.append({
                "content": content,
                "path": meta.get("path", doc_id),
                "score": 0.0,
                "source": "link_traversal",
            })
        logger.info(
            "vault_search: aggiunti %d documenti via link_traversal",
            len(traversal_ids),
        )

    # --- Knowledge graph expansion ---
    if graph_expand and results:
        try:
            try:
                from src.core.rag.graph import get_graph_store
            except ImportError:
                from core.rag.graph import get_graph_store  # type: ignore[no-redef]

            gs = get_graph_store()
            result_paths = [r["path"] for r in results if r.get("source") != "link_traversal"]
            expanded_ids = gs.expand(result_paths, hops=1)
            seen_paths = {r["path"] for r in results}
            new_ids = [d for d in expanded_ids if d not in seen_paths]

            if new_ids:
                # Recupera il chunk ::0 di ogni documento espanso
                chunk_ids = [f"{d}::0" for d in new_ids]
                fetched = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
                for doc_id, content, meta in zip(
                    fetched["ids"], fetched["documents"], fetched["metadatas"]
                ):
                    results.append({
                        "content": content,
                        "path": meta.get("path", doc_id.rsplit("::", 1)[0]),
                        "score": 0.0,
                        "source": "graph_traversal",
                        "chunk_index": 0,
                    })
                logger.info(
                    "vault_search: aggiunti %d documenti via graph_traversal",
                    len(fetched["ids"]),
                )
        except Exception as _exc:
            logger.debug("vault_search: graph_expand non disponibile: %s", _exc)

    return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import gc
    import os
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        from src.core.rag.indexer import index_vault
        import src.core.rag.indexer as _vi
    except ImportError:
        from core.rag.indexer import index_vault  # type: ignore[no-redef]
        import core.rag.indexer as _vi  # type: ignore[no-redef]

    async def _run_tests() -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            os.environ["VAULT_PATH"] = tmp
            os.environ["CHROMA_PATH"] = str(Path(tmp) / "chroma")
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
            os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

            vault = Path(tmp)
            wiki = vault / "wiki"
            wiki.mkdir()

            (wiki / "llm.md").write_text(
                "# LLM\nI Large Language Models sono modelli neurali addestrati su grandi corpora "
                "testuali. Vedi anche [[transformer]] per i dettagli architetturali.",
                encoding="utf-8",
            )
            (wiki / "transformer.md").write_text(
                "# Transformer\nArchitettura basata su meccanismi di self-attention, "
                "introdotta nel paper 'Attention is All You Need' (2017).",
                encoding="utf-8",
            )
            (wiki / "rag.md").write_text(
                "# RAG\nRetrieval-Augmented Generation combina recupero informazioni e generazione. "
                "Usa [[llm]] per la generazione e ChromaDB come vector store.",
                encoding="utf-8",
            )
            (wiki / "chromadb.md").write_text(
                "# ChromaDB\nDatabase vettoriale open-source per applicazioni AI. "
                "Supporta embedding e ricerca per similarita coseno.",
                encoding="utf-8",
            )

            print("\n=== setup: index_vault ===")
            report = await index_vault()
            assert report["indexed"] == 4, f"Attesi 4 indicizzati, trovati {report['indexed']}"
            assert not report["errors"]
            print(f"[OK] indicizzazione: {report['indexed']} file")

            print("\n=== vault_search ===")

            # 1. Ricerca semantica base
            results = await vault_search("modelli di linguaggio neurali", top_k=3)
            print(f"\nQuery: 'modelli di linguaggio neurali' — {len(results)} risultati")
            for r in results:
                print(f"  [{r['source']:15}] score={r['score']:.4f}  {r['path']}")
            assert len(results) > 0, "Nessun risultato"
            assert any(r["path"] == "wiki/llm.md" for r in results), "llm.md doveva apparire"
            print("[OK] ricerca semantica restituisce risultati rilevanti")

            # 2. follow_links=False
            results_nf = await vault_search("ChromaDB vettoriale", top_k=3, follow_links=False)
            assert all(r["source"] == "vector" for r in results_nf)
            print("[OK] follow_links=False disabilita il traversal")

            # 3. Filtro where
            results_w = await vault_search(
                "recupero informazioni generazione", where="wiki/rag", top_k=4
            )
            vector_results = [r for r in results_w if r["source"] == "vector"]
            print(f"\nFiltro where='wiki/rag' — vettoriali: {[r['path'] for r in vector_results]}")
            assert len(vector_results) > 0
            assert all("wiki/rag" in r["path"] for r in vector_results)
            print("[OK] filtro where restringe i risultati al path atteso")

            # 4. rerank=True senza server — fallback graceful a top_k candidati vettoriali
            results_r = await vault_search("architettura transformer", top_k=2, rerank=True)
            assert len(results_r) <= 2, f"rerank=True deve restituire al più top_k=2, trovati {len(results_r)}"
            assert all(r["source"] == "vector" for r in results_r if r["score"] > 0)
            print(f"[OK] rerank=True (fallback senza server): {len(results_r)} risultati ≤ top_k=2")

            # 5. rerank=True con instruction custom — fallback preserva i candidati vettoriali
            results_ri = await vault_search(
                "architettura transformer", top_k=2, rerank=True,
                instruction="Privilegia documenti con formule matematiche e pseudocodice.",
            )
            assert len(results_ri) <= 2
            print(f"[OK] rerank=True con instruction custom: {len(results_ri)} risultati ≤ top_k=2")

            _vi._collection = None
            _vi._model = None
            gc.collect()

        print("\nTutti i test vault_search passati.")

    asyncio.run(_run_tests())
