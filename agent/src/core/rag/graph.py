"""
rag/graph.py — Knowledge graph locale per RAG multi-hop.

Estrae entità e relazioni dai chunk (LLM call diretta, max_tokens=256)
e le persiste in SQLite (data/graph.db). Abilita traversal multi-hop
da doc_id sorgente a doc_id correlati via relazioni condivise.

Classi esposte:
    GraphStore      — CRUD SQLite + expand multi-hop
    EntityExtractor — estrazione triple via LLM diretta (background asyncio)

Funzioni esposte:
    get_graph_store()       — singleton GraphStore
    get_entity_extractor()  — singleton EntityExtractor
"""
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_CHUNK_WORDS = 100  # chunk troppo brevi → skip entity extraction


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """
    Persiste entità e relazioni estratte dai chunk in SQLite.

    Schema:
        entities(id TEXT PK, label TEXT, doc_id TEXT, chunk_idx INT)
        relations(src TEXT, rel TEXT, dst TEXT, doc_id TEXT)

    Gli ID entità sono normalizzati (lowercase/strip) per garantire
    deduplicazione cross-documento. I doc_id seguono il formato vault_search
    (path POSIX relativo al vault, es. "wiki/llm.md").
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    id        TEXT NOT NULL,
                    label     TEXT NOT NULL,
                    doc_id    TEXT NOT NULL,
                    chunk_idx INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (id, doc_id)
                );
                CREATE TABLE IF NOT EXISTS relations (
                    src    TEXT NOT NULL,
                    rel    TEXT NOT NULL,
                    dst    TEXT NOT NULL,
                    doc_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rel_src      ON relations(src);
                CREATE INDEX IF NOT EXISTS idx_rel_dst      ON relations(dst);
                CREATE INDEX IF NOT EXISTS idx_entities_doc ON entities(doc_id);
                CREATE INDEX IF NOT EXISTS idx_relations_doc ON relations(doc_id);
            """)

    def upsert(self, doc_id: str, triples: list[tuple[str, str, str]], chunk_idx: int = 0) -> None:
        """Inserisce/aggiorna entità e relazioni per un chunk di doc_id."""
        if not triples:
            return
        with self._conn() as conn:
            for src, rel, dst in triples:
                src_n = src.strip().lower()
                dst_n = dst.strip().lower()
                rel_n = rel.strip().lower()
                if not src_n or not dst_n or not rel_n:
                    continue
                # PK è (id, doc_id) — stessa entità in più doc crea righe separate
                conn.execute(
                    "INSERT OR REPLACE INTO entities (id, label, doc_id, chunk_idx) VALUES (?, ?, ?, ?)",
                    (src_n, src.strip(), doc_id, chunk_idx),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO entities (id, label, doc_id, chunk_idx) VALUES (?, ?, ?, ?)",
                    (dst_n, dst.strip(), doc_id, chunk_idx),
                )
                conn.execute(
                    "INSERT INTO relations (src, rel, dst, doc_id) VALUES (?, ?, ?, ?)",
                    (src_n, rel_n, dst_n, doc_id),
                )

    def delete(self, doc_id: str) -> None:
        """Rimuove tutte le entità e relazioni associate a doc_id."""
        with self._conn() as conn:
            conn.execute("DELETE FROM entities WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM relations WHERE doc_id = ?", (doc_id,))
            logger.debug("graph: rimosso doc_id '%s' dal grafo", doc_id)

    def expand(self, doc_ids: list[str], hops: int = 1) -> list[str]:
        """
        Dato un set di doc_id iniziali, segue le relazioni per N hop
        e restituisce i doc_id dei nodi raggiungibili (esclusi quelli iniziali).

        Algoritmo (basato su relations, non entities):
        1. Raccoglie le entity names (src/dst) da tutte le relazioni dei doc in frontier.
        2. Trova altri doc che condividono almeno una di quelle entity names.
        3. Ripete per hops iterazioni.
        """
        if not doc_ids or hops < 1:
            return []

        with self._conn() as conn:
            initial = set(doc_ids)
            seen = set(doc_ids)
            frontier = set(doc_ids)

            for _ in range(hops):
                if not frontier:
                    break

                # Entity names (src e dst) dai doc in frontier
                placeholders = ",".join("?" * len(frontier))
                rows = conn.execute(
                    f"SELECT DISTINCT src FROM relations WHERE doc_id IN ({placeholders}) "
                    f"UNION SELECT DISTINCT dst FROM relations WHERE doc_id IN ({placeholders})",
                    list(frontier) * 2,
                ).fetchall()
                entity_names = {r[0] for r in rows}
                if not entity_names:
                    break

                # Doc che condividono almeno una entity name, non ancora visti
                placeholders_e = ",".join("?" * len(entity_names))
                placeholders_s = ",".join("?" * len(seen))
                rows = conn.execute(
                    f"SELECT DISTINCT doc_id FROM relations "
                    f"WHERE (src IN ({placeholders_e}) OR dst IN ({placeholders_e})) "
                    f"AND doc_id NOT IN ({placeholders_s})",
                    list(entity_names) * 2 + list(seen),
                ).fetchall()

                next_docs = {r[0] for r in rows}
                seen |= next_docs
                frontier = next_docs

            return [d for d in seen if d not in initial]

    def count(self) -> tuple[int, int]:
        """Restituisce (n_entities, n_relations) per log/debug."""
        with self._conn() as conn:
            n_e = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            n_r = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        return n_e, n_r


# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Estrae triple (entità_A, relazione, entità_B) da chunk testuali via LLM.

    - LLM call diretta con max_tokens=256, thinking=False.
    - Skip automatico per chunk < _MIN_CHUNK_WORDS parole.
    - extract_and_store() è progettato per asyncio.create_task() — silent failure,
      non blocca la pipeline di indicizzazione.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._store = graph_store

    async def extract_and_store(
        self,
        doc_id: str,
        chunk_text: str,
        chunk_idx: int = 0,
    ) -> None:
        """Estrae triple e le salva nel GraphStore. Silent failure."""
        if len(chunk_text.split()) < _MIN_CHUNK_WORDS:
            return
        try:
            triples = await self._extract(chunk_text)
            if triples:
                self._store.upsert(doc_id, triples, chunk_idx)
                logger.debug(
                    "graph: %d triple estratte da '%s' chunk %d",
                    len(triples), doc_id, chunk_idx,
                )
        except Exception as exc:
            logger.debug(
                "graph: estrazione fallita per '%s' chunk %d: %s",
                doc_id, chunk_idx, exc,
            )

    async def _extract(self, chunk_text: str) -> list[tuple[str, str, str]]:
        """LLM call diretta per estrazione triple in formato JSON."""
        try:
            from src.core.llm.client import get_llm_client
        except ImportError:
            from core.llm.client import get_llm_client  # type: ignore[no-redef]

        llm = get_llm_client()
        system_prompt = (
            "Extract the most important knowledge graph triples from the text. "
            'Return ONLY a JSON object: {"triples": [["entity_A", "relation", "entity_B"], ...]}. '
            "Use concise lowercase relations (e.g. 'uses', 'improves', 'is-a', 'part-of'). "
            "Limit to 10 triples. Omit trivial or overly generic ones."
        )
        result = await llm.complete(
            system_prompt=system_prompt,
            user_message=chunk_text[:1500],
            thinking=False,
            temperature=0.1,
            max_tokens=256,
        )
        raw = result.get("triples", []) if isinstance(result, dict) else []
        return [
            (str(t[0]), str(t[1]), str(t[2]))
            for t in raw
            if isinstance(t, (list, tuple)) and len(t) == 3
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_graph_store: "GraphStore | None" = None
_entity_extractor: "EntityExtractor | None" = None


def get_graph_store() -> GraphStore:
    """Restituisce il singleton GraphStore, inizializzandolo al primo accesso."""
    global _graph_store
    if _graph_store is None:
        from config import Config
        db_path = getattr(Config, "graph_db_path", "./data/graph.db")
        _graph_store = GraphStore(db_path)
    return _graph_store


def get_entity_extractor() -> EntityExtractor:
    """Restituisce il singleton EntityExtractor, inizializzandolo al primo accesso."""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor(get_graph_store())
    return _entity_extractor


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

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

    print("\n=== GraphStore unit tests ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "test_graph.db")
        store = GraphStore(db)

        # --- upsert e count ---
        store.upsert("wiki/llm.md", [("LLM", "uses", "Transformer"), ("RAG", "improves", "LLM")], chunk_idx=0)
        n_e, n_r = store.count()
        check("upsert: 3 entità per 2 triple (LLM, Transformer, RAG)", n_e == 3)
        check("upsert: relazioni create", n_r == 2)

        # --- upsert secondo doc (RAG condiviso → riga separata per composite PK) ---
        store.upsert("wiki/rag.md", [("RAG", "uses", "ChromaDB"), ("ChromaDB", "stores", "Embeddings")], chunk_idx=0)
        n_e2, n_r2 = store.count()
        # RAG appare in entrambi i doc → 2 righe entities; + ChromaDB, Embeddings
        check("upsert secondo doc: entità aggiuntive", n_e2 > n_e)
        check("upsert secondo doc: relazioni aggiuntive", n_r2 == 4)

        # --- expand 1 hop ---
        expanded = store.expand(["wiki/llm.md"], hops=1)
        check("expand: wiki/rag.md raggiungibile da wiki/llm.md via RAG", "wiki/rag.md" in expanded)
        check("expand: wiki/llm.md escluso dal risultato", "wiki/llm.md" not in expanded)

        # --- expand con doc senza relazioni condivise ---
        store.upsert("wiki/isolated.md", [("IsolatedThing", "has", "Property")], chunk_idx=0)
        expanded_iso = store.expand(["wiki/isolated.md"], hops=1)
        check("expand: doc isolato → lista vuota o solo se-stesso", "wiki/isolated.md" not in expanded_iso)

        # --- delete ---
        store.delete("wiki/llm.md")
        n_e3, n_r3 = store.count()
        check("delete: entità del doc rimosso", n_e3 < n_e2)
        check("delete: relazioni del doc rimosso", n_r3 < n_r2)

        # --- expand dopo delete: wiki/llm.md cancellato, RAG relazioni rimosse ---
        expanded_after = store.expand(["wiki/rag.md"], hops=1)
        check("expand dopo delete: non segue relazioni cancellate", "wiki/llm.md" not in expanded_after)

        # --- edge cases ---
        check("expand lista vuota → []", store.expand([], hops=1) == [])
        check("expand hops=0 → []", store.expand(["wiki/rag.md"], hops=0) == [])

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
