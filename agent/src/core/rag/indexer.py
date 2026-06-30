"""
rag/indexer — Indicizzazione del vault Obsidian in ChromaDB.

Componente di sistema che gestisce il ciclo di vita del client ChromaDB e del
modello di embedding, e implementa l'indicizzazione incrementale di vault/wiki/
e vault/synthesis/.

Ottimizzazioni:
- Controllo mtime in bulk (un'unica query per tutti i file ::0)
- Skip parziale a livello di chunk tramite hash SHA-256 (solo contenuto cambiato)
- Un unico batch embedding cross-file per massimizzare l'utilizzo GPU
- Eliminazione zombie chunk per ID range senza query where

Funzioni esposte:
    get_chroma_client() -> chromadb.Collection   — singleton ChromaDB + embedding
    index_vault() -> dict                        — indicizza/aggiorna vault/wiki/ e vault/synthesis/
"""
import hashlib
import logging
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "wiki"
_HEADER_RE = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)

# Directory del vault indicizzate in ChromaDB.
# "wiki" è obbligatoria; le altre vengono skip silenzioso se assenti.
_INDEXED_DIRS = ["wiki", "synthesis"]

# Estensioni indicizzate per vault/raw/ — testo estratto da file binari.
_RAW_EXTENSIONS = {".pdf", ".docx"}

# Singleton: inizializzati una volta sola da get_chroma_client()
_collection: chromadb.Collection | None = None
_chroma_client: chromadb.ClientAPI | None = None
_model: SentenceTransformer | None = None


# ---------------------------------------------------------------------------
# Inizializzazione (singleton)
# ---------------------------------------------------------------------------

def get_chroma_client() -> chromadb.Collection:
    """
    Inizializza il modello di embedding e il client ChromaDB, restituisce la collection.

    Idempotente: chiamate successive restituiscono la stessa istanza già inizializzata.
    Se il modello non è presente in EMBEDDING_MODEL_PATH, viene scaricato da HuggingFace
    usando il nome del modello (ultimo componente del path).

    Returns:
        chromadb.Collection pronta all'uso (collection "wiki", metrica coseno).
    """
    global _collection, _chroma_client, _model
    if _collection is not None:
        return _collection

    from config import Config

    # --- Modello di embedding ---
    model_dir = Path(Config.embedding_model_dir)
    if model_dir.exists() and any(model_dir.iterdir()):
        model_str = str(model_dir)
        logger.info("Caricamento modello di embedding locale: %s", model_str)
    else:
        model_str = Config.embedding_model
        logger.warning(
            "Modello non trovato in '%s' — carico '%s' (download da HuggingFace se necessario)",
            model_dir,
            model_str,
        )

    try:
        from src.core.llm.gpu_manager import get_embedding_device
    except ImportError:
        from core.llm.gpu_manager import get_embedding_device  # type: ignore[no-redef]

    import time as _time
    _device = get_embedding_device()
    logger.info("Caricamento modello di embedding su %s (può richiedere minuti da filesystem Windows)...", _device)
    _t0 = _time.monotonic()
    _model = SentenceTransformer(model_str, device=_device)
    logger.info("Modello di embedding caricato in %.1fs", _time.monotonic() - _t0)

    # --- ChromaDB ---
    chroma_path = Config.chroma_path
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_path)
    _chroma_client = client
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        "ChromaDB pronto in '%s': %d documenti nella collection '%s'",
        chroma_path,
        _collection.count(),
        COLLECTION_NAME,
    )
    return _collection


def get_chroma_collection(name: str) -> "chromadb.Collection":
    """Crea o restituisce una collection ChromaDB per nome, riutilizzando il client singleton."""
    global _chroma_client
    if _chroma_client is None:
        get_chroma_client()  # inizializza client e modello
    return _chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> list[list[float]]:
    """Genera embeddings in batch con il modello singleton."""
    global _model
    if _model is None:
        get_chroma_client()
    vectors = _model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def _chunk_hash(text: str) -> str:
    """SHA-256 (16 hex chars) del testo di un chunk — usato per skip parziale."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _extract_text_pdf(path: Path) -> str:
    """Estrae testo da PDF via PyMuPDF; importazione lazy per non bloccare l'avvio."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    return "\n\n".join(page.get_text() for page in doc)


def _extract_text_docx(path: Path) -> str:
    """Estrae testo da DOCX via python-docx; importazione lazy per non bloccare l'avvio."""
    from docx import Document  # python-docx
    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_file_content(path: Path) -> str:
    """Legge il contenuto testuale di un file (Markdown, PDF o DOCX)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_text_pdf(path)
    if ext == ".docx":
        return _extract_text_docx(path)
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _segment_atomic(text: str) -> list[tuple[str, bool]]:
    """
    Segmenta il testo in blocchi atomici e testo normale.

    Blocchi atomici (is_atomic=True):
    - Fenced code block: da riga che inizia con ``` fino alla riga di chiusura ```.
    - Tabella Markdown: run consecutivo di righe che iniziano con | (incluso |---|).

    Returns: lista ordinata di (testo_blocco, is_atomic).
    """
    lines = text.split("\n")
    segments: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_fence = False

    def _flush(is_atomic: bool) -> None:
        if buf:
            segments.append(("\n".join(buf), is_atomic))
            buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_fence:
            buf.append(line)
            if stripped.startswith("```"):
                _flush(True)
                in_fence = False
        elif stripped.startswith("```"):
            _flush(False)
            in_fence = True
            buf.append(line)
        elif stripped.startswith("|"):
            _flush(False)
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            _flush(True)
            continue
        else:
            buf.append(line)

        i += 1

    _flush(in_fence)  # fence non chiusa → trattata come atomica
    return segments


def _atomic_extend(section_text: str, end: int) -> int:
    """
    Se il punto di taglio 'end' (numero di parole) cade dentro un blocco atomico,
    restituisce la posizione dopo la fine del blocco. Altrimenti ritorna 'end'.

    Garantisce che fenced code e tabelle Markdown non vengano mai spezzati
    dal sliding window della chunking pipeline.
    """
    pos = 0
    for seg_text, is_atomic in _segment_atomic(section_text):
        seg_end = pos + len(seg_text.split())
        if is_atomic and pos < end < seg_end:
            return seg_end
        pos = seg_end
    return end


def _chunk_document(
    content: str,
    source: str,
    mtime: float,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Divide un documento markdown in chunk con strategia a due livelli.

    Livello 1: split strutturale su header markdown (#, ##, ###).
    Livello 2: se una sezione supera chunk_size parole, sliding window
               con overlap di chunk_overlap parole e due guardie:
               - Orphan-fold: coda finale < chunk_size/2 parole → assorbita
                 nel chunk corrente (evita chunk stub inutilmente piccoli).
               - Atomic block guard: fenced code e tabelle MD non vengono
                 mai spezzati a metà (il punto di taglio viene posticipato
                 alla fine del blocco atomico).

    "Parole" = elementi di text.split() (whitespace split), non subword token BPE.

    Returns:
        Lista di dict: [{"id": str, "text": str, "metadata": dict}]
        ID formato: "{source}::{chunk_index}" (es. "wiki/llm.md::0").
        metadata include "path" (= source) per compatibilità con vault_search().
    """
    parts = _HEADER_RE.split(content)
    # parts = [pre_text, header1, body1, header2, body2, ...]

    sections: list[tuple[str, str]] = []

    if parts[0].strip():
        sections.append(("", parts[0]))

    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((header, body))

    chunks: list[dict] = []
    chunk_index = 0

    for header, body in sections:
        section_text = (header + "\n" + body).strip() if header else body.strip()
        if not section_text:
            continue

        section_title = header.lstrip("#").strip() if header else ""
        tokens = section_text.split()

        if len(tokens) <= chunk_size:
            chunks.append({
                "id": f"{source}::{chunk_index}",
                "text": section_text,
                "metadata": {
                    "path": source,
                    "source": source,
                    "section": section_title,
                    "chunk_index": chunk_index,
                    "mtime": mtime,
                },
            })
            chunk_index += 1
        else:
            # Sliding window con atomic block guard e orphan-fold.
            # tokens include le parole dell'header (già in section_text):
            # non si ri-prepend l'header per evitare duplicazione.
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))

                # Atomic block guard: se il taglio cade dentro un blocco atomico,
                # estende end fino alla fine del blocco.
                if end < len(tokens):
                    end = _atomic_extend(section_text, end)
                    end = min(end, len(tokens))

                # Orphan-fold: coda < chunk_size/2 → assorbita nel chunk corrente.
                remaining = len(tokens) - end
                if 0 < remaining < chunk_size // 2:
                    end = len(tokens)

                chunks.append({
                    "id": f"{source}::{chunk_index}",
                    "text": " ".join(tokens[start:end]),
                    "metadata": {
                        "path": source,
                        "source": source,
                        "section": section_title,
                        "chunk_index": chunk_index,
                        "mtime": mtime,
                    },
                })
                chunk_index += 1
                if end == len(tokens):
                    break
                start += chunk_size - chunk_overlap

    # Fallback: documento senza header e dentro chunk_size
    if not chunks and content.strip():
        chunks.append({
            "id": f"{source}::0",
            "text": content.strip(),
            "metadata": {
                "path": source,
                "source": source,
                "section": "",
                "chunk_index": 0,
                "mtime": mtime,
            },
        })

    return chunks


# ---------------------------------------------------------------------------
# Migrazione collection
# ---------------------------------------------------------------------------

def _ensure_collection_compatible() -> None:
    """
    Verifica che la collection ChromaDB sia compatibile con il modello corrente.

    Confronta la dimensione degli embedding salvati con quella prodotta dal modello
    attivo. Se differiscono (es. 384 → 768 dopo cambio modello), elimina la
    collection e ne crea una nuova vuota. index_vault() re-indicizzerà tutto.
    """
    global _collection

    if _collection is None or _collection.count() == 0:
        return

    test_vec = _embed(["test"])[0]
    new_dim = len(test_vec)

    sample = _collection.get(limit=1, include=["embeddings"])
    embeddings = sample.get("embeddings")
    if not sample["ids"] or embeddings is None or len(embeddings) == 0:
        return

    stored_dim = len(sample["embeddings"][0])
    if stored_dim == new_dim:
        return

    logger.warning(
        "Dimensioni embedding cambiate: elimino e ricreo la collection "
        "('%s': %d → %d dims)",
        COLLECTION_NAME, stored_dim, new_dim,
    )

    from config import Config
    client = chromadb.PersistentClient(path=Config.chroma_path)
    client.delete_collection(name=COLLECTION_NAME)
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Collection '%s' ricreata (vuota, %d dims)", COLLECTION_NAME, new_dim)


# ---------------------------------------------------------------------------
# Indicizzazione
# ---------------------------------------------------------------------------

async def index_vault(force: bool = False) -> dict:
    """
    Indicizza (o aggiorna) tutti i file .md in vault/wiki/ e vault/synthesis/ in ChromaDB.

    Ottimizzazioni rispetto alla versione naive:
    - Controllo mtime in bulk (un'unica query per tutti i file).
    - Skip parziale per chunk: solo i chunk con contenuto cambiato vengono re-embedded.
    - Un unico batch di embedding cross-file per massimizzare l'utilizzo GPU.
    - Eliminazione zombie chunk per ID range (nessuna query where).

    Chunk ID = "{path_relativo_vault}::{chunk_index}" (es. "wiki/llm.md::0").
    "indexed" conta file modificati/nuovi, non chunk prodotti.

    Args:
        force: se True, bypassa il check mtime e riprocessa tutti i file.
               Il check hash per chunk rimane attivo (evita re-embedding inutili).

    Returns:
        {"indexed": int, "skipped": int, "errors": list[str]}
    """
    from config import Config

    collection = get_chroma_client()
    _ensure_collection_compatible()

    vault_path = Config.vault_path

    # Raccoglie .md da tutte le directory indicizzate; wiki/ è obbligatoria.
    md_files: list[Path] = []
    for dir_name in _INDEXED_DIRS:
        dir_path = vault_path / dir_name
        if dir_path.exists():
            md_files.extend(dir_path.rglob("*.md"))
        elif dir_name == "wiki":
            logger.warning("directory wiki/ non trovata in '%s'", vault_path)
            return {"indexed": 0, "skipped": 0, "errors": [f"wiki/ non trovata: {vault_path / 'wiki'}"]}
        else:
            logger.debug("directory %s/ non trovata in '%s', skip", dir_name, vault_path)

    # Aggiunge PDF/DOCX da vault/raw/
    raw_dir = vault_path / "raw"
    if raw_dir.exists():
        for f in raw_dir.rglob("*"):
            if f.suffix.lower() in _RAW_EXTENSIONS:
                md_files.append(f)
    else:
        logger.debug("directory raw/ non trovata in '%s', skip", vault_path)

    md_files = sorted(md_files)

    total = len(md_files)
    _log_dirs = [str(vault_path / d) for d in _INDEXED_DIRS if (vault_path / d).exists()]
    if raw_dir.exists():
        _log_dirs.append(str(raw_dir))
    logger.info("trovati %d file da indicizzare in %s", total, _log_dirs)

    # Nessun file da indicizzare: esci subito (ChromaDB.get() rifiuta ids vuoti)
    if not md_files:
        logger.info("nessun file da indicizzare, skip")
        return {"indexed": 0, "skipped": 0, "errors": []}

    # ── Phase 1: bulk mtime check ──────────────────────────────────────────────
    # Un'unica query per recuperare ::0 di ogni file → evita N round-trip
    doc_ids = [md.relative_to(vault_path).as_posix() for md in md_files]
    bulk = collection.get(ids=[f"{d}::0" for d in doc_ids], include=["metadatas"])
    stored_meta: dict[str, tuple[float, int]] = {}
    for chunk_id, meta in zip(bulk["ids"], bulk["metadatas"]):
        did = chunk_id.rsplit("::", 1)[0]
        stored_meta[did] = (float(meta.get("mtime", -1.0)), int(meta.get("chunk_count", -1)))

    # ── Phase 2: classify files ────────────────────────────────────────────────
    to_process: list[tuple[Path, str, float]] = []
    skipped = 0
    for md, doc_id in zip(md_files, doc_ids):
        cur_mtime = md.stat().st_mtime
        if not force and doc_id in stored_meta and abs(stored_meta[doc_id][0] - cur_mtime) < 1e-3:
            logger.debug("skip '%s' (invariato)", doc_id)
            skipped += 1
        else:
            to_process.append((md, doc_id, cur_mtime))

    logger.info("%d documenti da (ri)indicizzare, %d invariati", len(to_process), skipped)

    indexed = 0
    total_chunks = 0
    errors: list[str] = []

    # ── Phase 3: leggi contenuto + hash chunk esistenti ───────────────────────
    # list of (doc_id, new_chunks, stored_hashes: dict[int→str], stored_count: int)
    file_data: list[tuple[str, list[dict], dict[int, str], int]] = []

    for i, (md, doc_id, cur_mtime) in enumerate(to_process, start=1):
        try:
            content = _read_file_content(md)
            if not content.strip():
                logger.info("[%d/%d] skip '%s' (file vuoto)", i, len(to_process), doc_id)
                skipped += 1
                continue

            new_chunks = _chunk_document(
                content, doc_id, cur_mtime,
                Config.chunk_size, Config.chunk_overlap,
            )

            stored_count = stored_meta.get(doc_id, (-1.0, -1))[1]
            stored_hashes: dict[int, str] = {}

            if stored_count >= 0:
                # chunk_count noto: costruiamo gli ID direttamente, nessuna query where
                ex = collection.get(
                    ids=[f"{doc_id}::{j}" for j in range(stored_count)],
                    include=["metadatas"],
                )
                for eid, emeta in zip(ex["ids"], ex["metadatas"]):
                    h = emeta.get("chunk_hash", "")
                    if h:
                        stored_hashes[int(eid.rsplit("::", 1)[1])] = h
            elif doc_id in stored_meta:
                # Retrocompatibilità: formato vecchio senza chunk_count
                ex = collection.get(
                    where={"source": {"$eq": doc_id}},
                    include=["metadatas"],
                )
                for eid, emeta in zip(ex["ids"], ex["metadatas"]):
                    h = emeta.get("chunk_hash", "")
                    if h:
                        stored_hashes[int(eid.rsplit("::", 1)[1])] = h
                stored_count = len(ex["ids"])

            file_data.append((doc_id, new_chunks, stored_hashes, stored_count))

        except Exception as exc:
            logger.error("[%d/%d] errore su '%s': %s", i, len(to_process), doc_id, exc)
            errors.append(f"{doc_id}: {exc}")

    # ── Phase 4: identifica chunk da re-embeddare ─────────────────────────────
    # Aggiunge chunk_hash e chunk_count ai metadati in-place
    embed_needed: list[tuple[int, int, str]] = []  # (file_data_idx, chunk_idx, text)

    for fi, (doc_id, new_chunks, stored_hashes, _) in enumerate(file_data):
        n = len(new_chunks)
        for ci, chunk in enumerate(new_chunks):
            h = _chunk_hash(chunk["text"])
            chunk["metadata"]["chunk_hash"] = h
            chunk["metadata"]["chunk_count"] = n
            if h != stored_hashes.get(ci, ""):
                embed_needed.append((fi, ci, chunk["text"]))

    # ── Phase 5: un unico batch embedding cross-file ──────────────────────────
    embed_map: dict[int, dict[int, list[float]]] = {}
    if embed_needed:
        logger.info(
            "batch embedding %d chunk da %d file...",
            len(embed_needed), len(file_data),
        )
        vecs = _embed([t for _, _, t in embed_needed])
        for (fi, ci, _), vec in zip(embed_needed, vecs):
            embed_map.setdefault(fi, {})[ci] = vec

    # ── Phase 6: upsert chunk cambiati; aggiorna metadata; rimuovi zombie ──────
    import asyncio as _asyncio
    try:
        from src.core.rag.graph import get_entity_extractor as _get_ee
    except ImportError:
        from core.rag.graph import get_entity_extractor as _get_ee  # type: ignore[no-redef]

    for fi, (doc_id, new_chunks, _, stored_count) in enumerate(file_data):
        try:
            file_embed = embed_map.get(fi, {})
            n_new = len(new_chunks)

            upsert_ids, upsert_embs, upsert_docs, upsert_metas = [], [], [], []
            update_ids, update_metas = [], []

            for ci, chunk in enumerate(new_chunks):
                if ci in file_embed:
                    upsert_ids.append(chunk["id"])
                    upsert_embs.append(file_embed[ci])
                    upsert_docs.append(chunk["text"])
                    upsert_metas.append(chunk["metadata"])
                else:
                    update_ids.append(chunk["id"])
                    update_metas.append(chunk["metadata"])

            if upsert_ids:
                collection.upsert(
                    ids=upsert_ids,
                    embeddings=upsert_embs,
                    documents=upsert_docs,
                    metadatas=upsert_metas,
                )
            if update_ids:
                collection.update(ids=update_ids, metadatas=update_metas)

            # Zombie: chunk con indice >= n_new rimasti dall'indicizzazione precedente
            if stored_count > n_new:
                zombie_ids = [f"{doc_id}::{j}" for j in range(n_new, stored_count)]
                logger.debug(
                    "rimozione %d chunk obsoleti per '%s'",
                    len(zombie_ids), doc_id,
                )
                collection.delete(ids=zombie_ids)
                try:
                    from src.core.rag.graph import get_graph_store as _get_gs
                except ImportError:
                    from core.rag.graph import get_graph_store as _get_gs  # type: ignore[no-redef]
                _get_gs().delete(doc_id)

            # Estrazione triple knowledge graph per chunk nuovi (background, silent failure)
            if upsert_ids:
                _ee = _get_ee()
                for ci, chunk in enumerate(new_chunks):
                    if ci in file_embed:
                        _asyncio.create_task(
                            _ee.extract_and_store(doc_id, chunk["text"], chunk_idx=ci)
                        )

            logger.info(
                "'%s' — %d chunk (%d re-embedded, %d invariati)",
                doc_id, n_new, len(upsert_ids), len(update_ids),
            )
            indexed += 1
            total_chunks += n_new

        except Exception as exc:
            logger.error("errore upsert '%s': %s", doc_id, exc)
            errors.append(f"{doc_id}: {exc}")

    logger.info(
        "index_vault completato: %d indicizzati (%d chunk totali), %d saltati, %d errori",
        indexed, total_chunks, skipped, len(errors),
    )
    return {"indexed": indexed, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import gc
    import os
    import sys
    import tempfile
    import time as _time

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

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

            print("\n=== index_vault ===")
            report = await index_vault()
            assert report["indexed"] == 4, f"Attesi 4 indicizzati, trovati {report['indexed']}"
            assert report["skipped"] == 0
            assert not report["errors"]
            print(f"[OK] indicizzazione completa (4 file, {report['indexed']} indicizzati)")

            report2 = await index_vault()
            assert report2["indexed"] == 0
            assert report2["skipped"] == 4
            print("[OK] seconda esecuzione: tutti i file saltati (mtime invariato)")

            _time.sleep(0.01)
            (wiki / "llm.md").write_text(
                "# LLM\nI Large Language Models sono modelli neurali addestrati su grandi corpora "
                "testuali. Vedi anche [[transformer]] per i dettagli architetturali.\n\n"
                "## Aggiornamento\nNuova sezione aggiunta.",
                encoding="utf-8",
            )
            report3 = await index_vault()
            assert report3["indexed"] == 1, f"Atteso 1 file re-indicizzato, trovati {report3['indexed']}"
            assert report3["skipped"] == 3
            assert not report3["errors"]
            print("[OK] re-indicizzazione parziale: solo il file modificato aggiornato")

            global _collection, _model  # type: ignore[misc]
            _collection = None
            _model = None
            gc.collect()

        print("\nTutti i test rag/indexer passati.")

    # ── Test _segment_atomic ──────────────────────────────────────────────────
    def _test_segment_atomic() -> None:
        print("\n=== _segment_atomic ===")

        # Testo senza blocchi atomici
        segs = _segment_atomic("Paragrafo normale.\nSeconda riga.")
        assert len(segs) == 1 and segs[0][1] is False, f"Atteso 1 blocco non-atomico, ottenuto {segs}"
        print("[OK] testo normale → 1 segmento non-atomico")

        # Fenced code block
        segs = _segment_atomic("Intro.\n```python\ncodice\n```\nOutro.")
        atomics = [s for s in segs if s[1]]
        assert len(atomics) == 1
        assert "```python" in atomics[0][0] and "codice" in atomics[0][0]
        print("[OK] fenced code block → rilevato come atomico")

        # Tabella Markdown
        segs = _segment_atomic("Intro.\n| A | B |\n|---|---|\n| 1 | 2 |\nOutro.")
        atomics = [s for s in segs if s[1]]
        assert len(atomics) == 1
        assert "| A | B |" in atomics[0][0]
        print("[OK] tabella Markdown → rilevata come atomica")

        # Conteggio parole: segmenti = stesso totale di section_text.split()
        text = "Intro.\n```python\ncodice qui\n```\n| A | B |\n|---|---|\nFine."
        total_seg = sum(len(s[0].split()) for s in _segment_atomic(text))
        assert total_seg == len(text.split()), f"Word count mismatch: {total_seg} vs {len(text.split())}"
        print("[OK] conteggio parole consistente tra segmenti e testo originale")

    _test_segment_atomic()

    # ── Test _chunk_document: orphan-fold e atomic guard ─────────────────────
    def _test_chunk_document() -> None:
        print("\n=== _chunk_document — orphan-fold e atomic guard ===")

        # Orphan-fold: sezione da 420 parole con chunk_size=300, overlap=50
        # → coda di 120 parole (< 150) assorbita nel primo chunk
        word = "parola"
        section_420 = " ".join([word] * 420)
        chunks = _chunk_document(section_420, "test.md", 0.0, 300, 50)
        assert len(chunks) == 1, f"Atteso 1 chunk (orphan-fold), ottenuti {len(chunks)}"
        assert len(chunks[0]["text"].split()) == 420
        print("[OK] orphan-fold: 420 parole con chunk_size=300 → 1 chunk (coda 120 < 150 assorbita)")

        # No orphan-fold: sezione da 520 parole → coda 220 ≥ 150 → 2 chunk
        section_520 = " ".join([word] * 520)
        chunks = _chunk_document(section_520, "test.md", 0.0, 300, 50)
        assert len(chunks) == 2, f"Attesi 2 chunk, ottenuti {len(chunks)}"
        print("[OK] no orphan-fold: 520 parole → 2 chunk (coda 220 ≥ 150)")

        # Atomic guard: tabella al confine di chunk — non deve essere spezzata
        # 290 parole normali + tabella da 15 parole = 305 total
        # chunk_size=300 → senza guard spezza la tabella a parola 300
        normal_words = " ".join([word] * 290)
        table = "| Col1 | Col2 | Col3 |\n|------|------|------|\n| A | B | C |\n| D | E | F |"
        text_with_table = normal_words + "\n" + table
        chunks = _chunk_document(text_with_table, "test.md", 0.0, 300, 50)
        # Il chunk 0 deve contenere l'intera tabella (esteso oltre 300)
        first_chunk_text = chunks[0]["text"]
        assert "| Col1 | Col2 | Col3 |" in first_chunk_text or \
               all(w in first_chunk_text for w in ["|", "Col1", "|", "A", "|", "D"]), \
               f"Tabella spezzata nel chunk 0: {first_chunk_text[-200:]}"
        print("[OK] atomic guard: tabella al confine → non spezzata nel primo chunk")

    _test_chunk_document()

    asyncio.run(_run_tests())
