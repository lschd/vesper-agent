"""
Watchdog — monitora vault/wiki/, vault/synthesis/ e vault/raw/ e re-indicizza i file modificati.

Usa watchdog per rilevare eventi filesystem in tempo reale. Il debounce
per-file evita re-indicizzazioni multiple su salvataggi rapidi consecutivi
(es. comportamento di Obsidian).
"""
import asyncio
import logging
import threading
from pathlib import Path

from watchdog.events import FileMovedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _WikiEventHandler(FileSystemEventHandler):
    """Handler degli eventi filesystem per vault/wiki/, vault/synthesis/ e vault/raw/."""

    def __init__(self, vault_path: Path, debounce_seconds: int) -> None:
        super().__init__()
        self._vault_path = vault_path
        self._debounce_seconds = debounce_seconds
        self._timers: dict[str, threading.Timer] = {}
        self._timers_lock = threading.Lock()
        self._reindex_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_indexed_md(self, path: str) -> bool:
        """Restituisce True se il path è un .md dentro una directory indicizzata (wiki/ o synthesis/)."""
        p = Path(path)
        if p.suffix != ".md":
            return False
        return (
            p.is_relative_to(self._vault_path / "wiki")
            or p.is_relative_to(self._vault_path / "synthesis")
        )

    def _is_raw_binary(self, path: str) -> bool:
        """Restituisce True se il path è un PDF/DOCX in vault/raw/."""
        p = Path(path)
        return (
            p.suffix.lower() in {".pdf", ".docx"}
            and p.is_relative_to(self._vault_path / "raw")
        )

    def _doc_id(self, path: str) -> str:
        """Converte un path assoluto nel doc_id relativo al vault (es. 'wiki/llm.md')."""
        return Path(path).relative_to(self._vault_path).as_posix()

    def _schedule_reindex(self, path: str) -> None:
        """Cancella il timer esistente per path e ne avvia uno nuovo (debounce)."""
        with self._timers_lock:
            existing = self._timers.pop(path, None)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(
                self._debounce_seconds,
                self._reindex_callback,
                args=(path,),
            )
            timer.daemon = True
            self._timers[path] = timer
            timer.start()
        logger.debug(
            "Watchdog: timer avviato per '%s' (%.0fs)",
            path, self._debounce_seconds,
        )

    def _delete_from_chroma(self, doc_id: str) -> None:
        """Rimuove tutti i chunk di doc_id dalla collection ChromaDB."""
        try:
            try:
                from src.core.rag.indexer import get_chroma_client
            except ImportError:
                from core.rag.indexer import get_chroma_client  # type: ignore[no-redef]

            collection = get_chroma_client()
            old = collection.get(where={"source": {"$eq": doc_id}}, include=[])
            if old["ids"]:
                collection.delete(ids=old["ids"])
                logger.info(
                    "Watchdog: eliminati %d chunk da ChromaDB per '%s'",
                    len(old["ids"]), doc_id,
                )
            else:
                logger.debug(
                    "Watchdog: nessun chunk in ChromaDB per '%s', nulla da eliminare",
                    doc_id,
                )
        except Exception as exc:
            logger.error(
                "Watchdog: errore durante eliminazione ChromaDB per '%s': %s",
                doc_id, exc,
            )

    def _reindex_callback(self, path: str) -> None:
        """Eseguito dal timer allo scadere del debounce: chiama index_vault()."""
        with self._timers_lock:
            self._timers.pop(path, None)

        try:
            try:
                from src.core.rag.indexer import index_vault
            except ImportError:
                from core.rag.indexer import index_vault  # type: ignore[no-redef]

            logger.info(
                "Watchdog: re-indicizzazione triggerata da modifica su '%s'",
                path,
            )
            # Serializza chiamate concorrenti: se un'altra re-indicizzazione è in
            # corso, questa attende. index_vault() skippa i file invariati via mtime,
            # quindi esecuzioni consecutive sono economiche.
            with self._reindex_lock:
                asyncio.run(index_vault())

        except Exception as exc:
            logger.error(
                "Watchdog: errore in index_vault (triggerato da '%s'): %s",
                path, exc,
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or not (self._is_indexed_md(event.src_path) or self._is_raw_binary(event.src_path)):
            return
        logger.info("Watchdog: CREATE '%s'", event.src_path)
        try:
            self._schedule_reindex(event.src_path)
        except Exception as exc:
            logger.error("Watchdog: errore su CREATE '%s': %s", event.src_path, exc)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or not (self._is_indexed_md(event.src_path) or self._is_raw_binary(event.src_path)):
            return
        logger.info("Watchdog: MODIFIED '%s'", event.src_path)
        try:
            self._schedule_reindex(event.src_path)
        except Exception as exc:
            logger.error("Watchdog: errore su MODIFIED '%s': %s", event.src_path, exc)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or not (self._is_indexed_md(event.src_path) or self._is_raw_binary(event.src_path)):
            return
        logger.info("Watchdog: DELETE '%s'", event.src_path)
        try:
            # Cancella eventuale timer pendente: il file non esiste più
            with self._timers_lock:
                timer = self._timers.pop(event.src_path, None)
                if timer is not None:
                    timer.cancel()
            self._delete_from_chroma(self._doc_id(event.src_path))
        except Exception as exc:
            logger.error("Watchdog: errore su DELETE '%s': %s", event.src_path, exc)

    def on_moved(self, event: FileMovedEvent) -> None:
        src_is_md = not event.is_directory and (self._is_indexed_md(event.src_path) or self._is_raw_binary(event.src_path))
        dest_is_md = not event.is_directory and (self._is_indexed_md(event.dest_path) or self._is_raw_binary(event.dest_path))

        if not src_is_md and not dest_is_md:
            return

        logger.info(
            "Watchdog: MOVED '%s' → '%s'",
            event.src_path, event.dest_path,
        )
        try:
            if src_is_md:
                with self._timers_lock:
                    timer = self._timers.pop(event.src_path, None)
                    if timer is not None:
                        timer.cancel()
                self._delete_from_chroma(self._doc_id(event.src_path))

            if dest_is_md:
                self._schedule_reindex(event.dest_path)
        except Exception as exc:
            logger.error(
                "Watchdog: errore su MOVED '%s' → '%s': %s",
                event.src_path, event.dest_path, exc,
            )

    def cancel_all_timers(self) -> None:
        """Cancella tutti i timer pendenti (chiamato da Watchdog.stop())."""
        with self._timers_lock:
            for timer in self._timers.values():
                timer.cancel()
            count = len(self._timers)
            self._timers.clear()
        if count:
            logger.debug("Watchdog: %d timer pendenti cancellati", count)


# ---------------------------------------------------------------------------
# Interfaccia pubblica
# ---------------------------------------------------------------------------

class Watchdog:
    """
    Monitora vault/wiki/, vault/synthesis/ e vault/raw/ e re-indicizza i file modificati in tempo reale.

    Uso tipico:
        watcher = Watchdog()
        watcher.start()
        # ... runtime ...
        watcher.stop()
    """

    def __init__(self) -> None:
        from config import Config

        self._vault_path: Path = Config.vault_path
        self._wiki_dir: Path = self._vault_path / "wiki"
        self._synthesis_dir: Path = self._vault_path / "synthesis"
        self._raw_dir: Path = self._vault_path / "raw"
        self._handler = _WikiEventHandler(
            vault_path=self._vault_path,
            debounce_seconds=Config.vault_watch_debounce_seconds,
        )
        self._observer: Observer | None = None

    def start(self) -> None:
        """Avvia l'Observer watchdog. Crea vault/wiki/, vault/synthesis/ e vault/raw/ se non esistono."""
        self._wiki_dir.mkdir(parents=True, exist_ok=True)
        self._synthesis_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            path=str(self._wiki_dir),
            recursive=True,
        )
        self._observer.schedule(
            self._handler,
            path=str(self._synthesis_dir),
            recursive=True,
        )
        self._observer.schedule(
            self._handler,
            path=str(self._raw_dir),
            recursive=True,
        )
        self._observer.start()
        logger.info(
            "Watchdog: avviato — monitora '%s', '%s' e '%s' (debounce=%ds)",
            self._wiki_dir,
            self._synthesis_dir,
            self._raw_dir,
            self._handler._debounce_seconds,
        )

    def stop(self) -> None:
        """Cancella i timer pendenti e ferma l'Observer."""
        self._handler.cancel_all_timers()
        if self._observer is not None and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Watchdog: fermato")

    def is_alive(self) -> bool:
        """Restituisce True se l'Observer è attivo."""
        return self._observer is not None and self._observer.is_alive()


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import tempfile
    import time
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

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

    print("\n=== Watchdog unit tests ===\n")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        vault = Path(tmp)
        wiki = vault / "wiki"
        wiki.mkdir(parents=True)

        os.environ["VAULT_PATH"] = str(vault)
        os.environ["VAULT_WATCH_DEBOUNCE_SECONDS"] = "1"

        # --- _WikiEventHandler helpers ---
        synthesis = vault / "synthesis"
        synthesis.mkdir(parents=True, exist_ok=True)
        raw = vault / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        handler = _WikiEventHandler(vault_path=vault, debounce_seconds=1)

        check(
            "_is_indexed_md: .md in wiki/ → True",
            handler._is_indexed_md(str(wiki / "test.md")),
        )
        check(
            "_is_indexed_md: .md in synthesis/ → True",
            handler._is_indexed_md(str(synthesis / "2025-06-01-test.md")),
        )
        check(
            "_is_indexed_md: .txt in wiki/ → False",
            not handler._is_indexed_md(str(wiki / "test.txt")),
        )
        check(
            "_is_indexed_md: .md fuori directory indicizzate → False",
            not handler._is_indexed_md(str(vault / "raw" / "test.md")),
        )
        check(
            "_is_raw_binary: .pdf in raw/ → True",
            handler._is_raw_binary(str(raw / "documento.pdf")),
        )
        check(
            "_is_raw_binary: .docx in raw/ → True",
            handler._is_raw_binary(str(raw / "report.docx")),
        )
        check(
            "_is_raw_binary: .pdf in wiki/ → False",
            not handler._is_raw_binary(str(wiki / "doc.pdf")),
        )
        check(
            "_is_raw_binary: .md in raw/ → False",
            not handler._is_raw_binary(str(raw / "note.md")),
        )
        check(
            "_doc_id: path assoluto → relativo al vault",
            handler._doc_id(str(wiki / "llm.md")) == "wiki/llm.md",
        )

        # --- Debounce: schedule + reset ---
        fired: list[str] = []

        def _fake_reindex(path: str) -> None:
            fired.append(path)

        with patch.object(handler, "_reindex_callback", side_effect=_fake_reindex):
            test_path = str(wiki / "doc.md")
            handler._schedule_reindex(test_path)
            handler._schedule_reindex(test_path)  # reset timer
            handler._schedule_reindex(test_path)  # reset timer

            with handler._timers_lock:
                timer_count = len(handler._timers)
            check("debounce: un solo timer attivo per file", timer_count == 1)

            time.sleep(1.5)
            check("debounce: callback chiamato esattamente una volta", len(fired) == 1)

        # --- cancel_all_timers ---
        handler2 = _WikiEventHandler(vault_path=vault, debounce_seconds=60)
        handler2._schedule_reindex(str(wiki / "a.md"))
        handler2._schedule_reindex(str(wiki / "b.md"))
        with handler2._timers_lock:
            before = len(handler2._timers)
        handler2.cancel_all_timers()
        with handler2._timers_lock:
            after = len(handler2._timers)
        check("cancel_all_timers: tutti i timer rimossi", before == 2 and after == 0)

        # --- Watchdog start/stop/is_alive ---
        watcher = Watchdog()
        check("is_alive: False prima di start()", not watcher.is_alive())
        watcher.start()
        check("is_alive: True dopo start()", watcher.is_alive())
        watcher.stop()
        check("is_alive: False dopo stop()", not watcher.is_alive())

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
