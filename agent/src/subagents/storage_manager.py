"""
STORAGE_MANAGER — gestisce lettura, scrittura e aggiornamento di documenti nel vault.

Richiede inferenza LLM solo per operazioni di riorganizzazione.
Tutte le altre operazioni (read/write/update) sono hard-coded.
Tool disponibili: write_document, update_document, read_document.
"""
import logging

try:
    from src.subagents.base import BaseSubagent
except ImportError:
    from base import BaseSubagent  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class StorageManager(BaseSubagent):
    """
    Gestisce le operazioni CRUD sul vault in modo hard-coded (senza LLM).

    L'inferenza LLM è riservata all'azione "reorganize", delegata a run()
    della classe base. Per tutte le altre operazioni, usare i metodi diretti:
    store(), retrieve(), update().
    """

    tool_whitelist = ["read_document", "write_document", "update_document"]

    def __init__(self) -> None:
        super().__init__("storage_manager")

    async def store(self, path: str, content: str) -> dict:
        """
        Scrive un nuovo documento nel vault.

        Args:
            path:    Path relativo a VAULT_PATH o assoluto. Non sovrascrive.
            content: Contenuto testuale del documento .md.

        Returns:
            {"success": bool, "output": str}
        """
        try:
            from src.tools.write_document import write_document
        except ImportError:
            from write_document import write_document  # type: ignore[no-redef]

        try:
            await write_document(path, content)
            logger.info("StorageManager.store: '%s' scritto", path)
            return {"success": True, "output": f"documento scritto: {path}"}
        except FileExistsError as exc:
            logger.warning("StorageManager.store: file già esistente — %s", exc)
            return {"success": False, "output": str(exc)}
        except (ValueError, OSError) as exc:
            logger.error("StorageManager.store: errore su '%s' — %s", path, exc)
            return {"success": False, "output": str(exc)}

    async def retrieve(self, path: str) -> dict:
        """
        Legge un documento dal vault.

        Args:
            path: Path relativo a VAULT_PATH o assoluto.

        Returns:
            {"success": bool, "output": str}  — output è il contenuto del file.
        """
        try:
            from src.tools.read_document import read_document
        except ImportError:
            from read_document import read_document  # type: ignore[no-redef]

        try:
            content = await read_document(path)
            logger.info("StorageManager.retrieve: '%s' letto (%d car)", path, len(content))
            return {"success": True, "output": content}
        except FileNotFoundError as exc:
            logger.warning("StorageManager.retrieve: file non trovato — %s", exc)
            return {"success": False, "output": str(exc)}
        except (ValueError, OSError) as exc:
            logger.error("StorageManager.retrieve: errore su '%s' — %s", path, exc)
            return {"success": False, "output": str(exc)}

    async def update(
        self, path: str, content: str, section: str | None = None
    ) -> dict:
        """
        Aggiorna un documento esistente nel vault.

        Args:
            path:    Path relativo a VAULT_PATH o assoluto. Il file deve esistere.
            content: Testo da inserire o sostituire.
            section: Heading Markdown della sezione target (es. "## Note").
                     Se None, il contenuto viene aggiunto in append.

        Returns:
            {"success": bool, "output": str}
        """
        try:
            from src.tools.update_document import update_document
        except ImportError:
            from update_document import update_document  # type: ignore[no-redef]

        try:
            await update_document(path, content, section)
            logger.info(
                "StorageManager.update: '%s' aggiornato (sezione=%r)", path, section
            )
            return {"success": True, "output": f"documento aggiornato: {path}"}
        except FileNotFoundError as exc:
            logger.warning("StorageManager.update: file non trovato — %s", exc)
            return {"success": False, "output": str(exc)}
        except (ValueError, OSError) as exc:
            logger.error("StorageManager.update: errore su '%s' — %s", path, exc)
            return {"success": False, "output": str(exc)}


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

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

    async def _run_tests() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["VAULT_PATH"] = tmp
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

            vault = Path(tmp)
            (vault / "raw").mkdir()

            sm = StorageManager()

            print("\n=== StorageManager — metodi diretti ===\n")

            # store(): scrive un nuovo file
            r = await sm.store("raw/nota.md", "# Nota\nContenuto di prova.")
            check("store() success=True su file nuovo", r["success"] is True)
            check("store() file effettivamente scritto", (vault / "raw" / "nota.md").exists())

            # store(): file già esistente → FileExistsError
            r2 = await sm.store("raw/nota.md", "contenuto duplicato")
            check("store() success=False su file esistente", r2["success"] is False)

            # retrieve(): legge il file scritto
            r3 = await sm.retrieve("raw/nota.md")
            check("retrieve() success=True", r3["success"] is True)
            check("retrieve() restituisce il contenuto", "Contenuto di prova." in r3["output"])

            # retrieve(): file inesistente
            r4 = await sm.retrieve("raw/inesistente.md")
            check("retrieve() success=False su file mancante", r4["success"] is False)

            # update(): append
            r5 = await sm.update("raw/nota.md", "Riga aggiunta.")
            check("update() append success=True", r5["success"] is True)
            updated_text = (vault / "raw" / "nota.md").read_text(encoding="utf-8")
            check("update() append ha aggiunto il contenuto", "Riga aggiunta." in updated_text)
            check("update() contenuto originale preservato", "Contenuto di prova." in updated_text)

            # update(): sostituzione sezione
            (vault / "raw" / "sezioni.md").write_text(
                "# Doc\n\n## Dati\nVecchi dati.\n\n## Fine\nOk.\n", encoding="utf-8"
            )
            r6 = await sm.update("raw/sezioni.md", "Nuovi dati.\n", section="## Dati")
            check("update() sezione success=True", r6["success"] is True)
            sect_text = (vault / "raw" / "sezioni.md").read_text(encoding="utf-8")
            check("update() sezione ha sostituito il contenuto", "Nuovi dati." in sect_text)
            check("update() sezione adiacente intatta", "## Fine" in sect_text)

            # update(): file inesistente
            r7 = await sm.update("raw/fantasma.md", "nulla")
            check("update() success=False su file mancante", r7["success"] is False)

        print(f"\nRisultato: {passed} OK, {failed} FAIL")

    asyncio.run(_run_tests())
    sys.exit(0 if failed == 0 else 1)
