"""
read_document(path: str) -> str

Legge e restituisce il contenuto testuale di un documento .md dal vault.

Args:
    path: Path assoluto o relativo a VAULT_PATH.

Returns:
    Contenuto testuale del file come stringa UTF-8.

Raises:
    ValueError: se il file non ha estensione .md.
    FileNotFoundError: se il file non esiste al path risolto.
    OSError: per altri errori di I/O.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    from config import Config
    return Config.vault_path / p


async def read_document(path: str) -> str:
    """
    Legge e restituisce il contenuto testuale di un documento .md.

    Args:
        path: Path assoluto o relativo a VAULT_PATH.

    Returns:
        Contenuto testuale del file.

    Raises:
        ValueError: se l'estensione non è .md.
        FileNotFoundError: se il file non esiste.
        OSError: per altri errori di I/O.
    """
    resolved = _resolve_path(path)

    if resolved.suffix.lower() != ".md":
        raise ValueError(
            f"read_document supporta solo file .md, "
            f"ricevuto: '{resolved.suffix or '(nessuna estensione)'}' — path: {resolved}"
        )

    if not resolved.exists():
        raise FileNotFoundError(
            f"File non trovato: {resolved}"
        )

    try:
        content = resolved.read_text(encoding="utf-8")
        logger.info("read_document: letto '%s' (%d caratteri)", resolved, len(content))
        return content
    except OSError as exc:
        logger.error("read_document: errore I/O su '%s': %s", resolved, exc)
        raise


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import tempfile

    # Aggiunge la root del progetto al path per consentire 'from config import Config'
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def _run_tests() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Config è importato pigramente: impostiamo le env var prima della prima chiamata.
            os.environ["VAULT_PATH"] = tmp
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

            vault = Path(tmp)
            test_file = vault / "note.md"
            test_file.write_text("# Test\nContenuto di prova.", encoding="utf-8")

            # 1. Lettura corretta tramite path relativo a VAULT_PATH
            content = await read_document("note.md")
            assert "Contenuto di prova." in content, "Lettura fallita"
            print("[OK] lettura da path relativo")

            # 2. Lettura corretta tramite path assoluto
            content_abs = await read_document(str(test_file))
            assert content == content_abs, "Path assoluto e relativo danno risultati diversi"
            print("[OK] lettura da path assoluto")

            # 3. FileNotFoundError su file inesistente
            try:
                await read_document("inesistente.md")
                assert False, "Doveva sollevare FileNotFoundError"
            except FileNotFoundError as exc:
                assert "inesistente" in str(exc)
                print("[OK] FileNotFoundError su file mancante")

            # 4. ValueError su estensione non-.md
            wrong = vault / "data.txt"
            wrong.write_text("testo", encoding="utf-8")
            try:
                await read_document("data.txt")
                assert False, "Doveva sollevare ValueError"
            except ValueError as exc:
                assert ".txt" in str(exc)
                print("[OK] ValueError su estensione non-.md")

        print("\nTutti i test read_document passati.")

    asyncio.run(_run_tests())
