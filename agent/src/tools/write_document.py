"""
write_document(path: str, content: str) -> bool

Scrive content in un nuovo file .md nel vault. Non sovrascrive file esistenti.

Args:
    path: Path assoluto, relativo a VAULT_PATH, o solo nome file.
          Se il path non include una directory esplicita, il file viene scritto
          in vault/raw/.
    content: Contenuto da scrivere nel file.

Returns:
    True se la scrittura ha avuto successo.

Raises:
    ValueError: se il file non ha estensione .md.
    FileExistsError: se il file esiste già al path risolto.
    OSError: per errori di I/O (re-sollevata con contesto aggiuntivo).
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    from config import Config
    vault = Config.vault_path
    # Nessuna directory esplicita → landing zone vault/raw/
    if p.parent == Path("."):
        return vault / "raw" / p
    return vault / p


async def write_document(path: str, content: str) -> bool:
    """
    Scrive content in un nuovo file .md nel vault.

    Args:
        path: Path assoluto, relativo a VAULT_PATH, o solo nome file
              (in tal caso scritto in vault/raw/).
        content: Contenuto da scrivere.

    Returns:
        True se la scrittura ha avuto successo.

    Raises:
        ValueError: se l'estensione non è .md.
        FileExistsError: se il file esiste già.
        OSError: per altri errori di I/O.
    """
    resolved = _resolve_path(path)

    if resolved.suffix.lower() != ".md":
        raise ValueError(
            f"write_document supporta solo file .md, "
            f"ricevuto: '{resolved.suffix or '(nessuna estensione)'}' — path: {resolved}"
        )

    if resolved.exists():
        raise FileExistsError(
            f"Il file esiste già, write_document non sovrascrive: {resolved}"
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        logger.info("write_document: scritto '%s' (%d caratteri)", resolved, len(content))
        return True
    except OSError as exc:
        logger.error("write_document: errore I/O su '%s': %s", resolved, exc)
        raise OSError(f"Impossibile scrivere '{resolved}': {exc}") from exc


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def _run_tests() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["VAULT_PATH"] = tmp
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

            vault = Path(tmp)

            # 1. Solo nome file → atterraggio in vault/raw/
            result = await write_document("nuova_nota.md", "# Nuova nota\nContenuto.")
            assert result is True
            raw_file = vault / "raw" / "nuova_nota.md"
            assert raw_file.exists(), f"File non trovato in raw/: {raw_file}"
            assert raw_file.read_text(encoding="utf-8") == "# Nuova nota\nContenuto."
            print("[OK] scrittura in vault/raw/ da nome file")

            # 2. Path relativo con directory esplicita
            result2 = await write_document("wiki/argomento.md", "# Argomento")
            assert result2 is True
            assert (vault / "wiki" / "argomento.md").exists()
            print("[OK] scrittura in sottodirectory con path relativo")

            # 3. FileExistsError su file già esistente
            try:
                await write_document("nuova_nota.md", "Contenuto duplicato")
                assert False, "Doveva sollevare FileExistsError"
            except FileExistsError as exc:
                assert "nuova_nota" in str(exc)
                print("[OK] FileExistsError su file esistente")

            # 4. ValueError su estensione non-.md
            try:
                await write_document("documento.txt", "testo")
                assert False, "Doveva sollevare ValueError"
            except ValueError as exc:
                assert ".txt" in str(exc)
                print("[OK] ValueError su estensione non-.md")

            # 5. Path assoluto
            abs_path = str(vault / "assoluto.md")
            await write_document(abs_path, "# Assoluto")
            assert Path(abs_path).exists()
            print("[OK] scrittura da path assoluto")

        print("\nTutti i test write_document passati.")

    asyncio.run(_run_tests())
