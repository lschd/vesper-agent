"""
update_document(path: str, content: str, section: str | None = None) -> bool

Aggiorna un documento .md nel vault. Se il file non esiste, lo crea.

Args:
    path: Path assoluto o relativo a VAULT_PATH.
    content: Testo da inserire o con cui sostituire la sezione.
    section: Heading Markdown della sezione da aggiornare (es. "## Note").
             Se None, content viene aggiunto in append preceduto da una riga vuota.
             Se il heading non viene trovato, la sezione viene aggiunta in fondo.
             Se il file non esiste e path è un bare filename, viene creato in vault/raw/.

Returns:
    True se l'operazione ha avuto successo.

Raises:
    FileNotFoundError: se il file non esiste e path contiene una directory esplicita.
    OSError: per errori di I/O.
"""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    from config import Config
    return Config.vault_path / p


def _apply_section_update(text: str, content: str, section: str) -> str:
    """
    Funzione pura: applica la sostituzione di una sezione al testo del documento.

    Cerca il heading esatto (es. "## Note"), sostituisce il corpo della sezione
    fino al prossimo heading di livello uguale o superiore (≤ # del target).
    Se il heading non esiste, lo aggiunge in fondo.
    """
    heading_match = re.match(r"^(#{1,6})\s+", section.strip())
    if not heading_match:
        # section non è un heading Markdown valido: append grezzo
        sep = "\n" if not text.endswith("\n") else ""
        body = content if content.endswith("\n") else content + "\n"
        return text + sep + "\n" + section.strip() + "\n" + body

    target_level = len(heading_match.group(1))
    target_heading = section.strip()

    lines = text.splitlines(keepends=True)

    # Cerca il heading nel documento (confronto esatto dopo strip)
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if line.rstrip("\r\n") == target_heading:
            start_idx = i
            break

    if start_idx is None:
        # Sezione non trovata: aggiunge il heading e il contenuto in fondo
        sep = "\n" if not text.endswith("\n") else ""
        body = content if content.endswith("\n") else content + "\n"
        return text + sep + "\n" + target_heading + "\n" + body

    # Trova la fine della sezione: prossimo heading con livello ≤ target
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        m = re.match(r"^(#{1,6})\s+", lines[i])
        if m and len(m.group(1)) <= target_level:
            end_idx = i
            break

    # Ricostruisce: heading invariato, corpo sostituito, resto invariato
    body = content if content.endswith("\n") else content + "\n"
    return "".join(lines[: start_idx + 1] + [body] + lines[end_idx:])


async def update_document(
    path: str, content: str, section: str | None = None
) -> bool:
    """
    Aggiorna un documento .md nel vault. Se il file non esiste, lo crea.

    Args:
        path: Path assoluto o relativo a VAULT_PATH.
        content: Testo da inserire.
        section: Heading Markdown della sezione target (es. "## Note").
                 Se None, aggiunge content in append preceduto da una riga vuota.
                 Se il file non esiste, section diventa il primo heading del nuovo file.

    Returns:
        True se l'operazione ha avuto successo.

    Raises:
        OSError: per errori di I/O.
    """
    try:
        from src.tools.read_document import read_document
        from src.tools.write_document import write_document
    except ImportError:
        from read_document import read_document  # type: ignore[no-redef]
        from write_document import write_document  # type: ignore[no-redef]

    resolved = _resolve_path(path)

    if not resolved.exists():
        if Path(path).parent != Path("."):
            raise FileNotFoundError(
                f"update_document richiede un file esistente, non trovato: {resolved}"
            )
        # bare filename → write_document instrada in vault/raw/
        if section is None:
            initial = content
        else:
            body = content if content.endswith("\n") else content + "\n"
            initial = section.strip() + "\n" + body
        await write_document(path, initial)
        logger.info(
            "update_document: creato in raw/ '%s' (sezione=%r, %d caratteri)",
            Path(path).name,
            section,
            len(initial),
        )
        return True

    try:
        existing = await read_document(str(resolved.resolve()))

        if section is None:
            sep = "\n" if not existing.endswith("\n") else ""
            updated = existing + sep + "\n" + content
        else:
            updated = _apply_section_update(existing, content, section)

        resolved.write_text(updated, encoding="utf-8")
        logger.info(
            "update_document: aggiornato '%s' (sezione=%r, %d→%d caratteri)",
            resolved,
            section,
            len(existing),
            len(updated),
        )
        return True
    except OSError as exc:
        logger.error("update_document: errore I/O su '%s': %s", resolved, exc)
        raise


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

    def _check(label: str, condition: bool) -> None:
        assert condition, f"FAIL: {label}"
        print(f"[OK] {label}")

    async def _run_tests() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["VAULT_PATH"] = tmp
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

            vault = Path(tmp)

            # --- Funzione pura: _apply_section_update ---

            base = "# Titolo\n\n## Note\nVecchie note.\n\n## Conclusione\nFine.\n"

            # Sostituzione sezione esistente
            result = _apply_section_update(base, "Nuove note.\n", "## Note")
            _check("sostituzione sezione esistente", "Nuove note." in result)
            _check("heading conservato", "## Note\n" in result)
            _check("sezione successiva intatta", "## Conclusione" in result)
            _check("vecchio contenuto rimosso", "Vecchie note." not in result)

            # Sezione non trovata → aggiunta in fondo
            result2 = _apply_section_update(base, "Nuova sezione.\n", "## Extra")
            _check("sezione non trovata aggiunta in fondo", "## Extra" in result2)
            _check("contenuto originale preservato", "Vecchie note." in result2)

            # Heading di livello superiore chiude la sezione corrente
            nested = "# Top\n## A\nContenuto A.\n### A.1\nSotto A.\n## B\nContenuto B.\n"
            result3 = _apply_section_update(nested, "Nuovo A.\n", "## A")
            _check("heading superiore chiude la sezione", "Nuovo A." in result3)
            _check("### A.1 rimossa con il corpo", "A.1" not in result3)
            _check("## B preservato", "## B" in result3)

            # Append senza sezione (testato indirettamente via update_document)
            doc = vault / "documento.md"
            doc.write_text("# Doc\nContenuto iniziale.", encoding="utf-8")

            # --- update_document: append ---
            await update_document("documento.md", "Nuova appendice.")
            updated_text = doc.read_text(encoding="utf-8")
            _check("append aggiunge il contenuto", "Nuova appendice." in updated_text)
            _check("contenuto originale preservato nell'append", "Contenuto iniziale." in updated_text)

            # --- update_document: sostituzione sezione ---
            sectioned = vault / "sezioni.md"
            sectioned.write_text(
                "# Documento\n\n## Dati\nDati vecchi.\n\n## Fine\nContenuto finale.\n",
                encoding="utf-8",
            )
            await update_document("sezioni.md", "Dati nuovi.\n", section="## Dati")
            text_after = sectioned.read_text(encoding="utf-8")
            _check("sostituzione sezione via update_document", "Dati nuovi." in text_after)
            _check("sezione adiacente intatta", "## Fine" in text_after)

            # --- update_document: bare filename → creato in vault/raw/ ---
            raw_dir = vault / "raw"
            raw_dir.mkdir(exist_ok=True)
            await update_document("nuovo.md", "Contenuto creato.")
            _check("bare filename creato in raw/", (raw_dir / "nuovo.md").exists())
            _check("contenuto scritto nel file nuovo", "Contenuto creato." in (raw_dir / "nuovo.md").read_text())

            # --- update_document: path con directory → FileNotFoundError ---
            try:
                await update_document("ricerche/inesistente.md", "Contenuto")
                assert False, "Doveva sollevare FileNotFoundError"
            except FileNotFoundError as exc:
                _check("FileNotFoundError per path con directory mancante", "inesistente" in str(exc))

        print("\nTutti i test update_document passati.")

    asyncio.run(_run_tests())
