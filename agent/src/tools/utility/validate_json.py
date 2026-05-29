"""
validate_json — utility di sistema, non chiamata direttamente dai subagents.

Richiamata automaticamente su ogni output JSON generato dall'LLM prima che
venga usato dal sistema. Estrae il JSON da testo grezzo (che può contenere
prefissi, suffissi, markdown, ecc.), applica correzioni automatiche ai
problemi sintattici comuni e restituisce sempre un dict Python.

Funzioni esposte:
    validate_json(raw: str) -> dict
    extract_json_str(raw: str) -> str
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

# Regex precompilate per le correzioni
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_UNQUOTED_KEY_RE = re.compile(r"(?<=[{,])(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)")
_MARKDOWN_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")


# ---------------------------------------------------------------------------
# Correzione automatica degli errori sintattici comuni
# ---------------------------------------------------------------------------

def _fix_common_errors(s: str) -> tuple[str, list[str]]:
    """
    Applica correzioni sintattiche comuni al testo JSON.

    Returns:
        (testo_corretto, lista_descrizioni_correzioni_applicate)

    Correzioni (nell'ordine in cui vengono applicate):
    1. Trailing comma prima di } o ]
    2. Apici singoli come delimitatori (solo se non sono presenti apici doppi)
    3. Chiavi non quotate (es. {key: "v"} → {"key": "v"})
    """
    fixes: list[str] = []

    # 1. Trailing comma: {"a": 1,} → {"a": 1}
    if _TRAILING_COMMA_RE.search(s):
        s = _TRAILING_COMMA_RE.sub(r"\1", s)
        fixes.append("trailing comma rimossa prima di } o ]")

    # 2. Apici singoli → doppi (solo se i singoli sono gli unici delimitatori)
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
        fixes.append("apici singoli convertiti in apici doppi")

    # 3. Chiavi non quotate: {key: "v"} → {"key": "v"}
    # Limitazione nota: potrebbe falsamente correggere contenuti di stringhe
    # che contengono ", key: value". Accettabile per output LLM strutturati.
    if _UNQUOTED_KEY_RE.search(s):
        s = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', s)
        fixes.append("chiavi non quotate: aggiunte virgolette doppie")

    return s, fixes


# ---------------------------------------------------------------------------
# Estrazione geometrica (senza parsing)
# ---------------------------------------------------------------------------

def extract_json_str(raw: str) -> str:
    """
    Estrae la stringa JSON grezza dal testo senza effettuare il parsing.

    Strategie (in ordine):
    a. raw.strip() se inizia con { e finisce con } (o [ e ])
    b. Sottostringa dal primo { all'ultimo }
    c. Contenuto del primo blocco ```json ... ``` o ``` ... ```
    d. ValueError se nessuna struttura JSON è rilevabile

    Args:
        raw: Testo grezzo prodotto dall'LLM.

    Returns:
        Stringa JSON estratta, non corretta e non validata.

    Raises:
        ValueError: se nessuna struttura JSON è rilevabile nel testo.
    """
    stripped = raw.strip()

    # a. Il testo è già un oggetto/array JSON (dopo strip)
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        return stripped

    # b. Prima { ... ultima }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and start < end:
        return raw[start : end + 1]

    # c. Blocco markdown
    m = _MARKDOWN_BLOCK_RE.search(raw)
    if m:
        return m.group(1).strip()

    # d. Nessuna struttura trovata
    raise ValueError(
        f"Nessuna struttura JSON rilevabile nel testo "
        f"({len(raw)} caratteri): {raw[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Validazione e correzione principale
# ---------------------------------------------------------------------------

def validate_json(raw: str) -> dict:
    """
    Estrae, corregge e fa il parsing del JSON prodotto dall'LLM.

    Strategia di estrazione (in ordine, ognuna provata con e senza correzioni):
    a. raw.strip() direttamente
    b. Sottostringa dal primo { all'ultimo }
    c. Contenuto di blocchi ```json ... ``` o ``` ... ```

    Correzioni automatiche applicate se il parsing diretto fallisce:
    - Trailing comma prima di } o ]
    - Apici singoli usati come delimitatori di stringa
    - Chiavi non quotate (es. {key: "value"})

    Args:
        raw: Testo grezzo prodotto dall'LLM, può contenere prefissi, suffissi
             o formattazione markdown.

    Returns:
        dict Python estratto dal JSON.

    Raises:
        ValueError: se nessuna strategia produce un dict valido.
    """
    logger.debug("validate_json: input grezzo (%d car): %.300r", len(raw), raw)

    stripped = raw.strip()
    _seen: list[str] = []

    def _add_candidate(s: str, name: str) -> None:
        if s not in _seen:
            _seen.append(s)
            candidates.append((s, name))

    candidates: list[tuple[str, str]] = []

    # a. Diretto
    _add_candidate(stripped, "diretto")

    # b. Bracket extraction
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and start < end:
        _add_candidate(raw[start : end + 1], "bracket_extraction")

    # c. Markdown block
    m = _MARKDOWN_BLOCK_RE.search(raw)
    if m:
        _add_candidate(m.group(1).strip(), "markdown_block")

    last_type_error: str | None = None

    for candidate, strategy in candidates:
        # Tenta parsing diretto
        try:
            result = json.loads(candidate)
            if not isinstance(result, dict):
                last_type_error = type(result).__name__
            else:
                logger.debug(
                    "validate_json: parsing OK (strategia='%s'): %r", strategy, result
                )
                return result
        except json.JSONDecodeError:
            pass

        # Tenta con correzioni automatiche
        fixed, fixes = _fix_common_errors(candidate)
        if fixed == candidate:
            continue  # Nessuna correzione applicabile, inutile riprovare
        try:
            result = json.loads(fixed)
            if not isinstance(result, dict):
                last_type_error = type(result).__name__
            else:
                for fix in fixes:
                    logger.warning("validate_json: correzione applicata — %s", fix)
                logger.debug(
                    "validate_json: parsing OK con correzioni (strategia='%s'): %r",
                    strategy,
                    result,
                )
                return result
        except json.JSONDecodeError:
            pass

    if last_type_error is not None:
        raise ValueError(
            f"validate_json: il JSON estratto non è un oggetto (dict) "
            f"ma {last_type_error} — input: {raw[:300]!r}"
        )
    raise ValueError(
        f"Impossibile estrarre o correggere un JSON valido dal testo "
        f"({len(raw)} caratteri): {raw[:500]!r}"
    )


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

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

    def expect_raises(label: str, exc_type: type, fn, *args) -> None:
        global passed, failed
        try:
            fn(*args)
            print(f"[FAIL] {label} — nessuna eccezione sollevata")
            failed += 1
        except exc_type:
            print(f"[OK] {label}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {label} — eccezione sbagliata: {type(e).__name__}: {e}")
            failed += 1

    print("\n=== validate_json ===\n")

    # 1. JSON valido diretto
    result = validate_json('{"action": "RETRIEVE", "source": "vault"}')
    check("JSON valido diretto", result == {"action": "RETRIEVE", "source": "vault"})

    # 2. Testo prima e dopo il JSON
    result = validate_json(
        'Ecco il risultato: {"RETRIEVE": {"query": "ultime notizie"}} '
        "Ho pianificato questa azione."
    )
    check("testo prima e dopo", result == {"RETRIEVE": {"query": "ultime notizie"}})

    # 3. Blocco markdown ```json
    result = validate_json(
        "Il piano è il seguente:\n```json\n"
        '{"GENERATE": {"format": "briefing"}}\n'
        "```\nProcedi."
    )
    check("blocco markdown ```json", result == {"GENERATE": {"format": "briefing"}})

    # 4. Blocco markdown generico ```
    result = validate_json('```\n{"success": true, "output": "ok"}\n```')
    check("blocco markdown generico", result == {"success": True, "output": "ok"})

    # 5. Trailing comma
    result = validate_json('{"key": "value", "nested": [1, 2, 3,],}')
    check(
        "trailing comma (oggetto e array)",
        result == {"key": "value", "nested": [1, 2, 3]},
    )

    # 6. Apici singoli
    result = validate_json("{'action': 'ANALYZE', 'target': 'documento.md'}")
    check(
        "apici singoli",
        result == {"action": "ANALYZE", "target": "documento.md"},
    )

    # 7. Chiavi non quotate
    result = validate_json('{action: "STORE", path: "vault/wiki/note.md"}')
    check(
        "chiavi non quotate",
        result == {"action": "STORE", "path": "vault/wiki/note.md"},
    )

    # 8. Combinazione: trailing comma + apici singoli
    result = validate_json("{'success': true, 'output': 'fatto',}")
    check(
        "trailing comma + apici singoli combinati",
        result == {"success": True, "output": "fatto"},
    )

    # 9. JSON con whitespace abbondante e a capo
    result = validate_json(
        "\n\n  {\n    \"RETRIEVE\": {\"query\": \"notizie\"},\n    "
        "\"GENERATE\": {\"format\": \"summary\"}\n  }\n\n"
    )
    check(
        "whitespace abbondante e newline",
        result == {"RETRIEVE": {"query": "notizie"}, "GENERATE": {"format": "summary"}},
    )

    # 10. Input non recuperabile → ValueError
    expect_raises(
        "input non recuperabile solleva ValueError",
        ValueError,
        validate_json,
        "Questo è solo testo senza JSON.",
    )

    # 11. Input vuoto → ValueError
    expect_raises(
        "input vuoto solleva ValueError",
        ValueError,
        validate_json,
        "",
    )

    print("\n=== extract_json_str ===\n")

    # 12. Stringa già JSON
    s = extract_json_str('{"key": "value"}')
    check("estrazione diretta", s == '{"key": "value"}')

    # 13. Testo attorno
    s = extract_json_str('Risposta: {"ok": true} fine.')
    check("estrazione da testo", s == '{"ok": true}')

    # 14. Markdown block
    s = extract_json_str("```json\n{\"result\": 42}\n```")
    check("estrazione da markdown", s == '{"result": 42}')

    # 15. Nessuna struttura → ValueError
    expect_raises(
        "extract_json_str su testo senza JSON",
        ValueError,
        extract_json_str,
        "nessun json qui",
    )

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
