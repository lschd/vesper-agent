"""Ottimizzatore di token per Vesper. Port deterministico di tokenjuice.

Basato su tokenjuice di @vincentkoc (https://github.com/vincentkoc/tokenjuice).
Riduce il volume del testo prima che venga inserito nel prompt LLM, usando
tecniche rule-driven senza dipendenze esterne: strip ANSI, deduplicazione,
troncatura head/tail, compattazione JSON, hash-clip.

Struttura:
  Layer 1: primitive testuali (port da tokenjuice/src/core/text.ts)
  Layer 2: utility JSON      (port da tokenjuice/src/core/reduce-utils.ts)
  Layer 3: regole Vesper     (adattato da tokenjuice/src/core/rules.ts)
  Layer 4: reduce_content()  (adattato da tokenjuice/src/core/reduce.ts)
  Layer 5: optimize_context_dict()  (integrazione Vesper-specifica)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti (da tokenjuice)
# ---------------------------------------------------------------------------

_TINY_MAX_CHARS = 240
_PASSTHROUGH_MIN_SAVED = 120
_PASSTHROUGH_MAX_RATIO = 0.85
_MIDDLE_MARKER = "\n... omitted ...\n"
_TAIL_SUFFIX = "\n... truncated ..."

# ---------------------------------------------------------------------------
# Layer 1 — Text primitives
# ---------------------------------------------------------------------------

_ANSI_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_CSI_INCOMPLETE = re.compile(r"\x1b\[[0-?]*[ -/]*$")
_ANSI_OSC_INCOMPLETE = re.compile(r"\x1b\][^\x07\x1b]*$")
_ANSI_SINGLE = re.compile(r"\x1b[@-_]")


def strip_ansi(text: str) -> str:
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_CSI.sub("", text)
    text = _ANSI_OSC_INCOMPLETE.sub("", text)
    text = _ANSI_CSI_INCOMPLETE.sub("", text)
    text = _ANSI_SINGLE.sub("", text)
    return text.replace("\x1b", "")


def normalize_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]


def trim_empty_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def dedupe_adjacent(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        if not result or result[-1] != line:
            result.append(line)
    return result


def head_tail(lines: list[str], head: int, tail: int) -> tuple[list[str], str | None]:
    """Mantieni i primi `head` e gli ultimi `tail` righe; ometti il middle.

    Returns (lines, compaction_kind) dove compaction_kind è None se non c'è stato taglio.
    """
    h = max(0, head)
    t = max(0, tail)
    if h == 0 and t == 0:
        return lines, None
    if len(lines) <= h + t:
        return lines, None
    omitted = len(lines) - h - t
    middle_marker = f"... {omitted} lines omitted ..."
    return [*lines[:h], middle_marker, *(lines[-t:] if t > 0 else [])], "head-tail-omission"


def clamp_text(text: str, max_chars: int) -> str:
    """Tronca dalla coda con marker."""
    if len(text) <= max_chars:
        return text
    body = max(0, max_chars - len(_TAIL_SUFFIX))
    head = text[:body]
    last_nl = head.rfind("\n")
    if last_nl != -1 and last_nl >= len(head) // 2:
        head = head[:last_nl]
    return head + _TAIL_SUFFIX


def clamp_text_middle(text: str, max_chars: int) -> str:
    """Mantieni 70% testa + 30% coda con marker centrale (come tokenjuice)."""
    if len(text) <= max_chars:
        return text
    marker_len = len(_MIDDLE_MARKER)
    body = max(0, max_chars - marker_len)
    head_chars = int(body * 0.7)
    tail_chars = max(0, body - head_chars)
    head = text[:head_chars]
    last_nl = head.rfind("\n")
    if last_nl != -1 and last_nl >= len(head) // 2:
        head = head[:last_nl]
    tail = text[-tail_chars:] if tail_chars else ""
    first_nl = tail.find("\n")
    if first_nl != -1 and first_nl <= len(tail) // 2:
        tail = tail[first_nl + 1:]
    return head + _MIDDLE_MARKER + tail


# ---------------------------------------------------------------------------
# Layer 2 — JSON utilities
# ---------------------------------------------------------------------------

def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def clip_middle_with_hash(text: str, max_chars: int) -> str:
    """Clip con hash SHA-256 della parte omessa (come tokenjuice)."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    marker = f" ...[{omitted} chars omitted, sha256:{_short_hash(text)}]... "
    body = max(0, max_chars - len(marker))
    head_chars = max(0, int(body * 0.55))
    tail_chars = max(0, body - head_chars)
    return text[:head_chars] + marker + text[-tail_chars:]


def _minify_json(text: str) -> str:
    """Minifica JSON rimuovendo whitespace fuori dalle stringhe."""
    text = text.strip()
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
            out.append(ch)
        elif ch in " \t\n\r":
            pass
        else:
            out.append(ch)
    return "".join(out)


def compact_json(text: str, max_chars: int) -> str | None:
    """Minifica JSON e applica hash-clip se necessario. None se non è JSON valido."""
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    minified = _minify_json(text)
    if len(minified) <= max_chars:
        return minified
    # usa hash-clip strict: marker conta nei max_chars
    omitted = len(minified) - max_chars
    marker = f"...[{omitted} chars omitted, sha256:{_short_hash(minified)}]..."
    marker_len = len(marker)
    if max_chars <= marker_len:
        return marker[:max_chars]
    body = max_chars - marker_len
    head_chars = int(body * 0.7)
    tail_chars = max(0, body - head_chars)
    return minified[:head_chars] + marker + (minified[-tail_chars:] if tail_chars else "")


# ---------------------------------------------------------------------------
# Layer 3 — Content rules per Vesper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContentRule:
    id: str
    head: int = 15
    tail: int = 10
    strip_ansi: bool = False
    trim_empty_edges: bool = True
    dedupe_adjacent: bool = True
    skip_patterns: tuple[str, ...] = field(default_factory=tuple)
    keep_patterns: tuple[str, ...] = field(default_factory=tuple)


_BUILTIN_RULES: dict[str, ContentRule] = {
    "json/data": ContentRule(
        id="json/data",
        head=0,
        tail=0,
    ),
    "vault/document": ContentRule(
        id="vault/document",
        head=30,
        tail=15,
        strip_ansi=False,
        trim_empty_edges=True,
        dedupe_adjacent=True,
        skip_patterns=("^---$",),
    ),
    "web/snippet": ContentRule(
        id="web/snippet",
        head=20,
        tail=8,
        strip_ansi=True,
        trim_empty_edges=True,
        dedupe_adjacent=True,
    ),
    "document/large": ContentRule(
        id="document/large",
        head=80,
        tail=40,
        strip_ansi=True,
        trim_empty_edges=True,
        dedupe_adjacent=False,
    ),
    "generic/text": ContentRule(
        id="generic/text",
        head=15,
        tail=10,
        strip_ansi=True,
        trim_empty_edges=True,
        dedupe_adjacent=True,
    ),
    "memory/capsule": ContentRule(
        id="memory/capsule",
        head=25,
        tail=12,
        strip_ansi=False,
        trim_empty_edges=True,
        dedupe_adjacent=True,
    ),
}

_FALLBACK_RULE = _BUILTIN_RULES["generic/text"]

BUILTIN_RULES = _BUILTIN_RULES


def _get_rule(content_type: str, rules: dict[str, ContentRule] | None = None) -> ContentRule:
    effective = {**_BUILTIN_RULES, **(rules or {})}
    return effective.get(content_type, _FALLBACK_RULE)


# ---------------------------------------------------------------------------
# Layer 4 — Core reducer
# ---------------------------------------------------------------------------

@dataclass
class ReduceResult:
    text: str
    raw_chars: int
    reduced_chars: int
    ratio: float
    compaction_kinds: list[str]

    @property
    def was_reduced(self) -> bool:
        return bool(self.compaction_kinds)


def reduce_content(text: str, content_type: str = "generic/text", max_chars: int = 4000, *, rules: dict[str, ContentRule] | None = None) -> ReduceResult:
    """Riduce `text` al budget `max_chars` usando regole content-type.

    Flusso (ispirato a tokenjuice/src/core/reduce.ts):
    1. Passthrough immediato se già piccolo
    2. Passthrough se il risparmio non vale la pena
    3. JSON compaction se content_type == "json/data" e nessuna rule custom
    4. Applica regola (transforms + head/tail)
    5. Clamp finale a max_chars
    """
    raw_chars = len(text)
    compaction_kinds: list[str] = []

    def _result(out: str, kinds: list[str]) -> ReduceResult:
        n = len(out)
        return ReduceResult(out, raw_chars, n, n / raw_chars if raw_chars else 1.0, kinds)

    # 1. già abbastanza piccolo
    if raw_chars <= max_chars:
        return _result(text, [])

    # 2. passthrough per output tiny (< 240 chars totali) — non vale ridurre
    if raw_chars <= _TINY_MAX_CHARS:
        return _result(text, [])

    # 3. JSON compaction — salta se il chiamante ha passato una rule custom per json/data
    if content_type == "json/data" and (rules is None or "json/data" not in rules):
        compacted = compact_json(text, max_chars)
        if compacted is not None:
            kinds = ["json-compact"] if len(compacted) < raw_chars else []
            return _result(compacted, kinds)
        # fallthrough a regola generica se non è JSON valido

    # 4. applica regola
    rule = _get_rule(content_type, rules)
    working = text

    if rule.strip_ansi:
        working = strip_ansi(working)

    lines = normalize_lines(working)

    if rule.trim_empty_edges:
        lines = trim_empty_edges(lines)

    if rule.dedupe_adjacent:
        lines = dedupe_adjacent(lines)

    if rule.skip_patterns:
        compiled_skip = [re.compile(p) for p in rule.skip_patterns]
        lines = [ln for ln in lines if not any(pat.search(ln) for pat in compiled_skip)]

    if rule.keep_patterns:
        compiled_keep = [re.compile(p) for p in rule.keep_patterns]
        kept = [ln for ln in lines if any(pat.search(ln) for pat in compiled_keep)]
        if kept:
            lines = kept

    lines, ht_kind = head_tail(lines, rule.head, rule.tail)
    if ht_kind:
        compaction_kinds.append(ht_kind)

    compact_text = "\n".join(lines).strip()

    # non vale la pena se risparmio < 120 chars o ratio > 0.85
    compact_len = len(compact_text)
    saved = raw_chars - compact_len
    ratio = compact_len / raw_chars if raw_chars else 1.0
    if saved < _PASSTHROUGH_MIN_SAVED or ratio > _PASSTHROUGH_MAX_RATIO:
        # torna alla versione solo con transforms ma senza head/tail
        working_no_ht = text
        if rule.strip_ansi:
            working_no_ht = strip_ansi(working_no_ht)
        compact_text = working_no_ht
        compaction_kinds = []

    # 5. clamp finale
    if len(compact_text) > max_chars:
        compact_text = clamp_text_middle(compact_text, max_chars)
        compaction_kinds.append("middle-truncation")

    return _result(compact_text, compaction_kinds)


# ---------------------------------------------------------------------------
# Layer 5 — Integrazione Vesper: optimize_context_dict
# ---------------------------------------------------------------------------

# Campi del context dict che non vanno mai toccati (brevi per definizione)
_SKIP_FIELDS = frozenset({"request", "instructions", "action", "format", "target_path"})


def _chars_to_budget(n_ctx: int) -> int:
    """Budget caratteri totale per il context (lascia 2000 token per output+overhead)."""
    return max(4000, (n_ctx - 2000) * 3)


def _reduce_item(item: object, content_type: str, per_budget: int) -> object:
    """Riduce un singolo item (str o dict con 'content') al budget."""
    if isinstance(item, str):
        r = reduce_content(item, content_type, per_budget)
        if r.was_reduced:
            logger.debug(
                "token_optimizer: %s %d → %d chars (-%.0f%%)",
                content_type, r.raw_chars, r.reduced_chars, (1 - r.ratio) * 100,
            )
        return r.text
    if isinstance(item, dict):
        content = item.get("content", "")
        if isinstance(content, str) and content:
            r = reduce_content(content, content_type, per_budget)
            if r.was_reduced:
                logger.debug(
                    "token_optimizer: %s %d → %d chars (-%.0f%%)",
                    content_type, r.raw_chars, r.reduced_chars, (1 - r.ratio) * 100,
                )
                return {**item, "content": r.text}
    return item


def optimize_context_dict(context: dict, n_ctx: int) -> dict:
    """Ottimizza i campi del context dict prima che vengano serializzati nel prompt.

    - Rispetta i campi in _SKIP_FIELDS (request, instructions, ecc.)
    - Riduce vault_results e web_results per item con le regole appropriate
    - Riduce document e campi stringa generici
    - Applica un secondo pass di troncatura uniforme se il totale supera il budget
    """
    if not context:
        return context

    budget = _chars_to_budget(n_ctx)

    try:
        initial_total = sum(len(json.dumps(v, ensure_ascii=False)) for v in context.values())
    except (TypeError, ValueError):
        initial_total = sum(len(str(v)) for v in context.values())

    logger.debug(
        "token_optimizer: budget=%d chars (n_ctx=%d), totale=%d chars, campi=%s",
        budget, n_ctx, initial_total, list(context.keys()),
    )

    result = dict(context)

    tool_results = result.get("tool_results")
    if isinstance(tool_results, dict):
        tr = dict(tool_results)

        vault = tr.get("vault_results")
        if isinstance(vault, list) and vault:
            per = max(500, budget // max(1, len(vault)))
            tr["vault_results"] = [_reduce_item(item, "vault/document", per) for item in vault]

        web = tr.get("web_results")
        if isinstance(web, list) and web:
            per = max(500, budget // max(1, len(web)))
            tr["web_results"] = [_reduce_item(item, "web/snippet", per) for item in web]

        doc = tr.get("document")
        if isinstance(doc, str) and doc:
            r = reduce_content(doc, "document/large", budget)
            if r.was_reduced:
                logger.debug(
                    "token_optimizer: document/large %d → %d chars (-%.0f%%)",
                    r.raw_chars, r.reduced_chars, (1 - r.ratio) * 100,
                )
                tr["document"] = r.text
        elif isinstance(doc, dict):
            tr["document"] = _reduce_item(doc, "document/large", budget)

        result["tool_results"] = tr

    # Riduzione capsule di memoria di sessione (priorità bassa: budget dimezzato rispetto a vault)
    ctx_mem = result.get("context_memory")
    if isinstance(ctx_mem, list) and ctx_mem:
        per = max(300, budget // max(1, len(ctx_mem) * 2))
        result["context_memory"] = [_reduce_item(item, "memory/capsule", per) for item in ctx_mem]

    # Riduzione generica degli altri campi stringa non in skip list
    for key, val in result.items():
        if key in _SKIP_FIELDS or key == "tool_results":
            continue
        if isinstance(val, str) and len(val) > budget:
            # tenta JSON, poi generic
            ctype = "json/data" if val.strip().startswith(("{", "[")) else "generic/text"
            r = reduce_content(val, ctype, budget)
            if r.was_reduced:
                logger.debug(
                    "token_optimizer: campo '%s' (%s) %d → %d chars (-%.0f%%)",
                    key, ctype, r.raw_chars, r.reduced_chars, (1 - r.ratio) * 100,
                )
                result[key] = r.text

    # Second pass: se il totale serializzato supera ancora il budget, tronca uniformemente
    try:
        total = sum(len(json.dumps(v, ensure_ascii=False)) for v in result.values())
    except (TypeError, ValueError):
        total = sum(len(str(v)) for v in result.values())

    if total > budget:
        # Secondo pass: context_memory prima (priorità bassa), poi vault/web
        ctx_mem2 = result.get("context_memory")
        if isinstance(ctx_mem2, list) and ctx_mem2:
            per_mem = max(150, budget // max(1, len(ctx_mem2) * 3))
            logger.debug(
                "token_optimizer: secondo pass context_memory — totale=%d > budget=%d, per_item=%d chars",
                total, budget, per_mem,
            )
            result["context_memory"] = [_reduce_item(i, "memory/capsule", per_mem) for i in ctx_mem2]

        tr2 = result.get("tool_results")
        if isinstance(tr2, dict):
            all_items = [
                *tr2.get("vault_results", []),
                *tr2.get("web_results", []),
            ]
            if all_items:
                per = max(200, budget // len(all_items))
                logger.debug(
                    "token_optimizer: secondo pass — totale=%d > budget=%d, per_item=%d chars",
                    total, budget, per,
                )
                tr2 = dict(tr2)
                tr2["vault_results"] = [_reduce_item(i, "vault/document", per) for i in tr2.get("vault_results", [])]
                tr2["web_results"] = [_reduce_item(i, "web/snippet", per) for i in tr2.get("web_results", [])]
                result["tool_results"] = tr2

    try:
        final_total = sum(len(json.dumps(v, ensure_ascii=False)) for v in result.values())
    except (TypeError, ValueError):
        final_total = sum(len(str(v)) for v in result.values())

    if final_total < initial_total:
        logger.debug(
            "token_optimizer: context %d → %d chars (-%.0f%%)",
            initial_total, final_total, (1 - final_total / initial_total) * 100,
        )

    return result


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

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

    print("=== Layer 1: text primitives ===\n")

    check("strip_ansi rimuove CSI", strip_ansi("\x1b[31mrosso\x1b[0m") == "rosso")
    check("strip_ansi testo pulito invariato", strip_ansi("ciao") == "ciao")
    check("dedupe_adjacent rimuove duplicati consecutivi", dedupe_adjacent(["a", "a", "b", "b", "b", "c"]) == ["a", "b", "c"])
    check("dedupe_adjacent preserva non-consecutivi", dedupe_adjacent(["a", "b", "a"]) == ["a", "b", "a"])
    check("trim_empty_edges rimuove bordi vuoti", trim_empty_edges(["", "  ", "a", "b", "", "  "]) == ["a", "b"])
    lines, kind = head_tail(["a", "b", "c", "d", "e", "f"], 2, 2)
    check("head_tail tronca il middle", kind == "head-tail-omission" and "omitted" in lines[2])
    check("head_tail preserva head e tail", lines[0] == "a" and lines[-1] == "f")
    lines_short, kind_short = head_tail(["a", "b"], 2, 2)
    check("head_tail no-op se abbastanza corto", kind_short is None and lines_short == ["a", "b"])
    check("clamp_text tronca dalla coda", "truncated" in clamp_text("a" * 500, 100))
    check("clamp_text_middle usa marker centrale", "omitted" in clamp_text_middle("a" * 200 + "\n" + "b" * 200, 100))
    check("clamp_text_middle passthrough se già corto", clamp_text_middle("ok", 100) == "ok")

    print("\n=== Layer 2: JSON utilities ===\n")

    json_text = '{"a": 1, "b": [1, 2, 3]}'
    compacted = compact_json(json_text, 1000)
    check("compact_json valido restituisce stringa", compacted is not None)
    check("compact_json rimuove whitespace", compacted is not None and " " not in compacted)
    check("compact_json non-JSON restituisce None", compact_json("ciao mondo", 1000) is None)
    big_json = json.dumps({"k": "v" * 1000})
    clipped = compact_json(big_json, 50)
    check("compact_json con clip non supera max_chars", clipped is not None and len(clipped) <= 50)
    check("clip_middle_with_hash rispetta max_chars", len(clip_middle_with_hash("x" * 500, 100)) <= 100)

    print("\n=== Layer 4: reduce_content ===\n")

    short_text = "testo breve"
    r = reduce_content(short_text, "generic/text", 4000)
    check("passthrough per testo breve", r.text == short_text and not r.was_reduced)

    long_text = "\n".join([f"Riga numero {i}: contenuto di esempio per test di troncatura." for i in range(100)])
    r2 = reduce_content(long_text, "generic/text", 500)
    check("reduce_content riduce testo lungo", r2.reduced_chars < r2.raw_chars)
    check("reduce_content ha compaction_kinds", r2.was_reduced)

    json_val = json.dumps({"key": "valore", "lista": list(range(500))})
    r3 = reduce_content(json_val, "json/data", 200)
    check("reduce_content compatta JSON", r3.reduced_chars <= 210)

    vault_text = "---\ntitle: Test\n---\n\n" + "\n".join([f"Paragrafo {i}: testo di esempio." for i in range(60)])
    r4 = reduce_content(vault_text, "vault/document", 500)
    check("vault/document riduce e salta frontmatter", r4.was_reduced)
    check("vault/document non contiene '---' del frontmatter", "---\n" not in r4.text[:10])

    print("\n=== Layer 5: optimize_context_dict ===\n")

    ctx = {
        "request": "domanda utente",
        "tool_results": {
            "vault_results": [
                {"id": "doc1", "content": "\n".join([f"Riga {i} del documento vault." for i in range(80)])},
                {"id": "doc2", "content": "\n".join([f"Riga {i} del secondo documento." for i in range(80)])},
            ],
            "web_results": [
                {"title": "Pagina web", "content": "\n".join([f"Snippet web {i}." for i in range(60)])},
            ],
        },
        "instructions": "genera una risposta",
    }
    optimized = optimize_context_dict(ctx, n_ctx=2048)
    check("optimize preserva 'request'", optimized["request"] == "domanda utente")
    check("optimize preserva 'instructions'", optimized["instructions"] == "genera una risposta")
    tr_opt = optimized["tool_results"]
    v1_content = tr_opt["vault_results"][0]["content"]
    check("vault_results ridotti", len(v1_content) < len(ctx["tool_results"]["vault_results"][0]["content"]))
    check("optimize context dict con n_ctx piccolo non crasha", isinstance(optimize_context_dict({}, 4096), dict))

    print("\n=== Extra: regression tests ===\n")

    lines_tail0, _ = head_tail(["a", "b", "c", "d"], 2, 0)
    check("head_tail tail=0 does not append the whole input as tail", lines_tail0 == ["a", "b", "... 2 lines omitted ..."])
    check("clip_middle_with_hash respects max_chars on tight budget", len(clip_middle_with_hash("x" * 500, 50)) <= 50)
    check("BUILTIN_RULES contains json/data", "json/data" in _BUILTIN_RULES)
    
    _big_json = json.dumps({"items": list(range(500))})
    r_json_override = reduce_content(_big_json, "json/data", 100, rules={"json/data": ContentRule(id="json/data", head=2, tail=1)})
    check("json/data custom rule override is not bypassed by the fast path", "json-compact" not in r_json_override.compaction_kinds)


    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
