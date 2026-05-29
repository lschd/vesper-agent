"""
LLMClient — client asincrono per llama-server (API OpenAI-compatibile).

Funzioni esposte:
    get_llm_client() -> LLMClient   — singleton globale

Classe:
    LLMClient
        async complete(system_prompt, user_message, temperature, max_tokens) -> dict
        async load_prompt(prompt_name) -> str
"""
import logging
import re
import time
from dataclasses import dataclass, field

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

logger = logging.getLogger(__name__)


def _read_n_ctx_from_gguf(model_path: "Path") -> "int | None":
    """Legge context_length dall'header GGUF. Restituisce None su qualsiasi errore."""
    import struct
    from pathlib import Path as _Path

    path = _Path(model_path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.read(4)   # version uint32
            f.read(8)   # tensor_count uint64
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def _read_str() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", errors="replace")

            _FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

            def _skip(vtype: int) -> None:
                if vtype in _FIXED:
                    f.read(_FIXED[vtype])
                elif vtype == 8:
                    _read_str()
                elif vtype == 9:
                    atype = struct.unpack("<I", f.read(4))[0]
                    alen = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(alen):
                        _skip(atype)

            for _ in range(kv_count):
                key = _read_str()
                vtype = struct.unpack("<I", f.read(4))[0]
                if key.endswith(".context_length") and vtype in _FIXED:
                    raw = f.read(_FIXED[vtype])
                    return int.from_bytes(raw, "little")
                _skip(vtype)
    except Exception:
        pass
    return None

_llm_client: "LLMClient | None" = None


@dataclass
class _Stats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_sec: float = 0.0


class LLMClient:
    """
    Client asincrono per llama-server con API OpenAI-compatibile.

    Il client è progettato per essere usato come singleton (get_llm_client()).
    Non condivide stato tra richieste: ogni chiamata a complete() è indipendente.

    Attributes:
        _base_url:      URL base del server LLM (es. "http://localhost:8000/v1").
        _model_name:    Nome del modello (ignorato dal server, incluso per conformità API).
        _http:          httpx.AsyncClient condiviso per connection pooling.
        _prompt_cache:  Cache in-memory per i prompt letti dal vault.
    """

    def __init__(self) -> None:
        from config import Config

        if Config.llm_managed:
            self._base_url = f"http://localhost:{Config.llm_port}/v1"
        else:
            self._base_url = Config.llm_base_url.rstrip("/")
        self._model_name: str = Config.llm_model_dir
        # connect/write corti per fail-fast se il server è irraggiungibile;
        # read=None perché la generazione LLM può durare minuti.
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        )
        self._prompt_cache: dict[str, str] = {}
        self._stats = _Stats()
        self._n_ctx: int = Config.llm_n_ctx if Config.llm_n_ctx is not None else 8192

    @property
    def n_ctx(self) -> int:
        """Contesto effettivo in token (aggiornato da fetch_n_ctx dopo l'avvio del server)."""
        return self._n_ctx

    async def fetch_n_ctx(self) -> int:
        """
        Legge n_ctx effettivo da GET /props del server llama.cpp.

        Interroga sempre il server per ottenere il valore reale del context window.
        Se Config.llm_n_ctx è impostato dall'utente, viene applicato come cap massimo
        (min(fetched, cap)) — così si può limitare il contesto ma non si può mai
        superare il limite reale del server, evitando errori 400 context_length_exceeded.

        Returns:
            Il valore n_ctx effettivo che sarà usato per il budget dei prompt.
        """
        from config import Config

        # Prova prima /v1/props?model=... poi /props (llama.cpp nativo classico).
        base_v1   = self._base_url.rstrip("/")
        base_root = base_v1.removesuffix("/v1")
        _candidates = [
            f"{base_v1}/props?model={self._model_name}",
            f"{base_root}/props",
        ]

        data: dict | None = None
        matched_url: str = ""
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)) as c:
            for url in _candidates:
                try:
                    resp = await c.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        matched_url = url
                        break
                except Exception:
                    continue

        if data is not None:
            gen = data.get("default_generation_settings", {})
            fetched: int | None = gen.get("n_ctx") or data.get("n_ctx")
            if isinstance(fetched, int) and fetched > 0:
                if Config.llm_n_ctx is not None:
                    self._n_ctx = min(fetched, Config.llm_n_ctx)
                    logger.info(
                        "llm_client: n_ctx=%d (server=%d, cap LLM_N_CTX=%d)",
                        self._n_ctx, fetched, Config.llm_n_ctx,
                    )
                else:
                    self._n_ctx = fetched
                    logger.info("llm_client: n_ctx=%d rilevato da %s", self._n_ctx, matched_url)
                return self._n_ctx
            logger.warning("llm_client: /props non contiene n_ctx valido — uso %d", self._n_ctx)
            return self._n_ctx

        # /props non raggiungibile: usa LLM_N_CTX se impostato, altrimenti leggi dal GGUF.
        if Config.llm_n_ctx is not None:
            self._n_ctx = Config.llm_n_ctx
            logger.info("llm_client: n_ctx=%d da LLM_N_CTX (/props non raggiungibile)", self._n_ctx)
            return self._n_ctx

        # Non leggiamo n_ctx dal GGUF: GGUF.context_length è il contesto di training
        # del modello (es. 32768 per Qwen3), non il --ctx-size configurato sul server.
        # Usarlo come budget farebbe saltare la troncatura e causerebbe 400 context_length_exceeded.
        logger.warning(
            "llm_client: /props non raggiungibile e LLM_N_CTX non impostato — "
            "uso n_ctx=%d (default). Imposta LLM_N_CTX nel .env per il valore corretto.",
            self._n_ctx,
        )
        return self._n_ctx

    def consume_stats(self) -> dict | None:
        """Restituisce le statistiche accumulate dall'ultima chiamata e le azzera.

        Returns None se non ci sono state chiamate dall'ultimo consume.
        """
        if self._stats.calls == 0:
            return None
        tok_per_sec = (
            round(self._stats.completion_tokens / self._stats.elapsed_sec, 1)
            if self._stats.elapsed_sec > 0 else 0.0
        )
        result = {
            "calls": self._stats.calls,
            "prompt_tokens": self._stats.prompt_tokens,
            "completion_tokens": self._stats.completion_tokens,
            "elapsed_sec": round(self._stats.elapsed_sec, 1),
            "tokens_per_sec": tok_per_sec,
        }
        self._stats = _Stats()
        return result

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        thinking: bool = False,
    ) -> dict:
        """
        Invia una richiesta chat/completions al server LLM e restituisce il JSON estratto.

        Args:
            system_prompt:  Istruzioni di sistema per il modello.
            user_message:   Messaggio utente (tipicamente il piano o la query).
            temperature:    Parametro di campionamento (0.0 = deterministico).
            max_tokens:     Limite di token nella risposta (-1 = nessun limite, usa n_ctx disponibile).
            top_p:          Nucleus sampling threshold.
            repeat_penalty: Penalità per la ripetizione di token.
            thinking:       Se True, prepone /think (abilita CoT di Qwen3);
                            se False, prepone /no_think (disabilita i blocchi <think>).

        Returns:
            dict estratto dal contenuto JSON della risposta del modello.

        Raises:
            ValueError:  se la risposta non contiene JSON valido.
            httpx.HTTPError: per errori di rete o HTTP non-2xx.
        """
        # /think o /no_think controllano il thinking mode di Qwen3 a livello di
        # tokenizer — più affidabile di chat_template_kwargs che il server Python
        # di llama-cpp-python ignora silenziosamente.
        think_prefix = "/think" if thinking else "/no_think"
        payload: dict = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{think_prefix}\n{user_message}"},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }
        # max_tokens <= 0 significa "nessun limite": omettere il campo è più
        # compatibile della spec OpenAI rispetto a inviare -1, che versioni
        # recenti di llama-server rifiutano con 400 Bad Request.
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        # repeat_penalty non è un campo standard OpenAI; lo includiamo solo quando
        # non è il valore neutro (1.0) per ridurre il rischio di 400 su server strict.
        if repeat_penalty != 1.0:
            payload["repeat_penalty"] = repeat_penalty

        t0 = time.perf_counter()
        try:
            resp = await self._http.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            )
            if not resp.is_success:
                logger.error(
                    "llm_client: HTTP %d — body: %.500s\npayload keys: %s",
                    resp.status_code, resp.text, list(payload.keys()),
                )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("llm_client: errore HTTP — %s: %s", type(exc).__name__, exc)
            raise
        elapsed = time.perf_counter() - t0

        data = resp.json()
        raw_content: str = data["choices"][0]["message"]["content"]
        # Rimuove blocchi <think>...</think> completi
        raw_content = _THINK_RE.sub("", raw_content)
        # Rimuove contenuto di thinking orfano: llama-cpp consuma <think> come special token
        # ma lascia passare </think>, producendo "reasoning text</think>\n\nJSON"
        if "</think>" in raw_content:
            raw_content = raw_content[raw_content.rfind("</think>") + len("</think>"):]
        raw_content = raw_content.strip()

        usage = data.get("usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        tok_per_sec = round(c_tok / elapsed, 1) if elapsed > 0 else 0.0
        logger.info(
            "llm_client: prompt=%d compl=%d t → %.1fs → %.1f tok/s",
            p_tok, c_tok, elapsed, tok_per_sec,
        )

        self._stats.calls += 1
        self._stats.prompt_tokens += p_tok
        self._stats.completion_tokens += c_tok
        self._stats.elapsed_sec += elapsed

        from src.tools.utility.validate_json import validate_json

        try:
            result = validate_json(raw_content)
        except ValueError as exc:
            logger.error(
                "llm_client: validate_json fallita — %s\nContenuto grezzo: %.500r",
                exc,
                raw_content,
            )
            raise

        logger.debug("llm_client: complete OK — %d chiavi nel dict", len(result))
        return result

    async def load_prompt(self, prompt_name: str) -> str:
        """
        Carica un prompt da src/prompts/{prompt_name}.md, con cache in-memory per sessione.

        Args:
            prompt_name: Nome del file senza estensione (es. "orchestrator").

        Returns:
            Contenuto del file .md come stringa.

        Raises:
            FileNotFoundError: se il file non esiste in src/prompts/.
        """
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]

        from pathlib import Path
        prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / f"{prompt_name}.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt non trovato: {prompt_path}")

        content = prompt_path.read_text(encoding="utf-8")
        self._prompt_cache[prompt_name] = content
        logger.info("llm_client: prompt '%s' caricato (%d caratteri)", prompt_name, len(content))
        return content


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def get_llm_client() -> LLMClient:
    """
    Restituisce il singleton LLMClient, creandolo al primo accesso.

    Thread-safety: sufficiente per un singolo event loop asyncio.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
        logger.info(
            "llm_client: singleton creato — model=%r base_url=%r",
            _llm_client._model_name,
            _llm_client._base_url,
        )
    return _llm_client


# ---------------------------------------------------------------------------
# Test minimale (richiede llama-server in esecuzione o mock)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

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

    async def _run_tests() -> None:
        import os

        os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
        os.environ.setdefault("VAULT_PATH", str(Path(__file__).resolve().parent.parent.parent / "vault"))
        os.environ.setdefault("LLM_BASE_URL", "http://localhost:8000/v1")
        os.environ.setdefault("LLM_MODEL_NAME", "local-model")

        print("\n=== LLMClient — test con mock HTTP ===\n")

        # --- Test 1: complete() con risposta JSON valida ---
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"action": "RETRIEVE", "query": "test"}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        client = LLMClient()

        with patch.object(client._http, "post", new=AsyncMock(return_value=mock_response)):
            result = await client.complete(
                system_prompt="Sei un orchestratore.",
                user_message="Pianifica la ricerca.",
            )
        check("complete() con JSON valido", result == {"action": "RETRIEVE", "query": "test"})

        # --- Test 2: complete() con JSON in blocco markdown ---
        mock_response2 = MagicMock()
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": "```json\n{\"step\": \"GENERATE\"}\n```"}}]
        }
        mock_response2.raise_for_status = MagicMock()

        with patch.object(client._http, "post", new=AsyncMock(return_value=mock_response2)):
            result2 = await client.complete("sys", "user")
        check("complete() con JSON in blocco markdown", result2 == {"step": "GENERATE"})

        # --- Test 3: complete() con risposta non-JSON → ValueError ---
        mock_response3 = MagicMock()
        mock_response3.json.return_value = {
            "choices": [{"message": {"content": "Risposta in testo libero senza JSON."}}]
        }
        mock_response3.raise_for_status = MagicMock()

        raised = False
        try:
            with patch.object(client._http, "post", new=AsyncMock(return_value=mock_response3)):
                await client.complete("sys", "user")
        except ValueError:
            raised = True
        check("complete() solleva ValueError su risposta non-JSON", raised)

        # --- Test 4: load_prompt() con read_document mockato ---
        fresh_client = LLMClient()
        _MOCK_PROMPT = "# Orchestrator\nSei un orchestratore AI."

        async def _mock_read(path: str) -> str:
            if "orchestrator" in path:
                return _MOCK_PROMPT
            raise FileNotFoundError(f"File non trovato: {path}")

        with patch("src.tools.read_document.read_document", new=_mock_read):
            prompt_text = await fresh_client.load_prompt("orchestrator")
            check("load_prompt() legge il file correttamente", "orchestratore AI" in prompt_text)

            prompt_cached = await fresh_client.load_prompt("orchestrator")
            check("load_prompt() usa la cache alla seconda chiamata", prompt_cached is prompt_text)

            raised_fnf = False
            try:
                await fresh_client.load_prompt("nonexistent")
            except FileNotFoundError:
                raised_fnf = True
            check("load_prompt() solleva FileNotFoundError su file mancante", raised_fnf)

        # --- Test 5: singleton ---
        global _llm_client
        _llm_client = None
        c1 = get_llm_client()
        c2 = get_llm_client()
        check("get_llm_client() restituisce sempre la stessa istanza", c1 is c2)

        print(f"\nRisultato: {passed} OK, {failed} FAIL")

    asyncio.run(_run_tests())
    sys.exit(0 if failed == 0 else 1)
