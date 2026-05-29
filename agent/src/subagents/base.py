"""Base class per tutti i subagents.

I subagents sono stateless: ogni chiamata porta con sé system prompt,
task da eseguire e tool disponibili. Rispondono sempre in JSON:
    {"success": true, "output": "..."}
"""
import json
import logging
from abc import ABC

logger = logging.getLogger(__name__)


def _format_context(context: dict) -> str:
    if not context:
        return "(nessun contesto)"
    try:
        return json.dumps(context, ensure_ascii=False, indent=2)
    except Exception:
        return str(context)


def _format_schema(schema: dict, indent: int = 0) -> str:
    """Formatta output_schema come blocco leggibile per il prompt.

    I valori dict vengono resi annidati; i valori stringa vengono racchiusi in <>.
    """
    pad = "  " * indent
    lines = ["{"]
    for key, val in schema.items():
        if isinstance(val, dict):
            nested = _format_schema(val, indent + 1)
            lines.append(f'{pad}  "{key}": {nested}')
        else:
            lines.append(f'{pad}  "{key}": <{val}>')
    lines.append(f"{pad}}}")
    return "\n".join(lines)


class BaseSubagent(ABC):
    """
    Classe base per tutti i subagents di Vesper.

    Ogni sottoclasse imposta il proprio prompt_name e tool_whitelist come
    attributi di classe, e chiama super().__init__(prompt_name) nel proprio __init__.

    Attributes:
        prompt_name:    Nome del file prompt in src/prompts/ (senza .md).
        tool_whitelist: Nomi dei tool che questo subagent può usare.
        _config:        SubagentConfig risolto da SUBAGENT_CONFIGS al momento
                        della costruzione.
    """

    tool_whitelist: list[str] = []

    def __init__(self, prompt_name: str) -> None:
        self.prompt_name = prompt_name
        try:
            from src.core.llm.client import get_llm_client
        except ImportError:
            from llm.client import get_llm_client  # type: ignore[no-redef]
        self._llm = get_llm_client()

        try:
            from src.subagents.config import DEFAULT_SUBAGENT_CONFIG, SUBAGENT_CONFIGS
        except ImportError:
            from subagents.config import DEFAULT_SUBAGENT_CONFIG, SUBAGENT_CONFIGS  # type: ignore[no-redef]
        self._config = SUBAGENT_CONFIGS.get(prompt_name.upper(), DEFAULT_SUBAGENT_CONFIG)

    async def run(self, task: str, context: dict, tools: list[str]) -> dict:
        """
        Esegue il subagent sul task fornito.

        Args:
            task:    Descrizione del task da eseguire.
            context: Informazioni rilevanti fornite dal Manager (non l'intero RequestContext).
            tools:   Nomi dei tool disponibili per questo invocation (sottoinsieme di tool_whitelist).

        Returns:
            {"success": bool, "output": any}
        """
        cfg = self._config

        # 1. Carica system prompt e appende lo schema di output dalla config
        try:
            system_prompt = await self._llm.load_prompt(self.prompt_name)
        except FileNotFoundError:
            logger.error(
                "%s: prompt '%s' non trovato nel vault",
                self.__class__.__name__, self.prompt_name,
            )
            return {"success": False, "output": f"prompt '{self.prompt_name}' non trovato"}

        system_prompt = (
            system_prompt
            + "\n\n## Output\n\n"
            "Rispondi SOLO con un oggetto JSON. Nessun testo fuori dal JSON.\n\n"
            + _format_schema(cfg.output_schema)
        )

        # 2. Costruisce la sezione tool
        try:
            from src.tools import TOOL_DESCRIPTIONS
        except ImportError:
            from tools import TOOL_DESCRIPTIONS  # type: ignore[no-redef]
        tool_lines = [
            f"- {t}: {TOOL_DESCRIPTIONS.get(t, '(nessuna descrizione)')}"
            for t in tools
        ]
        tools_section = (
            "\n".join(tool_lines) if tool_lines else "(nessun tool disponibile in questo invocation)"
        )

        # 3. Filtra il contesto ai soli campi definiti in context_fields
        if cfg.context_fields:
            filtered = {k: v for k, v in context.items() if k in cfg.context_fields and v not in (None, "", [], {})}
            dropped = [k for k in context if k not in cfg.context_fields]
            logger.debug(
                "%s: context %s → inclusi %s%s",
                self.__class__.__name__,
                list(context.keys()),
                list(filtered.keys()),
                f" | esclusi {dropped}" if dropped else "",
            )
        else:
            filtered = context

        # 3.5 Ottimizza il context prima di serializzarlo nel prompt
        try:
            from src.core.token_optimizer import optimize_context_dict
        except ImportError:
            from core.token_optimizer import optimize_context_dict  # type: ignore[no-redef]
        filtered = optimize_context_dict(filtered, self._llm.n_ctx)

        # 4. Costruisce user_message
        user_message = (
            f"## Task\n{task}\n\n"
            f"## Tool disponibili\n{tools_section}\n\n"
            f"## Contesto\n{_format_context(filtered)}"
        )

        # 5. Chiama LLM con i parametri di sampling del ruolo
        try:
            result = await self._llm.complete(
                system_prompt,
                user_message,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repeat_penalty=cfg.repeat_penalty,
                thinking=cfg.thinking,
            )
        except Exception as exc:
            logger.error(
                "%s: errore LLM per task '%.100s' — %s: %s",
                self.__class__.__name__, task, type(exc).__name__, exc,
            )
            return {"success": False, "output": str(exc)}

        # 6. Valida struttura {"success": bool, "output": any}
        if not isinstance(result.get("success"), bool) or "output" not in result:
            logger.warning(
                "%s: risposta LLM senza struttura attesa per task '%.100s': %r",
                self.__class__.__name__, task, result,
            )
            return {"success": False, "output": "formato risposta non valido"}

        logger.info(
            "%s: task='%.100s' success=%s",
            self.__class__.__name__, task, result["success"],
        )
        return result


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
    os.environ.setdefault("VAULT_PATH", str(Path(__file__).resolve().parent.parent.parent / "vault"))

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

    # Concrete subclass per i test
    class _MockSubagent(BaseSubagent):
        tool_whitelist = ["vault_search", "web_search"]

        def __init__(self) -> None:
            super().__init__("test_prompt")

    async def _run_tests() -> None:
        print("\n=== BaseSubagent ===\n")

        agent = _MockSubagent()

        # Patch sia load_prompt che complete sul client LLM
        with patch.object(agent._llm, "load_prompt", new=AsyncMock(return_value="Sei un subagent.")):

            # Test 1: run() con risposta valida
            with patch.object(
                agent._llm,
                "complete",
                new=AsyncMock(return_value={"success": True, "output": "risultato ok"}),
            ):
                result = await agent.run(
                    task="Cerca informazioni su Python asyncio.",
                    context={"query_result": "..."},
                    tools=["vault_search"],
                )
            check("run() con risposta valida", result == {"success": True, "output": "risultato ok"})

            # Test 2: run() con risposta malformata → fallback
            with patch.object(
                agent._llm,
                "complete",
                new=AsyncMock(return_value={"result": "senza struttura attesa"}),
            ):
                result2 = await agent.run("task", {}, [])
            check(
                "run() con risposta malformata restituisce fallback",
                result2 == {"success": False, "output": "formato risposta non valido"},
            )

            # Test 3: run() con eccezione LLM → errore strutturato
            with patch.object(
                agent._llm,
                "complete",
                new=AsyncMock(side_effect=ValueError("connessione rifiutata")),
            ):
                result3 = await agent.run("task", {}, [])
            check("run() con eccezione LLM restituisce success=False", result3["success"] is False)
            check("run() con eccezione LLM include messaggio", "connessione rifiutata" in result3["output"])

        # Test 4: prompt non trovato
        with patch.object(
            agent._llm,
            "load_prompt",
            new=AsyncMock(side_effect=FileNotFoundError("test_prompt.md non trovato")),
        ):
            result4 = await agent.run("task", {}, [])
        check(
            "run() con prompt mancante restituisce success=False",
            result4["success"] is False and "test_prompt" in result4["output"],
        )

        # Test 5: tool descriptions e contesto inclusi nel user_message
        captured_calls: list[dict] = []

        async def _capture_complete(system_prompt: str, user_message: str, **kw) -> dict:
            captured_calls.append({"system": system_prompt, "user": user_message, "kw": kw})
            return {"success": True, "output": "ok"}

        with patch.object(agent._llm, "load_prompt", new=AsyncMock(return_value="sys")):
            with patch.object(agent._llm, "complete", new=_capture_complete):
                await agent.run(
                    task="Cerca dati.",
                    context={"k": "v"},
                    tools=["vault_search", "web_search"],
                )
        call = captured_calls[0]
        msg = call["user"]
        check("user_message contiene il task", "Cerca dati." in msg)
        check("user_message contiene descrizione vault_search", "vault_search" in msg)
        check("user_message contiene descrizione web_search", "web_search" in msg)
        check("user_message contiene il contesto JSON", '"k"' in msg and '"v"' in msg)

        # Test 6: output_schema iniettato nel system_prompt
        sys_prompt = call["system"]
        check("system_prompt contiene '## Output'", "## Output" in sys_prompt)
        check("system_prompt contiene 'success'", '"success"' in sys_prompt)

        # Test 7: parametri di sampling passati a complete()
        kw = call["kw"]
        check("complete() riceve temperature", "temperature" in kw)
        check("complete() riceve top_p", "top_p" in kw)
        check("complete() riceve repeat_penalty", "repeat_penalty" in kw)
        check("complete() riceve thinking", "thinking" in kw)

        # Test 8: contesto filtrato per context_fields (DEFAULT ha context_fields vuoto → passa tutto)
        check(
            "DEFAULT context_fields vuoto: contesto non filtrato",
            '"k"' in msg,
        )

        # Test 9: subagent con context_fields filtrati
        class _FilteredSubagent(BaseSubagent):
            tool_whitelist: list[str] = []

            def __init__(self) -> None:
                super().__init__("researcher")

        filtered_agent = _FilteredSubagent()
        filtered_calls: list[dict] = []

        async def _capture_filtered(system_prompt: str, user_message: str, **kw) -> dict:
            filtered_calls.append({"user": user_message})
            return {"success": True, "output": "ok"}

        with patch.object(filtered_agent._llm, "load_prompt", new=AsyncMock(return_value="sys")):
            with patch.object(filtered_agent._llm, "complete", new=_capture_filtered):
                await filtered_agent.run(
                    task="Ricerca.",
                    context={"request": "query utente", "tool_results": "risultati", "extra": "escluso"},
                    tools=[],
                )
        filtered_msg = filtered_calls[0]["user"]
        check("context filtrato include 'request'", "query utente" in filtered_msg)
        check("context filtrato include 'tool_results' (in RESEARCHER context_fields)", "risultati" in filtered_msg)
        check("context filtrato esclude 'extra'", "escluso" not in filtered_msg)

        print(f"\nRisultato: {passed} OK, {failed} FAIL")

    asyncio.run(_run_tests())
    sys.exit(0 if failed == 0 else 1)
